"""Binary hit / no-hit classifier: P(a PMT sees any light | event params, position)."""

import math
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from nugget.surrogates.FlowMatchLY import FlowMatchLY, _reseed_dataset_rng_in_worker


# --------------------------------------------------------------------------- #
#  Network                                                                     #
# --------------------------------------------------------------------------- #

class HitNet(torch.nn.Module):
    """Residual MLP: context -> one logit."""

    def __init__(self, context_dim, width=256, depth=6, dropout=0.0):
        super().__init__()
        self.inp = torch.nn.Linear(context_dim, width)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width), torch.nn.SiLU(),
                torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
                torch.nn.Linear(width, width),
            ) for _ in range(depth)])
        self.out_norm = torch.nn.LayerNorm(width)
        self.out = torch.nn.Linear(width, 1)

    def forward(self, c):
        h = self.inp(c)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(torch.nn.functional.silu(self.out_norm(h))).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class HitLabelDataset(Dataset):
    """Balanced hit / no-hit samples over (event, PMT) pairs.

    Positives are the parquet's hit rows. Negatives are (event, PMT) pairs that do
    NOT appear for that event, drawn uniformly over the whole geometry. Occupancy is
    ~0.01%, so a faithful sample would be >99.98% negatives and useless for training;
    we sample balanced and record the true prior so the logits can be corrected back
    (see HitClassifier.log_prior_odds).
    """

    def __init__(self, model, parquet_path, geometry_csv_path,
                 num_samples_per_epoch=1_000_000, seed=None, pos_frac=0.5,
                 uniform_energy_zenith=False, n_energy_bins=20, n_coszen_bins=20,
                 n_mult_bins=0, importance_weight=False,
                 filter_vertex_in_domain=True, event_filter=None, verbose=True):
        import pandas as pd

        self.model = model
        self.device = torch.device('cpu')
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.pos_frac = float(pos_frac)

        # ---- geometry: every PMT is a candidate detector location ----
        geo = pd.read_csv(geometry_csv_path,
                          usecols=['string', 'om', 'pmt', 'om_x', 'om_y', 'om_z',
                                   'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'])
        geo['_gidx'] = np.arange(len(geo), dtype=np.int64)
        self._geo_pos = np.ascontiguousarray(
            geo[['om_x', 'om_y', 'om_z']].to_numpy(np.float32))
        self._geo_dir = np.ascontiguousarray(
            geo[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy(np.float32))
        self._n_geo = len(geo)

        df = pd.read_parquet(parquet_path,
                             columns=['run_id', 'event_id', 'string', 'om', 'pmt',
                                      'muon_x', 'muon_y', 'muon_z',
                                      'neutrino_energy', 'zenith', 'azimuth'])

        half = self._domain_half_extent()
        if filter_vertex_in_domain:
            n0 = len(df)
            df = df[(df.muon_x.abs() <= half[0]) & (df.muon_y.abs() <= half[1])
                    & (df.muon_z.abs() <= half[2])]
            if verbose and len(df) < n0:
                print(f"HitLabelDataset: dropped {n0 - len(df):,} row(s) with muon "
                      f"vertex outside {half.tolist()}")

        if event_filter is not None:
            pairs = list(zip(df.run_id.astype(int), df.event_id.astype(int)))
            mask = pd.Series(pairs, index=df.index, dtype=object).isin(set(event_filter))
            df = df[mask.to_numpy()]

        df = df.merge(geo[['string', 'om', 'pmt', '_gidx']],
                      on=['string', 'om', 'pmt'], how='inner', copy=False)
        if len(df) == 0:
            raise ValueError("No usable rows: parquet and geometry CSV do not overlap "
                             "(or every vertex fell outside the domain).")

        # ---- per-EVENT parameters (one entry per event, not per hit) ----
        codes, _ = pd.factorize(
            pd.Series(list(zip(df.run_id.astype(int), df.event_id.astype(int))),
                      dtype=object))
        codes = codes.astype(np.int64)
        self._n_events = int(codes.max()) + 1
        first = np.zeros(self._n_events, dtype=np.int64)
        first[codes[::-1]] = np.arange(len(codes) - 1, -1, -1)   # first row per event
        self._ev_vertex = np.ascontiguousarray(
            df[['muon_x', 'muon_y', 'muon_z']].to_numpy(np.float32)[first])
        self._ev_energy = df.neutrino_energy.to_numpy(np.float32)[first]
        self._ev_zenith = df.zenith.to_numpy(np.float32)[first]
        self._ev_azimuth = df.azimuth.to_numpy(np.float32)[first]

        # ---- hit lookup: one sorted int64 key per hit, (event, PMT) packed ----
        # searchsorted on this beats a Python set both in memory and in speed, and
        # 10.2e9 candidate cells makes enumeration impossible.
        gidx = df['_gidx'].to_numpy(np.int64)
        self._hit_key = np.sort(codes * self._n_geo + gidx)
        self._n_hits = len(self._hit_key)
        # hits per event: the stratification axis, and identical to _ev_hit_count
        self._ev_mult = np.bincount(self._hit_key // self._n_geo,
                                    minlength=self._n_events).astype(np.int64)

        cells = self._n_events * self._n_geo
        self.p_hit = self._n_hits / cells
        self.log_prior_odds = float(np.log(self._n_hits / max(cells - self._n_hits, 1)))

        self.num_samples_per_epoch = int(num_samples_per_epoch)
        self.uniform_energy_zenith = bool(uniform_energy_zenith)
        self.importance_weight = bool(importance_weight) and self.uniform_energy_zenith
        self._n_bins = 0
        self.n_mult_bins = int(n_mult_bins)
        if self.uniform_energy_zenith:
            self._build_bins(int(n_energy_bins), int(n_coszen_bins), self.n_mult_bins)

        if self.importance_weight:
            # Hits of one event occupy a contiguous run of the sorted key array, so a
            # positive can be drawn from a CHOSEN event rather than uniformly overall.
            starts = np.searchsorted(self._hit_key,
                                     np.arange(self._n_events) * self._n_geo)
            ends = np.searchsorted(self._hit_key,
                                   (np.arange(self._n_events) + 1) * self._n_geo)
            self._ev_hit_start = starts.astype(np.int64)
            self._ev_hit_count = self._ev_mult
            # g(ev) for the stratified draw: pick a bin uniformly, then a row in it.
            self._g_event = np.empty(self._n_events, dtype=np.float64)
            for b in range(self._n_bins):
                rows = self._bin_flat[self._bin_starts[b]:
                                      self._bin_starts[b] + self._bin_lens[b]]
                self._g_event[rows] = 1.0 / (self._n_bins * max(len(rows), 1))

        if verbose:
            print(f"HitLabelDataset: {self._n_hits:,} hits, {self._n_events:,} events, "
                  f"{self._n_geo:,} PMTs")
            print(f"  {cells:,} (event, PMT) cells -> occupancy "
                  f"P(hit) = {self.p_hit:.6g} ({100*self.p_hit:.4f}%)")
            if self.uniform_energy_zenith:
                print(f"  stratified over {self._n_bins} non-empty bins"
                      f"{' (incl. multiplicity)' if self.n_mult_bins else ''}; "
                      f"importance_weight={self.importance_weight}")
            print(f"  training at pos_frac={self.pos_frac}; calibrated logits need a "
                  f"shift of log_prior_odds = {self.log_prior_odds:.4f}")

    def _domain_half_extent(self):
        ds = self.model.domain_size
        if isinstance(ds, torch.Tensor):
            ds = ds.tolist() if ds.dim() > 0 else ds.item()
        if isinstance(ds, (tuple, list)) and len(ds) == 2:
            w, h = float(ds[0]), float(ds[1])
            return np.array([w / 2, w / 2, h / 2], dtype=np.float64)
        h = float(ds) / 2.0
        return np.array([h, h, h], dtype=np.float64)

    def _build_bins(self, n_e, n_c, n_m=0):
        """Stratify EVENTS over (log10 energy, cos zenith[, log10 multiplicity]).

        The multiplicity axis buys coverage of faint events, which are otherwise
        ~0.05% of positives because positives are drawn proportional to hit count.
        importance_weight=True corrects the resulting bias back out.
        """
        axes = [np.log10(np.clip(self._ev_energy, 1e-12, None)),
                np.cos(self._ev_zenith)]
        nbins = [int(n_e), int(n_c)]
        if n_m and int(n_m) > 0:
            axes.append(np.log10(np.maximum(self._ev_mult, 1)))
            nbins.append(int(n_m))

        def edges(v, nb):
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                hi = lo + 1e-6
            return np.linspace(lo, hi, nb + 1)

        flat = np.zeros(self._n_events, dtype=np.int64)
        for v, nb in zip(axes, nbins):
            i = np.clip(np.digitize(v, edges(v, nb)) - 1, 0, nb - 1)
            flat = flat * nb + i
        order = np.argsort(flat, kind='stable')
        bounds = np.flatnonzero(np.diff(flat[order])) + 1
        self._bin_flat = np.ascontiguousarray(order, dtype=np.int64)
        self._bin_starts = np.concatenate([[0], bounds]).astype(np.int64)
        self._bin_lens = np.diff(np.concatenate(
            [self._bin_starts, [len(self._bin_flat)]])).astype(np.int64)
        self._n_bins = len(self._bin_starts)

    def _sample_events(self, n):
        if self.uniform_energy_zenith and self._n_bins > 0:
            b = self._rng.integers(0, self._n_bins, size=n)
            lens = self._bin_lens[b]
            offs = (self._rng.random(n) * lens).astype(np.int64)
            np.minimum(offs, lens - 1, out=offs)
            return self._bin_flat[self._bin_starts[b] + offs]
        return self._rng.integers(0, self._n_events, size=n)

    def _is_hit(self, ev, gidx):
        key = ev * self._n_geo + gidx
        pos = np.searchsorted(self._hit_key, key)
        pos = np.minimum(pos, len(self._hit_key) - 1)
        return self._hit_key[pos] == key

    def __len__(self):
        return self.num_samples_per_epoch

    def get_batch(self, indices):
        n = len(np.asarray(indices).reshape(-1))
        n_pos = int(round(n * self.pos_frac))
        n_neg = n - n_pos

        # --- positives: existing hits ---
        if self.importance_weight:
            # stratified event, then one of ITS hits, so both classes share g(ev)
            ev_p = self._sample_events(n_pos)
            k = (self._rng.random(n_pos) * self._ev_hit_count[ev_p]).astype(np.int64)
            np.minimum(k, self._ev_hit_count[ev_p] - 1, out=k)
            hk = self._hit_key[self._ev_hit_start[ev_p] + k]
        else:
            hk = self._hit_key[self._rng.integers(0, self._n_hits, size=n_pos)]
        ev_p, gi_p = hk // self._n_geo, hk % self._n_geo

        # --- negatives: (event, PMT) pairs that are not hits ---
        ev_n = self._sample_events(n_neg)
        gi_n = self._rng.integers(0, self._n_geo, size=n_neg)
        bad = self._is_hit(ev_n, gi_n)
        for _ in range(20):                      # occupancy ~1e-4: converges at once
            k = int(bad.sum())
            if k == 0:
                break
            ev_n[bad] = self._sample_events(k)
            gi_n[bad] = self._rng.integers(0, self._n_geo, size=k)
            bad = self._is_hit(ev_n, gi_n)

        ev = np.concatenate([ev_p, ev_n])
        gi = np.concatenate([gi_p, gi_n])
        y = np.concatenate([np.ones(n_pos, np.float32), np.zeros(n_neg, np.float32)])

        ctx = self.model.build_context(
            torch.from_numpy(self._geo_pos[gi]),
            torch.from_numpy(self._ev_vertex[ev]),
            torch.from_numpy(self._ev_energy[ev]),
            torch.from_numpy(self._ev_zenith[ev]),
            torch.from_numpy(self._ev_azimuth[ev]),
            torch.from_numpy(self._geo_dir[gi]),
        )
        if not self.importance_weight:
            return ctx, torch.from_numpy(y)

        # Correct the stratified event draw g(ev) back to the true distribution:
        #   positives  target p(ev | hit) = n_hits(ev) / n_hits
        #   negatives  target uniform over events = 1 / n_events
        # Normalising each class to mean 1 leaves the class balance (and hence
        # log_prior_odds) untouched.
        w_p = (self._ev_hit_count[ev_p] / self._n_hits) / self._g_event[ev_p]
        w_n = (1.0 / self._n_events) / self._g_event[ev_n]
        w_p /= max(w_p.mean(), 1e-30)
        w_n /= max(w_n.mean(), 1e-30)
        w = np.concatenate([w_p, w_n]).astype(np.float32)
        return ctx, torch.from_numpy(y), torch.from_numpy(w)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        if isinstance(idx, (list, np.ndarray)):
            return self.get_batch(idx)
        c, y = self.get_batch([idx])
        return c[0], y[0]


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class HitClassifier(FlowMatchLY):
    """P(PMT is hit | event params, detector position), with prior correction.

    Subclasses FlowMatchLY purely to reuse `build_context` / `context_dim` and the
    context standardisation, so the feature vector is byte-for-byte the same one the
    light-yield and arrival-time flows consume. The inherited flow-specific methods
    (sample_z, log_prob_z, ...) are not meaningful here and raise.

    Together with a q>=1 flow this forms a hurdle model:
        P(q = 0 | c) = 1 - pi(c)
        p(q = k | c) = pi(c) * p_flow(k | c)     for k >= 1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_prior_odds = 0.0   # set from the dataset at train time
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    # ---------------- network ----------------

    def build_network(self):
        self.net = HitNet(self.context_dim, self.width, self.depth,
                          self.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        if self.reduce_lr_on_plateau:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience, min_lr=self.lr_scheduler_min_lr)
        n = sum(p.numel() for p in self.net.parameters())
        print(f"HitClassifier: context_dim={self.context_dim}, width={self.width}, "
              f"depth={self.depth}, params={n:,}")

    # ---------------- loss / prediction ----------------

    def hit_loss(self, context, labels, weights=None):
        c = self._apply_context_norm(self._prep(context))
        z = self.net(c)
        y = self._prep(labels).reshape(-1)
        if weights is None:
            return self.loss_fn(z, y)
        w = self._prep(weights).reshape(-1)
        per = torch.nn.functional.binary_cross_entropy_with_logits(
            z, y, reduction='none')
        return (per * w).sum() / w.sum().clamp(min=1e-30)

    def predict_hit_logit(self, context, calibrated=True):
        """Logit of P(hit | c).

        With calibrated=True the training-time class balance is undone by adding
        log_prior_odds, so the result reflects the true detector occupancy rather
        than the balanced sample the network was fitted on. This shift is exact for
        a prior change under a fixed likelihood ratio.
        """
        c = self._apply_context_norm(self._prep(context))
        z = self.net(c)
        if calibrated:
            # balanced training => training log-odds are 0, so the correction is a
            # straight addition of the true log prior odds.
            z = z + self.log_prior_odds
        return z

    def predict_hit_prob(self, context, calibrated=True):
        return torch.sigmoid(self.predict_hit_logit(context, calibrated=calibrated))

    def log_prob_hit(self, context, calibrated=True):
        """(log pi, log(1 - pi)) in a numerically stable form."""
        z = self.predict_hit_logit(context, calibrated=calibrated)
        return torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)

    # flow-only API that does not apply here
    def _flow_only(self, *a, **k):
        raise NotImplementedError(
            "HitClassifier models P(hit | c) only; use a FlowMatchLY / FlowMatchATime "
            "instance for the conditional density.")

    sample_z = log_prob_z = transport_to_base = cdf_z = _flow_only
    sample_light_yield = log_prob_light_yield = fm_loss = _flow_only

    # ---------------- normalisers ----------------

    def fit_normalisers(self, dataloader, max_batches=20):
        tot = tot_sq = None
        n = 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ctx = self._prep(batch[0])
            if tot is None:
                tot = torch.zeros(ctx.shape[1], device=self.device, dtype=ctx.dtype)
                tot_sq = torch.zeros(ctx.shape[1], device=self.device, dtype=ctx.dtype)
            tot += ctx.sum(0)
            tot_sq += (ctx ** 2).sum(0)
            n += ctx.shape[0]
        if n == 0:
            raise ValueError("fit_normalisers: dataloader yielded no batches")
        mean = tot / n
        std = torch.sqrt(torch.clamp(tot_sq / n - mean ** 2, min=0.0))
        self.context_mean = mean
        self.context_std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.target_mu, self.target_sigma = 0.0, 1.0   # unused, keeps save/load happy
        print(f"  normalisers: context fitted on {n:,} samples")

    # ---------------- training ----------------

    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                              verbose=True, early_stopping_patience=50, grad_clip=1.0,
                              save_every_n_epochs=None, checkpoint_path=None):
        if self.net is None:
            self.build_network()
        if self.context_mean is None and self.standardize_context:
            self.fit_normalisers(train_dataloader)
        # inherit the true occupancy from the training data
        self.log_prior_odds = float(getattr(train_dataloader.dataset,
                                            'log_prior_odds', self.log_prior_odds))

        if self.lr_schedule == 'onecycle':
            steps = len(train_dataloader) * epochs
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.learning_rate, total_steps=steps,
                pct_start=self.warmup_frac)
            print(f"  OneCycle: max_lr={self.learning_rate:g}, {steps:,} steps")

        best_val, patience = float('inf'), 0
        for epoch in range(epochs):
            self.net.train()
            tl, nb = 0.0, 0
            t0 = time.time()
            for batch in train_dataloader:
                ctx, y = batch[0], batch[1]
                w = batch[2] if len(batch) > 2 else None
                self.optimizer.zero_grad()
                loss = self.hit_loss(ctx, y, weights=w)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                self.optimizer.step()
                if self.lr_schedule == 'onecycle' and self.lr_scheduler is not None:
                    if self.lr_scheduler.last_epoch + 1 < self.lr_scheduler.total_steps:
                        self.lr_scheduler.step()
                tl += loss.item(); nb += 1
            tl /= max(nb, 1)
            self.train_losses.append(tl)

            vl = acc = auc = None
            if val_dataloader is not None:
                self.net.eval()
                vs, vn = 0.0, 0
                probs, ys = [], []
                with torch.no_grad():
                    for batch in val_dataloader:
                        ctx, y = batch[0], batch[1]
                        w = batch[2] if len(batch) > 2 else None
                        vs += self.hit_loss(ctx, y, weights=w).item(); vn += 1
                        # uncalibrated: matches the balanced validation sample
                        probs.append(torch.sigmoid(
                            self.predict_hit_logit(ctx, calibrated=False)).cpu())
                        ys.append(y.reshape(-1).cpu())
                vl = vs / max(vn, 1)
                self.val_losses.append(vl)
                p = torch.cat(probs).numpy(); yy = torch.cat(ys).numpy()
                acc = float(((p > 0.5) == (yy > 0.5)).mean())
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = float(roc_auc_score(yy, p))
                except Exception:
                    auc = float('nan')

                if vl < best_val:
                    best_val, patience = vl, 0
                    self.best_state_dict = {k: v.detach().clone()
                                            for k, v in self.net.state_dict().items()}
                else:
                    patience += 1

            if (self.lr_schedule != 'onecycle' and self.reduce_lr_on_plateau
                    and self.lr_scheduler is not None):
                self.lr_scheduler.step(vl if vl is not None else tl)

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                msg = f"Epoch {epoch+1}/{epochs}  train {tl:.5f}"
                if vl is not None:
                    msg += f"  val {vl:.5f}  acc {acc:.4f}  auc {auc:.4f}"
                print(msg + f"  ({time.time()-t0:.1f}s)", flush=True)

            if val_dataloader is not None and patience >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (best val {best_val:.5f})")
                break

            if save_every_n_epochs and ((epoch + 1) % save_every_n_epochs == 0
                                        or epoch + 1 == epochs):
                d = os.path.dirname(checkpoint_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self.save_model(checkpoint_path, state=self.best_state_dict)

        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)
            if verbose:
                print(f"Restored best weights (val {best_val:.5f})")
        self.is_trained = True
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses}

    # ---------------- calibration diagnostics ----------------

    @torch.no_grad()
    def reliability(self, context, labels, n_bins=12, calibrated=False):
        """Reliability table. Use calibrated=False against a BALANCED sample.

        Accuracy is a useless metric at 0.01% occupancy (predicting 'never hit'
        scores 99.99%), so judge this model by calibration and AUC instead.
        """
        p = self.predict_hit_prob(context, calibrated=calibrated).cpu().numpy()
        y = self._prep(labels).reshape(-1).cpu().numpy()
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        rows = []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 10:
                continue
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
        return rows

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        super().save_model(filepath, state=state)
        ck = torch.load(filepath, map_location='cpu', weights_only=False)
        ck['log_prior_odds'] = self.log_prior_odds
        torch.save(ck, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        self.log_prior_odds = float(ck.get('log_prior_odds', 0.0))
        return super().load_model(filepath)

    # ---------------- dataloaders ----------------

    def create_hit_parquet_dataloader(self, parquet_path, geometry_csv_path,
                                      num_samples_per_epoch=1_000_000, batch_size=4096,
                                      shuffle=True, num_workers=0, seed=None,
                                      pos_frac=0.5, uniform_energy_zenith=False,
                                      n_energy_bins=20, n_coszen_bins=20,
                                      n_mult_bins=0, importance_weight=False,
                                      filter_vertex_in_domain=True,
                                      test_save_path=None, test_frac=0.1,
                                      split_seed=None, pin_memory=None):
        import pandas as pd

        train_filter = None
        if test_save_path is not None:
            ev = pd.read_parquet(parquet_path, columns=['run_id', 'event_id'])
            pairs = list(zip(ev.run_id.astype(int), ev.event_id.astype(int)))
            uniq = sorted(set(pairs))
            rng = np.random.default_rng(split_seed if split_seed is not None else seed)
            perm = rng.permutation(len(uniq))
            n_test = int(round(test_frac * len(uniq)))
            test_events = {uniq[i] for i in perm[:n_test]}
            train_filter = {uniq[i] for i in perm[n_test:]}
            mask = pd.Series(pairs, dtype=object).isin(test_events).to_numpy()
            d = os.path.dirname(test_save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            pd.read_parquet(parquet_path).loc[mask].to_parquet(test_save_path,
                                                               index=False)
            print(f"held out {len(test_events)}/{len(uniq)} events "
                  f"({int(mask.sum()):,} rows) -> {test_save_path}")

        ds = HitLabelDataset(
            self, parquet_path, geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch, seed=seed, pos_frac=pos_frac,
            uniform_energy_zenith=uniform_energy_zenith,
            n_energy_bins=n_energy_bins, n_coszen_bins=n_coszen_bins,
            n_mult_bins=n_mult_bins, importance_weight=importance_weight,
            filter_vertex_in_domain=filter_vertex_in_domain,
            event_filter=train_filter)

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        kw = dict(dataset=ds, batch_size=None,
                  sampler=BatchSampler(base, batch_size=batch_size, drop_last=False),
                  num_workers=num_workers, pin_memory=pin_memory)
        if num_workers > 0:
            kw['worker_init_fn'] = _reseed_dataset_rng_in_worker
            kw['persistent_workers'] = True
        return DataLoader(**kw)

    def create_hit_parquet_val_dataloader(self, parquet_path, geometry_csv_path,
                                          num_samples_per_epoch=200_000,
                                          batch_size=4096, num_workers=0, seed=0, **kw):
        return self.create_hit_parquet_dataloader(
            parquet_path=parquet_path, geometry_csv_path=geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, seed=seed,
            test_save_path=None, **kw)

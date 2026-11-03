"""Transformer classifier P(signal flavour | detector response), e.g. nu_mu CC against nu_e and
nu_tau, on the PMT tokens and encoder of FlowMatchReco."""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from nugget.surrogates.FlowMatchLY import _reseed_dataset_rng_in_worker, _event_split
from nugget.surrogates.FlowMatchReco import (
    FEATURE_SETS, DetectorGeometry, PMTEncoder, RecoFlowDataset, event_tokens, pad_events,
)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class FlavorDataset(Dataset):
    """Events of several parquets as padded PMT tokens, labelled 1 (signal) or 0 (background).

    A draw is signal with probability signal_frac, then picks a source of that class by
    source_weights. uniform_energy_zenith flattens every source in (log10 E, cos zen) over the
    bins all of them populate, so the classes share those priors. Remaining keywords go to
    RecoFlowDataset (pulses, trigger, sub-geometry masking, tokens).
    """

    def __init__(self, parquets, signal, geometry_csv_path, num_events_per_epoch=None,
                 seed=None, signal_frac=0.5, source_weights=None, uniform_energy_zenith=False,
                 n_energy_bins=20, n_coszen_bins=20, event_filters=None, max_redraws=1000,
                 verbose=True, **reco_kw):
        signal = [signal] if isinstance(signal, str) else list(signal)
        self.names = list(parquets)
        source_weights = source_weights or {}
        unknown = (set(signal) | set(source_weights)) - set(self.names)
        if unknown:
            raise ValueError(f"{sorted(unknown)} not among the parquets {self.names}")
        if len(set(signal)) == len(self.names):
            raise ValueError("at least one parquet must be background")
        if not 0.0 < signal_frac < 1.0:
            raise ValueError("signal_frac must be in (0, 1)")
        self.is_signal = np.array([n in signal for n in self.names])
        w = np.array([float(source_weights.get(n, 1.0)) for n in self.names])
        self._class_src = [np.flatnonzero(self.is_signal == c) for c in (False, True)]
        if (w < 0).any() or any(w[s].sum() <= 0 for s in self._class_src):
            raise ValueError("source_weights must be >= 0 with a positive sum per class")
        self._class_p = [w[s] / w[s].sum() for s in self._class_src]
        self.signal_frac = float(signal_frac)
        self.max_redraws = int(max_redraws)
        self._seed = seed

        geometry = (geometry_csv_path if isinstance(geometry_csv_path, DetectorGeometry)
                    else DetectorGeometry.from_csv(geometry_csv_path))
        event_filters = event_filters or {}
        self.sources = []
        for n in self.names:
            if verbose:
                print(f"FlavorDataset [{n}] ({'signal' if n in signal else 'background'})")
            self.sources.append(RecoFlowDataset(
                parquets[n], geometry, event_filter=event_filters.get(n), verbose=verbose,
                **reco_kw))
        self._rng = np.random.default_rng(seed)
        self.num_samples_per_epoch = (int(num_events_per_epoch) if num_events_per_epoch
                                      else sum(s._n_events for s in self.sources))

        self.uniform_energy_zenith = bool(uniform_energy_zenith)
        if self.uniform_energy_zenith:
            self._build_bins(int(n_energy_bins), int(n_coszen_bins), verbose)

        self.config = {**self.sources[0].config, 'sources': list(self.names),
                       'signal': signal, 'signal_frac': self.signal_frac,
                       'source_weights': dict(zip(self.names, w.tolist())),
                       'uniform_energy_zenith': self.uniform_energy_zenith}

    @property
    def _rng(self):
        return self._shared_rng

    @_rng.setter
    def _rng(self, rng):
        # one stream for all sources, so reseeding a DataLoader worker reseeds every source
        self._shared_rng = rng
        for s in self.sources:
            s._rng = rng

    def _build_bins(self, n_e, n_c, verbose=True):
        def index(vs, nb):
            lo, hi = min(float(v.min()) for v in vs), max(float(v.max()) for v in vs)
            edges = np.linspace(lo, hi if hi > lo else lo + 1e-6, nb + 1)
            return [np.clip(np.digitize(v, edges) - 1, 0, nb - 1) for v in vs]

        ei = index([s._log10e for s in self.sources], n_e)
        ci = index([np.cos(s._zenith) for s in self.sources], n_c)
        flat = [e * n_c + c for e, c in zip(ei, ci)]
        counts = np.stack([np.bincount(f, minlength=n_e * n_c) for f in flat])
        # only bins every source populates, else the classes' binned priors differ
        self._bins = np.flatnonzero((counts > 0).all(0))
        if len(self._bins) == 0:
            raise ValueError("the sources share no (log10 E, cos zen) bin")
        self._bin_order = [np.argsort(f, kind='stable') for f in flat]
        self._bin_start = np.concatenate([np.zeros((len(flat), 1), np.int64),
                                          np.cumsum(counts, 1)[:, :-1]], axis=1)
        self._bin_len = counts
        if verbose:
            kept = counts[:, self._bins].sum(1) / counts.sum(1)
            print(f"FlavorDataset: uniform in (log10 E, cos zen) over {len(self._bins)}/"
                  f"{n_e * n_c} shared bins, holding "
                  + ", ".join(f"{n} {k:.0%}" for n, k in zip(self.names, kept)))

    def _sample_events(self, s, n):
        rng = self._rng
        if not self.uniform_energy_zenith:
            return rng.integers(0, self.sources[s]._n_events, size=n)
        b = self._bins[rng.integers(0, len(self._bins), size=n)]
        lens = self._bin_len[s, b]
        offs = np.minimum((rng.random(n) * lens).astype(np.int64), lens - 1)
        return self._bin_order[s][self._bin_start[s, b] + offs]

    def _sample_sources(self, n):
        labels = (self._rng.random(n) < self.signal_frac).astype(np.int64)
        src = np.empty(n, dtype=np.int64)
        for c in (0, 1):
            m = labels == c
            src[m] = self._rng.choice(self._class_src[c], size=int(m.sum()), p=self._class_p[c])
        return labels, src

    def __len__(self):
        return self.num_samples_per_epoch

    def get_batch(self, indices, masked=True):
        """(feats, mask, glob, label, source index, log10 E)."""
        n = len(np.asarray(indices).reshape(-1))
        labels, src = self._sample_sources(n)
        out, le = [], np.empty(n, dtype=np.float32)
        for i, s in enumerate(src):
            ds = self.sources[s]
            # an event that loses every pulse or fails the trigger would not have been
            # recorded; redraw from the same source so the label is kept
            for _ in range(self.max_redraws):
                e = self._sample_events(s, 1)[0]
                tok = ds.tokens(e, masked=masked)
                if tok is not None:
                    break
            else:
                raise RuntimeError(f"no [{self.names[s]}] event survived masking/trigger in "
                                   f"{self.max_redraws} draws; use more strings")
            out.append(tok)
            le[i] = ds._log10e[e]
        feats, mask, glob = pad_events(out)
        return (feats, mask, glob, torch.from_numpy(labels.astype(np.float32)),
                torch.from_numpy(src), torch.from_numpy(le))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        if isinstance(idx, (list, np.ndarray)):
            return self.get_batch(idx)
        return self.get_batch([idx])


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class FlavorNet(torch.nn.Module):
    """PMTEncoder condition vector -> MLP -> one logit."""

    def __init__(self, encoder, cond_dim, width=256, depth=2, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        layers, w = [], cond_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(w, width), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            w = width
        self.head = torch.nn.Sequential(*layers, torch.nn.Linear(w, 1))

    def forward(self, feats, mask, glob):
        return self.head(self.encoder(feats, mask, glob)).squeeze(-1)


class FlavorClassifier:
    """P(signal | PMT pulses) from the FlowMatchReco transformer encoder and an MLP head."""

    _CONFIG_KEYS = ('feature_set', 'd_model', 'encoder_depth', 'n_heads', 'ffn_dim',
                    'cond_dim', 'encoder_dropout', 'head_width', 'head_depth', 'head_dropout',
                    'length_scale', 'time_scale', 'time_transform')

    def __init__(self, device=None, feature_set='essential', d_model=96, encoder_depth=20,
                 n_heads=4, ffn_dim=512, cond_dim=256, encoder_dropout=0.0, head_width=256,
                 head_depth=2, head_dropout=0.0, length_scale=500.0, time_scale=1e4,
                 time_transform='linear', learning_rate=2e-4, weight_decay=1e-5,
                 lr_schedule='warmup_cosine', warmup_steps=2000, warmup_frac=0.1, min_lr=1e-5,
                 amp=False):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        if time_transform not in ('linear', 'asinh'):
            raise ValueError("time_transform must be 'linear' (thesis) or 'asinh'")
        if lr_schedule not in ('warmup_cosine', 'onecycle', 'constant'):
            raise ValueError("lr_schedule must be 'warmup_cosine', 'onecycle' or 'constant'")
        self.device = device if device is not None else torch.device('cpu')
        self.feature_set = feature_set
        self.d_model, self.encoder_depth, self.n_heads = d_model, encoder_depth, n_heads
        self.ffn_dim, self.cond_dim, self.encoder_dropout = ffn_dim, cond_dim, encoder_dropout
        self.head_width, self.head_depth, self.head_dropout = head_width, head_depth, head_dropout
        self.length_scale, self.time_scale = length_scale, time_scale
        self.time_transform = time_transform
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.lr_schedule, self.warmup_steps = lr_schedule, warmup_steps
        self.warmup_frac, self.min_lr = warmup_frac, min_lr
        self.amp = amp

        self.data_config = None
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_losses, self.val_losses, self.metrics = [], [], []
        self.best_state_dict = None
        self.is_trained = False

    # ---------------- dtype / device plumbing ----------------

    @property
    def param_dtype(self):
        return torch.float32 if self.net is None else next(self.net.parameters()).dtype

    def _prep(self, x):
        return x.to(device=self.device, dtype=self.param_dtype)

    def _autocast(self):
        dev = torch.device(self.device).type
        return torch.autocast(dev, dtype=torch.bfloat16, enabled=bool(self.amp) and dev == 'cuda')

    # ---------------- network ----------------

    def build_network(self):
        encoder = PMTEncoder(self.feature_set, self.d_model, self.encoder_depth, self.n_heads,
                             self.ffn_dim, self.cond_dim, self.encoder_dropout,
                             self.length_scale, self.time_scale, self.time_transform)
        # explicit float32: nugget sets a float64 default dtype, far too slow for this
        self.net = FlavorNet(encoder, self.cond_dim, self.head_width, self.head_depth,
                             self.head_dropout).to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        n_enc = sum(p.numel() for p in encoder.parameters())
        n_head = sum(p.numel() for p in self.net.head.parameters())
        print(f"FlavorClassifier: {self.feature_set} features, encoder d_model={self.d_model} "
              f"depth={self.encoder_depth} ({n_enc:,} params), head width={self.head_width} "
              f"depth={self.head_depth} ({n_head:,} params)")

    def _build_scheduler(self, total_steps):
        if self.lr_schedule == 'onecycle':
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.learning_rate, total_steps=total_steps,
                pct_start=self.warmup_frac)
        elif self.lr_schedule == 'warmup_cosine':
            warm, floor = max(int(self.warmup_steps), 1), self.min_lr / self.learning_rate

            def factor(step):
                if step < warm:
                    return (step + 1) / warm
                p = min((step - warm) / max(total_steps - warm, 1), 1.0)
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        else:
            self.lr_scheduler = None

    def _scheduler_step(self):
        s = self.lr_scheduler
        if s is None:
            return
        if self.lr_schedule == 'onecycle' and s.last_epoch + 1 >= s.total_steps:
            return
        s.step()

    def tokenize(self, events, geometry, om_active=None):
        """List of (tok, t, q) pulse arrays, tok indexing `geometry` PMTs -> padded
        (feats, mask, glob), tokenised as in training (features, silent tokens, max_tokens)."""
        cfg = self.data_config or {}
        return pad_events([
            event_tokens(tok, t, q, geometry, cfg.get('max_tokens'), self.feature_set,
                         cfg.get('silent_tokens', False), om_active)
            for tok, t, q in events])

    # ---------------- prediction / loss ----------------

    def logits(self, feats, mask, glob):
        """(B,) signal logits at the training signal_frac."""
        with self._autocast():
            z = self.net(self._prep(feats), mask.to(self.device), self._prep(glob))
        return z.to(self.param_dtype)

    def predict_logit(self, feats, mask, glob, signal_prior=None):
        """Logit of P(signal | event). signal_prior (scalar or per event) replaces the
        training signal_frac; the shift is exact at a fixed likelihood ratio."""
        self.net.eval()
        z = self.logits(feats, mask, glob)
        if signal_prior is not None:
            f = (self.data_config or {}).get('signal_frac', 0.5)
            p = torch.as_tensor(signal_prior, device=z.device, dtype=z.dtype)
            z = z + torch.logit(p) - math.log(f / (1.0 - f))
        return z

    def predict_proba(self, feats, mask, glob, signal_prior=None):
        return torch.sigmoid(self.predict_logit(feats, mask, glob, signal_prior))

    def loss(self, feats, mask, glob, label):
        return F.binary_cross_entropy_with_logits(self.logits(feats, mask, glob),
                                                  self._prep(label).reshape(-1))

    @torch.no_grad()
    def predict_dataloader(self, dataloader, max_batches=None):
        """Numpy logit, label, source index and log10 E over (up to max_batches of) a loader."""
        self.net.eval()
        out = {k: [] for k in ('logit', 'label', 'source', 'log10_energy')}
        for i, (feats, mask, glob, y, s, le) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            out['logit'].append(self.logits(feats, mask, glob).float().cpu())
            out['label'].append(y)
            out['source'].append(s)
            out['log10_energy'].append(le)
        return {k: torch.cat(v).numpy() for k, v in out.items()}

    def evaluate(self, dataloader, max_batches=None):
        """BCE, accuracy and ROC AUC: overall, signal vs each background source (auc_<name>),
        and the mean over half-decade energy bins (macro_auc)."""
        from sklearn.metrics import roc_auc_score
        r = self.predict_dataloader(dataloader, max_batches)
        z = r['logit'].astype(np.float64)
        sig, src, le = r['label'] > 0.5, r['source'], r['log10_energy']

        def auc(m):
            return (float(roc_auc_score(sig[m], z[m])) if 0 < sig[m].sum() < m.sum()
                    else float('nan'))

        out = {'bce': float(np.mean(np.logaddexp(0.0, z) - sig * z)),
               'accuracy': float(((z > 0) == sig).mean()),
               'auc': auc(np.ones_like(sig))}
        names = getattr(dataloader.dataset, 'names', None)
        for k in np.unique(src[~sig]):
            out[f'auc_{names[k] if names else k}'] = auc(sig | (src == k))
        edges = np.arange(np.floor(2 * le.min()) / 2, le.max() + 0.5, 0.5)
        b = np.digitize(le, edges)
        per_bin = [auc(b == k) for k in np.unique(b)
                   if (sig & (b == k)).sum() >= 10 and (~sig & (b == k)).sum() >= 10]
        out['macro_auc'] = float(np.mean(per_bin)) if per_bin else float('nan')
        out['n_events'] = int(len(z))
        return out

    # ---------------- training ----------------

    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                              verbose=True, early_stopping_patience=20, grad_clip=3.0,
                              save_every_n_epochs=None, checkpoint_path=None, val_batches=None):
        """Early stopping and the best weights on validation BCE; validation AUCs per epoch
        land in self.metrics."""
        if self.net is None:
            self.build_network()
        cfg = getattr(train_dataloader.dataset, 'config', None)
        if cfg is not None:
            self.data_config = dict(cfg)
        self._build_scheduler(len(train_dataloader) * epochs)

        best_val, patience = float('inf'), 0
        for epoch in range(epochs):
            self.net.train()
            t0 = time.time()
            tl, nb = 0.0, 0
            for batch in train_dataloader:
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss(*batch[:4])
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                self.optimizer.step()
                self._scheduler_step()
                tl += loss.item()
                nb += 1
            tl /= max(nb, 1)
            self.train_losses.append(tl)

            m = None
            if val_dataloader is not None:
                m = self.evaluate(val_dataloader, max_batches=val_batches)
                self.val_losses.append(m['bce'])
                self.metrics.append({'epoch': epoch + 1, **m})
                if m['bce'] < best_val:
                    best_val, patience = m['bce'], 0
                    self.best_state_dict = {k: v.detach().clone()
                                            for k, v in self.net.state_dict().items()}
                else:
                    patience += 1

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}  train {tl:.5f}"
                if m is not None:
                    per = ", ".join(f"{k[4:]} {v:.4f}" for k, v in m.items()
                                    if k.startswith('auc_'))
                    msg += (f"  val {m['bce']:.5f}  acc {m['accuracy']:.4f}  "
                            f"auc {m['auc']:.4f} ({per})  macro {m['macro_auc']:.4f}")
                print(msg + f"  lr {self.optimizer.param_groups[0]['lr']:.2e}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

            if val_dataloader is not None and patience >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (best val {best_val:.5f})")
                break

            if save_every_n_epochs and checkpoint_path and (
                    (epoch + 1) % save_every_n_epochs == 0 or epoch + 1 == epochs):
                d = os.path.dirname(checkpoint_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self.save_model(checkpoint_path, state=self.best_state_dict)

        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)
            if verbose:
                print(f"Restored best weights (val {best_val:.5f})")
        self.is_trained = True
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses,
                'metrics': self.metrics}

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        torch.save({
            'config': {k: getattr(self, k) for k in self._CONFIG_KEYS},
            'net_state_dict': state if state is not None else self.net.state_dict(),
            'data_config': self.data_config,
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
            'metrics': self.metrics,
        }, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        for k, v in ck['config'].items():
            setattr(self, k, v)
        for k in ('data_config', 'train_losses', 'val_losses', 'metrics'):
            setattr(self, k, ck.get(k, getattr(self, k)))
        self.build_network()
        self.net.load_state_dict(ck['net_state_dict'])
        self.is_trained = True
        return self

    # ---------------- dataloaders ----------------

    def create_flavor_dataloader(self, parquets, signal, geometry_csv_path,
                                 num_events_per_epoch=None, batch_size=128, shuffle=True,
                                 num_workers=0, seed=None, signal_frac=0.5, source_weights=None,
                                 uniform_energy_zenith=False, n_energy_bins=20,
                                 n_coszen_bins=20, test_save_paths=None, test_frac=0.1,
                                 split_seed=None, event_frac=None, pin_memory=None, **reco_kw):
        """Event-level loader over {name: parquet}; `signal` names the signal source(s), the
        rest are background. test_save_paths {name: path} holds out test_frac of each
        source's events; event_frac (scalar or {name: frac}) subsamples the files. Remaining
        keywords go to RecoFlowDataset (pulses, trigger, sub-geometry masking, tokens)."""
        filters = {}
        for n, path in parquets.items():
            frac = event_frac.get(n) if isinstance(event_frac, dict) else event_frac
            filters[n] = _event_split(
                path, frac, None if test_save_paths is None else test_save_paths[n], test_frac,
                split_seed if split_seed is not None else seed)
        ds = FlavorDataset(
            parquets, signal, geometry_csv_path, num_events_per_epoch=num_events_per_epoch,
            seed=seed, signal_frac=signal_frac, source_weights=source_weights,
            uniform_energy_zenith=uniform_energy_zenith, n_energy_bins=n_energy_bins,
            n_coszen_bins=n_coszen_bins, event_filters=filters, feature_set=self.feature_set,
            **reco_kw)

        if pin_memory is None:
            pin_memory = torch.device(self.device).type == 'cuda'
        base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        kw = dict(dataset=ds, batch_size=None,
                  sampler=BatchSampler(base, batch_size=batch_size, drop_last=False),
                  num_workers=num_workers, pin_memory=pin_memory)
        if num_workers > 0:
            kw['worker_init_fn'] = _reseed_dataset_rng_in_worker
            kw['persistent_workers'] = True
        return DataLoader(**kw)

    def create_flavor_val_dataloader(self, parquets, signal, geometry_csv_path,
                                     num_events_per_epoch=None, batch_size=128, num_workers=0,
                                     seed=0, **kw):
        return self.create_flavor_dataloader(
            parquets=parquets, signal=signal, geometry_csv_path=geometry_csv_path,
            num_events_per_epoch=num_events_per_epoch, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, seed=seed, test_save_paths=None, **kw)

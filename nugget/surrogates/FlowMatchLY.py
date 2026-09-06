"""Conditional flow matching for p(light yield | event params, detector position)."""

import os
import time
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from nugget.surrogates.base_surrogate import Surrogate

LN10 = math.log(10.0)


def _reseed_dataset_rng_in_worker(worker_id):
    """Give each DataLoader worker an independent numpy RNG stream."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if not hasattr(ds, '_rng'):
        return
    parts = [int(info.seed)]
    base = getattr(ds, '_seed', None)
    if base is not None:
        parts.append(int(base))
    ds._rng = np.random.default_rng(parts)


# --------------------------------------------------------------------------- #
#  Network                                                                     #
# --------------------------------------------------------------------------- #

class TimeEmbedding(torch.nn.Module):
    """Sinusoidal embedding of the flow time t in [0, 1]."""

    def __init__(self, dim=64, max_freq=64.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("TimeEmbedding dim must be even")
        self.dim = dim
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), dim // 2))
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        t = t.reshape(-1, 1)
        ang = t * self.freqs.to(t.dtype).unsqueeze(0) * 2.0 * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ResBlock(torch.nn.Module):
    """Pre-activation residual block with FiLM conditioning."""

    def __init__(self, width, cond_dim, dropout=0.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.film = torch.nn.Linear(cond_dim, 2 * width)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, h, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        x = self.norm(h) * (1.0 + scale) + shift
        x = torch.nn.functional.silu(x)
        x = self.fc1(x)
        x = torch.nn.functional.silu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return h + x


class VelocityNet(torch.nn.Module):
    """v_theta(z, t, c): predicted velocity of the 1-D target z at flow time t."""

    def __init__(self, context_dim, width=256, depth=6, time_dim=64,
                 cond_width=256, dropout=0.0):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.ctx = torch.nn.Sequential(
            torch.nn.Linear(context_dim, cond_width), torch.nn.SiLU(),
            torch.nn.Linear(cond_width, cond_width), torch.nn.SiLU(),
        )
        cond_dim = cond_width + time_dim
        self.inp = torch.nn.Linear(1, width)
        self.blocks = torch.nn.ModuleList(
            [ResBlock(width, cond_dim, dropout) for _ in range(depth)])
        self.out_norm = torch.nn.LayerNorm(width)
        self.out = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, z, t, c):
        z = z.reshape(-1, 1)
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        cond = torch.cat([self.ctx(c), self.time_emb(t)], dim=-1)
        h = self.inp(z)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


def _times_row_lengths(series):
    """Photons per row from a `times` column -- the light yield of that PMT.

    Handles the layouts these files come in: a ragged list/array per row, a plain
    scalar per row (already exploded, one photon), or a stringified list.
    """
    obj = series.to_numpy()
    n = len(obj)
    if n == 0:
        return np.zeros(0, np.float32)

    # stringified lists, e.g. '[16659.7, 16702.3]'
    if isinstance(obj[0], (str, bytes, np.str_)):
        import pandas as pd
        s = pd.Series(obj).astype(str).str.strip()
        if s.str.contains('...', regex=False).any():
            raise ValueError(
                "The 'times' column holds TRUNCATED string reprs (they contain "
                "'...'), so the photon counts on disk are already wrong.")
        s = (s.str.strip('[]()').str.replace(',', ' ', regex=False).str.strip())
        return s.str.split().str.len().fillna(0).to_numpy(np.float32)

    # plain numeric column: one photon per row (NaN = none)
    if getattr(obj, 'dtype', None) is not None and obj.dtype.kind in 'fiub':
        return np.isfinite(obj.astype(np.float64)).astype(np.float32)

    try:
        return np.fromiter((0 if x is None else len(x) for x in obj), np.float32, n)
    except TypeError:
        return np.fromiter((0 if x is None else np.size(x) for x in obj),
                           np.float32, n)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class LightYieldFlowDataset(Dataset):
    """Parquet rows -> (context features, light yield). One row = one sample."""

    def __init__(self, model, parquet_path, geometry_csv_path,
                 num_samples_per_epoch=None, seed=None,
                 uniform_energy_zenith=False, n_energy_bins=20, n_coszen_bins=20,
                 filter_vertex_in_domain=True, event_filter=None, verbose=True):
        import pandas as pd

        self.model = model
        self.device = torch.device('cpu')
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        geo = pd.read_csv(geometry_csv_path,
                          usecols=['string', 'om', 'pmt', 'om_x', 'om_y', 'om_z',
                                   'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'])
        base_cols = ['run_id', 'event_id', 'string', 'om', 'pmt',
                     'muon_x', 'muon_y', 'muon_z', 'neutrino_energy',
                     'zenith', 'azimuth']
        # Prefer an explicit 'count' column; otherwise derive the light yield from
        # the length of each row's 'times' list, which is the same quantity. Both
        # parquet engines validate column names before reading, so the failed
        # attempt is cheap.
        try:
            df = pd.read_parquet(parquet_path, columns=base_cols + ['count'])
            counts = df['count'].to_numpy(np.float32)
            src = 'count'
        except (ValueError, KeyError):
            df = pd.read_parquet(parquet_path, columns=base_cols + ['times'])
            counts = _times_row_lengths(df['times'])
            df = df.drop(columns=['times'])
            src = 'len(times)'
        df = df.assign(_ly=counts)
        if verbose:
            print(f"LightYieldFlowDataset: light yield taken from '{src}'")

        half = self._domain_half_extent()
        if filter_vertex_in_domain:
            n0 = len(df)
            df = df[(df.muon_x.abs() <= half[0]) & (df.muon_y.abs() <= half[1])
                    & (df.muon_z.abs() <= half[2])]
            if verbose and len(df) < n0:
                print(f"LightYieldFlowDataset: dropped {n0 - len(df)} row(s) with "
                      f"muon vertex outside {half.tolist()}")

        if event_filter is not None:
            ev_pairs = list(zip(df.run_id.astype(int), df.event_id.astype(int)))
            mask = pd.Series(ev_pairs, index=df.index, dtype=object).isin(set(event_filter))
            df = df[mask.to_numpy()]

        df = df.merge(geo, on=['string', 'om', 'pmt'], how='inner', copy=False)
        # The flow models q >= 1; a zero-photon row would give log10(0 + u) -> -inf.
        n_zero = int((df['_ly'] < 1).sum())
        if n_zero:
            if verbose:
                print(f"LightYieldFlowDataset: dropped {n_zero:,} row(s) with zero "
                      f"light yield (q >= 1 is modelled; zeros belong to the "
                      f"hit/no-hit classifier)")
            df = df[df['_ly'] >= 1]
        if len(df) == 0:
            raise ValueError("No usable rows: parquet and geometry CSV do not overlap "
                             "(or every vertex fell outside the domain).")

        self._point = np.ascontiguousarray(
            df[['om_x', 'om_y', 'om_z']].to_numpy(np.float32))
        self._pmt_direction = np.ascontiguousarray(
            df[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy(np.float32))
        self._muon_pos = np.ascontiguousarray(
            df[['muon_x', 'muon_y', 'muon_z']].to_numpy(np.float32))
        self._energy = df.neutrino_energy.to_numpy(np.float32)
        self._zenith = df.zenith.to_numpy(np.float32)
        self._azimuth = df.azimuth.to_numpy(np.float32)
        self._count = df['_ly'].to_numpy(np.float32)
        codes, _ = pd.factorize(
            pd.Series(list(zip(df.run_id.astype(int), df.event_id.astype(int))),
                      dtype=object))
        self._row_event_code = codes.astype(np.int64)

        self._n_rows = len(self._count)
        self._n_events = int(self._row_event_code.max()) + 1
        # Event -> row indices. Mirrors LLRnet's parquet dataset so this class can be
        # passed straight to vis_tools.plot_nll_landscape as `parquet_dataset`.
        _order = np.argsort(self._row_event_code, kind='stable')
        _codes = self._row_event_code[_order]
        _bounds = np.flatnonzero(np.diff(_codes)) + 1
        self._event_rows = {int(_codes[g[0]]): g for g in np.split(_order, _bounds)}
        self._events = np.fromiter(self._event_rows.keys(), dtype=np.int64)
        self.num_samples_per_epoch = (int(num_samples_per_epoch)
                                      if num_samples_per_epoch else self._n_rows)

        self.uniform_energy_zenith = bool(uniform_energy_zenith)
        self._n_bins = 0
        if self.uniform_energy_zenith:
            self._build_bins(int(n_energy_bins), int(n_coszen_bins))

        if verbose:
            print(f"LightYieldFlowDataset: {self._n_rows:,} rows, "
                  f"{self._n_events:,} events, "
                  f"{self.num_samples_per_epoch:,} samples/epoch")

    def _domain_half_extent(self):
        ds = self.model.domain_size
        if isinstance(ds, torch.Tensor):
            ds = ds.tolist() if ds.dim() > 0 else ds.item()
        if isinstance(ds, (tuple, list)) and len(ds) == 2:
            w, h = float(ds[0]), float(ds[1])
            return np.array([w / 2, w / 2, h / 2], dtype=np.float64)
        h = float(ds) / 2.0
        return np.array([h, h, h], dtype=np.float64)

    def _build_bins(self, n_e, n_c):
        log_e = np.log10(np.clip(self._energy, 1e-12, None))
        cz = np.cos(self._zenith)

        def edges(v, nb):
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                hi = lo + 1e-6
            return np.linspace(lo, hi, nb + 1)

        ei = np.clip(np.digitize(log_e, edges(log_e, n_e)) - 1, 0, n_e - 1)
        ci = np.clip(np.digitize(cz, edges(cz, n_c)) - 1, 0, n_c - 1)
        flat = ei * n_c + ci
        order = np.argsort(flat, kind='stable')
        bounds = np.flatnonzero(np.diff(flat[order])) + 1
        self._bin_flat = np.ascontiguousarray(order, dtype=np.int64)
        self._bin_starts = np.concatenate([[0], bounds]).astype(np.int64)
        self._bin_lens = np.diff(np.concatenate(
            [self._bin_starts, [len(self._bin_flat)]])).astype(np.int64)
        self._n_bins = len(self._bin_starts)

    def _sample_rows(self, n):
        if self.uniform_energy_zenith and self._n_bins > 0:
            b = self._rng.integers(0, self._n_bins, size=n)
            lens = self._bin_lens[b]
            offs = (self._rng.random(n) * lens).astype(np.int64)
            np.minimum(offs, lens - 1, out=offs)
            return self._bin_flat[self._bin_starts[b] + offs]
        return self._rng.integers(0, self._n_rows, size=n)

    def __len__(self):
        return self.num_samples_per_epoch

    def get_batch(self, indices):
        n = len(np.asarray(indices).reshape(-1))
        rows = self._sample_rows(n)
        ctx = self.model.build_context(
            torch.from_numpy(self._point[rows]),
            torch.from_numpy(self._muon_pos[rows]),
            torch.from_numpy(self._energy[rows]),
            torch.from_numpy(self._zenith[rows]),
            torch.from_numpy(self._azimuth[rows]),
            torch.from_numpy(self._pmt_direction[rows]),
        )
        return ctx, torch.from_numpy(self._count[rows])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        if isinstance(idx, (list, np.ndarray)):
            return self.get_batch(idx)
        ctx, q = self.get_batch([idx])
        return ctx[0], q[0]

    def _event_data(self, i, pmt_direction=None):
        """Event-parameter dict for row i (same keys as LLRnet's parquet dataset)."""
        zen = torch.tensor(float(self._zenith[i]))
        azi = torch.tensor(float(self._azimuth[i]))
        st, ct = torch.sin(zen), torch.cos(zen)
        direction = torch.stack([st * torch.cos(azi), st * torch.sin(azi), ct])
        pmt = self._pmt_direction[i] if pmt_direction is None else pmt_direction
        return {
            'position': torch.from_numpy(np.ascontiguousarray(self._muon_pos[i])),
            'energy': torch.tensor(float(self._energy[i])),
            'direction': direction,
            'pmt_direction': torch.from_numpy(
                np.ascontiguousarray(np.asarray(pmt, dtype=np.float32))),
            'zenith': zen,
            'azimuth': azi,
        }

    def count_stats(self):
        """(mu, sigma) of log10(count + U(0,1)), for target standardisation."""
        u = self._rng.random(self._n_rows).astype(np.float32)
        w = np.log10(self._count + u)
        return float(w.mean()), float(w.std())


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class FlowMatchLY(Surrogate):
    """Conditional flow matching model for p(light yield | event params, position)."""

    def __init__(self, device=None, dim=3, domain_size=8000,
                 width=256, depth=6, time_dim=64, cond_width=256, dropout=0.0,
                 learning_rate=1e-3, lr_schedule='onecycle', warmup_frac=0.1,
                 reduce_lr_on_plateau=False, lr_scheduler_patience=20,
                 lr_scheduler_factor=0.5, lr_scheduler_min_lr=1e-6,
                 weight_decay=0.0, sigma_min=1e-4,
                 rich_rel_pos_mode=True, include_vertex_position=True,
                 add_vertex_distance=False, add_distance_from_beam=False,
                 add_dist_long=False, track_dir_is_arrival=False,
                 add_pmt_direction=True, add_pmt_cosangle=False,
                 standardize_context=True, ly_eps=1e-6, **kwargs):
        super().__init__(device=device, dim=dim, domain_size=domain_size)

        self.width, self.depth = width, depth
        self.time_dim, self.cond_width, self.dropout = time_dim, cond_width, dropout
        self.learning_rate = learning_rate
        self.lr_schedule, self.warmup_frac = lr_schedule, warmup_frac
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.weight_decay = weight_decay
        self.sigma_min = sigma_min

        self.rich_rel_pos_mode = rich_rel_pos_mode
        self.include_vertex_position = include_vertex_position
        self.add_vertex_distance = add_vertex_distance
        self.add_distance_from_beam = add_distance_from_beam
        self.add_dist_long = add_dist_long
        self.track_dir_is_arrival = track_dir_is_arrival
        self.add_pmt_direction = add_pmt_direction
        self.add_pmt_cosangle = add_pmt_cosangle
        self.ly_eps = ly_eps

        self.standardize_context = standardize_context
        self.context_mean = None
        self.context_std = None
        self.target_mu = None      # mean of log10(q + u)
        self.target_sigma = None   # std  of log10(q + u)

        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.best_state_dict = None
        self.is_trained = False

    # ---------------- dtype / device plumbing ----------------

    @property
    def param_dtype(self):
        """Weight dtype; nugget may set a float64 default, inputs must match."""
        if self.net is None:
            return torch.get_default_dtype()
        return next(self.net.parameters()).dtype

    def _prep(self, x):
        return x.to(device=self.device, dtype=self.param_dtype)

    # ---------------- context features ----------------

    @property
    def context_dim(self):
        d = 3 + 3 + 1                                   # rel, direction, log10 E
        if not self.rich_rel_pos_mode:
            d += 3                                      # absolute det + vert
        elif self.include_vertex_position:
            d += 3
        if self.add_vertex_distance:
            d += 1
        d += 1                                          # cos_angle
        d += int(self.add_distance_from_beam) + int(self.add_dist_long)
        if self.add_pmt_direction:
            d += 3 + int(self.add_pmt_cosangle)
        return d

    def _norm_divisor(self):
        ds = self.domain_size
        if isinstance(ds, torch.Tensor):
            ds = ds.tolist() if ds.dim() > 0 else ds.item()
        if isinstance(ds, (tuple, list)) and len(ds) == 2:
            return float(ds[0]) / 2.0
        return float(ds) / 2.0

    def build_context(self, points, vertices, energies, zeniths=None, azimuths=None,
                      pmt_directions=None, directions=None):
        """(B,3),(B,3),(B,) + angles or a (B,3) unit vector -> (B, context_dim).

        Pass ``directions`` instead of ``zeniths``/``azimuths`` when the caller already
        holds a unit vector, which avoids a lossy round trip through angles.
        Differentiable throughout.
        """
        pts = points.reshape(-1, 3)
        vert_raw = vertices.reshape(-1, 3).to(pts.dtype)
        E = energies.reshape(-1).to(pts.dtype)
        norm = self._norm_divisor()

        det = pts / norm
        vert = vert_raw / norm
        if directions is not None:
            direction = directions.reshape(-1, 3).to(pts.dtype)
        else:
            if zeniths is None or azimuths is None:
                raise ValueError("build_context needs either (zeniths, azimuths) "
                                 "or directions")
            zen = zeniths.reshape(-1).to(pts.dtype)
            azi = azimuths.reshape(-1).to(pts.dtype)
            st, ct = torch.sin(zen), torch.cos(zen)
            direction = torch.stack([st * torch.cos(azi), st * torch.sin(azi), ct], dim=1)
        log_e = torch.log10(E + self.ly_eps) / 8.0
        rel = det - vert
        vert_dist = torch.linalg.norm(rel, dim=1)
        dir_norm = torch.linalg.norm(direction, dim=1)
        cos_angle = (direction * rel).sum(1) / (dir_norm * vert_dist + 1e-8)

        cols = []
        if self.rich_rel_pos_mode:
            cols += [rel, direction, log_e.unsqueeze(1)]
            if self.include_vertex_position:
                cols.append(vert)
        else:
            cols += [det, vert, direction, log_e.unsqueeze(1)]
        if self.add_vertex_distance:
            cols.append(vert_dist.unsqueeze(1))
        cols.append(cos_angle.unsqueeze(1))
        if self.add_distance_from_beam or self.add_dist_long:
            track_dir = -direction if self.track_dir_is_arrival else direction
            rel_m = pts - vert_raw
            d_long = (rel_m * track_dir).sum(1)
            d_perp = torch.linalg.norm(rel_m - d_long.unsqueeze(1) * track_dir, dim=1)
            if self.add_distance_from_beam:
                cols.append((d_perp / norm).unsqueeze(1))
            if self.add_dist_long:
                cols.append((d_long / norm).unsqueeze(1))
        if self.add_pmt_direction:
            if pmt_directions is None:
                raise ValueError("pmt_directions required when add_pmt_direction=True")
            pdv = pmt_directions.reshape(-1, 3).to(pts.dtype)
            cols.append(pdv)
            if self.add_pmt_cosangle:
                pn = torch.linalg.norm(pdv, dim=1)
                cols.append(((direction * pdv).sum(1)
                             / (dir_norm * pn + 1e-8)).unsqueeze(1))
        return torch.cat(cols, dim=1)

    def _apply_context_norm(self, c):
        if self.standardize_context and self.context_mean is not None:
            return (c - self.context_mean.to(c.dtype)) / self.context_std.to(c.dtype)
        return c

    # ---------------- target transform ----------------

    def dequantize(self, counts, generator=None):
        """Integer counts -> continuous q~ in [q, q+1)."""
        u = torch.rand(counts.shape, device=counts.device, dtype=counts.dtype,
                       generator=generator)
        return counts + u

    def to_z(self, q_deq):
        """q~ -> standardised log10 space."""
        return (torch.log10(q_deq) - self.target_mu) / self.target_sigma

    def from_z(self, z):
        return torch.pow(10.0, z * self.target_sigma + self.target_mu)

    def log_det_dz_dq(self, q_deq):
        """log |dz/dq~| for the change of variables."""
        return -torch.log(q_deq) - math.log(LN10) - math.log(self.target_sigma)

    # ---------------- network ----------------

    def build_network(self):
        self.net = VelocityNet(self.context_dim, self.width, self.depth,
                               self.time_dim, self.cond_width, self.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        if self.reduce_lr_on_plateau:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience, min_lr=self.lr_scheduler_min_lr)
        n = sum(p.numel() for p in self.net.parameters())
        print(f"FlowMatchLY: context_dim={self.context_dim}, "
              f"width={self.width}, depth={self.depth}, params={n:,}")

    # ---------------- flow matching loss ----------------

    def fm_loss(self, context, counts, generator=None):
        context, counts = self._prep(context), self._prep(counts)
        c = self._apply_context_norm(context)
        q_deq = self.dequantize(counts.reshape(-1), generator=generator)
        z1 = self.to_z(q_deq).reshape(-1, 1)

        z0 = torch.randn(z1.shape, device=z1.device, dtype=z1.dtype, generator=generator)
        t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype, generator=generator)

        a = 1.0 - (1.0 - self.sigma_min) * t.unsqueeze(1)
        z_t = a * z0 + t.unsqueeze(1) * z1
        target = z1 - (1.0 - self.sigma_min) * z0

        return torch.nn.functional.mse_loss(self.net(z_t, t, c), target)

    # ---------------- ODE integration ----------------

    def _velocity(self, z, t, c):
        return self.net(z, t, c)

    @torch.no_grad()
    def sample_z(self, context, n_steps=64, method='midpoint', generator=None):
        c = self._apply_context_norm(self._prep(context))
        B = c.shape[0]
        z = torch.randn(B, 1, device=c.device, dtype=c.dtype, generator=generator)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0 = i * dt
            if method == 'euler':
                z = z + dt * self._velocity(z, torch.full((B,), t0, device=c.device,
                                                          dtype=c.dtype), c)
            else:
                tm = torch.full((B,), t0 + 0.5 * dt, device=c.device, dtype=c.dtype)
                k1 = self._velocity(z, torch.full((B,), t0, device=c.device,
                                                  dtype=c.dtype), c)
                z = z + dt * self._velocity(z + 0.5 * dt * k1, tm, c)
        return z

    def _v_and_div(self, z, t, c, create_graph=False):
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            v = self._velocity(z, t, c)
            div = torch.autograd.grad(v.sum(), z, create_graph=create_graph)[0]
        return (v.detach(), div.detach()) if not create_graph else (v, div)

    def log_prob_z(self, z1, context, n_steps=64):
        """log p(z | c) by integrating the flow backwards with the divergence."""
        c = self._apply_context_norm(self._prep(context))
        z = self._prep(z1).reshape(-1, 1)
        B = z.shape[0]
        div_acc = torch.zeros(B, 1, device=z.device, dtype=z.dtype)
        dt = 1.0 / n_steps
        for i in reversed(range(n_steps)):
            tm = torch.full((B,), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
            v, div = self._v_and_div(z, tm, c)
            z = z - dt * v
            div_acc = div_acc + dt * div
        log_p0 = -0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi)
        return (log_p0 - div_acc).reshape(-1)

    @torch.no_grad()
    def transport_to_base(self, z1, context, n_steps=64):
        """Integrate the flow backwards: data-space z -> base-space z0."""
        c = self._apply_context_norm(self._prep(context))
        z = self._prep(z1).reshape(-1, 1)
        B = z.shape[0]
        dt = 1.0 / n_steps
        for i in reversed(range(n_steps)):
            tm = torch.full((B,), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
            k1 = self._velocity(z, tm, c)
            z = z - dt * self._velocity(z - 0.5 * dt * k1, tm, c)
        return z.reshape(-1)

    def cdf_z(self, z1, context, n_steps=64):
        """P(Z <= z1 | c). Exact: the 1-D flow map is monotone, so the CDF is
        the base-space Gaussian CDF of the transported point."""
        z0 = self.transport_to_base(z1, context, n_steps=n_steps)
        return 0.5 * (1.0 + torch.erf(z0 / math.sqrt(2.0)))

    def cdf_light_yield(self, q_deq, context, n_steps=64):
        """P(q~ <= q_deq | c) for the dequantised light yield."""
        return self.cdf_z(self.to_z(self._prep(q_deq).reshape(-1)), context,
                          n_steps=n_steps)

    # ---------------- light-yield level API ----------------

    def sample_light_yield(self, context, n_steps=64, discrete=True, generator=None):
        z = self.sample_z(context, n_steps=n_steps, generator=generator)
        q = self.from_z(z).reshape(-1)
        return torch.floor(q).clamp(min=1.0) if discrete else q

    def log_prob_light_yield(self, counts, context, n_steps=64, n_dequant=1,
                             generator=None):
        """log p(q~ | c) averaged over dequantisation draws (ELBO on log P(q))."""
        counts = self._prep(counts).reshape(-1)
        out = []
        for _ in range(n_dequant):
            q_deq = self.dequantize(counts, generator=generator)
            z = self.to_z(q_deq)
            out.append(self.log_prob_z(z, context, n_steps=n_steps)
                       + self.log_det_dz_dq(q_deq))
        return torch.stack(out).mean(0)

    def pmf_light_yield(self, counts, context, n_steps=64, n_dequant=8):
        """P(q = k | c) = E_u[p(k + u | c)], estimated by averaging densities."""
        counts = self._prep(counts).reshape(-1)
        acc = []
        for _ in range(n_dequant):
            q_deq = self.dequantize(counts)
            z = self.to_z(q_deq)
            lp = self.log_prob_z(z, context, n_steps=n_steps) + self.log_det_dz_dq(q_deq)
            acc.append(lp)
        return torch.logsumexp(torch.stack(acc), dim=0) - math.log(n_dequant)

    def expected_light_yield(self, context, n_samples=256, n_steps=64, generator=None):
        """E[q | c] by Monte Carlo over the flow."""
        B = context.shape[0]
        rep = context.repeat_interleave(n_samples, dim=0)
        q = self.sample_light_yield(rep, n_steps=n_steps, discrete=False,
                                    generator=generator)
        return q.reshape(B, n_samples).mean(dim=1)

    # ---------------- fitting the normalisers ----------------

    def fit_normalisers(self, dataloader, max_batches=20):
        tot = None
        tot_sq = None
        n = 0
        w_sum = 0.0
        w_sq = 0.0
        for i, (ctx, cnt) in enumerate(dataloader):
            if i >= max_batches:
                break
            ctx, cnt = self._prep(ctx), self._prep(cnt)
            if tot is None:
                tot = torch.zeros(ctx.shape[1], device=self.device, dtype=ctx.dtype)
                tot_sq = torch.zeros(ctx.shape[1], device=self.device, dtype=ctx.dtype)
            tot += ctx.sum(0)
            tot_sq += (ctx ** 2).sum(0)
            w = torch.log10(cnt + torch.rand_like(cnt))
            w_sum += w.sum().item()
            w_sq += (w ** 2).sum().item()
            n += ctx.shape[0]
        if n == 0:
            raise ValueError("fit_normalisers: dataloader yielded no batches")

        mean = tot / n
        std = torch.sqrt(torch.clamp(tot_sq / n - mean ** 2, min=0.0))
        self.context_mean = mean
        self.context_std = torch.where(std < 1e-6, torch.ones_like(std), std)
        mu = w_sum / n
        sigma = math.sqrt(max(w_sq / n - mu ** 2, 1e-12))
        self.target_mu, self.target_sigma = mu, sigma
        print(f"  normalisers: target log10(q+u) mu={mu:.4f} sigma={sigma:.4f} "
              f"(from {n:,} samples)")

    # ---------------- training ----------------

    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                              verbose=True, early_stopping_patience=50, grad_clip=1.0,
                              save_every_n_epochs=None, checkpoint_path=None):
        if self.net is None:
            self.build_network()
        if self.target_mu is None:
            self.fit_normalisers(train_dataloader)

        if self.lr_schedule == 'onecycle':
            steps = len(train_dataloader) * epochs
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.learning_rate, total_steps=steps,
                pct_start=self.warmup_frac)
            print(f"  OneCycle: max_lr={self.learning_rate:g}, {steps:,} steps, "
                  f"warmup {self.warmup_frac:.0%}")

        best_val = float('inf')
        patience = 0
        for epoch in range(epochs):
            self.net.train()
            tl, nb = 0.0, 0
            t0 = time.time()
            for ctx, cnt in train_dataloader:
                self.optimizer.zero_grad()
                loss = self.fm_loss(ctx, cnt)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                self.optimizer.step()
                if self.lr_schedule == 'onecycle' and self.lr_scheduler is not None:
                    if self.lr_scheduler.last_epoch + 1 < self.lr_scheduler.total_steps:
                        self.lr_scheduler.step()
                tl += loss.item()
                nb += 1
            tl /= max(nb, 1)
            self.train_losses.append(tl)

            vl = None
            if val_dataloader is not None:
                self.net.eval()
                vs, vn = 0.0, 0
                with torch.no_grad():
                    for ctx, cnt in val_dataloader:
                        vs += self.fm_loss(ctx, cnt).item()
                        vn += 1
                vl = vs / max(vn, 1)
                self.val_losses.append(vl)

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
                    msg += f"  val {vl:.5f}"
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

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        torch.save({
            'net_state_dict': state if state is not None else self.net.state_dict(),
            'width': self.width, 'depth': self.depth, 'time_dim': self.time_dim,
            'cond_width': self.cond_width, 'dropout': self.dropout,
            'domain_size': self.domain_size, 'dim': self.dim,
            'sigma_min': self.sigma_min, 'ly_eps': self.ly_eps,
            'rich_rel_pos_mode': self.rich_rel_pos_mode,
            'include_vertex_position': self.include_vertex_position,
            'add_vertex_distance': self.add_vertex_distance,
            'add_distance_from_beam': self.add_distance_from_beam,
            'add_dist_long': self.add_dist_long,
            'track_dir_is_arrival': self.track_dir_is_arrival,
            'add_pmt_direction': self.add_pmt_direction,
            'add_pmt_cosangle': self.add_pmt_cosangle,
            'standardize_context': self.standardize_context,
            'context_mean': None if self.context_mean is None else self.context_mean.cpu(),
            'context_std': None if self.context_std is None else self.context_std.cpu(),
            'target_mu': self.target_mu, 'target_sigma': self.target_sigma,
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
        }, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        for k in ['width', 'depth', 'time_dim', 'cond_width', 'dropout', 'domain_size',
                  'dim', 'sigma_min', 'ly_eps', 'rich_rel_pos_mode',
                  'include_vertex_position', 'add_vertex_distance',
                  'add_distance_from_beam', 'add_dist_long', 'track_dir_is_arrival',
                  'add_pmt_direction', 'add_pmt_cosangle', 'standardize_context',
                  'target_mu', 'target_sigma', 'train_losses', 'val_losses']:
            if k in ck:
                setattr(self, k, ck[k])
        self.build_network()
        self.net.load_state_dict(ck['net_state_dict'])
        cm, cs = ck.get('context_mean'), ck.get('context_std')
        self.context_mean = None if cm is None else cm.to(self.device)
        self.context_std = None if cs is None else cs.to(self.device)
        self.is_trained = True
        return self

    # ---------------- dataloaders ----------------

    def create_flow_parquet_dataloader(self, parquet_path, geometry_csv_path,
                                       num_samples_per_epoch=None, batch_size=4096,
                                       shuffle=True, num_workers=0, seed=None,
                                       uniform_energy_zenith=True,
                                       n_energy_bins=20, n_coszen_bins=20,
                                       filter_vertex_in_domain=True,
                                       test_save_path=None, test_frac=0.1,
                                       split_seed=None, pin_memory=None):
        import pandas as pd

        train_filter = None
        if test_save_path is not None:
            df = pd.read_parquet(parquet_path, columns=['run_id', 'event_id'])
            pairs = list(zip(df.run_id.astype(int), df.event_id.astype(int)))
            uniq = sorted(set(pairs))
            rng = np.random.default_rng(split_seed if split_seed is not None else seed)
            perm = rng.permutation(len(uniq))
            n_test = int(round(test_frac * len(uniq)))
            test_events = {uniq[i] for i in perm[:n_test]}
            train_filter = {uniq[i] for i in perm[n_test:]}
            full = pd.read_parquet(parquet_path)
            mask = pd.Series(pairs, index=full.index, dtype=object).isin(test_events)
            d = os.path.dirname(test_save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            full.loc[mask.to_numpy()].to_parquet(test_save_path, index=False)
            print(f"held out {len(test_events)}/{len(uniq)} events "
                  f"({int(mask.sum()):,} rows) -> {test_save_path}")
            del full

        ds = LightYieldFlowDataset(
            self, parquet_path, geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch, seed=seed,
            uniform_energy_zenith=uniform_energy_zenith,
            n_energy_bins=n_energy_bins, n_coszen_bins=n_coszen_bins,
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

    def create_flow_parquet_val_dataloader(self, parquet_path, geometry_csv_path,
                                           num_samples_per_epoch=None, batch_size=4096,
                                           num_workers=0, seed=0, **kw):
        return self.create_flow_parquet_dataloader(
            parquet_path=parquet_path, geometry_csv_path=geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, seed=seed,
            test_save_path=None, **kw)

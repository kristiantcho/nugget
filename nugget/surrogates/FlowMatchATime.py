"""Conditional flow matching for p(photon arrival time | event params, detector position)."""

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from nugget.surrogates.FlowMatchLY import (
    FlowMatchLY, _reseed_dataset_rng_in_worker,
)

C_VAC = 0.299792458          # speed of light in vacuum, m/ns


def _ragged_indices(starts, counts):
    """Flat indices for the concatenation of slices [s, s+c) — a vectorised
    equivalent of ``np.concatenate([np.arange(s, s+c) for s, c in zip(starts, counts)])``.
    Assumes every count is > 0.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    if len(starts) > 1:
        ends = np.cumsum(counts)[:-1]
        out[ends] = starts[1:] - (starts[:-1] + counts[:-1]) + 1
    return np.cumsum(out)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class ArrivalTimeFlowDataset(Dataset):
    """Parquet rows with a `times` list column -> (context, time residual).

    One PHOTON is one sample: the ragged `times` lists are flattened once and each
    photon carries an index back to its parent row's event/geometry parameters.
    """

    def __init__(self, model, parquet_path, geometry_csv_path,
                 num_samples_per_epoch=None, seed=None,
                 uniform_energy_zenith=False, n_energy_bins=20, n_coszen_bins=20,
                 filter_vertex_in_domain=True, event_filter=None,
                 max_photons_per_row=None, verbose=True):
        import pandas as pd

        self.model = model
        self.device = torch.device('cpu')
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        geo = pd.read_csv(geometry_csv_path,
                          usecols=['string', 'om', 'pmt', 'om_x', 'om_y', 'om_z',
                                   'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'])

        scalar_cols = ['run_id', 'event_id', 'string', 'om', 'pmt',
                       'muon_x', 'muon_y', 'muon_z', 'neutrino_energy',
                       'zenith', 'azimuth']
        df = pd.read_parquet(parquet_path, columns=scalar_cols + ['times'])

        # Flatten the ragged `times` column up front, then drop it: the per-row
        # object column is by far the heaviest thing here, and carrying it through
        # the filter/merge below would copy it repeatedly.
        counts_all = df['times'].str.len().to_numpy(np.int64)
        flat_all = np.concatenate(df['times'].to_numpy()).astype(np.float32)
        starts_all = np.concatenate([[0], np.cumsum(counts_all)[:-1]]).astype(np.int64)
        df = df.drop(columns=['times'])
        df['_orig'] = np.arange(len(df), dtype=np.int64)

        half = self._domain_half_extent()
        if filter_vertex_in_domain:
            n0 = len(df)
            df = df[(df.muon_x.abs() <= half[0]) & (df.muon_y.abs() <= half[1])
                    & (df.muon_z.abs() <= half[2])]
            if verbose and len(df) < n0:
                print(f"ArrivalTimeFlowDataset: dropped {n0 - len(df):,} row(s) with "
                      f"muon vertex outside {half.tolist()}")

        if event_filter is not None:
            ev_pairs = list(zip(df.run_id.astype(int), df.event_id.astype(int)))
            mask = pd.Series(ev_pairs, index=df.index, dtype=object).isin(set(event_filter))
            df = df[mask.to_numpy()]

        df = df.merge(geo, on=['string', 'om', 'pmt'], how='inner', copy=False)
        if len(df) == 0:
            raise ValueError("No usable rows: parquet and geometry CSV do not overlap "
                             "(or every vertex fell outside the domain).")

        keep = df['_orig'].to_numpy(np.int64)

        # ---- per-row (per-PMT) arrays ----
        self._point = np.ascontiguousarray(
            df[['om_x', 'om_y', 'om_z']].to_numpy(np.float32))
        self._pmt_direction = np.ascontiguousarray(
            df[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy(np.float32))
        self._muon_pos = np.ascontiguousarray(
            df[['muon_x', 'muon_y', 'muon_z']].to_numpy(np.float32))
        self._energy = df.neutrino_energy.to_numpy(np.float32)
        self._zenith = df.zenith.to_numpy(np.float32)
        self._azimuth = df.azimuth.to_numpy(np.float32)
        codes, _ = pd.factorize(
            pd.Series(list(zip(df.run_id.astype(int), df.event_id.astype(int))),
                      dtype=object))
        self._row_event_code = codes.astype(np.int64)
        self._n_rows = len(self._energy)

        # ---- gather the photons belonging to the kept rows ----
        counts = counts_all[keep]
        starts = starts_all[keep]
        if max_photons_per_row is not None:
            # Cap very bright PMTs so a handful of rows cannot dominate an epoch.
            cap = int(max_photons_per_row)
            capped = np.minimum(counts, cap)
            if verbose and capped.sum() < counts.sum():
                print(f"ArrivalTimeFlowDataset: capped photons/row at {cap} "
                      f"({int(counts.sum() - capped.sum()):,} photons dropped)")
            counts = capped

        nonzero = counts > 0
        if not nonzero.all():
            keep, counts, starts = keep[nonzero], counts[nonzero], starts[nonzero]
            df = df[nonzero]
            for name in ('_point', '_pmt_direction', '_muon_pos'):
                setattr(self, name, getattr(self, name)[nonzero])
            self._energy = self._energy[nonzero]
            self._zenith = self._zenith[nonzero]
            self._azimuth = self._azimuth[nonzero]
            self._row_event_code = self._row_event_code[nonzero]
            self._n_rows = int(nonzero.sum())

        flat = flat_all[_ragged_indices(starts, counts)]
        del flat_all

        self._t_hit = flat                                   # (n_photons,)
        self._ph_row = np.repeat(np.arange(self._n_rows, dtype=np.int64), counts)
        self._n_photons = len(self._t_hit)
        self._photons_per_row = counts
        # First photon index of each row, precomputed: the stratified sampler needs
        # it per batch and an O(n_rows) cumsum there would dominate the batch cost.
        self._row_photon_start = np.concatenate(
            [[0], np.cumsum(counts)[:-1]]).astype(np.int64)

        self._n_events = int(self._row_event_code.max()) + 1
        _order = np.argsort(self._row_event_code, kind='stable')
        _codes = self._row_event_code[_order]
        _bounds = np.flatnonzero(np.diff(_codes)) + 1
        self._event_rows = {int(_codes[g[0]]): g for g in np.split(_order, _bounds)}
        self._events = np.fromiter(self._event_rows.keys(), dtype=np.int64)

        self.num_samples_per_epoch = (int(num_samples_per_epoch)
                                      if num_samples_per_epoch else self._n_photons)

        self.uniform_energy_zenith = bool(uniform_energy_zenith)
        self._n_bins = 0
        if self.uniform_energy_zenith:
            self._build_bins(int(n_energy_bins), int(n_coszen_bins))

        if verbose:
            print(f"ArrivalTimeFlowDataset: {self._n_rows:,} rows (PMTs), "
                  f"{self._n_photons:,} photons, {self._n_events:,} events, "
                  f"{self.num_samples_per_epoch:,} samples/epoch")
            print(f"  photons/row: mean {counts.mean():.2f}, median "
                  f"{np.median(counts):.0f}, max {counts.max()}")

    # ---- geometry / binning helpers (mirror LightYieldFlowDataset) ----

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

    def _sample_photons(self, n):
        """Return (photon_idx, row_idx) for n draws.

        Default is uniform over photons. With uniform_energy_zenith the ROW is drawn
        stratified over (log10 E, cos zenith) bins and then one of its photons is
        picked uniformly -- either way the conditional p(t | theta, x) being learned
        is the same; only the coverage of (theta, x) changes.
        """
        if self.uniform_energy_zenith and self._n_bins > 0:
            b = self._rng.integers(0, self._n_bins, size=n)
            lens = self._bin_lens[b]
            offs = (self._rng.random(n) * lens).astype(np.int64)
            np.minimum(offs, lens - 1, out=offs)
            rows = self._bin_flat[self._bin_starts[b] + offs]
            npr = self._photons_per_row[rows]
            k = (self._rng.random(n) * npr).astype(np.int64)
            np.minimum(k, npr - 1, out=k)
            return self._row_photon_start[rows] + k, rows
        ph = self._rng.integers(0, self._n_photons, size=n)
        return ph, self._ph_row[ph]

    def __len__(self):
        return self.num_samples_per_epoch

    def get_batch(self, indices):
        n = len(np.asarray(indices).reshape(-1))
        ph, rows = self._sample_photons(n)

        pts = torch.from_numpy(self._point[rows])
        vert = torch.from_numpy(self._muon_pos[rows])
        zen = torch.from_numpy(self._zenith[rows])
        azi = torch.from_numpy(self._azimuth[rows])
        ctx = self.model.build_context(
            pts, vert, torch.from_numpy(self._energy[rows]), zen, azi,
            torch.from_numpy(self._pmt_direction[rows]),
        )
        t_res = self.model.time_residual(
            torch.from_numpy(self._t_hit[ph]), pts, vert, zeniths=zen, azimuths=azi)
        return ctx, t_res

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        if isinstance(idx, (list, np.ndarray)):
            return self.get_batch(idx)
        c, t = self.get_batch([idx])
        return c[0], t[0]

    def _event_data(self, i, pmt_direction=None):
        """Event-parameter dict for row i (same keys as the LLRnet parquet dataset)."""
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

    def residual_stats(self, max_samples=500000):
        """(mu, sigma) of the transformed time residual, for standardisation."""
        n = min(int(max_samples), self._n_photons)
        ph = self._rng.integers(0, self._n_photons, size=n)
        rows = self._ph_row[ph]
        t_res = self.model.time_residual(
            torch.from_numpy(self._t_hit[ph]),
            torch.from_numpy(self._point[rows]),
            torch.from_numpy(self._muon_pos[rows]),
            zeniths=torch.from_numpy(self._zenith[rows]),
            azimuths=torch.from_numpy(self._azimuth[rows]))
        w = self.model.transform_time(t_res)
        return float(w.mean()), float(w.std())


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class FlowMatchATime(FlowMatchLY):
    """Conditional flow matching for p(arrival time | event params, detector position).

    Reuses FlowMatchLY's velocity network, ODE solver, exact 1-D log-density and
    training loop; only the target variable changes. The modelled quantity is the
    Cherenkov time RESIDUAL

        t_res = t_hit - t_geom(theta, x)

    which removes the (large, geometry-driven) predictable part of the arrival time
    and leaves the scattering/jitter distribution. Because t_res is a pure
    translation of t_hit, log p(t_hit | c) = log p(t_res | c) with no extra Jacobian.
    """

    def __init__(self, *args, refractive_index=1.33, time_scale=10.0,
                 time_transform='asinh',
                 track_dir_is_arrival=True, **kwargs):
        # track_dir_is_arrival defaults to True here (unlike the light-yield model):
        # in this parquet zenith/azimuth are the ARRIVAL direction, and the geometric
        # time is badly wrong under the other convention -- measured median residual
        # 6.9 ns with -direction versus 2769 ns with +direction.
        super().__init__(*args, track_dir_is_arrival=track_dir_is_arrival, **kwargs)
        self.refractive_index = float(refractive_index)
        self.time_scale = float(time_scale)
        self.time_transform = time_transform
        if time_transform not in ('asinh', 'symlog'):
            raise ValueError("time_transform must be 'asinh' or 'symlog'")

    # ---------------- geometric (Cherenkov) arrival time ----------------

    def geometric_time(self, points, vertices, zeniths=None, azimuths=None,
                       directions=None):
        """Earliest direct-Cherenkov arrival time at each point, in ns.

        t_geom = [ d_along + d_perp * (n - cos_c) / sin_c ] / c,  cos_c = 1/n
        with d_along/d_perp measured along the muon TRAVEL direction from the vertex.
        Differentiable in the event parameters.
        """
        pts = points.reshape(-1, 3)
        vert = vertices.reshape(-1, 3).to(pts.dtype)
        if directions is not None:
            u = directions.reshape(-1, 3).to(pts.dtype)
        else:
            zen = zeniths.reshape(-1).to(pts.dtype)
            azi = azimuths.reshape(-1).to(pts.dtype)
            st, ct = torch.sin(zen), torch.cos(zen)
            u = torch.stack([st * torch.cos(azi), st * torch.sin(azi), ct], dim=1)
        travel = -u if self.track_dir_is_arrival else u

        rel = pts - vert
        d_along = (rel * travel).sum(1)
        d_perp = torch.linalg.norm(rel - d_along.unsqueeze(1) * travel, dim=1)

        n = self.refractive_index
        cos_c = 1.0 / n
        sin_c = math.sqrt(max(1.0 - cos_c ** 2, 1e-12))
        return (d_along + d_perp * (n - cos_c) / sin_c) / C_VAC

    def time_residual(self, t_hit, points, vertices, zeniths=None, azimuths=None,
                      directions=None):
        """t_hit - t_geom. Negative values are physical (PMT timing jitter)."""
        t_geom = self.geometric_time(points, vertices, zeniths, azimuths, directions)
        return t_hit.reshape(-1).to(t_geom.dtype) - t_geom

    # ---------------- target transform ----------------

    def transform_time(self, t_res):
        """t_res (ns) -> a roughly Gaussian, sign-preserving, MONOTONIC scale.

        Both options are strictly increasing, which the flow relies on: the exact CDF
        (cdf_time_residual) is only valid because t_res -> z is monotone.

        Note this is deliberately NOT the raw LLRnet PATD scaling
        ``sign(t)*log10(|t|+eps)``. That form folds at |t| = 1 ns -- t = -0.5 and
        t = +0.5 map to opposite signs -- so it is not invertible. It was fine there
        because it only ever fed a network input; as a density transform it would be
        wrong. 'symlog' below is the monotonic repair of it.
        """
        s = self.time_scale
        if self.time_transform == 'asinh':
            return torch.asinh(t_res / s)
        # symlog: sign(t) * log10(1 + |t|/s); linear near 0, logarithmic in the tails
        return torch.sign(t_res) * torch.log10(1.0 + torch.abs(t_res) / s)

    def inverse_transform_time(self, w):
        s = self.time_scale
        if self.time_transform == 'asinh':
            return torch.sinh(w) * s
        return torch.sign(w) * s * (torch.pow(10.0, torch.abs(w)) - 1.0)

    def log_abs_dw_dt(self, t_res):
        """log |dw/dt_res| for the transform above."""
        s = self.time_scale
        if self.time_transform == 'asinh':
            # d/dt asinh(t/s) = 1 / sqrt(s^2 + t^2)
            return -0.5 * torch.log(s * s + t_res * t_res)
        # d/dt symlog = 1 / (ln10 * (s + |t|))
        return -torch.log(s + torch.abs(t_res)) - math.log(math.log(10.0))

    # ---- hooks used by the inherited fm_loss / log_prob machinery ----

    def dequantize(self, values, generator=None):
        """Arrival times are continuous; no dequantisation is needed."""
        return values

    def to_z(self, t_res):
        return (self.transform_time(t_res) - self.target_mu) / self.target_sigma

    def from_z(self, z):
        return self.inverse_transform_time(z * self.target_sigma + self.target_mu)

    def log_det_dz_dq(self, t_res):
        """log |dz/dt_res|, i.e. the transform Jacobian plus standardisation."""
        return self.log_abs_dw_dt(t_res) - math.log(self.target_sigma)

    # ---------------- time-level API ----------------

    def sample_time_residual(self, context, n_steps=64, generator=None):
        z = self.sample_z(context, n_steps=n_steps, generator=generator)
        return self.from_z(z).reshape(-1)

    def sample_arrival_time(self, context, points, vertices, zeniths=None,
                            azimuths=None, directions=None, n_steps=64,
                            generator=None):
        """Sample t_hit = t_geom + t_res."""
        t_res = self.sample_time_residual(context, n_steps=n_steps, generator=generator)
        t_geom = self.geometric_time(points, vertices, zeniths, azimuths, directions)
        return t_geom.reshape(-1) + t_res

    def log_prob_time_residual(self, t_res, context, n_steps=64):
        """log p(t_res | c). Equals log p(t_hit | c): the shift has unit Jacobian."""
        t_res = self._prep(t_res).reshape(-1)
        return self.log_prob_z(self.to_z(t_res), context, n_steps=n_steps) \
            + self.log_det_dz_dq(t_res)

    def log_prob_arrival_time(self, t_hit, context, points, vertices, zeniths=None,
                              azimuths=None, directions=None, n_steps=64):
        t_res = self.time_residual(self._prep(t_hit), self._prep(points),
                                   self._prep(vertices), zeniths, azimuths, directions)
        return self.log_prob_time_residual(t_res, context, n_steps=n_steps)

    def cdf_time_residual(self, t_res, context, n_steps=64):
        """P(T_res <= t | c). Exact: the 1-D flow map is monotone and so is the
        transform, so the CDF is the base-space Gaussian CDF of the transported point."""
        return self.cdf_z(self.to_z(self._prep(t_res).reshape(-1)), context,
                          n_steps=n_steps)

    # ---------------- normalisers ----------------

    def fit_normalisers(self, dataloader, max_batches=20):
        """Context mean/std and the transformed-residual mean/std."""
        tot = tot_sq = None
        n = 0
        w_sum = w_sq = 0.0
        for i, (ctx, t_res) in enumerate(dataloader):
            if i >= max_batches:
                break
            ctx, t_res = self._prep(ctx), self._prep(t_res)
            if tot is None:
                tot = torch.zeros(ctx.shape[1], device=self.device, dtype=ctx.dtype)
                tot_sq = torch.zeros(ctx.shape[1], device=self.device, dtype=ctx.dtype)
            tot += ctx.sum(0)
            tot_sq += (ctx ** 2).sum(0)
            w = self.transform_time(t_res.reshape(-1))
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
        print(f"  normalisers: transformed t_res mu={mu:.4f} sigma={sigma:.4f} "
              f"(from {n:,} photons)")

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        super().save_model(filepath, state=state)
        ck = torch.load(filepath, map_location='cpu', weights_only=False)
        ck.update({'refractive_index': self.refractive_index,
                   'time_scale': self.time_scale,
                   'time_transform': self.time_transform,
                   })
        torch.save(ck, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        for k in ('refractive_index', 'time_scale', 'time_transform'):
            if k in ck:
                setattr(self, k, ck[k])
        return super().load_model(filepath)

    # ---------------- dataloaders ----------------

    def create_atime_parquet_dataloader(self, parquet_path, geometry_csv_path,
                                        num_samples_per_epoch=None, batch_size=4096,
                                        shuffle=True, num_workers=0, seed=None,
                                        uniform_energy_zenith=False,
                                        n_energy_bins=20, n_coszen_bins=20,
                                        filter_vertex_in_domain=True,
                                        max_photons_per_row=None,
                                        test_save_path=None, test_frac=0.1,
                                        split_seed=None, pin_memory=None):
        import os
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
            full = pd.read_parquet(parquet_path)
            full.loc[mask].to_parquet(test_save_path, index=False)
            print(f"held out {len(test_events)}/{len(uniq)} events "
                  f"({int(mask.sum()):,} rows) -> {test_save_path}")
            del full

        ds = ArrivalTimeFlowDataset(
            self, parquet_path, geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch, seed=seed,
            uniform_energy_zenith=uniform_energy_zenith,
            n_energy_bins=n_energy_bins, n_coszen_bins=n_coszen_bins,
            filter_vertex_in_domain=filter_vertex_in_domain,
            max_photons_per_row=max_photons_per_row,
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

    def create_atime_parquet_val_dataloader(self, parquet_path, geometry_csv_path,
                                            num_samples_per_epoch=None,
                                            batch_size=4096, num_workers=0, seed=0,
                                            **kw):
        return self.create_atime_parquet_dataloader(
            parquet_path=parquet_path, geometry_csv_path=geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, seed=seed,
            test_save_path=None, **kw)

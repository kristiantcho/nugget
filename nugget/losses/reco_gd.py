"""Reconstruction resolution from per-event gradient-descent maximum-likelihood fits with the
hit / light-yield / arrival-time flows. Not differentiable in the geometry: register it with
weight 0 (it still reaches the visualizer) or call it directly."""
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import ConvexHull, cKDTree

from nugget.losses.base_loss import LossFunction

# fitted parameters; zenith/azimuth in the flow models' convention (the parquet's, i.e. the
# arrival direction for track_dir_is_arrival models). No t0: photon times are used as they are,
# on the clock of the parquet and of NuSmoothie (interaction at t = 0), so the arrival-time
# residual t - t_geom depends on the vertex and the track only
PARAMS = ('x', 'y', 'z', 'zenith', 'azimuth', 'log10E')
# one row per event; zenith/azimuth in the same convention, energy in GeV
TRUTH_COLS = ('run_id', 'event_id', 'x', 'y', 'z', 'zenith', 'azimuth', 'energy')

LOG_FLOOR = np.log(1e-4) - np.log(1e4)   # arrival-time noise floor: 1e-4 over 10 us
ADAM_BETAS, ADAM_EPS = (0.9, 0.999), 1e-8


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _kw(m):
    return dict(device=m.device, dtype=m.param_dtype)


def _unit(zen, azi):
    return np.stack([np.sin(zen) * np.cos(azi), np.sin(zen) * np.sin(azi), np.cos(zen)], -1)


def _angles(d):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.arccos(np.clip(d[..., 2], -1, 1)), np.arctan2(d[..., 1], d[..., 0]) % (2 * np.pi)


def _canonical(th):
    """zenith/azimuth back to [0, pi] x [0, 2 pi)."""
    th = th.copy()
    th[3], th[4] = _angles(_unit(th[3], th[4]))
    return th


def _dpsi(a, b):
    return float(np.degrees(np.arccos(np.clip(_unit(*a[3:5]) @ _unit(*b[3:5]), -1, 1))))


def half_extent(ds):
    ds = np.atleast_1d(np.asarray(ds.tolist() if torch.is_tensor(ds) else ds, float))
    return np.array([ds[0], ds[0], ds[-1]]) / 2.0


def hull_distance(points, vertices):
    """Signed distance [m] of vertices to the convex hull of points: < 0 inside (exact),
    > 0 outside (a lower bound). NaN if the points span no volume."""
    vertices = np.asarray(vertices, float).reshape(-1, 3)
    try:
        eq = ConvexHull(np.asarray(points, float).reshape(-1, 3)).equations
    except Exception:
        return np.full(len(vertices), np.nan)
    eq = np.unique(eq.round(6), axis=0)    # the triangles of one face share a plane
    return np.concatenate([(v @ eq[:, :3].T + eq[:, 3]).max(1)
                           for v in np.array_split(vertices, max(len(vertices) // 4096, 1))])


def resolution_summary(res):
    """Resolution figures of a fit-result table."""
    if len(res) == 0:
        return dict(n_events=0, dpsi_median=np.nan, dpsi_68=np.nan, dlog10e_bias=np.nan,
                    dlog10e_68=np.nan, dvertex_median=np.nan, nll_gap_median=np.nan)
    dle = res.dlog10E_fit
    return dict(n_events=len(res),
                dpsi_median=float(res.dpsi_fit.median()),
                dpsi_68=float(np.percentile(res.dpsi_fit, 68)),
                dlog10e_bias=float(dle.median()),
                dlog10e_68=float((dle.quantile(0.84) - dle.quantile(0.16)) / 2),
                dvertex_median=float(res.dvertex_fit.median()),
                nll_gap_median=float((res.nll_fit - res.nll_true).median()))


def energy_table(res, width=0.5):
    """resolution_summary per true log10 E bin; the index is the bin's lower edge."""
    lo = (np.floor(res.true_log10E / width) * width).round(6)
    return pd.DataFrame({b: resolution_summary(g) for b, g in res.groupby(lo)}).T \
        .rename_axis('log10E_lo')


# --------------------------------------------------------------------------- #
#  Detector responses                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class Response:
    """Lit (event, PMT) pairs of many events in one geometry."""
    pmt_pos: np.ndarray     # (n_pmt, 3)
    pmt_dir: np.ndarray     # (n_pmt, 3)
    truth: pd.DataFrame     # one row per event: TRUTH_COLS, extra columns go into the results
    hits: pd.DataFrame      # ev (truth row), pmt_idx, n (photons), start (into times)
    times: np.ndarray = None   # flat photon times [ns]; None if the source has counts only


def read_geometry(path):
    """Geometry CSV -> (PMT table, (n, 3) positions, (n, 3) directions)."""
    g = pd.read_csv(path, usecols=['string', 'om', 'pmt', 'om_x', 'om_y', 'om_z',
                                   'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'])
    return (g, g[['om_x', 'om_y', 'om_z']].to_numpy(np.float64),
            g[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy(np.float64))


def event_table(parquet, energy_column='neutrino_energy'):
    """One row per event of a parquet (TRUTH_COLS), streamed batch by batch."""
    from nugget.utils.parquet_io import LEPTON_ALIASES, _iter_batches, _schema
    names = set(_schema(parquet).names)
    src = dict(x='muon_x', y='muon_y', z='muon_z', zenith='zenith', azimuth='azimuth',
               energy=energy_column)
    src = {k: c if c in names else LEPTON_ALIASES.get(c, c) for k, c in src.items()}
    parts = [b.to_pandas().drop_duplicates(['run_id', 'event_id'])
             for b in _iter_batches(parquet, ['run_id', 'event_id', *src.values()])]
    t = pd.concat(parts).rename(columns={c: k for k, c in src.items()})
    return (t.drop_duplicates(['run_id', 'event_id']).sort_values(['run_id', 'event_id'])
            [list(TRUTH_COLS)].reset_index(drop=True))


def parquet_response(parquet, geometry_csv, truth):
    """The parquet's own photons of the events in `truth`, in the geometry CSV."""
    from nugget.surrogates.FlowMatchATime import _times_to_flat
    from nugget.utils.parquet_io import _schema, read_parquet_columns
    names = set(_schema(parquet).names)
    tcol = 'times' if 'times' in names else 'time' if 'time' in names else None
    if tcol is None and 'count' not in names:
        raise ValueError("parquet needs a 'times', 'time' or 'count' column")
    truth = truth.reset_index(drop=True)
    key = ['run_id', 'event_id', 'string', 'om', 'pmt']
    df = read_parquet_columns(parquet, key + [tcol or 'count'],
                              events=list(zip(truth.run_id.tolist(), truth.event_id.tolist())))
    times = None
    if tcol == 'time':                                  # photon mode: one row per photon
        df = df.groupby(key, sort=False)['time'].agg(list).rename('times').reset_index()
        tcol = 'times'
    if tcol:
        times, n = _times_to_flat(df['times'])
        df = df.drop(columns='times').assign(n=n, start=np.cumsum(n) - n)
    else:
        df = df.assign(n=df['count'].to_numpy(np.int64), start=0)
    df = df[df.n > 0]

    geo, pos, dirs = read_geometry(geometry_csv)
    geo['pmt_idx'] = np.arange(len(geo))
    ev = truth[['run_id', 'event_id']].reset_index().rename(columns={'index': 'ev'})
    df = df.merge(geo[key[2:] + ['pmt_idx']], on=key[2:]).merge(ev, on=['run_id', 'event_id'])
    return Response(pos, dirs, truth, df[['ev', 'pmt_idx', 'n', 'start']].reset_index(drop=True),
                    times)


# --------------------------------------------------------------------------- #
#  Loss                                                                        #
# --------------------------------------------------------------------------- #

class RecoGDLoss(LossFunction):
    """Fit (x, y, z, zenith, azimuth, log10 E) of each event by projected Adam on the flows'
    -log L, from a random start, and report the resolution. The event time is not fitted: the
    interaction is at t = 0 and t_geom follows from the vertex and the track.

    __call__ simulates `signal_event_params` once with NuSmoothie in geom_dict's OMs and fits
    them; `fit` takes any Response (e.g. `parquet_response`). Returns 'reco_loss' (the
    `reco_metric` of resolution_summary, no gradient), 'reco_results' (one row per event) and
    every summary figure as 'reco_<name>'.

    start: 'perturbed' (truth + pert_scale * N(0, 1)), 'random' (vertex uniform in the lit
    PMTs' bounding box, isotropic direction, log10 E uniform in the bounds) or 'truth'.
    vertex_start: None (the vertex comes from `start`) or 'hit_weighted' (the photon-count
    weighted mean position of the lit PMTs; direction and energy still come from `start`).
    hit_radius: unlit PMTs within this of a lit one enter the hit term [m]; None = all.
    pert_scale, scale: one entry per PARAMS; scale is the unit of the Adam step (lr is in
    these units).
    """

    def __init__(self, device=None, hit_model=None, ly_model=None, atime_model=None,
                 hit_checkpoint=None, ly_checkpoint=None, atime_checkpoint=None,
                 terms=('hit', 'ly', 'atime'), hit_radius=None, n_steps=8,
                 start='perturbed', vertex_start=None,
                 pert_scale=(70.0, 70.0, 70.0, np.radians(10), np.radians(10), 1.0),
                 n_starts=1,
                 scale=(20.0, 20.0, 20.0, np.radians(5), np.radians(5), 0.3),
                 lr=0.05, max_iter=1000, patience=200, min_delta=1e-2, lr_factor=0.5,
                 lr_patience=20, min_lr=1e-4, log10e_bounds=(2.0, 6.0), bound_vertex=True,
                 events_per_batch=128, flow_chunk_rows=30000, hit_chunk_rows=200_000,
                 pmt_directions=None, geometry_csv_path=None, response_n_steps=64,
                 response_events_per_batch=128, seed=0, empty_cache=True, verbose=True):
        super().__init__(device=device)
        from nugget.surrogates.HitClassifier import HitClassifier
        from nugget.surrogates.FlowMatchLY import FlowMatchLY
        from nugget.surrogates.FlowMatchATime import FlowMatchATime
        if start not in ('perturbed', 'random', 'truth'):
            raise ValueError("start must be 'perturbed', 'random' or 'truth'")
        if vertex_start not in (None, 'hit_weighted'):
            raise ValueError("vertex_start must be None or 'hit_weighted'")
        for name, v in (('pert_scale', pert_scale), ('scale', scale)):
            if len(v) != len(PARAMS):
                raise ValueError(f"{name} needs {len(PARAMS)} entries {PARAMS} (t0 is no "
                                 f"longer fitted), got {len(v)}")

        def load(m, ckpt, cls):
            if m is None and ckpt is not None:
                m = cls(device=self.device).load_model(ckpt)
                for p in m.net.parameters():
                    p.requires_grad_(False)
            if m is not None:
                m.net.eval()
            return m

        self.hit_model = load(hit_model, hit_checkpoint, HitClassifier)
        self.ly_model = load(ly_model, ly_checkpoint, FlowMatchLY)
        self.atime_model = load(atime_model, atime_checkpoint, FlowMatchATime)
        if self.hit_model is None or self.ly_model is None:
            raise ValueError("needs a hit and a light-yield model")
        # the fitted log10E feeds all three, so they must share one energy definition
        etypes = {getattr(m, 'energy_type', 'neutrino')
                  for m in (self.hit_model, self.ly_model, self.atime_model) if m is not None}
        if len(etypes) > 1:
            raise ValueError(f"the models were trained on different energies {sorted(etypes)}")
        self.terms = tuple(t for t in terms if t != 'atime' or self.atime_model is not None)
        self.arrival = bool(getattr(self.hit_model, 'track_dir_is_arrival', False))
        self.half = half_extent(self.hit_model.domain_size)

        self.hit_radius, self.n_steps = hit_radius, n_steps
        self.start, self.vertex_start = start, vertex_start
        self.pert_scale, self.n_starts = np.asarray(pert_scale, float), int(n_starts)
        self.scale = np.asarray(scale, float)
        self.lr, self.max_iter, self.patience, self.min_delta = lr, max_iter, patience, min_delta
        self.lr_factor, self.lr_patience, self.min_lr = lr_factor, lr_patience, min_lr
        self.log10e_bounds, self.bound_vertex = log10e_bounds, bound_vertex
        self.events_per_batch = events_per_batch
        self.flow_chunk_rows, self.hit_chunk_rows = flow_chunk_rows, hit_chunk_rows
        self.pmt_directions, self.geometry_csv_path = pmt_directions, geometry_csv_path
        self.response_n_steps = response_n_steps
        self.response_events_per_batch = response_events_per_batch
        self.seed, self.empty_cache, self.verbose = seed, empty_cache, verbose
        self._smoothie = None

    # ---------------- geometry / events -> Response ----------------

    @staticmethod
    def om_positions(geom_dict, weight_threshold=None):
        """(n, 3) OM positions; with weight_threshold only the strings whose
        sigmoid(string_weights) exceeds it."""
        pts = _np(geom_dict['points_3d']).astype(np.float64).reshape(-1, 3)
        w, sxy = geom_dict.get('string_weights'), geom_dict.get('string_xy')
        if weight_threshold is None or w is None or sxy is None:
            return pts
        on = 1.0 / (1.0 + np.exp(-_np(w).reshape(-1))) > weight_threshold
        sxy = _np(sxy).reshape(-1, 2)[on]
        keep = ((pts[:, None, 0] == sxy[None, :, 0]) & (pts[:, None, 1] == sxy[None, :, 1])).any(1)
        return pts[keep]

    def truth_from_event_params(self, params):
        """Sampler event dicts (travel direction) -> truth table in the models' convention."""
        rows = []
        for i, p in enumerate(params):
            d = p.get('direction')
            d = (_np(d).reshape(3).astype(float) if d is not None
                 else _unit(float(_np(p['zenith']).reshape(-1)[0]),
                            float(_np(p['azimuth']).reshape(-1)[0])))
            zen, azi = _angles(-d if self.arrival else d)
            x, y, z = _np(p['position']).reshape(3).astype(float)
            rows.append(dict(run_id=0, event_id=i, x=x, y=y, z=z, zenith=float(zen),
                             azimuth=float(azi),
                             energy=float(_np(p['energy']).reshape(-1)[0])))
        return pd.DataFrame(rows, columns=list(TRUTH_COLS))

    def simulate_response(self, om_pos, truth, seed=None):
        """Sample the events' responses once with NuSmoothie at om_pos (16-PMT template)."""
        from nugget.surrogates.NuSmoothie import NuSmoothie, DEFAULT_GEOMETRY
        if self.atime_model is None:
            raise ValueError("simulating a response needs the arrival-time model")
        tmpl = (np.asarray(self.pmt_directions, float) if self.pmt_directions is not None
                else NuSmoothie.pmt_directions_from_geometry(
                    self.geometry_csv_path or DEFAULT_GEOMETRY))
        if self._smoothie is None:
            self._smoothie = NuSmoothie(
                device=self.device, hit_model=self.hit_model, ly_model=self.ly_model,
                atime_model=self.atime_model, pmt_directions=tmpl, pmt_mode='split',
                patd_keys=('hit_times',), n_steps=self.response_n_steps, verbose=False)
        truth = truth.reset_index(drop=True)
        om_pos = np.asarray(om_pos, np.float64).reshape(-1, 3)
        gen = torch.Generator(device=self.device).manual_seed(
            int(self.seed if seed is None else seed))
        om_t = torch.as_tensor(om_pos, device=self.device)
        d = _unit(truth.zenith.to_numpy(float), truth.azimuth.to_numpy(float))
        travel = -d if self.arrival else d
        xyz, E = truth[['x', 'y', 'z']].to_numpy(float), truth.energy.to_numpy(float)
        params = [dict(position=xyz[i], energy=E[i], direction=travel[i])
                  for i in range(len(truth))]

        hits, times, n_flat = [], [], 0
        epb = self.response_events_per_batch or max(len(params), 1)
        for s in range(0, len(params), epb):
            with torch.no_grad():
                r = self._smoothie.sample_response_batched(om_t, params[s:s + epb],
                                                           generator=gen)
            cnt = r['counts'].cpu().numpy().astype(np.int64)    # (B, n_pmt)
            ev, row = np.nonzero(cnt)                           # photon order
            n = cnt[ev, row]
            hits.append(pd.DataFrame(dict(ev=s + ev, pmt_idx=row, n=n,
                                          start=n_flat + np.cumsum(n) - n)))
            times.append(r['t_hit'].cpu().numpy())
            n_flat += int(n.sum())
        return Response(np.repeat(om_pos, len(tmpl), 0), np.tile(tmpl, (len(om_pos), 1)),
                        truth, pd.concat(hits, ignore_index=True),
                        np.concatenate(times) if times else np.empty(0))

    # ---------------- per-event tensors ----------------

    def _event_tensors(self, g, resp, tree, rng, terms):
        """Fixed tensors of one event's -log L (the LY dequantisation is drawn once)."""
        hit, ly, flow = self.hit_model, self.ly_model, self.atime_model
        gi, n = g.pmt_idx.to_numpy(np.int64), g.n.to_numpy(np.int64)
        pos, dirs = resp.pmt_pos, resp.pmt_dir
        d = dict(n_hit_pmts=len(gi), n_photons=int(n.sum()),
                 lit_lo=pos[gi].min(0), lit_hi=pos[gi].max(0),
                 hit_center=(pos[gi] * n[:, None]).sum(0) / n.sum())

        if tree is None:
            sel = np.arange(len(pos))
        else:
            near = tree.query_ball_point(pos[gi], self.hit_radius)
            sel = np.union1d(np.concatenate([np.asarray(l, np.int64) for l in near]), gi)
        d['P_h'] = torch.as_tensor(pos[sel], **_kw(hit))
        d['D_h'] = torch.as_tensor(dirs[sel], **_kw(hit))
        d['y_h'] = torch.as_tensor(np.isin(sel, gi), **_kw(hit))

        d['P_l'] = torch.as_tensor(pos[gi], **_kw(ly))
        d['D_l'] = torch.as_tensor(dirs[gi], **_kw(ly))
        d['z_l'] = ly.to_z(torch.as_tensor(n + rng.random(len(n)), **_kw(ly)))

        if 'atime' in terms:
            first = np.cumsum(n) - n
            ph = np.repeat(g.start.to_numpy(np.int64) - first, n) + np.arange(n.sum())
            owner = np.repeat(np.arange(len(gi)), n)
            d['P_a'] = torch.as_tensor(pos[gi[owner]], **_kw(flow))
            d['D_a'] = torch.as_tensor(dirs[gi[owner]], **_kw(flow))
            d['t_hit'] = torch.as_tensor(resp.times[ph].astype(np.float64), **_kw(flow))
        return d

    _TERM_KEYS = (('h', ('P_h', 'D_h', 'y_h')), ('l', ('P_l', 'D_l', 'z_l')),
                  ('a', ('P_a', 'D_a', 't_hit')))

    def _batch(self, ds):
        """Concatenated per-event tensors; i_* is each row's event index in the batch."""
        b = {'B': len(ds)}
        for key, cols in self._TERM_KEYS:
            if cols[0] not in ds[0]:
                continue
            for c in cols:
                b[c] = torch.cat([d[c] for d in ds])
            dev = b[cols[0]].device
            b['i_' + key] = torch.cat([torch.full((len(d[cols[0]]),), j, dtype=torch.long,
                                                  device=dev) for j, d in enumerate(ds)])
        return b

    # ---------------- likelihood ----------------

    def _chunks(self, b, terms):
        """(term, row slice) pairs covering every row of every term in the batch."""
        for term, key, size in (('hit', 'h', self.hit_chunk_rows),
                                ('ly', 'l', self.flow_chunk_rows),
                                ('atime', 'a', self.flow_chunk_rows)):
            if term in terms and 'i_' + key in b:
                n = len(b['i_' + key])
                size = size or max(n, 1)
                for s in range(0, n, size):
                    yield term, slice(s, min(s + size, n))

    def _term_ll(self, term, th, b, sl, differentiable):
        """Log-likelihood of rows sl of one term and each row's event index; th (B, 6)."""
        hit, ly, flow = self.hit_model, self.ly_model, self.atime_model

        def ctx(m, P, D, t):
            return m.build_context(P, t[:, :3], 10.0 ** t[:, 5], t[:, 3], t[:, 4],
                                   pmt_directions=D)

        if term == 'hit':
            idx, y = b['i_h'][sl], b['y_h'][sl]
            lp1, lp0 = hit.log_prob_hit(ctx(hit, b['P_h'][sl], b['D_h'][sl], th[idx]),
                                        calibrated=True)
            return (y * lp1 + (1 - y) * lp0).double(), idx
        if term == 'ly':
            idx = b['i_l'][sl]
            lp = ly.log_prob_z(b['z_l'][sl], ctx(ly, b['P_l'][sl], b['D_l'][sl], th[idx]),
                               n_steps=self.n_steps, differentiable=differentiable)
            return lp.double(), idx
        idx = b['i_a'][sl]
        t = th[idx]
        tg = flow.geometric_time(b['P_a'][sl], t[:, :3], zeniths=t[:, 3],
                                 azimuths=t[:, 4]).reshape(-1)
        lp = flow.log_prob_time_residual(b['t_hit'][sl] - tg,
                                         ctx(flow, b['P_a'][sl], b['D_a'][sl], t),
                                         n_steps=self.n_steps, differentiable=differentiable)
        return torch.logaddexp(lp, torch.full_like(lp, LOG_FLOOR)).double(), idx

    def _nll(self, th, b, terms):
        """(B,) -log L at th (B, 6), no gradient."""
        out = torch.zeros(b['B'], dtype=torch.float64, device=th.device)
        for term, sl in self._chunks(b, terms):
            ll, idx = self._term_ll(term, th, b, sl, differentiable=False)
            out.index_add_(0, idx, -ll.detach())
        return out

    def _nll_and_grad(self, th0, u, S, b, terms):
        """(B,) -log L at th0 + S u and its gradient in u, accumulated chunk by chunk so peak
        memory is set by the chunk sizes."""
        vals = torch.zeros(b['B'], dtype=torch.float64, device=u.device)
        grad = torch.zeros_like(u)
        for term, sl in self._chunks(b, terms):
            uu = u.detach().requires_grad_(True)
            ll, idx = self._term_ll(term, th0 + S * uu, b, sl, differentiable=True)
            part = torch.zeros_like(vals).index_add(0, idx, -ll)
            vals += part.detach()
            # a non-finite event ends its own fit; keep it out of the others' gradients
            g, = torch.autograd.grad(torch.where(torch.isfinite(part), part, 0.0).sum(), uu)
            grad += g
        return vals, grad

    # ---------------- optimiser ----------------

    def _bounds(self):
        lo, hi = np.full(len(PARAMS), -np.inf), np.full(len(PARAMS), np.inf)
        if self.bound_vertex:
            lo[:3], hi[:3] = -self.half, self.half
        if self.log10e_bounds is not None:
            lo[5], hi[5] = self.log10e_bounds
        return lo, hi

    def _start(self, d, truth, rng, lo, hi):
        if self.start == 'truth':
            th = truth.copy()
        elif self.start == 'perturbed':
            th = truth + self.pert_scale * rng.standard_normal(len(PARAMS))
        else:
            zen, azi = _angles(rng.standard_normal(3))
            e_lo, e_hi = self.log10e_bounds or (2.0, 6.0)
            th = np.r_[rng.uniform(d['lit_lo'], d['lit_hi']), zen, azi,
                       rng.uniform(e_lo, e_hi)]
        if self.vertex_start == 'hit_weighted':
            th[:3] = d['hit_center']
        return np.clip(th, lo, hi)

    def _init_state(self, entries):
        n, dev = len(entries), self.device
        th0 = torch.as_tensor(np.stack([e['start'] for e in entries]), device=dev,
                              dtype=torch.float64)
        z = torch.zeros_like(th0)
        f = lambda v: torch.full((n,), v, device=dev, dtype=torch.float64)
        i = lambda: torch.zeros(n, device=dev, dtype=torch.long)
        return dict(th0=th0, u=z, m=z.clone(), v=z.clone(), best_u=z.clone(),
                    it=i(), since=i(), bad=i(), best_it=i(), best=f(np.inf),
                    sched_best=f(np.inf), nll_start=f(np.nan), lr=f(self.lr), sec=f(0.0))

    def _step(self, s, vals, grad):
        """Adam + per-event plateau LR decay + early stopping, in place. Returns each
        event's stop reason ('' = keep going)."""
        b1, b2 = ADAM_BETAS
        ok = torch.isfinite(vals)
        v = torch.where(ok, vals, torch.full_like(vals, np.inf))
        s['nll_start'] = torch.where(s['it'] == 0, vals, s['nll_start'])
        s['since'] = torch.where(v < s['best'] - self.min_delta, torch.zeros_like(s['since']),
                                 s['since'] + 1)
        better = v < s['best']
        s['best_it'] = torch.where(better, s['it'], s['best_it'])
        s['best'] = torch.where(better, v, s['best'])
        s['best_u'] = torch.where(better[:, None], s['u'], s['best_u'])

        k = (s['it'] + 1).to(grad.dtype)[:, None]
        s['m'] = b1 * s['m'] + (1 - b1) * grad
        s['v'] = b2 * s['v'] + (1 - b2) * grad * grad
        s['u'] = s['u'] - s['lr'][:, None] * (s['m'] / (1 - b1 ** k)) / (
            (s['v'] / (1 - b2 ** k)).sqrt() + ADAM_EPS)

        sb = v < s['sched_best'] - self.min_delta
        s['sched_best'] = torch.where(sb, v, s['sched_best'])
        s['bad'] = torch.where(sb, torch.zeros_like(s['bad']), s['bad'] + 1)
        red = s['bad'] > self.lr_patience
        s['lr'] = torch.where(red, s['lr'] * self.lr_factor, s['lr'])
        s['bad'] = torch.where(red, torch.zeros_like(s['bad']), s['bad'])
        s['it'] = s['it'] + 1

        reason = np.full(len(vals), '', dtype=object)
        for mask, why in ((~ok, 'non_finite'), (s['since'] >= self.patience, 'patience'),
                          (s['lr'] < self.min_lr, 'min_lr'), (s['it'] >= self.max_iter, 'max_iter')):
            reason[(reason == '') & mask.cpu().numpy()] = why
        return reason

    def _finish(self, entries, s, idx, reason, terms, S, lo, hi):
        """Result rows for the events at positions idx of the active batch."""
        sub = [entries[j] for j in idx]
        b = self._batch([e['d'] for e in sub])
        truth = torch.as_tensor(np.stack([e['truth'] for e in sub]), device=self.device,
                                dtype=torch.float64)
        nll_true = self._nll(truth, b, terms).cpu().numpy()
        it = torch.as_tensor(idx, device=self.device)
        with torch.no_grad():
            th_best = (s['th0'][it] + S * s['best_u'][it]).cpu().numpy()
            th_start = s['th0'][it].cpu().numpy()
        cpu = {k: s[k][it].cpu().numpy() for k in ('it', 'best', 'best_it', 'sec', 'nll_start')}
        rows = []
        for k, (e, j) in enumerate(zip(sub, idx)):
            d, tr = e['d'], e['truth']
            fit, start = _canonical(th_best[k]), _canonical(th_start[k])
            rows.append(dict(
                ev=e['ev'], run_id=e['run'], event_id=e['evt'], start_index=e['k'],
                n_hit_pmts=d['n_hit_pmts'], n_photons=d['n_photons'],
                **{f'true_{p}': v for p, v in zip(PARAMS, tr)},
                **{f'start_{p}': v for p, v in zip(PARAMS, start)},
                **{f'fit_{p}': v for p, v in zip(PARAMS, fit)},
                nll_true=float(nll_true[k]), nll_fit=float(cpu['best'][k]),
                nll_start=float(cpu['nll_start'][k]), n_iter=int(cpu['it'][k]),
                best_iter=int(cpu['best_it'][k]), stop_reason=reason[j],
                seconds=float(cpu['sec'][k]),
                dpsi_start=_dpsi(start, tr), dpsi_fit=_dpsi(fit, tr),
                dvertex_start=float(np.linalg.norm(start[:3] - tr[:3])),
                dvertex_fit=float(np.linalg.norm(fit[:3] - tr[:3])),
                dlog10E_start=start[5] - tr[5], dlog10E_fit=fit[5] - tr[5],
                at_bound=bool((np.isclose(fit, lo) | np.isclose(fit, hi)).any())))
        return rows

    def fit(self, resp, tag='', save_path=None, save_every=100):
        """Fit every event of a Response -> one row per event (the best of n_starts), with
        the truth table's extra columns appended. save_path: write the rows every
        save_every finished fits."""
        dev = self.device
        terms = tuple(t for t in self.terms if t != 'atime' or resp.times is not None)
        tree = None if self.hit_radius is None else cKDTree(resp.pmt_pos)
        lo, hi = self._bounds()
        S = torch.as_tensor(self.scale, device=dev, dtype=torch.float64)
        lo_t, hi_t = (torch.as_tensor(x, device=dev, dtype=torch.float64) for x in (lo, hi))
        truth = resp.truth.reset_index(drop=True)
        if self.verbose:
            print(f"{tag}{resp.hits.ev.nunique()} lit events of {len(truth)}, terms {terms}, "
                  f"start {self.start!r}, vertex start {self.vertex_start or self.start!r}, "
                  f"{self.n_starts} start(s), t0 = 0")

        def entries():
            for ev, g in resp.hits.groupby('ev', sort=True):
                r = truth.iloc[ev]
                run, evt = int(r.run_id), int(r.event_id)
                rng = np.random.default_rng([self.seed, abs(run), abs(evt)])  # batch-independent
                d = self._event_tensors(g, resp, tree, rng, terms)
                tr = np.array([r.x, r.y, r.z, r.zenith, r.azimuth, np.log10(r.energy)],
                              dtype=np.float64)
                for k in range(self.n_starts):
                    yield dict(ev=ev, run=run, evt=evt, k=k, d=d, truth=tr,
                               start=self._start(d, tr, rng, lo, hi))

        src, active, state, batch, rows = entries(), [], None, None, []
        next_save, t_wall = save_every, time.time()
        while True:
            room = (None if self.events_per_batch is None
                    else max(self.events_per_batch - len(active), 0))
            new = list(itertools.islice(src, room))
            if new:
                st = self._init_state(new)
                active += new
                state = st if state is None else {k: torch.cat([state[k], st[k]]) for k in st}
                batch = None
            if not active:
                break
            if batch is None:
                batch = self._batch([e['d'] for e in active])

            t_it = time.time()
            vals, grad = self._nll_and_grad(state['th0'], state['u'], S, batch, terms)
            reason = self._step(state, vals, grad)
            state['u'] = (torch.minimum(torch.maximum(state['th0'] + S * state['u'], lo_t), hi_t)
                          - state['th0']) / S
            state['sec'] += (time.time() - t_it) / len(active)

            done = reason != ''
            if not done.any():
                continue
            new_rows = self._finish(active, state, np.flatnonzero(done), reason, terms,
                                    S, lo, hi)
            rows += new_rows
            if self.verbose:
                for r in new_rows:
                    print(f"{tag}[{len(rows):5d}] {r['run_id']}/{r['event_id']}: "
                          f"{r['n_hit_pmts']} PMTs | dpsi {r['dpsi_start']:.2f} -> "
                          f"{r['dpsi_fit']:.2f} deg, dvtx {r['dvertex_start']:.1f} -> "
                          f"{r['dvertex_fit']:.1f} m, dlogE {r['dlog10E_start']:+.3f} -> "
                          f"{r['dlog10E_fit']:+.3f} | NLL fit-truth "
                          f"{r['nll_fit'] - r['nll_true']:+.2f} | {r['n_iter']} it "
                          f"({r['stop_reason']})", flush=True)
            if save_path and save_every and len(rows) >= next_save:
                save_table(pd.DataFrame(rows), save_path)
                next_save += save_every
            keep = ~done
            active = [e for e, k in zip(active, keep) if k]
            kt = torch.as_tensor(keep, device=dev)
            state = {k: v[kt] for k, v in state.items()} if active else None
            batch = None
            if self.empty_cache and torch.device(dev).type == 'cuda':
                del grad
                torch.cuda.empty_cache()

        if self.verbose:
            wall = time.time() - t_wall
            print(f"{tag}done: {len(rows)} fits in {wall / 60:.1f} min")
        res = pd.DataFrame(rows)
        if res.empty:
            return res
        if self.n_starts > 1:
            key = res.nll_fit.where(np.isfinite(res.nll_fit), np.inf)
            res = res.loc[key.groupby(res.ev).idxmin()].assign(n_starts=self.n_starts)
        extra = [c for c in truth.columns if c not in TRUTH_COLS]
        if extra:
            res = res.merge(truth[extra], left_on='ev', right_index=True, how='left')
        return res.drop(columns='ev').reset_index(drop=True)

    # ---------------- loss ----------------

    def __call__(self, geom_dict, **kwargs):
        """kwargs: 'signal_event_params' or ('signal_sampler', 'num_events'); reco_metric (a
        resolution_summary key, default 'dpsi_median'); reco_weight_threshold (keep only
        strings above it); reco_starting_only and reco_containment_margin [m] (vertex inside
        the OMs' convex hull by at least the margin); reco_seed (response sampling)."""
        params = kwargs.get('signal_event_params')
        if params is None:
            sampler = kwargs.get('signal_sampler')
            if sampler is None:
                raise ValueError("provide signal_event_params or a signal_sampler")
            params = sampler.sample_events(int(kwargs.get('num_events', 100)))
        om_pos = self.om_positions(geom_dict, kwargs.get('reco_weight_threshold'))
        truth = self.truth_from_event_params(params)
        truth['hull_dist'] = hull_distance(om_pos, truth[['x', 'y', 'z']].to_numpy())
        if kwargs.get('reco_starting_only', False):
            truth = truth[truth.hull_dist <= -float(kwargs.get('reco_containment_margin', 0.0))]
        res = self.fit(self.simulate_response(om_pos, truth, seed=kwargs.get('reco_seed')))
        summ = resolution_summary(res)
        metric = kwargs.get('reco_metric', 'dpsi_median')
        return {'reco_loss': torch.tensor(summ[metric], dtype=torch.float64, device=self.device),
                'reco_results': res, 'reco_n_requested': len(truth),
                **{f'reco_{k}': v for k, v in summ.items()}}


def save_table(df, path):
    """Write a DataFrame to .parquet or .csv, creating the folder."""
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False) if path.endswith('.csv') else df.to_parquet(path, index=False)
    return df


def read_table(path):
    return pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)

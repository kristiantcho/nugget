"""Gradient-descent maximum-likelihood reconstruction of parquet events with the
hit / light-yield / arrival-time flows. Saves truth, start and best fit per event.
The events can also be re-simulated once in a new geometry (NEW_GEOMETRY).

Events are fitted in batches on each GPU, and sharded over DEVICES (one process each);
the shard files are merged into OUT and deleted at the end."""
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from nugget.surrogates.HitClassifier import HitClassifier
from nugget.surrogates.FlowMatchLY import FlowMatchLY
from nugget.surrogates.FlowMatchATime import FlowMatchATime, _times_to_flat
from nugget.utils.parquet_io import list_events, read_parquet_columns

# one worker process per entry; repeat a device to run several processes on it
DEVICES = ['cuda:1'] if torch.cuda.is_available() else ['cpu']
# PARQUET = './flow_models/mc_atime_muon_flow_v1r_test_samples.parquet'  # events held out from the models' training
PARQUET = './big_mu_accepted.parquet'  # events held out from the models' training
GEOM = '../other/800_40_40_geom.csv'
HIT_CKPT = './flow_models/best_mc_hit_model_v1r.pt'
LY_CKPT = './flow_models/best_mc_ly_muon_flow_model_v1r.pt'
ATIME_CKPT = './flow_models/best_mc_atime_muon_flow_model_v1r.pt'
OUT = './reco/gd_reco_v1r_hex70_pert2.parquet'                  # .parquet or .csv

N_EVENTS = 40000                 # random events drawn from PARQUET
EVENT_SEED = 0
MIN_HIT_PMTS = None               # skip events with fewer lit PMTs (None: no cut)
FILTER_VERTEX_IN_DOMAIN = True # skip vertices outside the models' training domain
TERMS = ('hit', 'ly', 'atime') # 'atime' is dropped automatically if the file has no times

# None fits the parquet's own hits. Otherwise the same events (truth from PARQUET) are
# simulated once with NuSmoothie in a new geometry and that response is fitted: a dict
# of DynamicString kwargs, or a CSV with om_x/om_y/om_z. Every OM gets the 16-PMT
# template of GEOM. The OM positions are saved next to OUT.
# NEW_GEOMETRY = None
NEW_GEOMETRY = dict(hex_type='hexagonal', domain_size=1200, n_strings=70,
                    points_per_string=20, custom_z_spacing=50)
RESPONSE_SEED = 0
RESPONSE_EVENTS_PER_BATCH = 128  # events per NuSmoothie call (~ events x PMTs rows)
RESPONSE_N_STEPS = 64           # ODE steps when sampling the response

# likelihood
# None in any of these three uses every measurement
N_PH_FIT = None                 # fixed random photon subset per event, reweighted
MAX_LY_PMTS = None             # fixed random subset of lit PMTs in the LY term, reweighted
HIT_RADIUS = 200            # unhit PMTs within this of a hit OM enter the hit term [m];
                               # None puts the whole geometry in the hit term
N_STEPS_FIT = 8                # ODE steps per flow evaluation
LOG_FLOOR = np.log(1e-4) - np.log(1e4)   # arrival-time noise floor: 1e-4 over 10 us
# 'fit': t0 is a free parameter. 'geom': t0 follows from the vertex and direction as
# the T0_QUANTILE quantile over lit PMTs of (first photon time - t_geom); 0 is the
# minimum, the latest vertex time at which no PMT sees light before direct light could
# reach it. 'profile': t0 minimises the arrival-time NLL at the current vertex and
# direction. That NLL has a ~1-2 ns wide cusp at the optimum and is concave further
# out, so it is found by a grid search when a fit starts and then followed by a
# three-point search every iteration. 'geom' and 'profile' fit 6 parameters.
T0_MODE = 'profile'
T0_QUANTILE = 0.1
# (half-width, spacing) [ns] of the start search around the 'geom' anchor, coarse to fine
T0_PROFILE_GRID = ((60.0, 5.0), (5.0, 1.0), (1.0, 0.25))
T0_PROFILE_H = 0.5             # per iteration: compare t0 and t0 +- this [ns]; while a
T0_PROFILE_MAX_STEPS = 8       # side wins, move there and double it, up to this many
                               # times (3 atime passes if t0 stays, +2 per move)

# batching: events are added to a batch while both row budgets hold (an event that
# alone exceeds them runs on its own). The budgets only set throughput: the gradient
# is accumulated over row chunks, so peak memory is set by the chunk sizes. Flow rows
# (LY PMTs + photons) cost far more than hit rows: they hold n_steps of double
# backward. Lower the chunk sizes on OOM. None means no limit (a None chunk size
# takes each term in one go).
EVENTS_PER_BATCH = 128
MAX_FLOW_ROWS = None
MAX_HIT_ROWS = None
FLOW_CHUNK_ROWS = 30000
HIT_CHUNK_ROWS = 200_000
EMPTY_CACHE = True             # torch.cuda.empty_cache() whenever events leave the batch

# start point and optimiser. Parameters: x, y, z [m], zenith, azimuth [rad, parquet
# convention], log10 E/GeV, t0 [ns, vertex time on a clock zeroed at the first hit]
START = 'perturbed'            # 'truth' or 'perturbed' (truth + SCALE * N(0, 1))
PARAMS = ['x', 'y', 'z', 'zenith', 'azimuth', 'log10E', 't0']
SCALE = np.array([20.0, 20.0, 20.0, np.radians(5), np.radians(5), 0.3, 20.0])
PERT_SCALE = np.array([70.0, 70.0, 70.0, np.radians(10), np.radians(10), 1, 50.0])
N_STARTS = 1                   # perturbed starts per event (START='perturbed'); the
                               # lowest-NLL fit is kept
# bounds: starts and every step are projected back inside them; None = unbounded
LOG10E_BOUNDS = (2.0, 6.0)     # the models' training range (1e2 .. 1e6 GeV)
BOUND_VERTEX = True            # keep the vertex inside the models' training domain
LR = 0.05                      # Adam step in units of SCALE
BETAS, ADAM_EPS = (0.9, 0.999), 1e-8
MAX_ITER = 1000
LR_FACTOR, LR_PATIENCE = 0.5, 20   # per-event plateau LR decay (as ReduceLROnPlateau)
PATIENCE = 200                 # stop after this many iterations without a MIN_DELTA gain
MIN_DELTA = 1e-2               # NLL units
MIN_LR = 1e-4
WITHIN = 0.5                   # report iterations to get within this NLL of the best
SAVE_EVERY = 100               # finished events between saves of a worker's file


def _kw(m):
    return dict(device=m.device, dtype=m.param_dtype)


def _half_extent(ds):
    ds = np.atleast_1d(np.asarray(ds.tolist() if torch.is_tensor(ds) else ds, float))
    return np.array([ds[0], ds[0], ds[-1]]) / 2.0


def _subset(n, cap, rng):
    """Indices of a fixed random subset of size cap, or all n if cap is None or >= n."""
    if cap is None or cap >= n:
        return np.arange(n)
    return rng.choice(n, cap, replace=False)


def _unit(z, a):
    return np.array([np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)])


def _canonical(th):
    """Map zenith/azimuth back to [0, pi] x [0, 2 pi) via the unit vector."""
    th = th.copy()
    u = _unit(th[3], th[4])
    th[3] = np.arccos(np.clip(u[2], -1, 1))
    th[4] = np.arctan2(u[1], u[0]) % (2 * np.pi)
    return th


def _dpsi(a, b):
    return np.degrees(np.arccos(np.clip(_unit(*a[3:5]) @ _unit(*b[3:5]), -1, 1)))


def sample_events():
    ev_all = list_events(PARQUET)
    rng = np.random.default_rng(EVENT_SEED)
    return [ev_all[i] for i in rng.choice(len(ev_all), min(N_EVENTS, len(ev_all)),
                                          replace=False)]


def load_events(events, terms):
    """The given events' hit rows joined to the geometry, plus counts/times."""
    import pyarrow.dataset as pads
    present = set(pads.dataset(PARQUET, format='parquet').schema.names)
    tcol = 'times' if 'times' in present else 'time' if 'time' in present else None
    if tcol is None and 'atime' in terms:
        print("no 'times'/'time' column: dropping the arrival-time term")
        terms = tuple(t for t in terms if t != 'atime')
    if tcol is None and 'count' not in present:
        raise ValueError("parquet needs a 'times', 'time' or 'count' column")

    cols = ['run_id', 'event_id', 'string', 'om', 'pmt', 'muon_x', 'muon_y', 'muon_z',
            'neutrino_energy', 'zenith', 'azimuth']   # lepton_* is read as muon_*
    extra = [tcol] if tcol else ['count']
    if tcol and 'count' in present:
        extra.append('count')
    df = read_parquet_columns(PARQUET, cols + extra, events=events)

    key = ['run_id', 'event_id', 'string', 'om', 'pmt']
    if tcol == 'time':                                 # photon mode: one row per photon
        first = df.groupby(key, sort=False).first().drop(columns='time')
        df = first.join(df.groupby(key, sort=False)['time'].agg(list).rename('times')
                        ).reset_index()
        tcol = 'times'
    if tcol:
        flat, cnt = _times_to_flat(df['times'])
        df = df.drop(columns='times').assign(_n=cnt)
        df['_start'] = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    else:
        flat = None
        df = df.assign(_n=df['count'].to_numpy(np.int64))
    df = df[df._n > 0]

    geo = pd.read_csv(GEOM, usecols=['string', 'om', 'pmt', 'om_x', 'om_y', 'om_z',
                                     'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'])
    geo_pos = geo[['om_x', 'om_y', 'om_z']].to_numpy(np.float32)
    geo_dir = geo[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy(np.float32)
    geo['_gidx'] = np.arange(len(geo))
    df = df.merge(geo[['string', 'om', 'pmt', '_gidx']], on=['string', 'om', 'pmt'])
    return df, flat, geo_pos, geo_dir, terms


def new_geometry_positions():
    """OM positions (N, 3) of NEW_GEOMETRY."""
    if isinstance(NEW_GEOMETRY, dict):
        from nugget.geometries.DynamicString import DynamicString
        g = DynamicString(device='cpu', dim=3, **NEW_GEOMETRY)
        return g.initialize_points()['points_3d'].detach().cpu().numpy().astype(np.float64)
    om = pd.read_csv(NEW_GEOMETRY, usecols=['om_x', 'om_y', 'om_z']).drop_duplicates()
    return om.to_numpy(np.float64)


def simulate_events(events, models, om_pos, device):
    """Sample the events' responses once with NuSmoothie at om_pos. Returns what
    load_events does, with _gidx indexing the new geometry's PMTs (OM-major)."""
    from nugget.surrogates.NuSmoothie import NuSmoothie
    hit, ly, flow = models
    cols = ['run_id', 'event_id', 'muon_x', 'muon_y', 'muon_z', 'neutrino_energy',
            'zenith', 'azimuth']
    truth = (read_parquet_columns(PARQUET, cols, events=events)
             .groupby(['run_id', 'event_id'], sort=False).first().reset_index())

    tmpl = NuSmoothie.pmt_directions_from_geometry(GEOM)            # (16, 3)
    geo_pos = np.repeat(om_pos, len(tmpl), axis=0).astype(np.float32)
    geo_dir = np.tile(tmpl, (len(om_pos), 1)).astype(np.float32)   # 'split' row order
    smoothie = NuSmoothie(device=device, hit_model=hit, ly_model=ly, atime_model=flow,
                          pmt_directions=tmpl, pmt_mode='split', patd_keys=('hit_times',),
                          n_steps=RESPONSE_N_STEPS, verbose=False)
    gen = torch.Generator(device=device).manual_seed(RESPONSE_SEED)
    om_t = torch.as_tensor(om_pos, device=device)

    # parquet angles are in the models' convention; NuSmoothie takes the travel direction
    zen, azi = truth.zenith.to_numpy(float), truth.azimuth.to_numpy(float)
    d = np.stack([np.sin(zen) * np.cos(azi), np.sin(zen) * np.sin(azi), np.cos(zen)], 1)
    travel = -d if getattr(hit, 'track_dir_is_arrival', False) else d
    params = [dict(position=r[['muon_x', 'muon_y', 'muon_z']].to_numpy(float),
                   energy=float(r.neutrino_energy), direction=travel[i])
              for i, (_, r) in enumerate(truth.iterrows())]

    parts, times, n_flat = [], [], 0
    for s in range(0, len(params), RESPONSE_EVENTS_PER_BATCH):
        with torch.no_grad():
            resp = smoothie.sample_response_batched(
                om_t, params[s:s + RESPONSE_EVENTS_PER_BATCH], generator=gen)
        cnt = resp['counts'].cpu().numpy().astype(np.int64)        # (B, n_pmts)
        ev, row = np.nonzero(cnt)              # (event, PMT) order = photon order
        n = cnt[ev, row]
        p = truth.iloc[s + ev].reset_index(drop=True)
        parts.append(p.assign(_gidx=row, _n=n, _start=n_flat + np.cumsum(n) - n))
        times.append(resp['t_hit'].cpu().numpy().astype(np.float32))
        n_flat += int(n.sum())
    df = pd.concat(parts, ignore_index=True)
    flat = np.concatenate(times) if times else np.empty(0, np.float32)
    n_lit = df.groupby(['run_id', 'event_id']).ngroups if len(df) else 0
    print(f'simulated {len(params)} events in {len(om_pos)} OMs x {len(tmpl)} PMTs: '
          f'{len(params) - n_lit} left no hits and are dropped')
    return df, flat, geo_pos, geo_dir, tuple(TERMS)


def event_data(g, flat, geo_pos, geo_dir, tree, models, rng):
    """Fixed tensors for one event (all subsets drawn once, so the NLL is smooth)."""
    hit, ly, flow = models
    gi = g['_gidx'].to_numpy(np.int64)
    n = g['_n'].to_numpy(np.int64)
    r0 = g.iloc[0]
    d = dict(n_hit_pmts=len(gi), n_photons=int(n.sum()))

    if HIT_RADIUS is None:
        sel = np.arange(len(geo_pos))
    else:
        near = tree.query_ball_point(geo_pos[gi], HIT_RADIUS)
        sel = np.union1d(np.concatenate([np.asarray(l, np.int64) for l in near]), gi)
    d['P_h'] = torch.as_tensor(geo_pos[sel], **_kw(hit))
    d['D_h'] = torch.as_tensor(geo_dir[sel], **_kw(hit))
    d['y_h'] = torch.as_tensor(np.isin(sel, gi), **_kw(hit))

    k = _subset(len(gi), MAX_LY_PMTS, rng)
    d['w_l'] = len(gi) / len(k)
    d['P_l'] = torch.as_tensor(geo_pos[gi[k]], **_kw(ly))
    d['D_l'] = torch.as_tensor(geo_dir[gi[k]], **_kw(ly))
    d['z_l'] = ly.to_z(torch.as_tensor(n[k] + rng.random(len(k)), **_kw(ly)))

    t_first = 0.0
    if flat is not None:
        st = g['_start'].to_numpy(np.int64)
        ph = np.concatenate([np.arange(s, s + c) for s, c in zip(st, n)])
        owner = np.repeat(np.arange(len(gi)), n)
        t_ph = flat[ph].astype(np.float64)
        t_first = t_ph.min()
        d['P_f'] = torch.as_tensor(geo_pos[gi], **_kw(flow))    # lit PMTs, first photon
        d['t_f'] = torch.as_tensor(np.minimum.reduceat(t_ph, np.cumsum(n) - n) - t_first,
                                   **_kw(flow))
        k = _subset(len(ph), N_PH_FIT, rng)
        d['w_a'] = len(ph) / len(k)
        d['P_a'] = torch.as_tensor(geo_pos[gi[owner[k]]], **_kw(flow))
        d['D_a'] = torch.as_tensor(geo_dir[gi[owner[k]]], **_kw(flow))
        d['t_rel'] = torch.as_tensor(t_ph[k] - t_first, **_kw(flow))

    d['t_first'] = t_first
    d['truth'] = np.array([r0.muon_x, r0.muon_y, r0.muon_z, r0.zenith, r0.azimuth,
                           np.log10(r0.neutrino_energy), -t_first], dtype=np.float64)
    d['n_flow_rows'] = len(d['P_l']) + (len(d['P_a']) if 'P_a' in d else 0)
    d['n_hit_rows'] = len(d['P_h'])
    return d


_TERM_KEYS = (('h', ('P_h', 'D_h', 'y_h')), ('l', ('P_l', 'D_l', 'z_l')),
              ('a', ('P_a', 'D_a', 't_rel')), ('f', ('P_f', 't_f')))


def build_batch(ds, device):
    """Concatenate per-event tensors; i_* is each row's event index in the batch."""
    b = {'B': len(ds)}
    for key, cols in _TERM_KEYS:
        if cols[0] not in ds[0]:
            continue
        for c in cols:
            b[c] = torch.cat([d[c] for d in ds])
        b['i_' + key] = torch.cat([torch.full((len(d[cols[0]]),), j, dtype=torch.long,
                                              device=b[cols[0]].device)
                                   for j, d in enumerate(ds)])
    for w in ('w_l', 'w_a'):
        if w in ds[0]:
            b[w] = torch.tensor([d[w] for d in ds], dtype=torch.float64, device=device)
    if 'P_f' in ds[0]:                  # (B, max lit PMTs) row indices, padded with n
        n = np.array([len(d['P_f']) for d in ds])
        cols = np.arange(n.max())
        pad = np.where(cols[None] < n[:, None], (np.cumsum(n) - n)[:, None] + cols[None],
                       n.sum())
        b['f_pad'] = torch.as_tensor(pad, device=b['P_f'].device)
        b['f_n'] = torch.as_tensor(n, device=b['P_f'].device)
    return b


def _anchor_t0(th, b, flow, q):
    """(B,) q-quantile over each event's lit PMTs of (first photon time - t_geom),
    differentiable in vertex and direction."""
    t = th[b['i_f']]
    tg = flow.geometric_time(b['P_f'], t[:, :3], zeniths=t[:, 3],
                             azimuths=t[:, 4]).reshape(-1).to(th.dtype)
    r = b['t_f'].to(th.dtype) - tg
    r = torch.cat([r, r.new_full((1,), float('inf'))])[b['f_pad']].sort(dim=1).values
    pos = q * (b['f_n'] - 1).to(th.dtype)
    lo = pos.floor().long()
    hi = torch.minimum(lo + 1, b['f_n'] - 1)
    w = pos - lo
    return (r.gather(1, lo[:, None])[:, 0] * (1 - w) + r.gather(1, hi[:, None])[:, 0] * w)


def _with_t0(th, b, flow, t0=None):
    """th with t0 replaced: by the given (B,) values, or by the geometric anchor when
    T0_MODE == 'geom'."""
    if t0 is not None:
        return torch.cat([th[:, :6], t0.to(th.dtype)[:, None]], 1)
    if T0_MODE != 'geom' or 'i_f' not in b:
        return th
    return torch.cat([th[:, :6], _anchor_t0(th, b, flow, T0_QUANTILE)[:, None]], 1)


def _atime_nll(th, b, models):
    """(B,) arrival-time NLL only, no gradient."""
    out = torch.zeros(b['B'], dtype=torch.float64, device=th.device)
    for term, sl in _chunks(b, ('atime',)):
        ll, idx = _term_ll(term, th, b, models, sl, differentiable=False)
        out.index_add_(0, idx, -ll.detach())
    return out


def _atime_nll_at(th, t0, b, models):
    return torch.nan_to_num(_atime_nll(_with_t0(th, b, None, t0), b, models),
                            nan=float('inf'))


def _profile_init(th, b, models):
    """(B,) profiled t0 from scratch: coarse-to-fine grid search around the 'geom' anchor."""
    th = th.detach()
    with torch.no_grad():
        t0 = _anchor_t0(th, b, models[2], T0_QUANTILE)
    for half, step in T0_PROFILE_GRID:
        offs = torch.arange(-half, half + 1e-9, step, dtype=t0.dtype, device=t0.device)
        F = torch.stack([_atime_nll_at(th, t0 + o, b, models) for o in offs])
        t0 = t0 + offs[F.argmin(0)]
    return t0


def _track_t0(th, t0, b, models):
    """Follow the minimum as th moves: compare t0 and t0 +- h; while a side wins, move
    there and double h (up to T0_PROFILE_MAX_STEPS). Once the centre wins everywhere,
    refine with the parabola through the three points."""
    th, t0 = th.detach(), t0.detach()
    h = torch.full_like(t0, T0_PROFILE_H)
    f0 = _atime_nll_at(th, t0, b, models)
    for _ in range(T0_PROFILE_MAX_STEPS):
        fm, fp = _atime_nll_at(th, t0 - h, b, models), _atime_nll_at(th, t0 + h, b, models)
        down = fm < torch.minimum(f0, fp)
        up = (fp < f0) & ~down
        if not (down | up).any():
            c = fp - 2 * f0 + fm
            para = torch.where(c > 0, 0.5 * h * (fm - fp) / c.clamp_min(1e-12),
                               torch.zeros_like(t0))
            return t0 + torch.nan_to_num(para, nan=0.0).clamp(-h, h)
        t0 = torch.where(down, t0 - h, torch.where(up, t0 + h, t0))
        f0 = torch.where(down, fm, torch.where(up, fp, f0))
        h = torch.where(down | up, 2 * h, h)
    return t0


def _chunks(b, terms):
    """(term, row slice) pairs covering every row of every term in the batch."""
    for term, key, size in (('hit', 'h', HIT_CHUNK_ROWS), ('ly', 'l', FLOW_CHUNK_ROWS),
                            ('atime', 'a', FLOW_CHUNK_ROWS)):
        if term in terms and 'i_' + key in b:
            n = len(b['i_' + key])
            size = size or max(n, 1)
            for s in range(0, n, size):
                yield term, slice(s, min(s + size, n))


def _term_ll(term, th, b, models, sl, differentiable):
    """Weighted log-likelihood of rows sl of one term, and each row's event index.
    th (B, 7) = [x, y, z, zenith, azimuth, log10 E, t0] per event."""
    hit, ly, flow = models

    def ctx(m, P, D, t):
        return m.build_context(P, t[:, :3], 10.0 ** t[:, 5], t[:, 3], t[:, 4],
                               pmt_directions=D)

    if term == 'hit':
        idx = b['i_h'][sl]
        y = b['y_h'][sl]
        lp1, lp0 = hit.log_prob_hit(ctx(hit, b['P_h'][sl], b['D_h'][sl], th[idx]),
                                    calibrated=True)
        return (y * lp1 + (1 - y) * lp0).double(), idx
    if term == 'ly':
        idx = b['i_l'][sl]
        lp = ly.log_prob_z(b['z_l'][sl], ctx(ly, b['P_l'][sl], b['D_l'][sl], th[idx]),
                           n_steps=N_STEPS_FIT, differentiable=differentiable)
        return b['w_l'][idx] * lp.double(), idx
    idx = b['i_a'][sl]
    t = th[idx]
    tg = flow.geometric_time(b['P_a'][sl], t[:, :3], zeniths=t[:, 3],
                             azimuths=t[:, 4]).reshape(-1)
    lp = flow.log_prob_time_residual(b['t_rel'][sl] - t[:, 6] - tg,
                                     ctx(flow, b['P_a'][sl], b['D_a'][sl], t),
                                     n_steps=N_STEPS_FIT, differentiable=differentiable)
    lp = torch.logaddexp(lp, torch.full_like(lp, LOG_FLOOR))
    return b['w_a'][idx] * lp.double(), idx


def nll_batch(th, b, models, terms, t0=None):
    """(B,) -log L at th (B, 7), no gradient."""
    out = torch.zeros(b['B'], dtype=torch.float64, device=th.device)
    th = _with_t0(th, b, models[2], t0)
    for term, sl in _chunks(b, terms):
        ll, idx = _term_ll(term, th, b, models, sl, differentiable=False)
        out.index_add_(0, idx, -ll.detach())
    return out


def nll_and_grad(th0, u, S, b, models, terms, t0=None):
    """(B,) -log L at th = th0 + S u and its gradient w.r.t. u (B, 7). The NLL is a
    sum over rows, so the gradient is accumulated chunk by chunk and each chunk's
    graph is freed before the next: peak memory is set by the chunk size."""
    vals = torch.zeros(b['B'], dtype=torch.float64, device=u.device)
    grad = torch.zeros_like(u)
    for term, sl in _chunks(b, terms):
        uu = u.detach().requires_grad_(True)
        th = _with_t0(th0 + S * uu, b, models[2], t0)
        ll, idx = _term_ll(term, th, b, models, sl, differentiable=True)
        part = torch.zeros_like(vals).index_add(0, idx, -ll)
        vals += part.detach()
        # a non-finite event ends its own fit; keep it out of the others' gradients
        g, = torch.autograd.grad(torch.where(torch.isfinite(part), part, 0.0).sum(), uu)
        grad += g
    return vals, grad


# per-event optimiser state, one row per active event
_TENSOR_STATE = ('th0', 'u', 'm', 'v', 'best_u', 'tp', 'tp0', 'best_tp')  # tp: profiled t0
_ARRAY_STATE = ('it', 'since', 'best', 'best_it', 'sched_best', 'bad', 'lr', 'sec')


def _init_state(entries, device):
    n = len(entries)
    th0 = torch.as_tensor(np.stack([e['start'] for e in entries]), device=device,
                          dtype=torch.float64)
    z = torch.zeros_like(th0)
    return dict(th0=th0, u=z, m=z.clone(), v=z.clone(), best_u=z.clone(),
                tp=th0[:, 6].clone(), tp0=th0[:, 6].clone(), best_tp=th0[:, 6].clone(),
                it=np.zeros(n, int), since=np.zeros(n, int), best=np.full(n, np.inf),
                best_it=np.zeros(n, int), sched_best=np.full(n, np.inf),
                bad=np.zeros(n, int), lr=np.full(n, LR), sec=np.zeros(n),
                hist=[[] for _ in range(n)])


def _cat_state(a, b):
    if a is None:
        return b
    out = {k: torch.cat([a[k], b[k]]) for k in _TENSOR_STATE}
    out.update({k: np.concatenate([a[k], b[k]]) for k in _ARRAY_STATE})
    out['hist'] = a['hist'] + b['hist']
    return out


def _keep_state(s, keep):
    kt = torch.as_tensor(keep, device=s['u'].device)
    out = {k: s[k][kt] for k in _TENSOR_STATE}
    out.update({k: s[k][keep] for k in _ARRAY_STATE})
    out['hist'] = [h for h, k in zip(s['hist'], keep) if k]
    return out


def _step(s, vals, ok, grad):
    """One Adam + plateau + early-stopping update for every active event (in place).
    Returns each event's stop reason ('' = keep going)."""
    b1, b2 = BETAS
    for j in np.flatnonzero(ok):
        s['hist'][j].append(float(vals[j]))
    v = np.where(ok, vals, np.inf)
    s['since'] = np.where(v < s['best'] - MIN_DELTA, 0, s['since'] + 1)
    better = v < s['best']
    s['best_it'] = np.where(better, s['it'], s['best_it'])
    s['best'] = np.where(better, v, s['best'])
    bt = torch.as_tensor(better, device=s['u'].device)
    s['best_u'] = torch.where(bt[:, None], s['u'], s['best_u'])
    s['best_tp'] = torch.where(bt, s['tp'], s['best_tp'])

    step = torch.as_tensor(s['it'] + 1, device=grad.device, dtype=grad.dtype)[:, None]
    s['m'] = b1 * s['m'] + (1 - b1) * grad
    s['v'] = b2 * s['v'] + (1 - b2) * grad * grad
    lr = torch.as_tensor(s['lr'], device=grad.device, dtype=grad.dtype)[:, None]
    s['u'] = s['u'] - lr * (s['m'] / (1 - b1 ** step)) / (
        (s['v'] / (1 - b2 ** step)).sqrt() + ADAM_EPS)

    sb = v < s['sched_best'] - MIN_DELTA
    s['sched_best'] = np.where(sb, v, s['sched_best'])
    s['bad'] = np.where(sb, 0, s['bad'] + 1)
    red = s['bad'] > LR_PATIENCE
    s['lr'] = np.where(red, s['lr'] * LR_FACTOR, s['lr'])
    s['bad'] = np.where(red, 0, s['bad'])
    s['it'] = s['it'] + 1

    reason = np.full(len(vals), '', dtype=object)
    for mask, why in ((~ok, 'non_finite'), (s['since'] >= PATIENCE, 'patience'),
                      (s['lr'] < MIN_LR, 'min_lr'), (s['it'] >= MAX_ITER, 'max_iter')):
        reason[(reason == '') & mask] = why
    return reason


def _bounds(half):
    """(lo, hi) over the 7 parameters, +-inf where unbounded."""
    lo, hi = np.full(len(PARAMS), -np.inf), np.full(len(PARAMS), np.inf)
    if BOUND_VERTEX:
        lo[:3], hi[:3] = -half, half
    if LOG10E_BOUNDS is not None:
        lo[5], hi[5] = LOG10E_BOUNDS
    return lo, hi


def _project(s, S, lo, hi):
    """Clamp th = th0 + S u into [lo, hi], in place on u (projected gradient descent)."""
    th = torch.minimum(torch.maximum(s['th0'] + S * s['u'], lo), hi)
    s['u'] = (th - s['th0']) / S


def _combine(rs, lo, hi):
    """One row per event from its starts' rows: the lowest finite NLL wins."""
    ok = [r for r in rs if np.isfinite(r['nll_fit'])] or rs
    best = dict(min(ok, key=lambda r: r['nll_fit']))
    nll = np.array([r['nll_fit'] for r in rs])
    fit = np.array([best[f'fit_{p}'] for p in PARAMS])
    best.update(n_starts=len(rs),
                n_starts_within=int((nll <= best['nll_fit'] + WITHIN).sum()),
                seconds=float(sum(r['seconds'] for r in rs)),
                at_bound=bool((np.isclose(fit, lo) | np.isclose(fit, hi)).any()))
    return best


def _finish(entries, s, idx, reason, models, terms, device):
    """Result rows for the events at positions idx of the active batch."""
    sub = [entries[j] for j in idx]
    truth = torch.as_tensor(np.stack([e['d']['truth'] for e in sub]), device=device)
    b_sub = build_batch([e['d'] for e in sub], device)
    profile = T0_MODE == 'profile' and 'i_a' in b_sub
    t0_true = _profile_init(truth, b_sub, models) if profile else None
    nll_true = nll_batch(truth, b_sub, models, terms, t0_true).cpu().numpy()
    S = torch.as_tensor(SCALE, device=device, dtype=torch.float64)
    with torch.no_grad():
        th_best = _with_t0(s['th0'][idx] + S * s['best_u'][idx], b_sub, models[2],
                           s['best_tp'][idx] if profile else None)
        th_start = _with_t0(s['th0'][idx], b_sub, models[2],
                            s['tp0'][idx] if profile else None)
    th_best, th_start = th_best.detach().cpu().numpy(), th_start.detach().cpu().numpy()
    rows = []
    for k, j in enumerate(idx):
        e, d = entries[j], entries[j]['d']
        hist = np.asarray(s['hist'][j])
        best = float(s['best'][j])
        th, tr = _canonical(th_best[k]), d['truth']
        rows.append(dict(
            run_id=e['run'], event_id=e['evt'], start_index=e['k'],
            n_hit_pmts=d['n_hit_pmts'],
            n_photons=d['n_photons'], t_first=d['t_first'],
            **{f'true_{p}': v for p, v in zip(PARAMS, tr)},
            **{f'start_{p}': v for p, v in zip(PARAMS, _canonical(th_start[k]))},
            **{f'fit_{p}': v for p, v in zip(PARAMS, th)},
            nll_true=float(nll_true[k]), nll_fit=best,
            nll_start=hist[0] if len(hist) else np.nan,
            n_iter=len(hist), best_iter=int(s['best_it'][j]),
            iter_within=int(np.argmax(hist <= best + WITHIN)) if len(hist) else -1,
            stop_reason=reason[j], seconds=float(s['sec'][j]),
            dpsi_start=_dpsi(e['start'], tr), dpsi_fit=_dpsi(th, tr),
            dvertex_fit=float(np.linalg.norm(th[:3] - tr[:3])),
            dlog10E_fit=th[5] - tr[5], dt0_fit=th[6] - tr[6]))
    return rows


def _save(rows, path):
    res = pd.DataFrame(rows)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    (res.to_csv(path, index=False) if path.endswith('.csv') else res.to_parquet(path, index=False))
    return res


def _read(path):
    return pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)


def run_worker(events, device, out_path, tag='', om_pos=None):
    """Fit the given events on one device, in batches; writes rows to out_path."""
    if T0_MODE not in ('fit', 'geom', 'profile'):
        raise ValueError("T0_MODE must be 'fit', 'geom' or 'profile'")
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    hit = HitClassifier(device=device).load_model(HIT_CKPT)
    ly = FlowMatchLY(device=device).load_model(LY_CKPT)
    flow = FlowMatchATime(device=device).load_model(ATIME_CKPT)
    models = (hit, ly, flow)
    for m in models:
        m.net.eval()
        for p in m.net.parameters():
            p.requires_grad_(False)

    if om_pos is None:
        df, flat, geo_pos, geo_dir, terms = load_events(events, TERMS)
    else:
        df, flat, geo_pos, geo_dir, terms = simulate_events(events, models, om_pos, device)
    tree = cKDTree(geo_pos)
    half = _half_extent(hit.domain_size)
    groups = df.groupby(['run_id', 'event_id'], sort=False)
    print(f'{tag}{groups.ngroups} events on {device}, terms {terms}, '
          f'atime particle_mode {flow.particle_mode}, start {START!r}')
    skipped = {'few_hits': 0, 'vertex_outside': 0}
    lo, hi = _bounds(half)
    n_starts = N_STARTS if START == 'perturbed' else 1
    profile = T0_MODE == 'profile' and 'atime' in terms

    def prepared():
        for (run, evt), g in groups:
            if MIN_HIT_PMTS is not None and len(g) < MIN_HIT_PMTS:
                skipped['few_hits'] += 1
                continue
            if FILTER_VERTEX_IN_DOMAIN and (np.abs(g.iloc[0][['muon_x', 'muon_y', 'muon_z']]
                                                   .to_numpy(float)) > half).any():
                skipped['vertex_outside'] += 1
                continue
            rng = np.random.default_rng([EVENT_SEED, int(run), int(evt)])  # batch-independent
            d = event_data(g, flat, geo_pos, geo_dir, tree, models, rng)
            pert = (PERT_SCALE * rng.standard_normal((n_starts, len(SCALE)))
                    if START == 'perturbed' else np.zeros((1, len(SCALE))))
            for k in range(n_starts):       # the starts share d's tensors
                yield dict(run=run, evt=evt, d=d, k=k,
                           start=np.clip(d['truth'] + pert[k], lo, hi))

    def fits(entries, e):
        nf = sum(x['d']['n_flow_rows'] for x in entries) + e['d']['n_flow_rows']
        nh = sum(x['d']['n_hit_rows'] for x in entries) + e['d']['n_hit_rows']
        return ((MAX_FLOW_ROWS is None or nf <= MAX_FLOW_ROWS)
                and (MAX_HIT_ROWS is None or nh <= MAX_HIT_ROWS))

    S = torch.as_tensor(SCALE, device=device, dtype=torch.float64)
    lo_t, hi_t = (torch.as_tensor(x, device=device, dtype=torch.float64) for x in (lo, hi))
    src, waiting, exhausted = prepared(), None, False
    active, state, batch, rows, pending = [], None, None, [], {}
    t_wall = time.time()
    while True:
        new = []
        while not exhausted and (EVENTS_PER_BATCH is None
                                 or len(active) + len(new) < EVENTS_PER_BATCH):
            if waiting is None:
                waiting = next(src, None)
                if waiting is None:
                    exhausted = True
                    break
            if (active or new) and not fits(active + new, waiting):
                break
            new.append(waiting)
            waiting = None
        if new:
            st_new = _init_state(new, device)
            if profile:
                b_new = build_batch([e['d'] for e in new], device)
                st_new['tp'] = _profile_init(st_new['th0'], b_new, models)
                st_new['tp0'], st_new['best_tp'] = st_new['tp'].clone(), st_new['tp'].clone()
            active += new
            state = _cat_state(state, st_new)
            batch = None
        if not active:
            break
        if batch is None:
            batch = build_batch([e['d'] for e in active], device)

        t_it = time.time()
        t0 = None
        if profile:      # envelope theorem: at the profiled t0 the other gradients hold
            state['tp'] = _track_t0(state['th0'] + S * state['u'], state['tp'], batch, models)
            t0 = state['tp']
        vals, grad = nll_and_grad(state['th0'], state['u'], S, batch, models, terms, t0)
        vals = vals.cpu().numpy()
        ok = np.isfinite(vals)
        reason = _step(state, vals, ok, grad)
        _project(state, S, lo_t, hi_t)
        state['sec'] = state['sec'] + (time.time() - t_it) / len(active)

        done = reason != ''
        if done.any():
            for r in _finish(active, state, np.flatnonzero(done), reason, models, terms, device):
                key = (r['run_id'], r['event_id'])
                pending.setdefault(key, []).append(r)
                if len(pending[key]) < n_starts:
                    continue
                r = _combine(pending.pop(key), lo, hi)
                rows.append(r)
                print(f'{tag}[{len(rows):5d}] {r["run_id"]}/{r["event_id"]}: '
                      f'{r["n_hit_pmts"]} PMTs, {r["n_photons"]} ph | dpsi '
                      f'{r["dpsi_start"]:.2f} -> {r["dpsi_fit"]:.2f} deg, dvtx '
                      f'{r["dvertex_fit"]:.1f} m, dlogE {r["dlog10E_fit"]:+.3f} | NLL fit-truth '
                      f'{r["nll_fit"] - r["nll_true"]:+.2f} | best @ {r["best_iter"]}/'
                      f'{r["n_iter"]} ({r["stop_reason"]}) | start {r["start_index"]}, '
                      f'{r["n_starts_within"]}/{n_starts} agree'
                      f'{" | at bound" if r["at_bound"] else ""}', flush=True)
                if len(rows) % SAVE_EVERY == 0:
                    _save(rows, out_path)
            keep = ~done
            active = [e for e, k in zip(active, keep) if k]
            state = _keep_state(state, keep) if active else None
            batch = None
            if EMPTY_CACHE and device.startswith('cuda'):
                del grad
                torch.cuda.empty_cache()

    _save(rows, out_path)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    wall = time.time() - t_wall
    print(f'{tag}done: {len(rows)} events in {wall / 60:.1f} min '
          f'({60 * len(rows) / max(wall, 1e-9):.1f} events/min); skipped {skipped}')


def _worker(i, events, device, path, om_pos):
    run_worker(events, device, path, tag=f'[shard {i}] ', om_pos=om_pos)


def _shard_path(i):
    root, ext = os.path.splitext(OUT)
    return f'{root}.shard{i}{ext}'


def summarize(res, wall=None):
    if res.empty:
        print('no events fitted')
        return
    q = lambda c, p=(50, 68, 90): ' / '.join(f'{np.nanpercentile(res[c], x):.3g}' for x in p)
    print(f'\n===== {len(res)} events =====')
    print('stop reasons     :', res.stop_reason.value_counts().to_dict())
    print(f'iterations run   : p50/p68/p90 {q("n_iter")}  (max {res.n_iter.max()})')
    print(f'iter of best NLL : p50/p68/p90 {q("best_iter")}')
    print(f'iter to within {WITHIN} of best: p50/p68/p90 {q("iter_within")}')
    if wall is not None:
        print(f'wall time        : {wall / 60:.1f} min, {60 * len(res) / wall:.1f} events/min')
    print(f'GPU time / event : median {res.seconds.median():.2f}s (share of its batches)')
    print(f'dpsi start [deg] : p50/p68/p90 {q("dpsi_start")}')
    print(f'dpsi fit   [deg] : p50/p68/p90 {q("dpsi_fit")}')
    print(f'|d vertex| [m]   : p50/p68/p90 {q("dvertex_fit")}')
    print(f'd log10E         : median {res.dlog10E_fit.median():+.3f}, '
          f'p16-p84 half-width {(res.dlog10E_fit.quantile(.84) - res.dlog10E_fit.quantile(.16)) / 2:.3f}')
    print(f'd t0 [ns]        : median {res.dt0_fit.median():+.1f}, '
          f'p16-p84 half-width {(res.dt0_fit.quantile(.84) - res.dt0_fit.quantile(.16)) / 2:.1f}')
    dn = res.nll_fit - res.nll_true
    print(f'NLL fit - truth  : median {dn.median():+.2f}; fit <= truth + {WITHIN} in '
          f'{(dn <= WITHIN).mean():.0%} of events (else the optimiser stopped short of the truth)')
    if START == 'perturbed':
        print(f'fit closer than start in direction: {(res.dpsi_fit < res.dpsi_start).mean():.0%}')
    if 'n_starts' in res and res.n_starts.max() > 1:
        print(f'starts within {WITHIN} NLL of the best: '
              f'{res.n_starts_within.value_counts().sort_index().to_dict()} '
              f'(of {res.n_starts.max()}; 1 = only one start found that minimum)')
    if 'at_bound' in res:
        print(f'best fit at a bound (vertex / energy): {res.at_bound.mean():.1%}')


def main():
    t0 = time.time()
    events = sample_events()
    om_pos = None
    if NEW_GEOMETRY is not None:
        om_pos = new_geometry_positions()
        path = os.path.splitext(OUT)[0] + '.om_positions.npy'
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.save(path, om_pos)
        print(f'new geometry: {len(om_pos)} OMs -> {path}')
    n = len(DEVICES)
    if n == 1:
        run_worker(events, DEVICES[0], OUT, om_pos=om_pos)
    else:
        ctx = torch.multiprocessing.get_context('spawn')
        procs = [ctx.Process(target=_worker,
                             args=(i, events[i::n], dev, _shard_path(i), om_pos))
                 for i, dev in enumerate(DEVICES)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        paths = [_shard_path(i) for i in range(n)]
        found = [p for p in paths if os.path.exists(p)]
        if not found:
            raise RuntimeError('no shard produced any output')
        res = pd.concat([_read(p) for p in found], ignore_index=True)
        _save(res, OUT)
        failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            print(f'shards {failed} failed: merged what exists, kept the shard files')
        else:
            for p in found:
                os.remove(p)
    res = _read(OUT)
    print(f'saved {len(res)} rows -> {OUT}')
    summarize(res, wall=time.time() - t0)


if __name__ == '__main__':
    main()

"""Gradient-descent maximum-likelihood reconstruction of parquet events with the
hit / light-yield / arrival-time flows. Saves truth, start and best fit per event."""
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

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# PARQUET = './flow_models/mc_atime_muon_flow_v1r_test_samples.parquet'  # events held out from the models' training
PARQUET = './big_mu_accepted.parquet'  # events held out from the models' training
GEOM = '../other/800_40_40_geom.csv'
HIT_CKPT = './flow_models/best_mc_hit_model_v1r.pt'
LY_CKPT = './flow_models/best_mc_ly_muon_flow_model_v1r.pt'
ATIME_CKPT = './flow_models/best_mc_atime_muon_flow_model_v1r.pt'
OUT = './reco/gd_reco_v1r.parquet'                  # .parquet or .csv

N_EVENTS = 20000                 # random events drawn from PARQUET
EVENT_SEED = 0
MIN_HIT_PMTS = None               # skip events with fewer lit PMTs
FILTER_VERTEX_IN_DOMAIN = True # skip vertices outside the models' training domain
TERMS = ('hit', 'ly', 'atime') # 'atime' is dropped automatically if the file has no times

# likelihood
# None in any of these three uses every measurement
N_PH_FIT = None                 # fixed random photon subset per event, reweighted
MAX_LY_PMTS = None             # fixed random subset of lit PMTs in the LY term, reweighted
HIT_RADIUS = 200            # unhit PMTs within this of a hit OM enter the hit term [m];
                               # None puts the whole geometry in the hit term
N_STEPS_FIT = 8                # ODE steps per flow evaluation
LOG_FLOOR = np.log(1e-4) - np.log(1e4)   # arrival-time noise floor: 1e-4 over 10 us

# start point and optimiser. Parameters: x, y, z [m], zenith, azimuth [rad, parquet
# convention], log10 E/GeV, t0 [ns, vertex time on a clock zeroed at the first hit]
START = 'truth'            # 'truth' or 'perturbed' (truth + SCALE * N(0, 1))
PARAMS = ['x', 'y', 'z', 'zenith', 'azimuth', 'log10E', 't0']
SCALE = np.array([20.0, 20.0, 20.0, np.radians(5), np.radians(5), 0.3, 20.0])
LR = 0.05                      # Adam step in units of SCALE
MAX_ITER = 1000
LR_FACTOR, LR_PATIENCE = 0.5, 30   # ReduceLROnPlateau on the NLL
PATIENCE = 100                 # stop after this many iterations without a MIN_DELTA gain
MIN_DELTA = 1e-2               # NLL units
MIN_LR = 1e-4
WITHIN = 0.5                   # report iterations to get within this NLL of the best
SAVE_EVERY = 10


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


def load_events(terms):
    """Random N_EVENTS events: their hit rows joined to the geometry, plus counts/times."""
    import pyarrow.dataset as pads
    present = set(pads.dataset(PARQUET, format='parquet').schema.names)
    tcol = 'times' if 'times' in present else 'time' if 'time' in present else None
    if tcol is None and 'atime' in terms:
        print("no 'times'/'time' column: dropping the arrival-time term")
        terms = tuple(t for t in terms if t != 'atime')
    if tcol is None and 'count' not in present:
        raise ValueError("parquet needs a 'times', 'time' or 'count' column")

    ev_all = list_events(PARQUET)
    rng = np.random.default_rng(EVENT_SEED)
    pick = [ev_all[i] for i in rng.choice(len(ev_all), min(N_EVENTS, len(ev_all)),
                                          replace=False)]
    cols = ['run_id', 'event_id', 'string', 'om', 'pmt', 'muon_x', 'muon_y', 'muon_z',
            'neutrino_energy', 'zenith', 'azimuth']   # lepton_* is read as muon_*
    extra = [tcol] if tcol else ['count']
    if tcol and 'count' in present:
        extra.append('count')
    df = read_parquet_columns(PARQUET, cols + extra, events=pick)

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
        k = _subset(len(ph), N_PH_FIT, rng)
        d['w_a'] = len(ph) / len(k)
        d['P_a'] = torch.as_tensor(geo_pos[gi[owner[k]]], **_kw(flow))
        d['D_a'] = torch.as_tensor(geo_dir[gi[owner[k]]], **_kw(flow))
        d['t_rel'] = torch.as_tensor(t_ph[k] - t_first, **_kw(flow))

    d['t_first'] = t_first
    d['truth'] = np.array([r0.muon_x, r0.muon_y, r0.muon_z, r0.zenith, r0.azimuth,
                           np.log10(r0.neutrino_energy), -t_first], dtype=np.float64)
    return d


def nll(th, d, models, terms, differentiable=True):
    """th = [x, y, z, zenith, azimuth, log10 E, t0] -> -log L."""
    hit, ly, flow = models
    V, Z, A, E, t0 = th[:3], th[3], th[4], 10.0 ** th[5], th[6]

    def ctx(m, P, D):
        n = len(P)
        return m.build_context(P, V.expand(n, 3), E.expand(n), Z.expand(n), A.expand(n),
                               pmt_directions=D)

    ll = 0.0
    if 'hit' in terms:
        lp1, lp0 = hit.log_prob_hit(ctx(hit, d['P_h'], d['D_h']), calibrated=True)
        ll = ll + (d['y_h'] * lp1 + (1 - d['y_h']) * lp0).sum()
    if 'ly' in terms:
        ll = ll + d['w_l'] * ly.log_prob_z(d['z_l'], ctx(ly, d['P_l'], d['D_l']),
                                           n_steps=N_STEPS_FIT,
                                           differentiable=differentiable).sum()
    if 'atime' in terms:
        n = len(d['P_a'])
        tg = flow.geometric_time(d['P_a'], V.expand(n, 3), zeniths=Z.expand(n),
                                 azimuths=A.expand(n)).reshape(-1)
        lp = flow.log_prob_time_residual(d['t_rel'] - t0 - tg,
                                         ctx(flow, d['P_a'], d['D_a']),
                                         n_steps=N_STEPS_FIT,
                                         differentiable=differentiable)
        ll = ll + d['w_a'] * torch.logaddexp(lp, torch.full_like(lp, LOG_FLOOR)).sum()
    return -ll


def fit(d, start, models, terms):
    """Adam in SCALE units with plateau LR decay and early stopping. Keeps the best point."""
    S = torch.as_tensor(SCALE, device=DEVICE, dtype=torch.float64)
    th0 = torch.as_tensor(start, device=DEVICE, dtype=torch.float64)
    u = torch.zeros(len(SCALE), device=DEVICE, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([u], lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=LR_FACTOR, patience=LR_PATIENCE, threshold=MIN_DELTA,
        threshold_mode='abs')
    hist, best, best_u, best_it, since, reason = [], np.inf, u.detach().clone(), 0, 0, 'max_iter'
    for it in range(MAX_ITER):
        loss = nll(th0 + S * u, d, models, terms)
        val = loss.item()
        if not np.isfinite(val):
            reason = 'non_finite'
            break
        hist.append(val)
        since = 0 if val < best - MIN_DELTA else since + 1
        if val < best:
            best, best_u, best_it = val, u.detach().clone(), it
        u.grad, = torch.autograd.grad(loss, u)     # grad w.r.t. th only, not the nets
        opt.step()
        sched.step(val)
        if since >= PATIENCE:
            reason = 'patience'
            break
        if opt.param_groups[0]['lr'] < MIN_LR:
            reason = 'min_lr'
            break
    hist = np.asarray(hist)
    within = int(np.argmax(hist <= best + WITHIN)) if len(hist) else -1
    th = (th0 + S * best_u).detach().cpu().numpy()
    return dict(theta=_canonical(th), nll_fit=best, nll_start=hist[0] if len(hist) else np.nan,
                n_iter=len(hist), best_iter=best_it, iter_within=within, stop_reason=reason)


def summarize(res):
    if res.empty:
        print('no events fitted')
        return
    q = lambda c, p=(50, 68, 90): ' / '.join(f'{np.nanpercentile(res[c], x):.3g}' for x in p)
    print(f'\n===== {len(res)} events =====')
    print('stop reasons     :', res.stop_reason.value_counts().to_dict())
    print(f'iterations run   : p50/p68/p90 {q("n_iter")}  (max {res.n_iter.max()})')
    print(f'iter of best NLL : p50/p68/p90 {q("best_iter")}')
    print(f'iter to within {WITHIN} of best: p50/p68/p90 {q("iter_within")}')
    print(f'time per event   : median {res.seconds.median():.1f}s, total {res.seconds.sum() / 60:.1f} min, '
          f'{(res.seconds / res.n_iter).median() * 1e3:.0f} ms/iter')
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


def save(rows):
    res = pd.DataFrame(rows)
    d = os.path.dirname(OUT)
    if d:
        os.makedirs(d, exist_ok=True)
    (res.to_csv(OUT, index=False) if OUT.endswith('.csv') else res.to_parquet(OUT, index=False))
    return res


def main():
    hit = HitClassifier(device=DEVICE).load_model(HIT_CKPT)
    ly = FlowMatchLY(device=DEVICE).load_model(LY_CKPT)
    flow = FlowMatchATime(device=DEVICE).load_model(ATIME_CKPT)
    models = (hit, ly, flow)
    for m in models:
        m.net.eval()
        for p in m.net.parameters():
            p.requires_grad_(False)
    print(f'atime particle_mode: {flow.particle_mode}')

    df, flat, geo_pos, geo_dir, terms = load_events(TERMS)
    tree = cKDTree(geo_pos)
    half = _half_extent(hit.domain_size)
    groups = list(df.groupby(['run_id', 'event_id'], sort=False))
    print(f'{len(groups)} events loaded, terms {terms}, start {START!r}, device {DEVICE}')

    rows, skipped = [], {'few_hits': 0, 'vertex_outside': 0}
    for i, ((run, evt), g) in enumerate(groups):
        if len(g) < MIN_HIT_PMTS:
            skipped['few_hits'] += 1
            continue
        if FILTER_VERTEX_IN_DOMAIN and (np.abs(g.iloc[0][['muon_x', 'muon_y', 'muon_z']]
                                               .to_numpy(float)) > half).any():
            skipped['vertex_outside'] += 1
            continue
        rng = np.random.default_rng([EVENT_SEED, i])
        d = event_data(g, flat, geo_pos, geo_dir, tree, models, rng)
        truth = d['truth']
        start = truth + (SCALE * rng.standard_normal(len(SCALE)) if START == 'perturbed' else 0.0)

        t = time.time()
        r = fit(d, start, models, terms)
        sec = time.time() - t
        nll_true = nll(torch.as_tensor(truth, device=DEVICE), d, models, terms,
                       differentiable=False).item()
        th = r.pop('theta')
        row = dict(run_id=run, event_id=evt, n_hit_pmts=d['n_hit_pmts'],
                   n_photons=d['n_photons'], t_first=d['t_first'],
                   **{f'true_{p}': v for p, v in zip(PARAMS, truth)},
                   **{f'start_{p}': v for p, v in zip(PARAMS, _canonical(start))},
                   **{f'fit_{p}': v for p, v in zip(PARAMS, th)},
                   nll_true=nll_true, **r, seconds=sec,
                   dpsi_start=_dpsi(start, truth), dpsi_fit=_dpsi(th, truth),
                   dvertex_fit=float(np.linalg.norm(th[:3] - truth[:3])),
                   dlog10E_fit=th[5] - truth[5], dt0_fit=th[6] - truth[6])
        rows.append(row)
        print(f'[{len(rows):4d}] {run}/{evt}: {d["n_hit_pmts"]} PMTs, {d["n_photons"]} ph | '
              f'dpsi {row["dpsi_start"]:.2f} -> {row["dpsi_fit"]:.2f} deg, '
              f'dvtx {row["dvertex_fit"]:.1f} m, dlogE {row["dlog10E_fit"]:+.3f} | '
              f'NLL fit-truth {row["nll_fit"] - nll_true:+.2f} | best @ {r["best_iter"]}/'
              f'{r["n_iter"]} ({r["stop_reason"]}), {sec:.1f}s')
        if len(rows) % SAVE_EVERY == 0:
            save(rows)

    res = save(rows)
    print(f'skipped: {skipped}; saved {len(res)} rows -> {OUT}')
    summarize(res)


if __name__ == '__main__':
    main()

"""Gradient-descent maximum-likelihood reconstruction test of many events with RecoGDLoss.

CASE picks the data and the event selection:
  'parquet'            the parquet's own photons in its geometry (GEOM)
  'parquet_starting'   same, only starting tracks (vertex inside GEOM's OM hull)
  'simulate'           the parquet events re-simulated once with NuSmoothie in NEW_GEOMETRY
  'simulate_starting'  same, only starting tracks (vertex inside NEW_GEOMETRY's OM hull)
Run as `python reco_flow_gd.py [CASE]`. Writes <stem>.events.parquet (the chosen events and
their truth, before fitting) and OUT (one row per fitted event), sharding over DEVICES."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from nugget.losses.reco_gd import (RecoGDLoss, energy_table, event_table, half_extent,
                                   hull_distance, parquet_response, read_geometry, read_table,
                                   resolution_summary, save_table)

# one worker process per entry; repeat a device to run several processes on it
DEVICES = ['cuda:1'] if torch.cuda.is_available() else ['cpu']
PARQUET = './big_mu_accepted.parquet'
GEOM = '../other/800_40_40_geom.csv'
HIT_CKPT = './flow_models/best_mc_muon_hit_model_v1rl.pt'
LY_CKPT = './flow_models/best_mc_ly_muon_flow_model_v1rl.pt'
ATIME_CKPT = './flow_models/best_mc_atime_muon_flow_model_v1rlg.pt'

CASES = {
    'parquet': dict(source='parquet', starting=False),
    'parquet_starting': dict(source='parquet', starting=True),
    'simulate': dict(source='simulate', starting=False),
    'simulate_starting': dict(source='simulate', starting=True),
}
CASE = 'parquet_starting'
OUT = './reco/gd_reco_v1rl_{case}.parquet'          # .parquet or .csv
EVENT_CACHE = './reco/event_table_big_mu.parquet'  # truth of every PARQUET event; None = rebuild

# event selection
N_EVENTS = 40000                 # random events drawn from those passing the cuts
EVENT_SEED = 0
FILTER_VERTEX_IN_DOMAIN = True   # skip vertices outside the models' training domain
CONTAINMENT_MARGIN = 0.0         # starting tracks: vertex at least this far inside the hull [m]

# 'simulate' cases: a dict of DynamicString kwargs, or a CSV with om_x/om_y/om_z. Every OM
# gets the 16-PMT template of GEOM; the OM positions are saved next to OUT.
NEW_GEOMETRY = dict(hex_type='hexagonal', domain_size=1200, n_strings=70,
                    points_per_string=20, custom_z_spacing=50)
RESPONSE_SEED = 0

# fit (see RecoGDLoss)
RECO = dict(
    terms=('hit', 'ly', 'atime'), hit_radius=200, n_steps=8,
    start='perturbed',
    # None: the vertex comes from `start`; 'hit_weighted': photon-weighted mean of the lit
    # PMT positions (direction and energy still from `start`)
    vertex_start='hit_weighted',
    # (x, y, z, zenith, azimuth, log10E); no t0, the interaction is at t = 0
    pert_scale=(70.0, 70.0, 70.0, np.radians(10), np.radians(10), 1.0), n_starts=4,
    scale=(20.0, 20.0, 20.0, np.radians(5), np.radians(5), 0.3),
    lr=0.05, max_iter=1000, patience=200, min_delta=1e-2, lr_factor=0.5, lr_patience=20,
    min_lr=1e-4, log10e_bounds=(2.0, 6.0), bound_vertex=True,
    events_per_batch=16, flow_chunk_rows=15000, hit_chunk_rows=100_000,
    response_n_steps=32, response_events_per_batch=16)
SAVE_EVERY = 100                 # finished events between saves of a worker's file
ENERGY_BIN = 0.5                 # log10 E bin width of the summary table


def new_geometry_positions():
    """OM positions (N, 3) of NEW_GEOMETRY."""
    if isinstance(NEW_GEOMETRY, dict):
        from nugget.geometries.DynamicString import DynamicString
        g = DynamicString(device='cpu', dim=3, **NEW_GEOMETRY)
        return g.initialize_points()['points_3d'].detach().cpu().numpy().astype(np.float64)
    om = pd.read_csv(NEW_GEOMETRY, usecols=['om_x', 'om_y', 'om_z']).drop_duplicates()
    return om.to_numpy(np.float64)


def all_events(energy_column):
    """Truth of every PARQUET event, with the energy the models were trained on."""
    cache = EVENT_CACHE
    if cache and energy_column != 'neutrino_energy':    # one cache per energy column
        root, ext = os.path.splitext(cache)
        cache = f'{root}.{energy_column}{ext}'
    if cache and os.path.exists(cache):
        return read_table(cache)
    t = time.time()
    ev = event_table(PARQUET, energy_column=energy_column)
    print(f'event table: {len(ev)} events from {PARQUET} in {time.time() - t:.0f}s')
    if cache:
        save_table(ev, cache)
    return ev


def select_events(om_pos, starting, half, energy_column):
    """N_EVENTS random events passing the cuts, with their truth and containment."""
    ev = all_events(energy_column)
    xyz = ev[['x', 'y', 'z']].to_numpy(float)
    ev = ev.assign(log10E=np.log10(ev.energy), cos_zenith=np.cos(ev.zenith),
                   hull_dist=hull_distance(om_pos, xyz),
                   in_domain=(np.abs(xyz) <= half).all(1))
    ev['contained'] = ev.hull_dist <= -CONTAINMENT_MARGIN
    # copies: under pandas copy-on-write to_numpy() is a read-only view
    keep = ev.in_domain.to_numpy(copy=True) if FILTER_VERTEX_IN_DOMAIN else np.ones(len(ev), bool)
    if starting:
        keep = keep & ev.contained.to_numpy()
    print(f'{len(ev)} events: {ev.in_domain.mean():.1%} in the model domain, '
          f'{ev.contained.mean():.1%} start inside the hull; {keep.sum()} pass the cuts')
    pool = ev[keep]
    rng = np.random.default_rng(EVENT_SEED)
    idx = np.sort(rng.choice(len(pool), min(N_EVENTS, len(pool)), replace=False))
    return pool.iloc[idx].reset_index(drop=True)


def run_worker(i, truth, device, out_path, source, om_pos):
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    loss = RecoGDLoss(device=device, hit_checkpoint=HIT_CKPT, ly_checkpoint=LY_CKPT,
                      atime_checkpoint=ATIME_CKPT, geometry_csv_path=GEOM, seed=EVENT_SEED,
                      **RECO)
    resp = (parquet_response(PARQUET, GEOM, truth) if source == 'parquet'
            else loss.simulate_response(om_pos, truth, seed=RESPONSE_SEED))
    n_lit = resp.hits.ev.nunique()
    print(f'[shard {i}] {len(truth)} events, {n_lit} with light ({len(truth) - n_lit} dropped)')
    res = loss.fit(resp, tag=f'[shard {i}] ', save_path=out_path, save_every=SAVE_EVERY)
    save_table(res, out_path)


def _q(s, p=(50, 68, 90)):
    return ' / '.join(f'{np.nanpercentile(s, x):.3g}' for x in p)


def summarize(res, events, case, wall):
    print(f'\n===== {case}: {len(res)} of {len(events)} chosen events fitted =====')
    if res.empty:
        return
    print('stop reasons     :', res.stop_reason.value_counts().to_dict())
    print(f'iterations       : p50/p68/p90 {_q(res.n_iter)}  (max {res.n_iter.max()})')
    print(f'wall time        : {wall / 60:.1f} min, {60 * len(res) / wall:.1f} events/min')
    print(f'dpsi start [deg] : p50/p68/p90 {_q(res.dpsi_start)}')
    print(f'dpsi fit   [deg] : p50/p68/p90 {_q(res.dpsi_fit)}')
    print(f'|d vertex| start : p50/p68/p90 {_q(res.dvertex_start)}')
    print(f'|d vertex| [m]   : p50/p68/p90 {_q(res.dvertex_fit)}')
    s = resolution_summary(res)
    print(f'd log10E         : median {s["dlog10e_bias"]:+.3f}, p16-p84 half-width '
          f'{s["dlog10e_68"]:.3f}')
    dn = res.nll_fit - res.nll_true
    print(f'NLL fit - truth  : median {dn.median():+.2f}; fit <= truth + 0.5 in '
          f'{(dn <= 0.5).mean():.0%} of events (else the optimiser stopped short of the truth)')
    print(f'fit closer than start in direction: {(res.dpsi_fit < res.dpsi_start).mean():.0%}')
    print(f'best fit at a bound (vertex / energy): {res.at_bound.mean():.1%}')
    if 'contained' in res and res.contained.nunique() > 1:
        for c, g in res.groupby('contained'):
            print(f'{"starting" if c else "outside "} ({len(g):5d}): dpsi p50 '
                  f'{g.dpsi_fit.median():.2f} deg, dlog10E 68% {resolution_summary(g)["dlog10e_68"]:.3f}')
    print(f'\nper true log10 E bin (width {ENERGY_BIN}):')
    print(energy_table(res, ENERGY_BIN).to_string(float_format=lambda x: f'{x:.3g}'))


def main(case):
    cfg = CASES[case]
    out = OUT.format(case=case)
    stem = os.path.splitext(out)[0]
    t_start = time.time()
    if cfg['source'] == 'parquet':
        g = read_geometry(GEOM)[0]       # the hull of the OM centres, not of every PMT
        om_pos = g.groupby(['string', 'om'])[['om_x', 'om_y', 'om_z']].mean().to_numpy()
        sim_pos = None
    else:
        om_pos = sim_pos = new_geometry_positions()
        os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
        np.save(stem + '.om_positions.npy', om_pos)
        print(f'new geometry: {len(om_pos)} OMs -> {stem}.om_positions.npy')

    from nugget.surrogates.HitClassifier import HitClassifier
    hit = HitClassifier(device='cpu').load_model(HIT_CKPT)
    half = half_extent(hit.domain_size)
    # the truth energy is the one the models were trained on
    print(f'models trained on {hit.energy_type} energy ({hit.energy_column})')
    events = select_events(om_pos, cfg['starting'], half, hit.energy_column)
    save_table(events, stem + '.events.parquet')
    print(f'{case}: {len(events)} events -> {stem}.events.parquet')

    n = len(DEVICES)
    if n == 1:
        run_worker(0, events, DEVICES[0], out, cfg['source'], sim_pos)
    else:
        paths = [f'{stem}.shard{i}{os.path.splitext(out)[1]}' for i in range(n)]
        ctx = torch.multiprocessing.get_context('spawn')
        procs = [ctx.Process(target=run_worker,
                             args=(i, events.iloc[i::n].reset_index(drop=True), dev, paths[i],
                                   cfg['source'], sim_pos))
                 for i, dev in enumerate(DEVICES)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        found = [p for p in paths if os.path.exists(p)]
        if not found:
            raise RuntimeError('no shard produced any output')
        save_table(pd.concat([read_table(p) for p in found], ignore_index=True), out)
        if any(p.exitcode != 0 for p in procs):
            print('some shards failed: merged what exists, kept the shard files')
        else:
            for p in found:
                os.remove(p)
    res = read_table(out)
    print(f'saved {len(res)} rows -> {out}')
    summarize(res, events, case, time.time() - t_start)


if __name__ == '__main__':
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case not in CASES:
        raise SystemExit(f'unknown case {case!r}; one of {sorted(CASES)}')
    main(case)

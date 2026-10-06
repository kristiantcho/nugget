import pickle

import numpy as np
import torch

import nugget
from nugget.surrogates.NuSmoothie import NuSmoothie
from nugget.losses.fisher_info_flow import FlowFisherResolutionLoss
from nugget.losses.fisher_info_flow_mc import FlowFisherMCResolutionLoss
from nugget.losses.fisher_info_om_net import (OMFisherNetLoss, OMFisherTargets,
                                              resolution_from_fisher)

DEVICE = 'cuda:3'
HIT_CKPT = './flow_models/best_mc_hit_model_v1r.pt'
LY_CKPT = './flow_models/best_mc_ly_muon_flow_model_v1r.pt'
ATIME_CKPT = './flow_models/best_mc_atime_muon_flow_model_v1r.pt'

CHECKPOINT = './fisher_models/best_om_muon_fisher_net_v3mc.pt'
HISTORY = './fisher_models/om_muon_fisher_net_v3mc_training_history.pkl'

# Parameters of the Fisher matrix. Any set FlowFisherResolutionLoss can scan works.
FISHER_PARAMS = ('energy', 'zenith', 'azimuth')

# 'exact': train on quadrature Fishers.  'mc': train on single-draw MC Fishers
# (unbiased, noisy, far cheaper); the net learns their expectation. Validation and
# the checks below always use exact targets.
TARGET_MODE = 'mc'

# Exact targets: all three terms, every PMT (no hit sampling), generous quadrature.
FISHER_N_QUAD = 48
FISHER_N_STEPS = 16
FISHER_CHUNK = 8192

# MC targets: one draw per PMT for the light yield and one for the arrival time.
MC_SAMPLES = 1

N_STEPS = 2_000            # each step draws EVENTS_PER_STEP fresh events
EVENTS_PER_STEP = 16 if TARGET_MODE == 'exact' else 128   # MC draws are ~20-40x cheaper
OMS_PER_EVENT = 64          # 90% around the track, 10% uniform in a cube

sampler = nugget.samplers.cyl_sampler.CylinderSampler(
    device=DEVICE,
    event_type='signal',
    domain_size=1200,
    E_min=1e2, E_max=1e6,
    energy_dist='log_uniform',
    uniform_zenith_sampling=True,
    random_position_along_ray=True,
    find_exact_intersection=False,
    cylinder_radius=600,
    cylinder_height=1000,
    cylinder_center=[0, 0, 0],
)

ns = NuSmoothie(device=DEVICE, domain_size=10000, hit_checkpoint=HIT_CKPT,
                ly_checkpoint=LY_CKPT, atime_checkpoint=ATIME_CKPT)
exact = FlowFisherResolutionLoss(
    hit_model=ns.hit_model, ly_model=ns.ly_model, atime_model=ns.atime_model,
    device=DEVICE, fisher_info_params=FISHER_PARAMS, mode='all',
    n_quad=FISHER_N_QUAD, n_steps=FISHER_N_STEPS, sample_hits=False)
# hit_sample_seed stays None: every call must draw fresh MC noise
mc = FlowFisherMCResolutionLoss(
    hit_model=ns.hit_model, ly_model=ns.ly_model, atime_model=ns.atime_model,
    device=DEVICE, fisher_info_params=FISHER_PARAMS, mode='all',
    n_steps=FISHER_N_STEPS, sample_hits=False,
    n_samples_ly=MC_SAMPLES, n_samples_t=MC_SAMPLES)

placement = dict(
    oms_per_event=OMS_PER_EVENT,
    d_perp_range=(1.0, 400.0),      # log-uniform distance from the track [m]
    d_long_range=(-200.0, 1500.0),  # along the track from the vertex [m]
    uniform_frac=0.1,
    uniform_half_size=2000,
    chunk=FISHER_CHUNK,
)
exact_targets = OMFisherTargets(exact, sampler, seed=0, **placement)
targets = (exact_targets if TARGET_MODE == 'exact'
           else OMFisherTargets(mc, sampler, seed=1, **placement))

om_loss = OMFisherNetLoss(
    device=DEVICE,
    fisher_params=FISHER_PARAMS,
    resolution_type='angular',

    # --- context features (same conventions as the hit / flow models, no PMT terms) ---
    domain_size=20000,
    rich_rel_pos_mode=True,
    include_vertex_position=False,
    add_vertex_distance=False,
    add_distance_from_beam=False,
    add_dist_long=False,
    add_log_distances=True,
    ly_eps=1e-6,

    # --- network ---
    width=256,
    depth=10,
    dropout=0.0,
    net_dtype=torch.float32,

    # --- optimisation ---
    learning_rate=1e-3,
    lr_schedule='onecycle',
    warmup_frac=0.05,
    weight_decay=1e-5,

    # --- target / loss ---
    target_mode=TARGET_MODE,
    eps_floor=1e-6,             # added to F' before the Cholesky factor
    weight_tau='median',        # 'exact': dark-OM down-weighting; 'mc': error floor
    geometry_grads='chunked',
)

history = om_loss.fit_online(
    targets,
    n_steps=N_STEPS,
    events_per_step=EVENTS_PER_STEP,
    updates_per_step=4,         # MLP steps per fresh draw
    batch_size=4096,
    buffer_size=200_000,        # replay buffer of (event, OM) pairs
    warmup_events=256,          # normalisers + initial buffer
    n_val_events=512,           # fixed validation draw
    val_every=20,
    early_stopping_patience=50, # in validations
    save_every=5,               # in validations (every new best is saved anyway)
    checkpoint_path=CHECKPOINT,
    val_targets=exact_targets,  # exact validation: the true error, not error + noise
)
pickle.dump(history, open(HISTORY, 'wb'))


# ---------------------------------------------------------------------------
# Check 1: per-event resolution on fresh events. Sum each event's OMs (exact vs
# predicted Fisher) and compare the resolutions.
# ---------------------------------------------------------------------------
if 'zenith' in FISHER_PARAMS and 'azimuth' in FISHER_PARAMS:
    P = len(FISHER_PARAMS)
    test = exact_targets(512)
    zen, azi = test['zenith'], test['azimuth']
    u = torch.stack([torch.sin(zen) * torch.cos(azi), torch.sin(zen) * torch.sin(azi),
                     torch.cos(zen)], dim=1)
    with torch.no_grad():
        F_pred = om_loss.predict_fisher(test['om_pos'], test['vertex'], test['energy'], u)
    F_true_ev = test['F'].reshape(-1, OMS_PER_EVENT, P, P).sum(1)
    F_pred_ev = F_pred.reshape(-1, OMS_PER_EVENT, P, P).sum(1)
    E_ev = test['energy'].reshape(-1, OMS_PER_EVENT)[:, 0]
    z_ev = zen.reshape(-1, OMS_PER_EVENT)[:, 0]
    r_true = resolution_from_fisher(F_true_ev, list(FISHER_PARAMS), E_ev, z_ev)
    r_pred = resolution_from_fisher(F_pred_ev, list(FISHER_PARAMS), E_ev, z_ev)
    ok = torch.isfinite(r_true) & torch.isfinite(r_pred) & (r_true > 0)
    ratio = (r_pred[ok] / r_true[ok]).cpu().numpy()
    print(f'\nper-event angular resolution, net / exact ({int(ok.sum())} events):')
    print(f'  p16 / p50 / p84 = {np.percentile(ratio, [16, 50, 84]).round(3)}   '
          f'mean |log ratio| = {np.mean(np.abs(np.log(ratio))):.3f}')


# ---------------------------------------------------------------------------
# Check 2: as a loss on a real string geometry, against the exact Fisher loss.
# ---------------------------------------------------------------------------
COMPARE_ON_GEOMETRY = True
if COMPARE_ON_GEOMETRY and 'zenith' in FISHER_PARAMS and 'azimuth' in FISHER_PARAMS:
    geometry = nugget.geometries.DynamicString.DynamicString(
        device=DEVICE, hex_type='hexagonal', domain_size=1200, dim=3,
        n_strings=50, points_per_string=20, custom_z_spacing=50.0)
    gd = geometry.initialize_points()
    evs = sampler.sample_events(20)
    r_net = om_loss(gd, signal_event_params=evs,
                    fisher_res_metric='mean')['angular_resolution_per_event']
    r_ex = exact(gd, signal_event_params=evs, fisher_res_metric='mean',
                 fisher_info_chunk_size=FISHER_CHUNK,
                 fisher_info_events_per_batch=4)['angular_resolution_per_event']
    q = (r_net / r_ex).detach().cpu().numpy()
    print(f'\ngeometry check (50 strings x 20 OMs, 20 events): net / exact resolution '
          f'p16 / p50 / p84 = {np.percentile(q, [16, 50, 84]).round(3)}')

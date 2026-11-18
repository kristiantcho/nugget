import nugget
import pickle
import torch
import numpy as np

GEOM = '../other/800_40_40_geom.csv'
TRAIN_PARQUET = 'big_mu_accepted.parquet'
TEST_PARQUET = './flow_models/mc_atime_muon_flow_v1rlg_test_samples.parquet'
CHECKPOINT = './flow_models/best_mc_atime_muon_flow_model_v1rlg.pt'

EPOCHS = 300

flow = nugget.surrogates.FlowMatchATime.FlowMatchATime(
    device="cuda:0",
    domain_size=20000,
    dim=3,
    particle_mode='track',
    # --- velocity network ---
    width=256,
    depth=12,
    time_dim=64,        # sinusoidal embedding of the FLOW time t (not the arrival time)
    cond_width=256,
    dropout=0.0,

    # --- optimisation ---
    learning_rate=5e-4,
    lr_schedule='onecycle',
    warmup_frac=0.1,
    weight_decay=1e-5,
    reduce_lr_on_plateau=False,

    # --- flow ---
    sigma_min=1e-6,

    # --- arrival-time target ---
    refractive_index=1.35,   # phase index: sets the Cherenkov angle in t_geom
    group_index=1.38,        # photon-speed index in t_geom (None = refractive_index). On the
                             # muon parquet 1.35 / 1.38 puts the prompt edge at ~0 ns for every
                             # d_perp (1.33 / 1.33 drifts 0.26 ns/m); retrain under a new name
    time_scale=2.0,        # ns; asinh(t/scale). ~ the width of the direct peak
    time_transform='asinh',  
    track_dir_is_arrival=True,

    # --- context features (identical to the light-yield model) ---
    rich_rel_pos_mode=True,
    include_vertex_position=False,
    add_vertex_distance=False,
    add_distance_from_beam=True,
    add_dist_long=True,
    add_pmt_direction=True,
    add_pmt_cosangle=True,
    standardize_context=True,
    include_direction=True,
    # True: infinite-track features (no vertex) and only samples whose emission point
    # is through_min_upstream m past the vertex; set it the same in all three scripts
    through_going=False,
    through_min_upstream=50.0,
    energy_type='lepton',     # 'neutrino' or 'lepton' energy feature; same in all three
    ly_eps=1e-6,
)

train_dataloader = flow.create_atime_parquet_dataloader(
    parquet_path=TRAIN_PARQUET,
    geometry_csv_path=GEOM,
    num_samples_per_epoch=1000_000,
    batch_size=10000,
    num_workers=0,
    shuffle=True,
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    n_count_bins=30,
    filter_vertex_in_domain=True,
    # A few PMTs record >1000 photons; capping keeps them from dominating an epoch.
    max_photons_per_row=None,
    test_save_path=TEST_PARQUET,
    test_frac=0.1,
)

val_dataloader = flow.create_atime_parquet_val_dataloader(
    parquet_path=TEST_PARQUET,
    geometry_csv_path=GEOM,
    num_samples_per_epoch=100_000,
    batch_size=10000,
    num_workers=0,
    # seed=0,

    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    n_count_bins=30,
    filter_vertex_in_domain=True,
)

history = flow.train_with_dataloader(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    # grad_clip=1.0,
    early_stopping_patience=50,
    save_every_n_epochs=10,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history,
            open('./flow_models/mc_atime_muon_flow_v1rlg_training_history.pkl', 'wb'))


# ---------------------------------------------------------------------------
# Post-training check: does the model reproduce the time-residual distribution?
# Averaging p(t_res | c) over the data's own contexts must return p(t_res).
# ---------------------------------------------------------------------------
ds = val_dataloader.dataset
ctx, t_true = ds.get_batch(range(20000))
ctx = ctx.to(flow.device).float()
t_pred = flow.sample_time_residual(ctx, n_steps=64).cpu()

for name, arr in [('data', t_true.numpy()), ('flow', t_pred.numpy())]:
    a = np.asarray(arr, dtype=float)
    print(f"{name:5s}  p1={np.percentile(a,1):8.2f}  median={np.median(a):7.2f}  "
          f"p99={np.percentile(a,99):9.2f}  frac<0={np.mean(a<0):.4f}  "
          f"max={a.max():.1f}")

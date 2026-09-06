import nugget
import pickle
import torch

GEOM = '../other/800_40_40_geom.csv'
TRAIN_PARQUET = 'big_mu_accepted.parquet'
TEST_PARQUET = './flow_models/mc_ly_muon_flow_v2_test_samples.parquet'
CHECKPOINT = './flow_models/best_mc_ly_muon_flow_model_v2.pt'

EPOCHS = 500

flow = nugget.surrogates.FlowMatchLY.FlowMatchLY(
    device="cuda:1",
    domain_size=10000,
    dim=3,

    # --- velocity network ---
    width=256,          # hidden width of the residual trunk
    depth=8,            # number of residual blocks
    time_dim=64,        # sinusoidal embedding size for the flow time t
    cond_width=256,     # width of the context encoder
    dropout=0.0,

    # --- optimisation ---
    learning_rate=5e-4,
    lr_schedule='onecycle',
    warmup_frac=0.1,
    weight_decay=1e-5,
    reduce_lr_on_plateau=False,

    # --- flow ---
    sigma_min=1e-6,     # 0 gives exactly straight (rectified-flow) paths

    # --- context features (same conventions as LLRnet) ---
    rich_rel_pos_mode=True,
    include_vertex_position=True,
    add_vertex_distance=False,
    add_distance_from_beam=True,
    add_dist_long=True,
    track_dir_is_arrival=True,
    add_pmt_direction=True,
    add_pmt_cosangle=False,
    standardize_context=True,
    ly_eps=1e-6,
)

train_dataloader = flow.create_flow_parquet_dataloader(
    parquet_path=TRAIN_PARQUET,
    geometry_csv_path=GEOM,
    num_samples_per_epoch=1000_000,
    batch_size=4096,
    num_workers=0,
    shuffle=True,
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    filter_vertex_in_domain=True,
    test_save_path=TEST_PARQUET,
    test_frac=0.1,
)

val_dataloader = flow.create_flow_parquet_val_dataloader(
    parquet_path=TEST_PARQUET,
    geometry_csv_path=GEOM,
    num_samples_per_epoch=200_000,
    batch_size=4096,
    num_workers=0,
    # seed=0,
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    filter_vertex_in_domain=True,
)

history = flow.train_with_dataloader(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    # grad_clip=1.0,
    early_stopping_patience=40,
    save_every_n_epochs=10,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history, open('./flow_models/mc_ly_muon_flow_v2_training_history.pkl', 'wb'))


# ---------------------------------------------------------------------------
# Quick post-training check: does the model reproduce the light-yield marginal?
# Averaging p(q | c) over the data's own contexts must return p(q).
# ---------------------------------------------------------------------------
ds = val_dataloader.dataset
ctx, q_true = ds.get_batch(range(20000))
ctx = ctx.to(flow.device).float()
q_pred = flow.sample_light_yield(ctx, n_steps=64).cpu()

import numpy as np
for name, arr in [('data', q_true.numpy()), ('flow', q_pred.numpy())]:
    a = np.asarray(arr, dtype=float)
    print(f"{name:5s}  P(q=1)={np.mean(a == 1):.4f}  median={np.median(a):.1f}  "
          f"mean={a.mean():.2f}  q99={np.quantile(a, 0.99):.0f}  max={a.max():.0f}")

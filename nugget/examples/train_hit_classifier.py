import nugget
import pickle
import torch
import numpy as np

GEOM = '../other/800_40_40_geom.csv'
TRAIN_PARQUET = 'ly_mu_mc_all.parquet'
TEST_PARQUET = './flow_models/mc_hit_v1_test_samples.parquet'
CHECKPOINT = './flow_models/best_mc_hit_model_v1.pt'

EPOCHS = 200

hit = nugget.surrogates.HitClassifier.HitClassifier(
    device="cuda:1",
    domain_size=8000,
    dim=3,

    # --- network ---
    width=256,
    depth=6,
    dropout=0.0,

    # --- optimisation ---
    learning_rate=1e-3,
    lr_schedule='onecycle',
    warmup_frac=0.1,
    weight_decay=1e-5,
    reduce_lr_on_plateau=False,

    # --- context features: identical to the light-yield / arrival-time flows ---
    rich_rel_pos_mode=True,
    include_vertex_position=True,
    add_vertex_distance=False,
    add_distance_from_beam=True,
    add_dist_long=False,
    track_dir_is_arrival=False,
    add_pmt_direction=True,
    add_pmt_cosangle=False,
    standardize_context=True,
    ly_eps=1e-6,
)

train_dataloader = hit.create_hit_parquet_dataloader(
    parquet_path=TRAIN_PARQUET,
    geometry_csv_path=GEOM,
    num_samples_per_epoch=1_000_000,
    batch_size=4096,
    num_workers=0,
    shuffle=True,
    # Occupancy is ~0.01%, so a faithful sample would be >99.98% negatives and carry
    # almost no gradient. Train balanced; predict_hit_prob() undoes this with the
    # exact prior shift (log_prior_odds), which is stored in the checkpoint.
    pos_frac=0.5,
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    filter_vertex_in_domain=True,
    test_save_path=TEST_PARQUET,
    test_frac=0.1,
)

val_dataloader = hit.create_hit_parquet_val_dataloader(
    parquet_path=TEST_PARQUET,
    geometry_csv_path=GEOM,
    num_samples_per_epoch=200_000,
    batch_size=4096,
    num_workers=0,
    seed=0,
    pos_frac=0.5,
    uniform_energy_zenith=True,
    filter_vertex_in_domain=True,
)

history = hit.train_with_dataloader(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    grad_clip=1.0,
    early_stopping_patience=30,
    save_every_n_epochs=10,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history, open('./flow_models/mc_hit_v1_training_history.pkl', 'wb'))


# ---------------------------------------------------------------------------
# Calibration check. Accuracy is meaningless here (predicting "never hit" scores
# 99.99%), so look at the reliability table and AUC instead.
# ---------------------------------------------------------------------------
ds = val_dataloader.dataset
ctx, y = ds.get_batch(range(200_000))
ctx = ctx.to(hit.device).float()

print(f"\nlog_prior_odds = {hit.log_prior_odds:.4f}  "
      f"(true occupancy P(hit) = {ds.p_hit:.6g})")
print("\nreliability on the balanced validation sample (uncalibrated):")
print(f"  {'mean pred':>10}  {'empirical':>10}  {'n':>8}")
for pred, emp, n in hit.reliability(ctx, y, n_bins=12, calibrated=False):
    print(f"  {pred:10.4f}  {emp:10.4f}  {n:8d}")

p_cal = hit.predict_hit_prob(ctx, calibrated=True).detach().cpu().numpy()
print(f"\ncalibrated P(hit) on the same sample: mean {p_cal.mean():.6f}, "
      f"max {p_cal.max():.4f}")
print("  (the mean is far below 0.5 because the calibrated model reports true "
      "detector occupancy, not the balanced training mix)")

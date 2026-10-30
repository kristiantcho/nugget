import nugget
import pickle

GEOM = '../other/800_40_40_geom.csv'
TRAIN_PARQUET = 'big_mu_accepted.parquet'
TEST_PARQUET = './flow_models/mc_reco_muon_flow_v1_test_samples.parquet'
CHECKPOINT = './flow_models/best_mc_reco_muon_flow_model_v1.pt'

EPOCHS = 100

flow = nugget.surrogates.FlowMatchReco.FlowMatchReco(
    device="cuda:0",
    # 'essential': relative position, PMT orientation, q_tot, t_first, t50 (+ token flags);
    # 'thesis': all 27 features of Bas 2026, Table 8.1
    feature_set='essential',
    # --- PMT transformer encoder (token/pooling layout of Bas 2026, Sec. 8.1-8.2) ---
    d_model=96,
    encoder_depth=20,
    n_heads=4,
    ffn_dim=512,
    cond_dim=256,
    encoder_dropout=0.0,

    # --- adaLN-Zero vector field on R x S^2 ---
    width=256,
    depth=8,
    time_dim=64,
    mlp_ratio=2,
    dropout=0.0,

    # --- token transforms (thesis Eq. 8.2); 'asinh' keeps ns-level timing visible ---
    length_scale=500.0,
    time_scale=1e4,
    time_transform='linear',

    # --- flow ---
    sigma_min=1e-4,
    direction_loss_weight=1.0,
    n_flow_samples=8,       # flow-time draws per encoded event; the encoder dominates cost

    # --- optimisation (thesis Table B.2: warm-up 2000 steps, cosine to 1e-5) ---
    learning_rate=2e-4,
    weight_decay=1e-5,
    lr_schedule='warmup_cosine',
    warmup_steps=2000,
    min_lr=1e-5,
    amp=True,               # bf16 autocast for training; sampling/log_prob stay float32
)

data_kw = dict(
    geometry_csv_path=GEOM,
    energy_column='neutrino_energy',
    # photons -> pulses with the thesis/P-ONE Offline PMT response (Table 5.3)
    use_pulses=True,
    pulse_params=None,
    # >= 3 PMTs of one OM within 10 ns, evaluated after masking
    require_trigger=True,
    # every event is seen in its own random sub-geometry of the 1378-string grid: 10-150
    # strings drawn around a random string with a Gaussian width log-uniform in 40-1000 m
    # (compact 40 m cluster ... scattered), then in half the events up to 50% of the OMs
    # dropped. ~14% of (event, geometry) draws trigger; the rest are redrawn.
    n_strings_range=(10, 150),
    string_spread_range=(40.0, 1000.0),
    om_drop_prob=0.5,
    om_drop_range=(0.0, 0.5),
    # one token per active silent OM on a hit string and per active silent string, so the
    # whole sub-geometry is seen: ~250 tokens/event median, ~1.4k p99. max_tokens is only a
    # guard against rare huge events; lower batch_size first if memory runs out.
    silent_tokens=True,
    max_tokens=4096,
    batch_size=64,
    num_workers=8,
)

train_dataloader = flow.create_reco_parquet_dataloader(
    parquet_path=TRAIN_PARQUET,
    num_events_per_epoch=200_000,
    shuffle=True,
    test_save_path=TEST_PARQUET,
    test_frac=0.1,
    **data_kw,
)

val_dataloader = flow.create_reco_parquet_val_dataloader(
    parquet_path=TEST_PARQUET,
    num_events_per_epoch=20_000,
    **data_kw,
)

history = flow.train_with_dataloader(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    grad_clip=3.0,
    early_stopping_patience=20,
    metrics_every_n_epochs=5,
    metrics_batches=20,
    save_every_n_epochs=5,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history,
            open('./flow_models/mc_reco_muon_flow_v1_training_history.pkl', 'wb'))


# ---------------------------------------------------------------------------
# Post-training check: a compact 100-string cluster vs. 50 scattered strings, all OMs.
# For a specific layout, pass a geometry CSV holding only its strings and
# n_strings_range=None, om_drop_prob=0.0.
# ---------------------------------------------------------------------------
eval_kw = {**data_kw, 'num_workers': 0, 'om_drop_prob': 0.0}
for name, mask in [('compact 100', dict(n_strings_range=(100, 100),
                                        string_spread_range=(40.0, 40.0))),
                   ('scattered 50', dict(n_strings_range=(50, 50),
                                         string_spread_range=(3000.0, 3000.0)))]:
    loader = flow.create_reco_parquet_val_dataloader(
        parquet_path=TEST_PARQUET, num_events_per_epoch=5_000, **{**eval_kw, **mask})
    m = flow.evaluate(loader, max_batches=40, n_samples=128, n_steps=32)
    print(f"{name:15s}  median angle {m['median_angle_deg']:.2f} deg  "
          f"macro {m['macro_median_angle_deg']:.2f} deg  "
          f"|dlog10E| median {m['median_abs_dlog10e']:.3f}  68% {m['dlog10e_68']:.3f}")

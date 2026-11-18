import nugget
import pickle

GEOM = '../other/800_40_40_geom.csv'
# the frozen classifier whose P(nu_mu) the surrogate learns
TEACHER_TAG = 'mu_vs_e_tau_v1'
TEACHER = f'./flavor_models/best_flavor_clf_{TEACHER_TAG}.pt'
# the classifier's held-out nu_mu events: left out of training, used for validation
TEACHER_TEST = f'./flavor_models/flavor_clf_{TEACHER_TAG}_test_mu.parquet'
TRAIN_PARQUET = 'big_mu_accepted.parquet'
TAG = f'{TEACHER_TAG}_s1'
CHECKPOINT = f'./flavor_models/best_flavor_surrogate_{TAG}.pt'

DEVICE = "cuda:0"
EPOCHS = 300

teacher = nugget.losses.flavor_classifier.FlavorClassifier(device=DEVICE, amp=True)
teacher.load_model(TEACHER)

sur = nugget.losses.flavor_surrogate.FlavorSurrogate(
    device=DEVICE,
    # energy the event context carries: 'neutrino' or 'muon' (must match the sampler's
    # 'energy' during optimisation)
    energy_type='muon',
    # --- OM tokens: attention within each string, then across strings ---
    d_model=64,
    om_depth=4,
    string_depth=4,
    n_heads=4,
    ffn_dim=256,
    dropout=0.0,
    length_scale=500.0,

    # --- adaLN MLP head conditioned on (log10 E, cos zenith) ---
    cond_dim=128,
    head_width=256,
    head_depth=4,
    mlp_ratio=2,
    head_dropout=0.0,

    # prior of the classifier's P(nu_mu) to learn; None = its training prior (50%)
    signal_prior=None,

    # --- optimisation ---
    learning_rate=3e-4,
    weight_decay=1e-5,
    lr_schedule='warmup_cosine',
    warmup_steps=1000,
    min_lr=1e-5,
    amp=True,
)

data_kw = dict(
    geometry_csv_path=GEOM,
    # sub-geometries as the classifier was trained (None = its data_config)
    n_strings_range=None,
    om_drop_prob=None,
    om_drop_range=None,
    # flat in (log10 E, cos zen), so every region is learned equally well
    uniform_energy_zenith=True,
    # an event left without pulses: 'redraw' learns E[P | detected], 'zero' E[P * detected]
    undetected='redraw',
    batch_size=128,
    num_workers=8,
)

train_dataloader = sur.create_dataloader(
    TRAIN_PARQUET, teacher,
    num_events_per_epoch=200_000,
    shuffle=True,
    exclude_parquet=TEACHER_TEST,
    **data_kw,
)

val_dataloader = sur.create_val_dataloader(
    TEACHER_TEST, teacher,
    num_events_per_epoch=20_000,
    **data_kw,
)

history = sur.train_with_dataloader(
    train_dataloader=train_dataloader,
    teacher=teacher,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    grad_clip=1.0,
    early_stopping_patience=30,
    save_every_n_epochs=5,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history, open(f'./flavor_models/flavor_surrogate_{TAG}_training_history.pkl', 'wb'))


# ---------------------------------------------------------------------------
# Post-training check: mean prediction vs mean classifier output, all OMs, a few sizes.
# ---------------------------------------------------------------------------
eval_kw = {**data_kw, 'num_workers': 0, 'om_drop_prob': 0.0}
for n_str in (50, 150, 300, 500):
    loader = sur.create_val_dataloader(
        TEACHER_TEST, teacher, num_events_per_epoch=5_000,
        **{**eval_kw, 'n_strings_range': (n_str, n_str)})
    m = sur.evaluate(loader, teacher)
    print(f"{n_str:4d} strings  mean P {m['mean_pred']:.4f} (classifier {m['mean_target']:.4f})"
          f"  r2 {m['r2']:.4f}  ece {m['ece']:.4f}  bce {m['bce']:.4f}")

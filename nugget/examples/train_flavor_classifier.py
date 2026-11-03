import nugget
import pickle

GEOM = '../other/800_40_40_geom.csv'
# one parquet per flavour; SIGNAL is labelled 1, every other source 0
PARQUETS = {
    'mu': 'big_mu_accepted.parquet',
    'e': 'small_e_accepted.parquet',
    'tau': 'small_tau_accepted.parquet',
}
SIGNAL = 'mu'
TAG = 'mu_vs_e_tau_v1'
TEST_PARQUETS = {k: f'./flavor_models/flavor_clf_{TAG}_test_{k}.parquet' for k in PARQUETS}
CHECKPOINT = f'./flavor_models/best_flavor_clf_{TAG}.pt'

EPOCHS = 500

clf = nugget.losses.flavor_classifier.FlavorClassifier(
    device="cuda:0",
    feature_set='thesis',
    # --- PMT transformer encoder, as in FlowMatchReco ---
    d_model=96,
    encoder_depth=10,
    n_heads=4,
    ffn_dim=512,
    cond_dim=256,
    encoder_dropout=0.0,

    # --- classifier head ---
    head_width=256,
    head_depth=2,
    head_dropout=0.1,

    # --- token transforms (thesis Eq. 8.2) ---
    length_scale=500.0,
    time_scale=1e4,
    time_transform='linear',

    # --- optimisation ---
    learning_rate=2e-4,
    weight_decay=1e-5,
    lr_schedule='warmup_cosine',
    warmup_steps=2000,
    min_lr=1e-5,
    amp=True,
)

data_kw = dict(
    signal=SIGNAL,
    geometry_csv_path=GEOM,
    # half the draws signal; the background draws split by source_weights (default equal)
    signal_frac=0.5,
    source_weights=None,    # e.g. {'e': 1.0, 'tau': 2.0}
    # every source flattened over the (log10 E_nu, cos zen) bins all of them populate, so
    # the classifier cannot separate the flavours by their generated spectra
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    energy_column='neutrino_energy',
    use_pulses=True,
    pulse_params=None,
    require_trigger=False,
    # every event in its own random sub-geometry, as for the reco flow
    n_strings_range=(10, 300),
    om_drop_prob=0.5,
    om_drop_range=(0.0, 0.5),
    silent_tokens='oms',
    max_tokens=20000,
    batch_size=64,
    num_workers=8,
)

train_dataloader = clf.create_flavor_dataloader(
    parquets=PARQUETS,
    num_events_per_epoch=200_000,
    shuffle=True,
    test_save_paths=TEST_PARQUETS,
    test_frac=0.1,
    **data_kw,
)

val_dataloader = clf.create_flavor_val_dataloader(
    parquets=TEST_PARQUETS,
    num_events_per_epoch=20_000,
    **data_kw,
)

history = clf.train_with_dataloader(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    grad_clip=3.0,
    early_stopping_patience=30,
    save_every_n_epochs=5,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history, open(f'./flavor_models/flavor_clf_{TAG}_training_history.pkl', 'wb'))


# ---------------------------------------------------------------------------
# Post-training check: AUC per background flavour for a few detector sizes, all OMs.
# ---------------------------------------------------------------------------
eval_kw = {**data_kw, 'num_workers': 0, 'om_drop_prob': 0.0}
for n_str in (50, 150, 500):
    loader = clf.create_flavor_val_dataloader(
        parquets=TEST_PARQUETS, num_events_per_epoch=5_000,
        **{**eval_kw, 'n_strings_range': (n_str, n_str)})
    m = clf.evaluate(loader, max_batches=40)
    per = ", ".join(f"{k[4:]} {v:.3f}" for k, v in m.items() if k.startswith('auc_'))
    print(f"{n_str:4d} strings  AUC {m['auc']:.3f} ({per})  macro {m['macro_auc']:.3f}  "
          f"acc {m['accuracy']:.3f}")

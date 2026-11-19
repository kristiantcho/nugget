import pickle

import numpy as np

import nugget
from nugget.surrogates.FlowMatchReco import DetectorGeometry

GEOM = '../other/800_40_40_geom.csv'
# one parquet per flavour; SIGNAL is labelled 1, every other source 0
PARQUETS = {
    'mu': 'big_mu_accepted.parquet',
    'e': 'small_e_accepted.parquet',
    'tau': 'small_tau_accepted.parquet',
}
SIGNAL = 'mu'

# Amortisation-gap test. None: every event in its own random sub-geometry (the reference
# model, REF_TAG). An int: every event in one fixed set of that many strings (all their
# OMs, drawn with FIXED_SEED), trained on the reference model's train / test split and then
# scored side by side with it on that geometry. Keep it inside n_strings_range for a fair test.
FIXED_N_STRINGS = 150
FIXED_SEED = 0
REF_TAG = 'mu_vs_e_tau_v1'
TAG = REF_TAG if FIXED_N_STRINGS is None else f'{REF_TAG}_fixed{FIXED_N_STRINGS}s{FIXED_SEED}'
# the reference run's held-out events: written by that run, reused by the fixed ones
TEST_PARQUETS = {k: f'./flavor_models/flavor_clf_{REF_TAG}_test_{k}.parquet' for k in PARQUETS}
CHECKPOINT = f'./flavor_models/best_flavor_clf_{TAG}.pt'

geometry = DetectorGeometry.from_csv(GEOM)
FIXED_STRINGS = (None if FIXED_N_STRINGS is None else np.sort(
    np.random.default_rng(FIXED_SEED).choice(geometry.n_strings, FIXED_N_STRINGS, replace=False)))

EPOCHS = 500

clf = nugget.losses.flavor_classifier.FlavorClassifier(
    device="cuda:1",
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
    head_dropout=0.0,

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
    geometry_csv_path=geometry,
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
    n_strings_range=(50, 300),
    om_drop_prob=0.2,
    om_drop_range=(0.0, 0.5),
    # set by FIXED_N_STRINGS: one fixed sub-geometry instead of the random ones above
    fixed_strings=FIXED_STRINGS,
    silent_tokens='strings',
    max_tokens=20000,
    batch_size=64,
    num_workers=4,
)

# the reference run holds out 10% of the events; a fixed run trains on the rest of the same split
split = (dict(test_save_paths=TEST_PARQUETS, test_frac=0.1) if FIXED_N_STRINGS is None
         else dict(exclude_parquets=TEST_PARQUETS))

train_dataloader = clf.create_flavor_dataloader(
    parquets=PARQUETS,
    num_events_per_epoch=200_000,
    shuffle=True,
    **split,
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
    early_stopping_patience=50,
    save_every_n_epochs=5,
    checkpoint_path=CHECKPOINT,
)

pickle.dump(history, open(f'./flavor_models/flavor_clf_{TAG}_training_history.pkl', 'wb'))


def summary(m):
    per = ", ".join(f"{k[4:]} {v:.4f}" for k, v in m.items() if k.startswith('auc_'))
    return (f"AUC {m['auc']:.4f} ({per})  macro {m['macro_auc']:.4f}  acc {m['accuracy']:.4f}  "
            f"bce {m['bce']:.4f}")


if FIXED_N_STRINGS is None:
    # -----------------------------------------------------------------------
    # Post-training check: AUC per background flavour for a few detector sizes, all OMs.
    # -----------------------------------------------------------------------
    eval_kw = {**data_kw, 'num_workers': 0, 'om_drop_prob': 0.0}
    for n_str in (50, 150, 500):
        loader = clf.create_flavor_val_dataloader(
            parquets=TEST_PARQUETS, num_events_per_epoch=5_000,
            **{**eval_kw, 'n_strings_range': (n_str, n_str)})
        print(f"{n_str:4d} strings  {summary(clf.evaluate(loader, max_batches=40))}")
else:
    # -----------------------------------------------------------------------
    # Amortisation gap: this model and the random-geometry reference on the same held-out
    # events and pulse draws, in the fixed geometry.
    # -----------------------------------------------------------------------
    class Cached(list):
        """A loader's batches drawn once, so every model sees the same events and pulses."""

        def __init__(self, loader):
            super().__init__(loader)
            self.dataset = loader.dataset

    ref = nugget.losses.flavor_classifier.FlavorClassifier(device=clf.device, amp=True)
    ref.load_model(f'./flavor_models/best_flavor_clf_{REF_TAG}.pt')
    batches = Cached(clf.create_flavor_val_dataloader(
        parquets=TEST_PARQUETS, num_events_per_epoch=10_000, **data_kw))
    m_fix, m_ref = clf.evaluate(batches), ref.evaluate(batches)
    print(f"\nfixed geometry: {FIXED_N_STRINGS} strings (seed {FIXED_SEED}), "
          f"{m_fix['n_events']:,} held-out events")
    print(f"  trained on it      {summary(m_fix)}")
    print(f"  random geometries  {summary(m_ref)}")
    gap = {k: m_fix[k] - m_ref[k] for k in m_fix if k.startswith('auc') or k == 'macro_auc'}
    print("  amortisation gap (fixed - random): "
          + ", ".join(f"{k} {v:+.4f}" for k, v in gap.items()))

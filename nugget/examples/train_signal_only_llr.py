import nugget  # Main NUGGET package for neutrino detector optimization
import pickle 

# lightsabre = nugget.surrogates.LightSabre.LightSabre(use_poisson=True, domain_size=2000, particle_mode='track')
# light_yield_surrogate = lightsabre.light_yield_surrogate
# signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
#             event_type='signal', 
#             domain_size=2000, 
#             E_min=1e2, 
#             E_max=1e8, 
#             energy_dist='log_uniform', 
#             find_exact_intersection=False, 
#             random_position_along_ray=True,
#             uniform_zenith_sampling=True
#             )


# Feature count for the settings below:
#   rel (3) + direction (3) + log10(E)/8 (1) + cos_angle (1)
#   + dist_perp (1) + dist_long (1) + pmt_direction (3) + log_ly (1) = 14
INPUT_DIM = 16

llr_net = nugget.surrogates.LLRnet.LLRnet(
    domain_size=5000,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    hidden_dims=[250 for _ in range(14)],  # Neural network architecture
    use_fourier_features=False,  # Use Fourier features for better spatial encoding
    num_parallel_branches=1,  # Multiple branches for ensemble learning
    frequency_scales=[0.1, 0.4],  # Different frequency scales for fourier features
    num_frequencies_per_branch=[64,64],  # Number of Fourier features per branch
    learnable_frequencies=False,  # Fixed frequency features
    dropout_rate=0,  # Regularization
    # All the signal in this task lives in interactions between the light yield and the
    # geometry (every feature on its own is identically distributed in both classes), so
    # the loss starts on a flat ln(2) plateau. A large batch with a warmup escapes it in
    # ~1k steps; batch 32 at lr 1e-4 was still at ~0.689 after 30k steps.
    learning_rate=1e-4,  # peak LR for the OneCycle schedule
    # lr_schedule='onecycle',
    # warmup_frac=0.15,
    standardize_inputs=True,
    shared_mlp=False,  # Independent MLPs for each branch
    use_residual_connections=True,  # Skip connections for better training
    signal_noise_scale=0,  # Noise level for signal events
    background_noise_scale=0,  # Noise level for background events
    add_relative_pos=False,  # Whether to include relative position features
    log_scale_ly=True,  # Whether to log-scale the light yield inputs
    norm_pos=True,  # Whether to normalize position inputs
    log_scale_energy=True,  # Whether to log-scale the energy inputs
    # Track geometry. dist_perp alone is blind to up/downstream, so pair it with the
    # signed dist_long. Together these were worth ~0.04 nats of achievable loss.
    add_distance_from_beam=False,
    add_dist_long=False,
    # zenith/azimuth in this parquet are the ARRIVAL direction, so the muon travels along
    # -direction. Verified on the data: 95% of hit PMTs are downstream of the vertex
    # under this convention versus 5% under the other.
    track_dir_is_arrival=False,
    reduce_lr_on_plateau=False,  # superseded by lr_schedule='onecycle'
    lr_scheduler_patience=15,  # Patience for LR scheduler
    use_rich_features=True,  # Whether to use rich features from the surrogate model
    add_vertex_distance=False,
    rich_rel_pos_mode=True,
    log_charge_scale=4,
    ly_eps=1e-6,
    add_pmt_direction=True,
    add_pmt_cosangle=True
    )

# llr_net.load_model('llrnet_models/best_mc_ly_muon_llr_model_v4.pt')
# train_dataloader = llr_net.create_signal_only_dataloader(
#     signal_sampler=signal_sampler, 
#     signal_surrogate_func=light_yield_surrogate,
#     num_samples_per_epoch=2048,
#     batch_size=16,       # Size of each batch (N pairs)
#     num_workers=4,
#     event_labels=['position','energy', 'direction'],
#     shuffle=True,
#     samples_per_event = 1,
#     min_light_yield=0.1,         
#     max_resample_attempts=200,
#     vary_cylinder=False,
#     pin_memory=True,
#     pin_memory_device=None,
#     record_marginal_lys=True
    
#     # cylinder_sampler=nugget.samplers.cyl_sampler.CylinderSampler
#     )

GEOM = '../other/800_40_40_geom.csv'
TEST_PARQUET = './llrnet_models/mc_ly_muon_llr_v4_test_samples.parquet'

EPOCHS = 200

train_dataloader = llr_net.create_light_yield_parquet_dataloader(
    parquet_path='new_accepted_photons_ly_all.parquet',
    geometry_csv_path=GEOM,
    # Each epoch is num_samples_per_epoch matched/mismatched pairs = 2x that many
    # samples. Batches of 4096 need a much bigger epoch than 2048 samples to be
    # meaningful, and it also makes the reported per-epoch loss far less noisy.
    num_samples_per_epoch=500000,
    batch_size=4096,     # large batch: the interaction gradient is tiny at init
    # num_workers=0 is ~12x FASTER here (measured 12,100 vs 980 samples/s at batch 4096).
    # Building one item costs only 0.09 ms, so the per-item worker IPC dominates; the
    # workers also each fork a copy of the multi-GB dataset.
    num_workers=0,
    shuffle=True,
    zero_ly_prob=0.00,   # zeros are handled by a separate hit/no-hit network
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    filter_vertex_in_domain=True,
    test_save_path=TEST_PARQUET,
    test_frac=0.1,
    )

# Event-disjoint validation from the held-out split, so early stopping and the saved
# checkpoint track generalisation rather than the training loss.
# val_dataloader = llr_net.create_light_yield_parquet_val_dataloader(
#     parquet_path=TEST_PARQUET,
#     geometry_csv_path=GEOM,
#     num_samples_per_epoch=10000,
#     batch_size=5000,
#     num_workers=0,
#     seed=0,
#     uniform_energy_zenith=True,
#     n_energy_bins=20,
#     n_coszen_bins=20,
#     filter_vertex_in_domain=True,
#     )

history = llr_net.train_with_dataloader(
    train_dataloader=train_dataloader,
    # val_dataloader=val_dataloader,
    epochs=EPOCHS,
    input_dim=INPUT_DIM,
    grad_clip=None,
    early_stopping_patience=300,
    save_every_n_epochs=10,
    checkpoint_path='llrnet_models/best_mc_ly_muon_llr_model_v4.pt',
)

# Save the best model for later use
# llr_net.save_model('best_cascade_charge_llr_model_v1')

# #save history as pickle
pickle.dump(history, open('llrnet_models/mc_ly_muon_llr_v4_training_history.pkl', 'wb'))
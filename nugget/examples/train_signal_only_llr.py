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


llr_net = nugget.surrogates.LLRnet.LLRnet(
    domain_size=3000,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    hidden_dims=[128, 128, 128, 128, 128, 128],  # Neural network architecture
    use_fourier_features=False,  # Use Fourier features for better spatial encoding
    num_parallel_branches=1,  # Multiple branches for ensemble learning
    frequency_scales=[0.1, 0.4],  # Different frequency scales for fourier features
    num_frequencies_per_branch=[64,64],  # Number of Fourier features per branch
    learnable_frequencies=False,  # Fixed frequency features
    dropout_rate=0,  # Regularization
    learning_rate=1e-4,  # Optimizer learning rate
    shared_mlp=False,  # Independent MLPs for each branch
    use_residual_connections=True,  # Skip connections for better training
    signal_noise_scale=0,  # Noise level for signal events
    background_noise_scale=0,  # Noise level for background events
    add_relative_pos=False,  # Whether to include relative position features
    log_scale_ly=True,  # Whether to log-scale the light yield inputs
    norm_pos=True,  # Whether to normalize position inputs
    log_scale_energy=True,  # Whether to log-scale the energy inputs
    add_distance_from_beam=False,  # Whether to include distance from beam as a feature
    reduce_lr_on_plateau=True,  # Reduce learning rate on plateau
    lr_scheduler_patience=15,  # Patience for LR scheduler
    use_rich_features=True,  # Whether to use rich features from the surrogate model
    add_vertex_distance=False,
    rich_rel_pos_mode=True,
    log_charge_scale=4,
    ly_eps=1e-6,
    add_pmt_direction=True
    )

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

train_dataloader = llr_net.create_light_yield_parquet_dataloader(
    parquet_path='new_accepted_photons_ly_all.parquet',
    geometry_csv_path='../other/800_40_40_geom.csv',
    num_samples_per_epoch=2048,
    batch_size=32,       # Size of each batch (N pairs)
    num_workers=4,
    shuffle=True,
    zero_ly_prob=0.00,
    uniform_energy_zenith=True,
    n_energy_bins=20,
    n_coszen_bins=20,
    filter_vertex_in_domain=True,
    test_save_path='./llrnet_models/mc_ly_muon_llr_v2_test_samples.parquet',
    test_frac=0.1,
    )
history = llr_net.train_with_dataloader(
    train_dataloader=train_dataloader,
    epochs=2000,
    input_dim=12,
    grad_clip=None,
    save_every_n_epochs=20,
    checkpoint_path='llrnet_models/best_mc_ly_muon_llr_model_v2.pt',
)

# Save the best model for later use
# llr_net.save_model('best_cascade_charge_llr_model_v1')

# #save history as pickle
pickle.dump(history, open('llrnet_models/mc_ly_muon_llr_v2_training_history.pkl', 'wb'))
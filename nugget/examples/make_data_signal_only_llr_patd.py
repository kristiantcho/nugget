import nugget  # Main NUGGET package for neutrino detector optimization

light_yield_surrogate = nugget.surrogates.LightSabre.LightSabrePATD(use_poisson=True, num_track_points=500, domain_size=2500).light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(event_type='signal', domain_size=2500, E_min=1e2, E_max=1e8, energy_dist='log_uniform', find_exact_intersection=True)


llr_net = nugget.surrogates.LLRnet.LLRnet(
    domain_size=2500,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    hidden_dims=[256, 256, 256, 256],  # Neural network architecture
    use_fourier_features=True,  # Use Fourier features for better spatial encoding
    num_parallel_branches=3,  # Multiple branches for ensemble learning
    frequency_scales=[0.1, 0.2, 0.4],  # Different frequency scales for fourier features
    num_frequencies_per_branch=[64,64, 64],  # Number of Fourier features per branch
    learnable_frequencies=False,  # Fixed frequency features
    dropout_rate=0,  # Regularization
    learning_rate=1e-3,  # Optimizer learning rate
    shared_mlp=False,  # Independent MLPs for each branch
    use_residual_connections=True,  # Skip connections for better training
    signal_noise_scale=0,  # Noise level for signal events
    background_noise_scale=0.2,  # Noise level for background events
    add_relative_pos=False,  # Whether to include relative position features
    log_scale_ly=True,  # Whether to log-scale the light yield inputs
    norm_pos=True,  # Whether to normalize position inputs
    log_scale_energy=True,  # Whether to log-scale the energy inputs
    add_distance_from_beam=False,  # Whether to include distance from beam as a feature
    reduce_lr_on_plateau=True,  # Reduce learning rate on plateau
    lr_scheduler_patience=35,  # Patience for LR scheduler
    use_patd=True,
)

stats = llr_net.pregenerate_training_data(
    signal_sampler=signal_sampler, 
    signal_surrogate_func=light_yield_surrogate,
    num_events=1000000,
    event_labels=['position','energy', 'direction'],
    max_photons=200,  # Maximum number of photons to consider per event
    output_filepath='1e6_200_patd_dataset.h5'
)

print(stats)


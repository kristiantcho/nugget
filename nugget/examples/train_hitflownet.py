import nugget  # Main NUGGET package for neutrino detector optimization
import pickle


# Use LightSabrePATD for photon arrival time distributions
lightsabre_patd = nugget.surrogates.LightSabre.LightSabrePATD(
    use_poisson=True, 
    num_track_points=1000, 
    domain_size=2500, 
    use_max_energy_dist=True
)
patd_surrogate = lightsabre_patd.light_yield_surrogate

signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
    event_type='signal', 
    domain_size=2500, 
    E_min=1e2, 
    E_max=1e8, 
    energy_dist='log_uniform', 
    find_exact_intersection=True
)


hitflownet = nugget.surrogates.HitFlowNet.HitFlowNet(
    domain_size=2500,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    hidden_dims=[256, 128, 64, 64],  # Neural network architecture
    use_fourier_features=True,  # Use Fourier features for better spatial encoding
    num_parallel_branches=2,  # Multiple branches for ensemble learning
    frequency_scales=[0.1, 0.4],  # Different frequency scales for fourier features
    num_frequencies_per_branch=[64, 64],  # Number of Fourier features per branch
    learnable_frequencies=True,  # Learnable frequency features
    dropout_rate=0,  # Regularization
    learning_rate=1e-3,  # Optimizer learning rate
    shared_mlp=False,  # Independent MLPs for each branch
    use_residual_connections=True,  # Skip connections for better training
    add_relative_pos=False,  # Whether to include relative position features
    norm_pos=True,  # Whether to normalize position inputs
    log_scale_energy=True,  # Whether to log-scale the energy inputs
    add_distance_from_beam=False,  # Whether to include distance from beam as a feature
    reduce_lr_on_plateau=True,  # Reduce learning rate on plateau
    lr_scheduler_patience=35,  # Patience for LR scheduler
    # Flow architecture parameters
    num_flow_layers=5,  # Number of PiecewiseRationalQuadraticCDF layers
    num_bins=4,  # Number of bins for spline transforms
    tail_bound=6.0,  # Tail bound for spline transforms
    tails='linear',  # Tail type
)

hitflownet.create_and_save_flow_dataset(
    sampler=signal_sampler,
    patd_surrogate=patd_surrogate,
    num_samples=int(100),
    min_photons=5,  # Minimum number of photons required
    save_every=10,
    save_path='hitflownet_flow_dataset_test',
    flow_training_iterations=2000,  # Max iterations to train each flow
    flow_lr=1e-3,  # Learning rate for flow training
    flow_convergence_threshold=1e-4,  # Convergence threshold
    flow_patience=50,  # Patience for flow convergence
    resume=True,
    verbose_flow_training=True,
    )

# train_dataloader = hitflownet.create_hitflow_dataloader(
#     sampler=signal_sampler, 
#     patd_surrogate=patd_surrogate,
#     num_samples_per_epoch=2048,
#     batch_size=8,  # Size of each batch
#     num_workers=4,
#     event_labels=['position', 'energy', 'zenith', 'azimuth'],
#     shuffle=True,
#     min_photons=5,  # Minimum number of photons required
#     # Flow training parameters for each sample
#     flow_training_iterations=1000,  # Max iterations to train each flow
#     flow_lr=1e-3,  # Learning rate for flow training
#     flow_convergence_threshold=1e-4,  # Convergence threshold
#     flow_patience=50  # Patience for flow convergence
# )

# history = hitflownet.train_with_dataloader(
#     train_dataloader=train_dataloader,
#     # val_dataloader=val_dataloader,  # Optional validation dataloader
#     epochs=500,  # Maximum number of training epochs
#     # early_stopping_patience=30  # Stop if validation doesn't improve for 30 epochs
# )

# # Save the best model for later use
# hitflownet.save_model('best_hitflow_net_model')

# # Save history as pickle
# pickle.dump(history, open('hitflow_net_training_history.pkl', 'wb'))

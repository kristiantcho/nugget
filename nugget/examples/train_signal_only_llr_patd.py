import nugget  # Main NUGGET package for neutrino detector optimization
import pickle
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
light_yield_surrogate = nugget.surrogates.LightSabre.LightSabrePATD(use_poisson=True, num_track_points=1000, domain_size=2500, use_max_energy_dist=True).light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(event_type='signal', domain_size=2500, E_min=1e2, E_max=1e8, energy_dist='log_uniform', find_exact_intersection=True)


llr_net = nugget.surrogates.LLRnet.LLRnet(
    # device='cuda',
    domain_size=2500,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    hidden_dims=[256, 256, 256, 256],  # Neural network architecture
    use_fourier_features=True,  # Use Fourier features for better spatial encoding
    num_parallel_branches=2,  # Multiple branches for ensemble learning
    frequency_scales=[0.1, 0.4],  # Different frequency scales for fourier features
    num_frequencies_per_branch=[64, 64],  # Number of Fourier features per branch
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
    min_photons=1,  # Minimum number of photons to consider an event valid
    num_photons_per_sample=128  # Number of photons to sample from each valid event
)

train_dataloader = llr_net.create_patd_dataloader(
    signal_sampler=signal_sampler, 
    signal_surrogate_func=light_yield_surrogate,
    num_samples_per_epoch=4096,
    batch_size=128,       # Size of each batch (N pairs)
    num_workers=4,
    event_labels=['position','energy', 'direction'],
    # shuffle=False,
    shuffle_photons=True,
)

# train_dataloader = llr_net.create_pregenerated_patd_dataloader(
#     h5_filepath='5e4_200_patd_dataset.h5',
#     num_samples_per_epoch=2048,
#     batch_size=16,       # Size of each batch (N pairs)
#     num_workers=4,
#     event_labels=['position','energy', 'direction'],
#     shuffle=False,
#     preload_to_memory=True,
#     keep_file_open=True
# )



history = llr_net.train_with_dataloader(
    train_dataloader=train_dataloader,
    # val_dataloader=val_dataloader,
    epochs=300,  # Maximum number of training epochs
    # early_stopping_patience=30  # Stop if validation doesn't improve for 50 epochs
    input_dim=11 # manually specify input dimension 
)
# Save the best model for later use
llr_net.save_model('best_ls_signal_only_model_large_uniform_logE_poisson_cylinder_patd')

#save history as pickle
pickle.dump(history, open('signal_only_uniform_logE_poisson_cylinder_patd_training_history.pkl', 'wb'))
import nugget  # Main NUGGET package for neutrino detector optimization
import pickle


lightsabre = nugget.surrogates.LightSabre.LightSabre(use_poisson=False, domain_size=2500)
light_yield_surrogate = lightsabre.light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(event_type='signal', domain_size=2500, E_min=1e2, E_max=1e8, energy_dist='log_uniform', find_exact_intersection=True)


chargenet = nugget.surrogates.ChargeNet.ChargeNet(
    domain_size=2500,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    hidden_dims=[64, 64, 64, 64],  # Neural network architecture
    use_fourier_features=True,  # Use Fourier features for better spatial encoding
    num_parallel_branches=2,  # Multiple branches for ensemble learning
    frequency_scales=[0.1, 0.4],  # Different frequency scales for fourier features
    num_frequencies_per_branch=[64,64],  # Number of Fourier features per branch
    learnable_frequencies=True,  # Fixed frequency features
    dropout_rate=0,  # Regularization
    learning_rate=1e-4,  # Optimizer learning rate
    shared_mlp=False,  # Independent MLPs for each branch
    use_residual_connections=True,  # Skip connections for better training
    add_relative_pos=False,  # Whether to include relative position features
    log_scale_ly=True,  # Whether to log-scale the light yield inputs
    norm_pos=True,  # Whether to normalize position inputs
    log_scale_energy=True,  # Whether to log-scale the energy inputs
    add_distance_from_beam=False,  # Whether to include distance from beam as a feature
    reduce_lr_on_plateau=True,  # Reduce learning rate on plateau
    lr_scheduler_patience=35,  # Patience for LR scheduler
)

train_dataloader = chargenet.create_charge_dataloader(
    sampler=signal_sampler, 
    surrogate_func=light_yield_surrogate,
    num_samples_per_epoch=2048,
    batch_size=8,       # Size of each batch (N pairs)
    num_workers=4,
    event_labels=['position','energy', 'direction'],
    shuffle=True,
    min_light_yield=1,         
    max_resample_attempts=200
)

history = chargenet.train_with_dataloader(
    train_dataloader=train_dataloader,
    # val_dataloader=val_dataloader,
    epochs=500,  # Maximum number of training epochs
    # early_stopping_patience=30  # Stop if validation doesn't improve for 50 epochs
)
# Save the best model for later use
chargenet.save_model('best_charge_net_model')

# #save history as pickle
pickle.dump(history, open('charge_net_training_history.pkl', 'wb'))
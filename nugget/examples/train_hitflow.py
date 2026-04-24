import nugget  # Main NUGGET package for neutrino detector optimization
import numpy as np


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
    find_exact_intersection=False,
    random_position_along_ray=True, 
)


hitflow = nugget.surrogates.HitFlow.HitFlow(
    domain_size=2500,  # Size of the detector domain
    dim=3,  # 3D spatial coordinates
    num_layers=12,
    hidden_features=64,  # Neural network architecture
    num_bins=12,  # Number of bins for PiecewiseRationalQuadraticCDF
    tail_bound=4,  # Tail bound for spline transforms
    vary_cylinder=False,  # Vary cylinder size during training
    min_domain_size=2500*0.01,  # Minimum cylinder size
    max_domain_size=2500,  # Maximum cylinder size  
    use_min_hit_time=False,  # Use minimum hit time as a feature
    shuffle_training_batches=True,  # Shuffle training batches each epoch
    reduce_lr_on_plateau=True,  # Reduce learning rate on plateau
    scale_fac=1.0,  # Scaling factor for hit times
)

history = hitflow.train_model(
    event_sampler=signal_sampler,
    light_yield_surrogate_func=patd_surrogate,
    num_iterations=3000,
    batch_size=32,
    epoch_size=5000,
    lr=1e-4,
    min_hits=10,  # Minimum number of hits to consider an event valid
    save_interval=100,
    save_path='best_hitflow_model_v4',
    verbose=True,
    max_hits_per_event=100,
    sampling_timeout=1000
    )
# # Save the best model for later use
# hitflow.save_model('best_hitflow_model')

# # Save history as pickle
np.save('hitflow_training_history_v4.npy', np.array(history))

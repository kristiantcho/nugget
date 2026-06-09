import nugget  # Main NUGGET package for neutrino detector optimization
import pickle
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


device = None

light_yield_surrogate = nugget.surrogates.LightSabre.LightSabrePATD(
        device=device,
        use_poisson=True,
        num_track_points=1000,
        domain_size=2000,
        use_max_energy_dist=True,
        use_perpendicular_distance_only=True,
        particle_mode='track',
        ).light_yield_surrogate

signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
        device=device,
        event_type='signal',
        domain_size=2000,
        E_min=1e2,
        E_max=1e8,
        energy_dist='log_uniform',
        find_exact_intersection=False,
        random_position_along_ray=True,
        uniform_zenith_sampling=True,
        )


llr_net = nugget.surrogates.LLRnet.LLRnet(
    device=device,
    domain_size=2000,
    dim=3,
    hidden_dims=[64, 64, 64, 64, 64, 64],
    use_fourier_features=False,
    num_parallel_branches=1,
    shared_mlp=False,
    use_residual_connections=True,
    learnable_frequencies=False,
    dropout_rate=0,
    
    learning_rate=1e-4,
    reduce_lr_on_plateau=True,
    lr_scheduler_patience=30,
    lr_scheduler_factor=0.5,
    lr_scheduler_min_lr=1e-6,
    
    add_relative_pos=False,
    log_scale_ly=True,
    norm_pos=True,
    log_scale_energy=True,
    input_charge=False,
    input_delta_time=False, 
    
    use_patd=True,
    min_photons=1,
    num_photons_per_sample=16,
    
    rel_time=True,
    use_rich_features=True,
    add_distance_from_beam=True, 
    add_vertex_distance=False,
)

train_dataloader = llr_net.create_patd_dataloader(
    signal_sampler=signal_sampler,
    signal_surrogate_func=light_yield_surrogate,
    num_samples_per_epoch=4096,
    batch_size=32,
    num_workers=4,
    shuffle_photons=True,
    manual_photons=True,
      
)

history = llr_net.train_with_dataloader(
    train_dataloader=train_dataloader,
    epochs=1500,
    input_dim=13,
    grad_clip=None,
    save_every_n_epochs=20,
    checkpoint_path='best_hit_llr_model_v7.pt',
)

# llr_net.save_model('best_cascade_hit_llr_model_v4')
pickle.dump(history, open('hit_llr_v7_training_history.pkl', 'wb'))

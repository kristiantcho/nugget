import nugget  # Main NUGGET package for neutrino detector optimization
import pickle
import torch
import numpy as np
# import os
# from nugget.losses.effective_area import get_bounding_cylinder

device = 'cuda:2'
num_events = 50000
version = 'r600_50_u_1'
print(f"Using device: {device}")
print(f"Using signal_version: {version}")
use_llrnet = False
center = [0,0,0]
radius = 600
height = 1000
use_patd=False
recompute=False

events_per_batch = 50 if not use_llrnet and not use_patd else 1
extra = '_patd' if use_patd else ''
if not use_patd:
    lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabre(device=device, use_poisson=False, domain_size=1600, particle_mode = 'track')
else:
    lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabrePATD(
        device=device,
        use_poisson=False,
        num_track_points=1000,
        domain_size=1600,
        use_max_energy_dist=True,
        use_perpendicular_distance_only=True,
        particle_mode='track',
        )
light_yield_surrogate = lightsabre_surrogate.light_yield_surrogate
# signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
#                                                     device=None, 
#                                                     event_type='signal', 
#                                                     domain_size=1600, 
#                                                     E_min=1e2, 
#                                                     E_max=1e8, 
#                                                     find_exact_intersection=False,
#                                                     random_position_along_ray=True,  
#                                                     energy_dist='log_uniform',
#                                                     cylinder_center=center,
#                                                     cylinder_radius=radius,
#                                                     cylinder_height=height,
#                                                     uniform_zenith_sampling=True,
#                                                     cos_range=torch.tensor([-1,1]),
#                                                     # point_towards_center=True,
#                                                     # cos_range=torch.tensor((np.cos(np.radians(155)),np.cos(np.radians(180))))
#                                                     )
# signal_events = signal_sampler.sample_events(num_events)

# signal_events = pickle.load(open(f'/u/kristiantcho/ptmp/nugget/nugget/examples/res_test/signal_events_{num_events}_{version}.pkl', 'rb'))
    # for event in signal_events:
    #     event['position'] = torch.tensor([0.0,0.0,0.0], device=device)  # Center events for testing
# events_file_name = f'res_test/signal_events_{num_events}_{version}'
# pickle.dump(signal_events, open(events_file_name +'.pkl', 'wb'))
signal_events = nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events_{num_events}_{version}.pt')[:]
# nugget.utils.data_tools.save_signal_events_parquet(signal_events, f'res_test/signal_events_{num_events}_{version}.pt')
# signal_events = nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events_{num_events}_{version}.pt')
# signal_events = nugget.utils.data_tools.load_signal_events_parquet(f'res_test/space_test/{version}/signal_events_{num_events}_e2-4_vertical.pt')
# signal_events += nugget.utils.data_tools.load_signal_events_parquet(f'res_test/space_test/{version}/signal_events_{num_events}_e4-6_vertical.pt')
# signal_events += nugget.utils.data_tools.load_signal_events_parquet(f'res_test/space_test/{version}/signal_events_{num_events}_e6-8_vertical.pt')
# # signal_events += nugget.utils.data_tools.load_signal_events_parquet(f'res_test/space_test/{version}/signal_events_{num_events}_e5-6_vertical.pt')
# # signal_events += nugget.utils.data_tools.load_signal_events_parquet(f'res_test/space_test/{version}/signal_events_{num_events}_e6-8_vertical.pt')
# num_events *= 3
# version += '_vertical'
# nugget.utils.data_tools.save_signal_events_parquet(signal_events, f'res_test/signal_events_{num_events}_{version}.pt')
for i, signal_event in enumerate(signal_events):
    for key, value in signal_event.items():
        if isinstance(value, torch.Tensor):
            signal_events[i][key] = value.to(device)
# for geom_name in ['800main']:
for geom_name in ['800main_full_hex', '340grid']:
    print(f"Running fisher info calculations for geometry: {geom_name}")
    # if geom_name != '800main':
    #     n_strings = 70
    #     domain_size = 1200
    # else:
    n_strings = 1027 # number of strings for a complete hexagon with 18 rings
    domain_size = 1600
    geometry = nugget.geometries.EvanescentString.EvanescentString(
        device=device,
    # geometry = nugget.geometries.SpaceString.SpaceString(
    #     hex_type='hexagonal',
        domain_size=domain_size,  # Size of detector domain
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use GPU if available
        dim=3,  # 3D geometry
        n_strings=n_strings,  # Initial number of detector strings
        points_per_string=20,  # Number of PMTs/sensors per string
        # random_weights=True,
        custom_z_spacing=50,
        starting_weight=6
        # custom_string_spacing=0.09  # Custom spacing between strings
        # starting_spacing=0.05
    )
    if geom_name == '340grid':
        geom_dict = geometry.initialize_points(
            {"string_xy": np.load(f'/u/kristiantcho/ptmp/nugget/nugget/other/{geom_name}_xy.npy')}
        )
    else:
        geom_dict = geometry.initialize_points()

    # center, radius, height = get_bounding_cylinder(geom_dict['points_3d'])
    # print(f"Bounding cylinder center: {center}, radius: {radius}, height: {height}")
    
    angular_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
        device=device,
        resolution_type='angular',
        fisher_info_params=['energy', 'direction']
    )

    signal_yield_loss_func = nugget.losses.light_yield.WeightedLightYieldLoss(
        device=device,
    )
    if not use_patd and use_llrnet:
        llr_net = nugget.surrogates.LLRnet.LLRnet(
            device=device,
            domain_size=2000,  # Size of the detector domain
            dim=3,  # 3D spatial coordinates
            hidden_dims=[64, 64, 64, 64],  # Neural network architecture
            use_fourier_features=False,  # Use Fourier features for better spatial encoding
            num_parallel_branches=1,  # Multiple branches for ensemble learning
            frequency_scales=[0.1, 0.4],  # Different frequency scales for fourier features
            num_frequencies_per_branch=[64,64],  # Number of Fourier features per branch
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
            use_rich_features=True,  # Whether to use rich features from the surrogate model
        )

        llr_net.load_model('best_charge_llr_model_v5.pt')
    elif use_patd and use_llrnet:
        llr_net = nugget.surrogates.LLRnet.LLRnet(
        device=device,
        domain_size=2000,
        dim=3,
        hidden_dims=[64, 64, 64, 64, 64, 64],
        use_fourier_features=False,
        num_parallel_branches=1,
        learnable_frequencies=False,
        dropout_rate=0.05,
        learning_rate=3e-4,
        shared_mlp=False,
        use_residual_connections=True,
        signal_noise_scale=0,
        background_noise_scale=0,
        # add_relative_pos / log_scale_ly / norm_pos / log_scale_energy are unused
        # when use_rich_features=True; prepare_features_patd handles normalisation internally.
        add_relative_pos=False,
        log_scale_ly=True,
        norm_pos=True,
        log_scale_energy=True,
        
        reduce_lr_on_plateau=True,
        lr_scheduler_patience=30,
        lr_scheduler_factor=0.5,
        lr_scheduler_min_lr=1e-6,
        min_photons=1,
        num_photons_per_sample=None,
        input_charge=False,
        rel_time=True,
        input_delta_time=False,  # residual time is baked into prepare_features_patd as t_geom_min
        use_rich_features=True,
        add_distance_from_beam=True,
        use_patd=True,
        add_vertex_distance=False,
        )

        llr_net.load_model('best_hit_llr_model_v7.pt')
    # llr_net.load_model('best_cascade_charge_llr_model_v1')


    # for i in range(9):
        # print(f"Sampling events and computing losses for iteration {i+1}/9...")
    
    #check if file exists and add number to filename

    # while os.path.isfile(events_file_name + str(i) + '.pkl'):
    #     i += 1
    if recompute:
        events_file_name = f'res_test/fisher_info_per_string_per_event_{num_events}_{geom_name}_{version}{extra}.pt'
        try: 
            fisher_info_recompute = torch.load(events_file_name)
            fisher_info_recompute = fisher_info_recompute.to(device)
            print(f"Loaded precomputed fisher info from file: {events_file_name}")
        except FileNotFoundError:
            print(f"No precomputed fisher info found at {events_file_name}, computing from scratch...")
            recompute = False
    fisher_info_per_string_per_event = angular_resolution_loss.compute_fisher_info_per_string_per_event(
                string_xy=geom_dict['string_xy'],
                points_3d=geom_dict['points_3d'],
                signal_event_params=signal_events,
                signal_surrogate_func=light_yield_surrogate,
                llr_net=llr_net if use_llrnet else None,
                llr_iterations=100 if use_llrnet else 1,
                skip_zero_response=True,
                verbose=True,
                jacrev_chunk_size=50000,
                point_chunk_size=22000,
                grad_chunk_size=7,
                llr_autodiff_mode='jvp',
                use_rich_features=True,
                use_patd=use_patd,
                use_patd_quadrature=use_patd,
                t_offset_ns=0,
                t_max_ns=5000,
                zero_response_threshold=0.001,
                use_charge_quadrature=not use_patd and use_llrnet,  # for patd, charge quadrature is handled inside the surrogate
                charge_center_on_llr_peak=not use_patd,  # for patd, the surrogate is already centered on the llr peak
                charge_peak_scan_points=64,
                adaptive_grid_retry=True,
                adaptive_t_max_floor_ns=10,
                uninformative_fisher_value=1e-6,
                precomputed_fisher_per_string_per_event=fisher_info_recompute if recompute else None,
                recompute_bad_points=recompute,
                empty_cache_after_event=True,
                events_per_batch=events_per_batch,
                )

    # precomputed_ly = signal_yield_loss_func.light_yield_per_string(
    #             surrogate_func=light_yield_surrogate,
    #             event_params=signal_events,
    #             string_xy=geom_dict['string_xy'],
    #             points_3d=geom_dict['points_3d']
    #             )


    
    extra += '_pois' if not use_llrnet else ''
    torch.save(fisher_info_per_string_per_event.cpu(), f'res_test/fisher_info_per_string_per_event_{num_events}_{geom_name}_{version}{extra}.pt')
    # torch.save(precomputed_ly.cpu(), f'res_test/light_yield_per_string_{num_events}_{geom_name}_{version}.pt')


# loss_params = {
# # 'cylinder_kwargs': {'find_exact_intersection': True},
# # 'num_events_per_bin': 10,
# 'signal_surrogate_func': light_yield_surrogate,
# 'bounding_cylinder_temperature': 10,
# # 'perfect_efficiency': False,
# # 'zenith_range': (-1, 0),
# # 'skip_zero_response': True,
# 'fisher_info_llr_net': llr_net,
# 'use_relative_energy': True,
# 'signal_event_params': signal_events,
# 'fisher_info_llr_iterations': 100,
# 'llr_event_labels': ['position','energy', 'direction'],
# }
    
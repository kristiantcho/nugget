import nugget  # Main NUGGET package for neutrino detector optimization
import pickle
import torch
import numpy as np
import os
from nugget.losses.effective_area import get_bounding_cylinder

device = 'cuda:1'
version = 'r600_50_1'
print(f"Using device: {device}")
print(f"Using signal_version: {version}")
center = [0,0,0]
radius = 600
height = 1000
lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabre(device=device, use_poisson=True, domain_size=2500)
light_yield_surrogate = lightsabre_surrogate.light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
                                                    device=None, 
                                                    event_type='signal', 
                                                    domain_size=2500, 
                                                    E_min=1e2, 
                                                    E_max=1e8, 
                                                    find_exact_intersection=True,  
                                                    energy_dist='log_uniform',
                                                    cylinder_center=center,
                                                    cylinder_radius=radius,
                                                    cylinder_height=height,
                                                    # point_towards_center=True,
                                                    # cos_range=torch.tensor((np.cos(np.pi/2),np.cos(np.pi/2)))
                                                    )
signal_events = signal_sampler.sample_events(10000)
    # signal_events = pickle.load(open('/u/kristiantcho/ptmp/nugget/nugget/examples/res_test/signal_events_10000_r800_1.pkl', 'rb'))
    # for event in signal_events:
    #     event['position'] = torch.tensor([0.0,0.0,0.0], device=device)  # Center events for testing
events_file_name = f'res_test/signal_events_10000_{version}'
pickle.dump(signal_events, open(events_file_name +'.pkl', 'wb'))
for i, signal_event in enumerate(signal_events):
    for key, value in signal_event.items():
        if isinstance(value, torch.Tensor):
            signal_events[i][key] = value.to(device)
for geom_name in ['compact', 'default', 'expanded', 'modified', 'large', '102geom', '160geom', '600hexagon', '800main']:
    print(f"Running fisher info calculations for geometry: {geom_name}")
    if geom_name != '800main':
        n_strings = 70
        domain_size = 1200
    else:
        n_strings = 1000
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
    if geom_name != '800main' and geom_name != '600hexagon':
        geom_dict = geometry.initialize_points(
            {"string_xy": np.load(f'/u/kristiantcho/ptmp/other/{geom_name}_xy.npy')}
        )
    else:
        geom_dict = geometry.initialize_points()

    # center, radius, height = get_bounding_cylinder(geom_dict['points_3d'])
    # print(f"Bounding cylinder center: {center}, radius: {radius}, height: {height}")
    
    angular_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
        device=device,
        resolution_type='angular',
        fisher_info_params=['position','energy', 'direction']
    )

    signal_yield_loss_func = nugget.losses.light_yield.WeightedLightYieldLoss(
        device=device,
    )

    llr_net = nugget.surrogates.LLRnet.LLRnet(
        device=device,
        domain_size=(radius*2,height),  # Size of the detector domain
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
    )

    llr_net.load_model('best_charge_llr_model_v2')


    # for i in range(9):
        # print(f"Sampling events and computing losses for iteration {i+1}/9...")
    
    #check if file exists and add number to filename

    # while os.path.isfile(events_file_name + str(i) + '.pkl'):
    #     i += 1

    fisher_info_per_string_per_event = angular_resolution_loss.compute_fisher_info_per_string_per_event(
                string_xy=geom_dict['string_xy'],
                points_3d=geom_dict['points_3d'],
                signal_event_params=signal_events,
                signal_surrogate_func=light_yield_surrogate,
                llr_net=llr_net,
                llr_iterations=100,
                skip_zero_response=True,
                verbose=True,
                jacrev_chunk_size=50000,
                point_chunk_size=7000,
                grad_chunk_size=7,
                llr_autodiff_mode='jvp'
                )

    precomputed_ly = signal_yield_loss_func.light_yield_per_string(
                surrogate_func=light_yield_surrogate,
                event_params=signal_events,
                string_xy=geom_dict['string_xy'],
                points_3d=geom_dict['points_3d']
                )


    
    torch.save(fisher_info_per_string_per_event.cpu(), f'res_test/fisher_info_per_string_per_event_10000_{geom_name}_{version}.pt')
    torch.save(precomputed_ly.cpu(), f'res_test/light_yield_per_string_10000_{geom_name}_{version}.pt')


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
    
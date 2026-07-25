import nugget  # Main NUGGET package for neutrino detector optimization
import torch
import numpy as np
import pickle

device='cuda:2'
center = [0,0,0]
radius = 600
height = 1000
num_strings = 61
event_type = 'cascade'  # 'track' or 'cascade'
lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabre(device=device, use_poisson=False, domain_size=1600, particle_mode = event_type)
light_yield_surrogate = lightsabre_surrogate.light_yield_surrogate_batched
# signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
#                                                     device=device, 
#                                                     event_type='signal', 
#                                                     domain_size=1600, 
#                                                     E_min=1e2, 
#                                                     E_max=1e8, 
#                                                     find_exact_intersection=True,
#                                                     random_position_along_ray=False,  
#                                                     energy_dist='log_uniform',
#                                                     cylinder_center=center,
#                                                     cylinder_radius=radius,
#                                                     cylinder_height=height,
#                                                     uniform_zenith_sampling=True,
#                                                     cos_range=torch.tensor([0,1]),
#                                                     # point_towards_center=True,
#                                                     # cos_range=torch.tensor((np.cos(np.radians(155)),np.cos(np.radians(180))))
#                                                     )
angular_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
        device=device,
        resolution_type='angular',
        fisher_info_params=['direction']
    )
energy_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
    device=device,
    resolution_type='energy',
    fisher_info_params=['energy']
)

num_events = 50000
angular_loss_dicts = {}
energy_loss_dicts = {}
spacing_min = 10
spacing_max = 250
spacing_count = 50

for energy in [2,3,4,5,6,7]:
    # print(f"Running fisher info calculations for energy range: 10^{energy} GeV to 10^{energy+1 if energy < 6 else energy+2} GeV")
    print(f"Running fisher info calculations for energy range: 10^{energy} GeV to 10^{energy+1} GeV")
    signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
                                                        device=None, 
                                                        event_type='signal', 
                                                        domain_size=1600, 
                                                        E_min=10**energy, 
                                                        E_max=10**(energy+1), #if energy < 6 else 10**(energy+2), 
                                                        find_exact_intersection=True if event_type == 'track' else False,
                                                        random_position_along_ray=False if event_type == 'track' else True,
                                                        energy_dist='log_uniform',
                                                        cylinder_center=center,
                                                        cylinder_radius=radius,
                                                        cylinder_height=height,
                                                        uniform_zenith_sampling=True,
                                                        cos_range=torch.tensor([-1,1]),
                                                        # point_towards_center=True,
                                                        # cos_range=torch.tensor((np.cos(np.radians(155)),np.cos(np.radians(180))))
                                                        )
    signal_events = signal_sampler.sample_events(num_events=num_events)
    # nugget.utils.data_tools.save_signal_events_parquet(signal_events, f'pois_tests/pois_space_127_signal_events_e{energy}-e{energy+1 if energy < 6 else energy+2}.parquet')
    nugget.utils.data_tools.save_signal_events_parquet(signal_events, f'pois_tests/pois_space_{num_strings}_{event_type}_signal_events_e{energy}-e{energy+1}.parquet')
    # if energy < 6:
    angular_loss_dicts[f'{energy}-{energy+1}'] = []
    energy_loss_dicts[f'{energy}-{energy+1}'] = []
    # else:
    #     angular_loss_dicts[f'{energy}-{energy+2}'] = []
    #     energy_loss_dicts[f'{energy}-{energy+2}'] = []
    for string_spacing in np.logspace(np.log10(spacing_min), np.log10(spacing_max), spacing_count):
        print(f"Running fisher info calculations for string spacing: {string_spacing}m", flush=True)
        geometry = nugget.geometries.SpaceString.SpaceString(
                device=device,
                hex_type='hexagonal',
                domain_size=2500,
                dim=3,
                n_strings=num_strings,
                points_per_string=20,
                starting_spacing=string_spacing,
                starting_z_spacing=50
            )
        geom_dict = geometry.initialize_points()
        
        
        loss_params = {
                    'signal_event_params': signal_events,
                    'signal_surrogate_func': light_yield_surrogate,
                    'llr_net': None,
                    'llr_iterations': 1,
                    'skip_zero_response': False,
                    'verbose': True,
                    'jacrev_chunk_size': 50000,
                    'point_chunk_size': 22000,
                    'grad_chunk_size': 7,
                    'llr_autodiff_mode': 'jvp',
                    'use_rich_features': True,
                    'use_patd': False,
                    'use_patd_quadrature': False,
                    't_offset_ns': 0,
                    't_max_ns': 5000,
                    'zero_response_threshold': 0.001,
                    'use_charge_quadrature': False,
                    'charge_center_on_llr_peak': False,
                    'adaptive_grid_retry': True,
                    'adaptive_t_max_floor_ns': 10,
                    'uninformative_fisher_value': 1e-6,
                    'precomputed_fisher_per_string_per_event': None,
                    'recompute_bad_points': False,
                    'empty_cache_after_event': True,
                    'events_per_batch': 100000,
                    'fisher_info_detach_tensors': True,
                    'fisher_info_use_patd': False,
                    'fisher_res_metric': 'fom',  # 'fom' 'median' 'mean'
                    'fisher_info_use_torch_compile': False,
                    }
        ang_loss_dict = angular_resolution_loss(
                    geom_dict=geom_dict,
                    **loss_params
                    )
        energy_loss_dict = energy_resolution_loss(
                    geom_dict=geom_dict,
                    **loss_params
                    )
        for key in ang_loss_dict:
            if isinstance(ang_loss_dict[key], torch.Tensor):
                ang_loss_dict[key] = ang_loss_dict[key].detach().cpu()
        for key in energy_loss_dict:
            if isinstance(energy_loss_dict[key], torch.Tensor):
                energy_loss_dict[key] = energy_loss_dict[key].detach().cpu()
        # if energy < 6:
        angular_loss_dicts[f'{energy}-{energy+1}'].append(ang_loss_dict)
        energy_loss_dicts[f'{energy}-{energy+1}'].append(energy_loss_dict)
        # else:
            # angular_loss_dicts[f'{energy}-{energy+2}'].append(ang_loss_dict)
            # energy_loss_dicts[f'{energy}-{energy+2}'].append(energy_loss_dict)




copy_angular_loss_dicts = angular_loss_dicts.copy()
copy_energy_loss_dicts = energy_loss_dicts.copy()
for energy in range(2,8):
    # for angular_loss_dict in copy_angular_loss_dicts[f'{energy}-{energy+1}' if energy < 6 else f'{energy}-{energy+2}']:
    for angular_loss_dict in copy_angular_loss_dicts[f'{energy}-{energy+1}']:
        del angular_loss_dict['resolution_params']
    # for energy_loss_dict in copy_energy_loss_dicts[f'{energy}-{energy+1}' if energy < 6 else f'{energy}-{energy+2}']:
    for energy_loss_dict in copy_energy_loss_dicts[f'{energy}-{energy+1}']:    
        del energy_loss_dict['resolution_params']
with open(f'pois_tests/pois_{num_strings}_{event_type}_angular_loss_dicts_energies.pkl', 'wb') as f:
    pickle.dump(copy_angular_loss_dicts, f)

with open(f'pois_tests/pois_{num_strings}_{event_type}_energy_loss_dicts_energies.pkl', 'wb') as f:
    pickle.dump(copy_energy_loss_dicts, f)
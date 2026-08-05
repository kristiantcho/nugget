import nugget  # Main NUGGET package for neutrino detector optimization
import torch
import numpy as np
import pickle

device='cuda:3'
center = [0,0,0]
radius = 600
height = 1000
num_strings = 61
event_type = 'track'  # 'track' or 'cascade'
limit_zenith = None  # 'horizontal' or 'vertical'
free_sim_volume = False  # If True, the simulation volume is not constrained to a cylinder. If False, the simulation volume is constrained to a cylinder with the specified center, radius, and height.
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
trigger_loss = nugget.losses.trigger.TriggerLoss(
    device=device,
    light_yield_threshold=6.0,
    distance_bar_length=550.0,
    distance_bar_step=20.0,
    min_points_threshold=30.0,
    t1_temperature=2.0,
    t3_temperature=2.0,
    t_temperature=4.0, 
    use_hard_cuts=True,
)
effective_area_loss = nugget.losses.effective_area.EffectiveAreaLoss(
    device=device,
    domain_size=2000,
    trigger=trigger_loss
    )



analysis_loss = nugget.losses.analysis_loss.AnalysisLoss(
    device=device,
    fisher_info_params=['direction'],
    effective_area_loss=effective_area_loss
)

num_events = 100000
angular_loss_dicts = {}
energy_loss_dicts = {}
spacing_min = 50
spacing_max = 300
spacing_count = 100


    # print(f"Running fisher info calculations for energy range: 10^{energy} GeV to 10^{energy+1 if energy < 6 else energy+2} GeV")
    
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
                                                    device=None, 
                                                    event_type='signal', 
                                                    domain_size=1600, 
                                                    E_min=10**2, 
                                                    E_max=10**8, #if energy < 6 else 10**(energy+2), 
                                                    find_exact_intersection=True if event_type == 'track' else False,
                                                    random_position_along_ray=False if event_type == 'track' else True,
                                                    energy_dist='log_uniform',
                                                    cylinder_center=center,
                                                    cylinder_radius=radius,
                                                    cylinder_height=height,
                                                    uniform_zenith_sampling=True,
                                                    cos_range=torch.tensor([-1,1]) if limit_zenith is None else limit_zenith,
                                                    # point_towards_center=True,
                                                    # cos_range=torch.tensor((np.cos(np.radians(155)),np.cos(np.radians(180))))
                                                    )
if not free_sim_volume:
    signal_events = signal_sampler.sample_events(num_events=num_events)
    signal_events = analysis_loss._ensure_weights(
                    signal_events,
                    friend_config='../other/friend_config.yaml',
                    pyff_config='../other/pyff_config.yaml',
                    signal_sampler=signal_sampler,
                    pid=14,
                    temp_path=None,
                )
    # nugget.utils.data_tools.save_signal_events_parquet(signal_events, f'pois_tests/pois_space_127_signal_events_e{energy}-e{energy+1 if energy < 6 else energy+2}.parquet')
    nugget.utils.data_tools.save_signal_events_parquet(signal_events, f'pois_tests/pois_space_{num_strings}_{event_type}_{limit_zenith+ "_" if limit_zenith is not None else ""}signal_events_analysis.parquet')
    signal_events = nugget.utils.data_tools.add_events_to_device(signal_events, device=device)
# if energy < 6:

# else:
#     angular_loss_dicts[f'{energy}-{energy+2}'] = []
#     energy_loss_dicts[f'{energy}-{energy+2}'] = []
analysis_loss_dicts = {}
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
    if free_sim_volume:
        # change cylinder parameters to match the geometry for bigger volume:
        radius = torch.max(torch.linalg.vector_norm(geom_dict['string_xy'], axis=1))
        signal_sampler.cylinder_radius = radius
        signal_events = signal_sampler.sample_events(num_events=num_events)
    
    loss_params = {
                'num_events': num_events,
                'signal_sampler': signal_sampler,
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
                'fisher_info_use_torch_compile': True,
                'use_relative_energy': True,


                'trigger_use_torch_compile': False,
                'use_batched_trigger': True,
                'use_batched_binned_trigger': True,
                'use_batched_effective_area': True,
                'bounding_cylinder_temperature': 1,
            
            
                'downweight_untriggerable': True,
                "trigger_neighbor_distance": 550.0,
                "trigger_min_neighbors": 30,
                'trigger_distance_sharpness': 0.05,
                'trigger_count_sharpness':1,
                # 'num_events_per_bin': 100,
                # 'num_energy_bins': 30,
                # 'num_zenith_bins': 30,
                'per_event_effective_area_loss': True,
                'fom_adjust_cylinder_to_geometry':False,
                'fom_include_uniform_log_e_term':True,
                'include_projected_area':True,
                'use_sampler_cyl_for_volume':True,
                # 'perfect_efficiency': False,
                # 'cos_zenith_range': (-1, 0),
                # 'binned_trigger_batch_size': 500,
                # 'batched_surrogate_func': batched_track_surrogate,
                # 'detach_trigger': False,
                'analysis_signal_flux_var_names': ['astro_norm'],
                'friend_config': '../other/friend_config.yaml',
                'pyff_config': '../other/pyff_config.yaml',
                'live_time': 10 * 365 * 24 * 60 * 60,  # 10 years in seconds
                }
    analysis_loss_dict = analysis_loss(
                geom_dict=geom_dict,
                **loss_params
                )

    for key in analysis_loss_dict:
        if isinstance(analysis_loss_dict[key], torch.Tensor):
            analysis_loss_dict[key] = analysis_loss_dict[key].detach().cpu()
    analysis_loss_dicts[f'{string_spacing}'] = analysis_loss_dict
    torch.cuda.empty_cache()

  
with open(f'pois_tests/pois_{num_strings}_{event_type}_{limit_zenith + "_" if limit_zenith is not None else ""}{"free_" if free_sim_volume else ""}analysis_loss_dicts.pkl', 'wb') as f:
    pickle.dump(analysis_loss_dicts, f)

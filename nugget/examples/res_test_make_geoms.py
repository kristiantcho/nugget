import torch
import numpy as np
import nugget
import pickle
import os

device="cuda:3"
string_number_penalty = nugget.losses.geometry_penalties.StringNumberPenalty(device=device)
string_boundary_penalty = nugget.losses.geometry_penalties.StringBoundaryPenaltyCircle(device=device)
weighted_binarization_penalty = nugget.losses.geometry_penalties.WeightBinarizationPenalty(device=device)
local_string_repulsion_penalty = nugget.losses.geometry_penalties.LocalStringRepulsionPenalty(device=device)
weighted_angular_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
    device=device,
    resolution_type='angular',
    fisher_info_params=['direction', 'position']
    )
weighted_energy_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
    device=device,
    resolution_type='energy',
    fisher_info_params=['energy', 'position']
)
rov_penalty = nugget.losses.geometry_penalties.ROVPenalty( 
    device=device,
    rov_rec_width=230,  # ROV dimensions
    rov_height=160, 
    rov_tri_length=160
)
fisher_res_metric = 'mean'  # 'fom' 'median' 'mean'
version = '_poisson'
use_rov = 'rov' # rov or no_rov
num_events = 'inf'
event_type = 'track' # cascade or track
res_param = 'angle' # angle or energy
use_weights = True
use_fold = False
n_folds = 6
spacing_test = False
num_strings = 61
limit_zenith = 'vertical' # vertical, horizontal, or None for full range
center = [0,0,0]
radius = 600
height = 1000
bin_energies = True
folder_name = f'res_test/opt_geoms/opt_geoms_dyn_{num_strings}_{num_events}_r{radius}_50{version}_{use_rov}_{event_type}_{res_param}_{fisher_res_metric}{"_"+limit_zenith if limit_zenith is not None else ""}{"_spacing_test" if spacing_test else ""}{"_6fold" if use_fold else ""}{"_weights" if use_weights else ""}'
print(f"Saving optimized geometries to folder: {folder_name}")
# if folder does not exist, create it

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# check if there are already optimized geometries in the folder, and if so add a number to the folder name
# count = 0
# while os.path.exists(f'{folder_name}geom_{count}.pkl'):
#     count += 1

loss_params = {
    # 'llr_net': llr_net,
    # 'signal_event_params': nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events/signal_events_{total_events}_{events_version}.pt'),  # Pre-sampled signal events for loss computation
    # 'signal_event_params': nugget.utils.data_tools.select_events(nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events_{total_events}_{events_version}.pt'),limits=selection_limits),  # Pre-sampled signal events for loss computation
    # 'background_event_params': new_background_events,
    # 'signal_surrogate_func': light_yield_surrogate,  # Function to compute light yield
    # 'background_surrogate_func': light_yield_surrogate,
    # 'signal_sampler': signal_sampler,
    # 'background_sampler': background_sampler,
    'num_events': 3000,  # Number of events to sample per optimization step
    'signal_noise_scale': 0,  # Noise level for signal events
    # 'background_noise_scale': 0.2,  # Noise level for background events
    'boundary_range': 1200,  # Size of boundary region
    'skip_zero_response': False,
    # 'fisher_info_llr_net': llr_net,
    'use_relative_energy': True,
    # 'event_paths': ['/u/kristiantcho/ptmp/nugget/nugget/examples/res_test_signal_events_100_1.pkl'],
    # 'fisher_info_paths': ['/u/kristiantcho/ptmp/nugget/nugget/examples/res_test_fisher_info_per_string_per_event_100_1.pt'],
    # 'sample_every': 50,
    # 'precomputed_signal_yield_per_string': torch.load('res_test/light_yield_per_string_50000_800main_full_hex_r600_50_u_1.pt')[selection_inds],
    # 'llr_event_labels': ['position','energy', 'direction'],
    # 'precomputed_fisher_info_per_string_per_event': fisher_info_precomp#[selection_inds]

    'eva_min_num_strings': num_strings,  # Minimum number of active strings
    'string_number_use_binarization_weight': False,
    'max_radius': 80,  # Maximum radius for string placement
    'num_angles': 360,  # Number of angles (divided into 360 degrees) to test for rov
    'rov_alt_mode': True,  # Whether to use alternative mode for rov penalty (see rov_penalty.py for details)
    'local_sharpness': 10,  # Sharpness parameter for local string repulsion
    'boundary_sharpness': 10,  # Sharpness parameter for boundary penalty
    'string_number_beta':5,
    'detach_other_probs':False,
    'rov_soft_inside':True,
    'rov_inside_sharpness': 10,
    'rov_angle_softmin_tau': 0.05,
    'rov_angle_chunk_size': 60,
    'rov_away_weight': 0.00,
    'rov_away_num_neighbours':5,
    'rov_use_torch_compile': False,
    'local_repulsion_use_torch_compile': False,
    'rov_inside_use_softplus': True,


    # 'other_geoms':[pickle.load(open('./800main_full_hex_r600_50_pl_1_ang_res_rov/geom_17.pkl', 'rb'))],
    'diversity_use_hungarian': True,
    'diversity_use_sinkhorn':False,
    'diversity_min':1,
    'sinkhorn_niter':10,
    'sinkhorn_epsilon':0.5,
    'diversity_use_mmd':False,
    
    # 'fisher_info_llr_net': llr_net,
    # 'fisher_info_llr_iterations':50,
    # 'llr_event_labels': ['position','energy', 'direction'],
    'fisher_info_grad_chunk_size': 14,
    'fisher_info_jacrev_chunk_size': 50000,
    'fisher_info_point_chunk_size': 40100,
    'fisher_info_llr_autodiff_mode': 'jvp',
    'fisher_info_detach_tensors': False if not use_weights else True,
    'fisher_info_use_patd': False,
    'fisher_res_metric': 'mean',  # 'fom' 'median' 'mean'
    'fisher_info_use_torch_compile': True,
    # 'eval_patd_log_probs':patd_surrogate.eval_patd_log_probs,
    'use_rich_features': True,
    'use_patd_quadrature': False,
    't_offset_ns': 0.0,
    't_max_ns': 5000.0,
    'use_charge_quadrature': False,
    'charge_center_on_llr_peak': True,
    'charge_peak_scan_points':32,
    'adaptive_grid_retry':True,
    'adaptive_t_max_floor_ns':10,
    'uninformative_fisher_value':1e-6,
    'empty_cache_after_event': False,
    'events_per_batch': 1000,

    'trigger_use_torch_compile': False,
    'use_batched_trigger': True,
    'use_batched_binned_trigger': True,
    'use_batched_effective_area': True,
    'use_irregular_cylinder': False,
    'bounding_cylinder_temperature': 1,
    'binned_trigger_chunk_size': 500,
    # 'num_events_per_bin': 100,
    # 'num_energy_bins': 30,
    # 'num_zenith_bins': 30,
    'per_event_effective_area_loss': True,
    'fom_adjust_cylinder_to_geometry':False,
    'normalize_fom_by_energy':True,
    # 'perfect_efficiency': False,
    # 'cos_zenith_range': (-1, 0),
    # 'binned_trigger_batch_size': 500,
    # 'batched_surrogate_func': batched_track_surrogate,
    # 'detach_trigger': False,
    'constraints_list': [
                # 'energy_resolution_loss',
                # 'angular_resolution_loss',
                # 'signal_yield_loss', 
                'rov_penalty', 
                # 'string_boundary_penalty', 
                'local_string_repulsion_penalty', 
                'string_number_penalty', 
                # 'string_weights_penalty',
                'weight_binarization_penalty',
                # 'diversity_penalty',
                ],  # Constraints to enforce
}
loss_weights_dict = {
    'angular_resolution_loss': 1,
    'pointsource_fom_loss': 1e2,
    'energy_resolution_loss': 0.05,
    # 'fisher_loss': 0.005, 
    'signal_yield_loss': 0.01,        # High weight: maximize light collection
    # 'signal_llr_loss': 2.5,          # Moderate weight: good signal discrimination
    'string_boundary_penalty': 2,  # Very high: hard constraint
    'local_string_repulsion_penalty': 1,
    # 'string_repulsion_penalty': 0.000001,
    # 'string_weights_penalty': 0,     # Encourage sparse solutions
    'string_weights_penalty': 0.05,     # Encourage sparse solutions
    'string_number_penalty': 10,      # Limit detector complexity
    'weight_binarization_penalty': 0.0001,
    'rov_penalty': 1,
    'diversity_penalty': 0.001
}

loss_sigmoid_list = [
    'angular_resolution_loss',
    'energy_resolution_loss',
    # 'fisher_loss', 
    # 'signal_yield_loss',        # High weight: maximize light collection
    # 'signal_llr_loss',
    'string_boundary_penalty',  # Very high: hard constraint
    'local_string_repulsion_penalty',
    # 'string_repulsion_penalty': 0.000001,
    # 'string_weights_penalty': 0,     # Encourage sparse solutions
    'string_weights_penalty',     # Encourage sparse solutions
    'string_number_penalty',      # Limit detector complexity
    'weight_binarization_penalty',
    'rov_penalty'
]
# loss_func_dict = {
#     'angular_resolution_loss': weighted_angular_resolution_loss,
#     # 'energy_resolution_loss': weighted_energy_resolution_loss,
#     # 'fisher_loss': fisher_info_loss_func, 
#     # 'signal_yield_loss': signal_yield_loss_func,  # Maximize light collection
#     # 'signal_llr_loss': signal_llr_loss_func,      # Maximize signal discrimination
#     # 'local_string_repulsion_penalty': local_string_repulsion_penalty,
#     # 'string_repulsion_penalty': string_repulsion_penalty,
#     # 'string_weights_penalty': string_weights_penalty,  # Encourage using less strings
#     # 'string_number_penalty': string_number_penalty,    # Limit number of strings
#     # 'string_boundary_penalty': string_boundary_penalty,  # Keep strings in bounds
#     # 'weight_binarization_penalty': weighted_binarization_penalty
# }

loss_func_dict = {}
if res_param == 'angle':
    loss_func_dict['angular_resolution_loss'] = weighted_angular_resolution_loss
elif res_param == 'energy':
    loss_func_dict['energy_resolution_loss'] = weighted_energy_resolution_loss

if use_rov == 'rov':
    loss_func_dict['rov_penalty'] = rov_penalty
    loss_func_dict['local_string_repulsion_penalty'] = local_string_repulsion_penalty

if use_weights:
    # loss_func_dict['weight_binarization_penalty'] = weighted_binarization_penalty
    loss_func_dict['string_number_penalty'] = string_number_penalty

lightsabre = nugget.surrogates.LightSabre.LightSabre(
                device=device,
                use_poisson=False, 
                num_track_points=2000, 
                domain_size=1600,
                particle_mode=event_type,
                )
light_yield_surrogate = lightsabre.light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
    device=device,
    event_type='signal', 
    domain_size=1600, 
    E_min=10**2, 
    E_max=10**8, 
    energy_dist='log_uniform', 
    find_exact_intersection=True if event_type == 'track' else False,
    random_position_along_ray=False if event_type == 'track' else True,
    uniform_zenith_sampling=True,
    cylinder_center=center,
    cylinder_radius=radius,
    cylinder_height=height,
    cos_range = torch.tensor([-1,0]) if limit_zenith is None else limit_zenith
    # cos_range=torch.tensor((np.cos(np.pi/2),np.cos(np.pi/2)))
    )

loss_params.update({
    'signal_surrogate_func': light_yield_surrogate,
    'signal_sampler': signal_sampler,
    })

num_trials = 15
if bin_energies:
    num_trials = 6
for i in range(num_trials):
    if not bin_energies:
        print(f"Running optimization iteration {i+1}/{num_trials}")
    else:
        print(f"Running optimization iteration in energy range e{i+2}-e{i+3}")
# for i in range(6):
#     print(f"Running optimization iteration in energy range e{i+2}-e{i+3}")
    # if the geometry in the folder already exists, skip this iteration
    # if os.path.exists(f'{folder_name}/geom_{i}.pkl'):
    #     print(f"Geometry for iteration {i} already exists, skipping optimization.")
    #     continue
    # same for energy range, if the geometry for this energy range already exists, skip this iteration
    # if os.path.exists(f'{folder_name}geom_e{i+2}_e{i+3}.pkl'):
    #     print(f"Geometry for energy range e{i+2}-e{i+3} already exists, skipping optimization.")
    #     continue

    
    # selection_limits = {
    #     'energy': (10**((i*2)+2), 10**(2*(i+1) + 2)),  # Example energy range
    #     }
    # selection_inds = nugget.utils.data_tools.select_event_indices(
    #         nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events/signal_events_{num_events}_r600_50{version}.pt')[:],
    #         limits=selection_limits  # Example limit
    #         )
    # signal_events = nugget.utils.data_tools.select_events(nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events/signal_events_{num_events}_r600_50{version}.pt'),limits=selection_limits)
    # fisher_info = torch.load(f'res_test/fisher_info/fisher_info_per_string_per_event_{num_events}_800main_full_hex_r600_50{version}.pt')#[selection_inds]
    # signal_events = nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events/signal_events_{num_events}_r600_50{version}.pt')
    # loss_params.update({
    #     'signal_event_params': signal_events,
    #     'precomputed_fisher_info_per_string_per_event': fisher_info
    # })
    if bin_energies:
        signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
        device=device,
        event_type='signal', 
        domain_size=1600, 
        E_min=10**(i+2), 
        E_max=10**(i+3), 
        energy_dist='log_uniform', 
        find_exact_intersection=True,
        random_position_along_ray=False,
        uniform_zenith_sampling=True,
        cylinder_center=center,
        cylinder_radius=radius,
        cylinder_height=height,
        cos_range = torch.tensor([-1,0]) if limit_zenith is None else limit_zenith
        # cos_range=torch.tensor((np.cos(np.pi/2),np.cos(np.pi/2)))
        )
        loss_params.update({
            'signal_sampler': signal_sampler,
            })
        if i > 1:
            loss_weights_dict['energy_resolution_loss'] = 0.5
        if i > 3:
            loss_weights_dict['energy_resolution_loss'] = 5
        if i > 5:
            loss_weights_dict['energy_resolution_loss'] = 10
    if use_fold:
        geometry = nugget.geometries.NFoldString.NFoldString(
                device=device,
                domain_size=1000,  # Size of detector domain
                dim=3,  # 3D geometry
                n_folds=n_folds,
                strings_per_fold=(num_strings-1)//n_folds,
                points_per_string=20,  # Number of PMTs/sensors per string
                custom_z_spacing=50,
                use_weights=False,
                starting_weight=6,
                random_weights=False,
                slice_init = 'sunflower',
                add_center_string=True,
                # fold_offset=2*np.pi/3
            )
    elif spacing_test:
        geometry = nugget.geometries.SpaceString.SpaceString(
                            device=device,
                            hex_type='hexagonal',
                            # random_xy=True,
                            domain_size=300,  # Size of detector domain
                            dim=3,  # 3D geometry
                            n_strings=num_strings,  # Initial number of detector strings
                            points_per_string=20,  # Number of PMTs/sensors per string
                            # starting_weight=-0.5,  # Initial weight for each string (controls visibility)
                            # custom_string_spacing=0.09  # Custom spacing between strings
                            # custom_string_spacing=300.0,
                            starting_z_spacing=50,
                            starting_spacing=50.0,
                        )
    elif use_weights:
        geometry = nugget.geometries.EvanescentString.EvanescentString(
                        device=device,
                        hex_type='hexagonal',
                        domain_size=2000,  # Size of detector domain
                        dim=3,  # 3D geometry
                        n_strings=1951,  # Initial number of detector strings
                        points_per_string=20,  # Number of PMTs/sensors per string
                        starting_weight=6,  # Initial weight for each string (controls visibility)
                        random_weights=False,
                        custom_z_spacing=50,
                    )
    else:
        geometry = nugget.geometries.DynamicString.DynamicString(
                        device=device,
                        hex_type='hexagonal',
                        domain_size=1000,  # Size of detector domain
                        dim=3,  # 3D geometry
                        n_strings=num_strings,  # Initial number of detector strings
                        points_per_string=20,  # Number of PMTs/sensors per string
                        custom_z_spacing=50,
                        # random_weights=True
                        # starting_weight = 100,
                    )
    optimizer = nugget.utils.basic_optimizer.Optimizer(
        device=geometry.device, 
        geometry=geometry,
        conflict_free=False,  # Single-objective optimization (vs multi-objective)
        use_custom_cf_weight=False,
        use_alm=True,  # Use Augmented Lagrangian Method for constraints
        sigmoid_losses=True,
        sigmoid_softness=1,
        alm_params={
            'gamma': 1e-2,
            'alpha': 0.95,
            'epsilon': 1e-8,
            }
        )
    if spacing_test:
        opt_list = [('string_spacing', 5)]
    elif use_fold:
        opt_list = [('slice_radius', 5), ('slice_angle', 0.05)]
    elif use_weights:
        opt_list = [('string_weights', 0.5)]
    else:
        opt_list = [('string_xy', 5)]
    optimizer.init_geometry(
        # opt_list=[('string_weights', 0.5)],  # Learning rate for string weights (without sigmoid applied)
        opt_list=opt_list  # Learning rate for string weights (without sigmoid applied)
    )  
    if spacing_test:
        save_geom_folder = f'{folder_name}/geom_e{i+2}_e{i+3}' if bin_energies else f'{folder_name}/geom_{i}'
    else:
        save_geom_folder = None 
    geom_dict = optimizer.optimize(
        clear_cuda_cache=True,
        loss_func_dict=loss_func_dict,          # Dictionary of loss functions to use
        loss_weights_dict=loss_weights_dict,    # Weights for combining multiple losses
        loss_params_dict=loss_params,           # Parameters for loss function computation
        n_iter=1500,                           # Maximum number of optimization iterations
        print_freq=50,                          # Print progress every N iterations
        sigmoid_loss_list=loss_sigmoid_list,         # Which losses to apply sigmoid to (for better optimization dynamics)
        save_geom_folder=save_geom_folder,  # Folder to save intermediate geometries
        save_geom_freq=25,
        # save_best_geom_file = f'{folder_name}geom_e{(2*i)+2}_e{2*(i+1) + 2}.pkl',  # File to save best geometry found
        save_best_geom_file = f'{folder_name}/geom_{i}.pkl' if not bin_energies else f'{folder_name}/geom_e{i+2}_e{i+3}.pkl',  # File to save best geometry found
        # save_best_geom_file = f'{folder_name}geom_e{i+2}_e{i+3}.pkl',
        save_last_geom = True,
        revert_on_nan=True,                      # Revert geometry if loss becomes NaN
        max_nan_retries=5,  
    )
    if spacing_test:
        pickle.dump(optimizer.uw_loss_dict, open(f'{folder_name}/geom_e{i+2}_e{i+3}_loss_dict.pkl', 'wb') if bin_energies else open(f'{folder_name}/geom_{i}_loss_dict.pkl', 'wb'))

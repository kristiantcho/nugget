import torch
import numpy as np
import nugget
import pickle
import os

string_number_penalty = nugget.losses.geometry_penalties.StringNumberPenalty()
string_boundary_penalty = nugget.losses.geometry_penalties.StringBoundaryPenaltyCircle()
weighted_binarization_penalty = nugget.losses.geometry_penalties.WeightBinarizationPenalty()
weighted_angular_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
    resolution_type='angular',
    fisher_info_params=['position','energy', 'direction']
    )
weighted_energy_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
    resolution_type='energy',
    fisher_info_params=['position','energy', 'direction']
)
rov_penalty = nugget.losses.geometry_penalties.ROVPenalty( 
    rov_rec_width=230,  # ROV dimensions
    rov_height=159.9, 
    rov_tri_length=159.9
)
version = '_u_1'
use_rov = 'rov'
num_events = 50000
folder_name = f'res_test/opt_geoms_full_hex_{num_events}_r600_50{version}_{use_rov}/'
print(f"Saving optimized geometries to folder: {folder_name}")
# if folder does not exist, create it

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# check if there are already optimized geometries in the folder, and if so add a number to the folder name
# count = 0
# while os.path.exists(f'{folder_name}geom_{count}.pkl'):
#     count += 1

loss_params = {
    # 'signal_event_params': pickle.load(open(f'res_test/signal_events_{num_events}_r600_50{version}.pkl', 'rb'))[:],
    # 'signal_event_params': nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events_{num_events}_r600_50{version}.pt')[:],
    'num_events': 300,  # Number of events to sample per optimization step
    'boundary_range': 1200,  # Size of boundary region
    'skip_zero_response': True,
    'use_relative_energy': True,
    # 'precomputed_signal_yield_per_string': torch.load(f'res_test/light_yield_per_string_{num_events}_800main_full_hex_r600_50{version}.pt')[:],
    # 'precomputed_fisher_info_per_string_per_event': torch.load(f'res_test/fisher_info_per_string_per_event_{num_events}_800main_full_hex_r600_50{version}.pt')[:]
    }
loss_params.update({
    'eva_min_num_strings': 61,  # Minimum number of active strings
    'max_radius': 80,  # Maximum radius for string placement
    'num_angles': 360,  # Number of angles (divided into 360 degrees) to test for rov
    'rov_alt_mode': True,  # Whether to use alternative mode for rov penalty (see rov_penalty.py for details)
    'local_sharpness': 5,  # Sharpness parameter for local string repulsion
    'boundary_sharpness': 10,  # Sharpness parameter for boundary penalty
    'string_number_beta':5,
    'detach_other_probs':True,
    'rov_soft_inside':False,
    'rov_inside_sharpness': 5,
    'rov_angle_softmin_tau': -1,
    'constraints_list': [
                # 'energy_resolution_loss',
                # 'angular_resolution_loss',
                # 'signal_yield_loss', 
                'rov_penalty', 
                'string_boundary_penalty', 
                'local_string_repulsion_penalty', 
                'string_number_penalty', 
                # 'string_weights_penalty',
                'weighted_binarization_penalty'
                ],  # Constraints to enforce
    })
loss_weights_dict = {
    'angular_resolution_loss': 1e3,
    'energy_resolution_loss': 1e8,
    # 'fisher_loss': 0.005, 
    'signal_yield_loss': 0.01,        # High weight: maximize light collection
    # 'signal_llr_loss': 2.5,          # Moderate weight: good signal discrimination
    'string_boundary_penalty': 2,  # Very high: hard constraint
    'local_string_repulsion_penalty': 5,
    # 'string_repulsion_penalty': 0.000001,
    # 'string_weights_penalty': 0,     # Encourage sparse solutions
    'string_weights_penalty': 0.05,     # Encourage sparse solutions
    'string_number_penalty': 1,      # Limit detector complexity
    'weight_binarization_penalty': 0.1,
    'rov_penalty': 1
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
loss_func_dict = {
    'angular_resolution_loss': weighted_angular_resolution_loss,
    # 'energy_resolution_loss': weighted_energy_resolution_loss,
    # 'fisher_loss': fisher_info_loss_func, 
    # 'signal_yield_loss': signal_yield_loss_func,  # Maximize light collection
    # 'signal_llr_loss': signal_llr_loss_func,      # Maximize signal discrimination
    # 'local_string_repulsion_penalty': local_string_repulsion_penalty,
    # 'string_repulsion_penalty': string_repulsion_penalty,
    # 'string_weights_penalty': string_weights_penalty,  # Encourage using less strings
    'string_number_penalty': string_number_penalty,    # Limit number of strings
    # 'string_boundary_penalty': string_boundary_penalty,  # Keep strings in bounds
    # 'weight_binarization_penalty': weighted_binarization_penalty
}
if use_rov == 'rov':
    loss_func_dict['rov_penalty'] = rov_penalty

for i in range(6):
    # print(f"Running optimization iteration {i+1}/15")
    selection_limits = {
        'energy': (10**(i+2), 10**(i+3)),  # Example energy range
        }
    selection_inds = nugget.utils.data_tools.select_event_indices(
            nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events_{num_events}_r600_50{version}.pt')[:],
            limits=selection_limits  # Example limit
            )
    signal_events = nugget.utils.data_tools.select_events(nugget.utils.data_tools.load_signal_events_parquet(f'res_test/signal_events_{num_events}_r600_50{version}.pt'),limits=selection_limits)
    fisher_info = torch.load(f'res_test/fisher_info_per_string_per_event_{num_events}_800main_full_hex_r600_50{version}.pt')[selection_inds]
    loss_params.update({
        'signal_event_params': signal_events,
        'precomputed_fisher_info_per_string_per_event': fisher_info
    })
    geometry = nugget.geometries.EvanescentString.EvanescentString(
            # device="cuda:1",
            hex_type='hexagonal',
            domain_size=1600,  # Size of detector domain
            dim=3,  # 3D geometry
            n_strings=1027,  # Initial number of detector strings
            points_per_string=20,  # Number of PMTs/sensors per string
            custom_z_spacing=50,
            random_weights=True
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
    optimizer.init_geometry(
        opt_list=[('string_weights', 0.5)],  # Learning rate for string weights (without sigmoid applied)
    )  
        
    geom_dict = optimizer.optimize(
        loss_func_dict=loss_func_dict,          # Dictionary of loss functions to use
        loss_weights_dict=loss_weights_dict,    # Weights for combining multiple losses
        loss_params_dict=loss_params,           # Parameters for loss function computation
        n_iter=2000,                           # Maximum number of optimization iterations
        print_freq=100,                          # Print progress every N iterations
        sigmoid_loss_list=loss_sigmoid_list,         # Which losses to apply sigmoid to (for better optimization dynamics)
        save_best_geom_file = f'{folder_name}geom_e{i+2}_e{i+3}.pkl',  # File to save best geometry found
        save_last_geom = True, 
    )
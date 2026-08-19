import torch
import numpy as np
import nugget
import pickle
import os

device="cuda:1"
string_number_penalty = nugget.losses.geometry_penalties.StringNumberPenalty(device=device)
string_boundary_penalty = nugget.losses.geometry_penalties.StringBoundaryPenaltyCircle(device=device)
weighted_binarization_penalty = nugget.losses.geometry_penalties.WeightBinarizationPenalty(device=device)
local_string_repulsion_penalty = nugget.losses.geometry_penalties.LocalStringRepulsionPenalty(device=device)

rov_penalty = nugget.losses.geometry_penalties.ROVPenalty( 
    device=device,
    rov_rec_width=230,  # ROV dimensions
    rov_height=160, 
    rov_tri_length=160
)
light_yield_loss = nugget.losses.light_yield.LightYieldLoss(device=device)

num_strings = 61
folder_name = f'res_test/opt_geoms/opt_geoms_dyn_{num_strings}_rov'
print(f"Saving optimized geometries to folder: {folder_name}")
# if folder does not exist, create it
light_yield_surrogate = nugget.surrogates.Uniform.Uniform(device=device, dim=3)

signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
    device=device,
    event_type='signal', 
    domain_size=1600, 
    E_min=10**2, 
    E_max=10**8, 
    energy_dist='log_uniform', 
    find_exact_intersection=False,
    random_position_along_ray=True,
    uniform_zenith_sampling=True,
    )

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# check if there are already optimized geometries in the folder, and if so add a number to the folder name
# count = 0
# while os.path.exists(f'{folder_name}geom_{count}.pkl'):
#     count += 1

loss_params = {
    'num_events': 1,  
    'signal_surrogate_func': light_yield_surrogate,
    'signal_sampler': signal_sampler,
    'boundary_range': 1200,  # Size of boundary region
    'eva_min_num_strings': 70,  # Minimum number of active strings
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
    'rov_angle_chunk_size': 360,
    'rov_inside_use_softplus': True,
    # 'rov_away_weight': 0.00,
    # 'rov_away_num_neighbours':5,
    # 'rov_use_torch_compile': False,
    'local_repulsion_use_torch_compile': False,
   
    'constraints_list': [
              
                'rov_penalty', 
                'string_boundary_penalty', 
                'local_string_repulsion_penalty', 
                'string_number_penalty', 
                # 'string_weights_penalty',
                # 'weight_binarization_penalty',
                # 'diversity_penalty',
                ],  # Constraints to enforce
}
loss_weights_dict = {
    'signal_yield_loss': 0.01,        # High weight: maximize light collection
    'string_boundary_penalty': 2,  # Very high: hard constraint
    'local_string_repulsion_penalty': 1,
    'string_weights_penalty': 0.05,     # Encourage sparse solutions
    'string_number_penalty': 10,      # Limit detector complexity
    'weight_binarization_penalty': 0.1,
    'rov_penalty': 1,
 

}

loss_sigmoid_list = [

    'signal_yield_loss',        
    'string_boundary_penalty', 
    'local_string_repulsion_penalty',   
    'string_weights_penalty',     
    'string_number_penalty',     
    'weight_binarization_penalty',
    'rov_penalty'
]
loss_func_dict = {
    'signal_yield_loss': light_yield_loss,
    'rov_penalty': rov_penalty,
    'string_boundary_penalty': string_boundary_penalty,
    'local_string_repulsion_penalty': local_string_repulsion_penalty,
}

num_trials = 15
for i in range(num_trials):
   
    print(f"Running optimization iteration {i+1}/{num_trials}")
  
    geometry = nugget.geometries.DynamicString.DynamicString(
            device=device,
            random_xy=True,  # Randomize initial string positions
            domain_size=300,  # Size of detector domain
            dim=3,  # 3D geometry
            n_strings=num_strings,  # Initial number of detector strings
            points_per_string=1,  # Number of PMTs/sensors per strin
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
        # opt_list=[('string_weights', 0.5)],  # Learning rate for string weights (without sigmoid applied)
        opt_list=[('string_xy', 10)],  # Learning rate for string weights (without sigmoid applied)
    )  
    
    save_geom_folder = f'{folder_name}/geom_{i}'
    geom_dict = optimizer.optimize(
        clear_cuda_cache=True,
        loss_func_dict=loss_func_dict,          # Dictionary of loss functions to use
        loss_weights_dict=loss_weights_dict,    # Weights for combining multiple losses
        loss_params_dict=loss_params,           # Parameters for loss function computation
        n_iter=200,                           # Maximum number of optimization iterations
        print_freq=50,                          # Print progress every N iterations
        sigmoid_loss_list=loss_sigmoid_list,         # Which losses to apply sigmoid to (for better optimization dynamics)
        save_geom_folder=save_geom_folder,  # Folder to save intermediate geometries
        save_geom_freq=10,
        save_last_geom = True,
        revert_on_nan=True,                      # Revert geometry if loss becomes NaN
        max_nan_retries=5,  
    )
    
    pickle.dump(optimizer.uw_loss_dict, open(f'{folder_name}/geom_{i}_loss_dict.pkl', 'wb'))

import nugget  # Main NUGGET package for neutrino detector optimization
import pickle
import os

light_yield_surrogate = nugget.surrogates.Uniform.Uniform()
background_sampler = nugget.samplers.toy_sampler.ToySampler(x_bias=0, event_type='background', domain_size=2500)
signal_sampler = nugget.samplers.toy_sampler.ToySampler(x_bias=0, event_type='signal', domain_size=2500)

geometry = nugget.geometries.EvanescentString.EvanescentString(
# geometry = nugget.geometries.SpaceString.SpaceString(
#     hex_type='hexagonal',
    domain_size=2500,  # Size of detector domain
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use GPU if available
    dim=3,  # 3D geometry
    n_strings=1000,  # Initial number of detector strings
    points_per_string=1,  # Number of PMTs/sensors per string
    starting_weight=-6
    # random_weights=True,
    # custom_string_spacing=0.09  # Custom spacing between strings
    # starting_spacing=0.05
)

new_signal_events = signal_sampler.sample_events(1)  # Fresh signal events for optimization
new_background_events = background_sampler.sample_events(1)  # Fresh background events

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

# exp_scheduler = nugget.utils.schedulers.CosineScheduler
# Initialize the geometry with optimization parameters
# string_weights controls which strings are active (learnable parameter)
optimizer.init_geometry(
    # opt_list=[('string_spacing', 0.01)],  # Learning rate for string positions
    opt_list=[('string_weights', 0.1)],  # Learning rate for string weights (without sigmoid applied)
    # schedule_creator=exp_scheduler,
    # schedule_params={'string_weights': {'eta_min': 5e-2, 'num_iterations': 1000}},
    # geom_dict=pickle.load(open('./best_geom_1_event.pkl', 'rb'))
    )

signal_yield_loss_func = nugget.losses.light_yield.WeightedLightYieldLoss(
    device=geometry.device,
)

# Local String Repulsion: Prevents strings from getting too close to each other
local_string_repulsion_penalty = nugget.losses.geometry_penalties.LocalStringRepulsionPenalty(
    device=geometry.device
)

# String Boundary Penalty: Keeps strings within the detector domain
string_boundary_penalty = nugget.losses.geometry_penalties.StringBoundaryPenaltyCircle(
    device=geometry.device,
)



# String Number Penalty: Penalizes having too many active strings
string_number_penalty = nugget.losses.geometry_penalties.StringNumberPenalty(
    device=geometry.device
)

# Weight Binarization: Encourages string weights to be either 0 or 1
weighted_binarization_penalty = nugget.losses.geometry_penalties.WeightBinarizationPenalty(
    device=geometry.device
)

# ROV Penalty: Accounts for physical constraints from Remotely Operated Vehicle
rov_penalty = nugget.losses.geometry_penalties.ROVPenalty(
    device=geometry.device, 
    rov_rec_width=230,  # ROV dimensions
    rov_height=159, 
    rov_tri_length=159,
)

# === LOSS FUNCTION PARAMETERS ===
loss_params = {
    # 'llr_net': llr_net,
    'signal_event_params': new_signal_events,   
    # 'background_event_params': new_background_events,
    'signal_surrogate_func': light_yield_surrogate,  # Function to compute light yield
    # 'background_surrogate_func': light_yield_surrogate,
    'signal_sampler': signal_sampler,
    'background_sampler': background_sampler,
    'num_events': 1,  # Number of events to sample per optimization step
    'signal_noise_scale': 0.0,  # Noise level for signal events
    'background_noise_scale': 0.2,  # Noise level for background events
    'boundary_range': 2000,  # Size of boundary region
    'skip_zero_response': True,
    # 'fisher_info_llr_net': llr_net,
    'use_relative_energy': True,
    # 'sample_every': 50,
}

# Additional parameters for evanescent string optimization
loss_params.update({
    'eva_min_num_strings': 74,  # Minimum number of active strings
    'max_radius': 80,  # Maximum radius for string placement
    'num_angles': 72,  # Number of angles (divided into 360 degrees) to test for rov
    'local_sharpness': 8,  # Sharpness parameter for local string repulsion
    'boundary_sharpness': 5,  # Sharpness parameter for boundary penalty
    'string_number_beta':1,
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

# === ACTIVE LOSS FUNCTIONS ===
# Select which loss functions to use in optimization
loss_func_dict = {
    # 'angular_resolution_loss': weighted_angular_resolution_loss,
    # 'energy_resolution_loss': weighted_energy_resolution_loss,
    # 'fisher_loss': fisher_info_loss_func, 
    'signal_yield_loss': signal_yield_loss_func,  # Maximize light collection
    # 'signal_llr_loss': signal_llr_loss_func,      # Maximize signal discrimination
    'local_string_repulsion_penalty': local_string_repulsion_penalty,
    # 'string_repulsion_penalty': string_repulsion_penalty,
    # 'string_weights_penalty': string_weights_penalty,  # Encourage using less strings
    'string_number_penalty': string_number_penalty,    # Limit number of strings
    'rov_penalty': rov_penalty,
    'string_boundary_penalty': string_boundary_penalty,  # Keep strings in bounds
    'weight_binarization_penalty': weighted_binarization_penalty
}

# === LOSS WEIGHTS ===
# Relative importance of each loss component
loss_weights_dict = {
    'angular_resolution_loss': 1,
    # 'energy_resolution_loss': 0.5,
    # 'fisher_loss': 0.005, 
    'signal_yield_loss': 0.01,        # High weight: maximize light collection
    # 'signal_llr_loss': 2.5,          # Moderate weight: good signal discrimination
    'string_boundary_penalty': 1,  # Very high: hard constraint
    'local_string_repulsion_penalty': 8,
    # 'string_repulsion_penalty': 0.000001,
    # 'string_weights_penalty': 0,     # Encourage sparse solutions
    'string_weights_penalty': 0.05,     # Encourage sparse solutions
    'string_number_penalty': 1,      # Limit detector complexity
    'weight_binarization_penalty': 0.04,
    'rov_penalty': 1
}
# check if the folder exists anf if yes then make a new folder with an incremented number at the end
geom_folder_path = "/afs/ipp-garching.mpg.de/home/k/kristiantcho/ptmp/nugget/nugget/examples/rov_uniform_geom_zero_weights"
# if os.path.exists(geom_folder_path):
#     existing_folders = [f for f in os.listdir("/afs/ipp-garching.mpg.de/home/k/kristiantcho/ptmp/nugget/nugget/examples/") if f.startswith("rov_uniform_geom_iter_")]
#     existing_numbers = [int(f.split("_")[-1]) for f in existing_folders]
#     new_number = max(existing_numbers) + 1 if existing_numbers else 0
#     geom_folder_path = f"/afs/ipp-garching.mpg.de/home/k/kristiantcho/ptmp/nugget/nugget/examples/rov_uniform_geom_iter_{new_number}"
os.makedirs(geom_folder_path, exist_ok=True)
# geom_file_path = os.path.join(geom_folder_path, f"geom_0.pkl")
# if not os.path.exists(geom_file_path):
#     pickle.dump(optimizer.geom_dict, open(geom_file_path, "wb"))

# check if there is already saved geometries with geom_(number).pkl and take one number after the highest
# for _ in range(14):
    # existing_files = os.listdir(geom_folder_path)
    # existing_geom_numbers = []
    # if existing_files is not None:
    #     for file in existing_files:
    #         if "geom_" in file and file.endswith(".pkl"):
    #             geom_number = int(file.split("_")[1].split(".")[0])
    #             existing_geom_numbers.append(geom_number)
    # if existing_geom_numbers:
    #     new_geom_number = max(existing_geom_numbers) + 1
    # else:
    #     new_geom_number = 0
    # geom_file_path = os.path.join(geom_folder_path, f"geom_{new_geom_number}.pkl")
    # pickle.dump(optimizer.geom_dict, open(geom_file_path, "wb"))

geom_dict = optimizer.optimize(
    loss_func_dict=loss_func_dict,          # Dictionary of loss functions to use
    loss_weights_dict=loss_weights_dict,    # Weights for combining multiple losses
    loss_params_dict=loss_params,           # Parameters for loss function computation
    n_iter=1000,                           # Maximum number of optimization iterations
    print_freq=5,                          # Print progress every N iterations                          # Save animation frames every N iterations
    # save_best_geom_file = geom_file_path,  # File to save best geometry found
    # save_last_geom = True,
    save_geom_folder=geom_folder_path,  # Directory to save geometry checkpoints
    save_geom_freq=100, 
    # cf_loss_weights_dict=cf_loss_weights_dict,  # custom weights for conflict-free optimization
    # loss_dict=optimizer.loss_dict,         # resupply loss histories if continuing optimization (only for visualization)
    # uw_loss_dict=optimizer.uw_loss_dict,   
    # vis_loss_dict=optimizer.vis_loss_dict, 
    # vis_uw_loss_dict=optimizer.vis_uw_loss_dict  
)

# pickle.dump(optimizer.loss_dict, open(os.path.join(geom_folder_path, f'loss_dict_{new_geom_number}.pkl'), 'wb'))
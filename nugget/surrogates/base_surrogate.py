import torch
import numpy as np
    
class Surrogate:
    
    def __init__(self, device=None, dim=3, domain_size=2):
        """Base class for surrogate models."""
        self.device = device if device is not None else torch.device("cpu")
        self.dim = dim
        self.domain_size = domain_size
        self.half_domain = domain_size / 2
     
    def __call__(self, opt_point=None, event_params=None):
        """
        Call the surrogate model with keyword arguments.
        
        Parameters:
        -----------
        opt_point : torch.Tensor
            Optimization point (dim)
        event_params : dict
            Dictionary of event parameters
            
        Returns:
        --------
        torch.Tensor
            Output of the surrogate model
        """
        raise NotImplementedError("Surrogate model not implemented.")
    
    
    # def generate_surrogate_functions(self, 
    #                                num_funcs = 1, 
    #                                func_type = 'background',
    #                                parameter_ranges = None, 
    #                                position_ranges = None,
    #                                optimize_params = None,
    #                                parameter_grid_size = 10,
    #                                batch_num = 0,
    #                                keep_param_const = None,
    #                                randomly_sample_grid= False,
    #                                random_other_grid = False,
    #                             #    surrogate_model = None
    #                                ):
    #     """
    #     Generate surrogate functions for optimization.
        
    #     Parameters:
    #     -----------
    #     num_funcs : int
    #         Number of functions to generate
    #     func_type : str
    #         Type of functions to generate ('background' or 'signal')
    #     parameter_ranges : dict or None
    #         Dictionary of parameter ranges for surrogate functions
    #         Example: {'amp': [0.8, 1.0], 'phi': [0, 2*pi]}
    #     position_ranges : dict or None
    #         Dictionary of position ranges for surrogate functions
    #         Example: {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [-0.5, 0.5]}
    #     optimize_params : list or None
    #         List of parameters to optimize over (for signal functions)
    #     parameter_grid_size : int
    #         Size of the parameter grid for signal functions
    #     batch_num : int
    #         Batch number for signal functions
    #     keep_param_const : dict or None
    #         Dictionary of parameters to keep constant and their values
    #     randomly_sample_grid : bool
    #         Whether to randomly sample points from the optimization parameter grid
    #     random_other_grid : bool
    #         Whether to randomly sample points from the non-optimized parameters for each optimization parameter in the grid
    #     Returns:
    #     --------
    #     list
    #         List of generated surrogate functions
    #     """
    #     # Set default parameter ranges if not provided
    #     if parameter_ranges is None:
    #         parameter_ranges = {
    #             'amp': [0.8, 1.0],
    #             'entry_angle': [0, 2 * np.pi] if self.dim == 2 else None,
    #             'phi': [0, 2 * np.pi] if self.dim == 3 else None,
    #             'theta': [0, np.pi] if self.dim == 3 else None,
    #             # 'sigma_front': [0.1, 0.3],
    #             # 'sigma_back': [0.02, 0.1],
    #             # 'sigma_perp': [0.03, 0.15]
    #         }
    #         # Remove None entries
    #         parameter_ranges = {k: v for k, v in parameter_ranges.items() if v is not None}
        
    #     # Set default position ranges if not provided
    #     if position_ranges is None:
    #         position_ranges = {
    #             'x': [-self.half_domain, self.half_domain],
    #             'y': [-self.half_domain, self.half_domain]
    #         }
    #         if self.dim == 3:
    #             position_ranges['z'] = [-self.half_domain, self.half_domain]
                
    #     # Apply constant parameters if provided
    #     if keep_param_const is not None:
    #         for param, value in keep_param_const.items():
    #             if len(list(value) ) > 1:
    #                 value = value[batch_num % len(value)]
    #             # else:
    #             #     # Extract scalar value from single-element list
    #             #     value = value[0] if isinstance(value, (list, tuple)) else value
    #             if param in parameter_ranges:
    #                 parameter_ranges[param] = [value, value]
    #             if param in position_ranges:
    #                 position_ranges[param] = [value, value]
        
    #     # Different generation strategies for background vs. signal functions
    #     if func_type.lower() == 'background' or optimize_params is None or len(optimize_params) == 0:
    #         # For background functions or when not optimizing specific parameters,
    #         # generate random functions across the entire parameter space
    #         surrogate_funcs = []
    #         for _ in range(num_funcs):
    #             # Generate random parameters
    #             amp = np.random.uniform(*parameter_ranges['amp'])
                
    #             # Generate random position
    #             position = []
    #             for dim_name in ['x', 'y', 'z'][:self.dim]:
    #                 if dim_name in position_ranges:
    #                     position.append(np.random.uniform(*position_ranges[dim_name]))
    #                 else:
    #                     position.append(0.0)
    #             position = torch.tensor(position, device=self.device)
                
    #             # Other parameters depend on dimension
    #             if self.dim == 2:
    #                 entry_angle = np.random.uniform(*parameter_ranges['entry_angle'])
    #                 surrogate_func = self.__call__(
    #                     1, amp=amp, position=position, entry_angle=entry_angle,
    #                     # sigma_front=np.random.uniform(*parameter_ranges['sigma_front']),
    #                     # sigma_back=np.random.uniform(*parameter_ranges['sigma_back']),
    #                     # sigma_perp=np.random.uniform(*parameter_ranges['sigma_perp'])
    #                 )
    #             else:  # 3D
    #                 phi = np.random.uniform(*parameter_ranges['phi'])
    #                 theta = np.random.uniform(*parameter_ranges['theta'])
    #                 surrogate_func = self.__call__(
    #                     1, amp=amp, position=position, phi=phi, theta=theta,
    #                     # sigma_front=np.random.uniform(*parameter_ranges['sigma_front']),
    #                     # sigma_back=np.random.uniform(*parameter_ranges['sigma_back']),
    #                     # sigma_perp=np.random.uniform(*parameter_ranges['sigma_perp'])
    #                 )
                
    #             surrogate_funcs.append(surrogate_func)
                
    #         return surrogate_funcs
            
    #     else:  # Signal functions with parameter optimization
    #         # For signal functions when optimizing specific parameters,
    #         # create a grid over the parameter space
            
    #         # Store parameter values for visualization
    #         param_values = {}
    #         surrogate_funcs = []
            
    #         # First, generate a single random set of values for non-optimized parameters
    #         non_optimized_params = {}
            
    #         # Handle non-optimized parameters from parameter_ranges
    #         for param_name, bounds in parameter_ranges.items():
    #             if param_name not in optimize_params:
    #                 if not random_other_grid and not randomly_sample_grid:    
    #                     non_optimized_params[param_name] = np.random.uniform(*bounds)
    #                 else:
    #                     # Randomly sample values within the range
    #                     non_optimized_params[param_name] = np.random.uniform(*bounds, parameter_grid_size**len(optimize_params))
            
    #         # Handle position parameters - one random value for each non-optimized position
    #         for pos_param, bounds in position_ranges.items():
    #             if pos_param not in optimize_params:
    #                 non_optimized_params[pos_param] = np.random.uniform(*bounds)
            
    #         # Generate a grid of values for the optimized parameters
    #         if len(optimize_params) == 1:
    #             param_name = optimize_params[0]
    #             if param_name in parameter_ranges or param_name in position_ranges:
    #                 bounds = parameter_ranges.get(param_name, position_ranges.get(param_name))
                    
    #                 if randomly_sample_grid:
    #                     # Randomly sample values within the range
    #                     param_vals = np.random.uniform(bounds[0], bounds[1], parameter_grid_size)
    #                 else:
    #                     # Create evenly spaced grid
    #                     param_vals = np.linspace(bounds[0], bounds[1], parameter_grid_size)
    #                     # Take a subset for this batch
    #                     # indices = np.arange(batch_num * num_funcs, min((batch_num + 1) * num_funcs, parameter_grid_size))
    #                     # param_vals = param_vals[indices % len(param_vals)]
                    
    #                 param_values[param_name] = torch.tensor(param_vals, device=self.device)
                    
    #                 # Create functions for each parameter value
    #                 for i, val in enumerate(param_vals):
    #                     kwargs = {**non_optimized_params, param_name: val}
                        
    #                     # Extract position and other parameters
    #                     position = []
                        
    #                     for dim_name in ['x', 'y', 'z'][:self.dim]:
    #                         val = np.array([kwargs.get(dim_name, 0.0)]).squeeze()
    #                         if np.ndim(val) > 0:
    #                             position.append(kwargs.pop(dim_name, 0.0)[i])
    #                         else:
    #                             position.append(kwargs.pop(dim_name, 0.0))
    #                     val = np.array([kwargs.get('amp', 1.0)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_amp = kwargs.pop('amp', 1.0)[i]
    #                     else:
    #                         temp_amp = kwargs.pop('amp', 1.0)
    #                     val = np.array([kwargs.get('entry_angle', 0.0)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_entry_angle = kwargs.pop('entry_angle', 0.0)[i]
    #                     else:
    #                         temp_entry_angle = kwargs.pop('entry_angle', 0.0)
    #                     val = np.array([kwargs.get('phi', 0.0)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_phi = kwargs.pop('phi', 0.0)[i]
    #                     else:
    #                         temp_phi = kwargs.pop('phi', 0.0)
    #                     val = np.array([kwargs.get('theta', np.pi/2)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_theta = kwargs.pop('theta', np.pi/2)[i]
    #                     else:
    #                         temp_theta = kwargs.pop('theta', np.pi/2)
                            
                        
    #                     position = torch.tensor(position, device=self.device)
                        
    #                     # Create the surrogate function with the specific parameters
    #                     if self.dim == 2:
    #                         surrogate_func = self.__call__(
    #                             1, amp=temp_amp, 
    #                             position=position, 
    #                             entry_angle=temp_entry_angle,
    #                             # sigma_front=kwargs.pop('sigma_front', 0.2),
    #                             # sigma_back=kwargs.pop('sigma_back', 0.05),
    #                             # sigma_perp=kwargs.pop('sigma_perp', 0.1)
    #                         )
    #                     else:  # 3D
    #                         surrogate_func = self.__call__(
    #                             1, amp=temp_amp, 
    #                             position=position, 
    #                             phi=temp_phi, 
    #                             theta=temp_theta,
    #                             # sigma_front=kwargs.pop('sigma_front', 0.2),
    #                             # sigma_back=kwargs.pop('sigma_back', 0.05),
    #                             # sigma_perp=kwargs.pop('sigma_perp', 0.1)
    #                         )
                        
    #                     surrogate_funcs.append(surrogate_func)
                
    #         elif len(optimize_params) == 2:
    #             param1, param2 = optimize_params
    #             if (param1 in parameter_ranges or param1 in position_ranges) and \
    #                (param2 in parameter_ranges or param2 in position_ranges):
                    
    #                 bounds1 = parameter_ranges.get(param1, position_ranges.get(param1))
    #                 bounds2 = parameter_ranges.get(param2, position_ranges.get(param2))
                    
    #                 if randomly_sample_grid:
    #                     # Randomly sample parameter combinations
    #                     param1_vals = np.random.uniform(bounds1[0], bounds1[1], num_funcs)
    #                     param2_vals = np.random.uniform(bounds2[0], bounds2[1], num_funcs)
    #                 else:
    #                     # Create a 2D grid of parameter values
    #                     grid_size1 = parameter_grid_size
    #                     grid_size2 = parameter_grid_size 
                        
    #                     param1_vals_full = np.linspace(bounds1[0], bounds1[1], grid_size1)
    #                     param2_vals_full = np.linspace(bounds2[0], bounds2[1], grid_size2)
                        
    #                     P1, P2 = np.meshgrid(param1_vals_full, param2_vals_full)
    #                     param1_vals = P1.flatten()
    #                     param2_vals = P2.flatten()
                        
    #                     # Take a subset for this batch
    #                     # start_idx = batch_num * num_funcs
    #                     # end_idx = min((batch_num + 1) * num_funcs, len(param1_vals))
                        
    #                     # if start_idx < len(param1_vals):
    #                     #     param1_vals = param1_vals[start_idx:end_idx]
    #                     #     param2_vals = param2_vals[start_idx:end_idx]
    #                     # else:
    #                     #     # Wrap around if we've exhausted the grid
    #                     #     indices = np.arange(start_idx, end_idx) % len(param1_vals)
    #                     #     param1_vals = param1_vals[indices]
    #                     #     param2_vals = param2_vals[indices]
                    
    #                 param_values[param1] = torch.tensor(param1_vals, device=self.device)
    #                 param_values[param2] = torch.tensor(param2_vals, device=self.device)
                    
    #                 # Create functions for each parameter combination
    #                 for i in range(len(param1_vals)):
    #                     kwargs = {**non_optimized_params, param1: param1_vals[i], param2: param2_vals[i]}
                        
    #                     # Extract position and other parameters
    #                     position = []
    #                     for dim_name in ['x', 'y', 'z'][:self.dim]:
    #                         val = np.array([kwargs.get(dim_name, 0.0)]).squeeze()
    #                         if np.ndim(val) > 0:
    #                             position.append(kwargs.pop(dim_name, 0.0)[i])
    #                         else:
    #                             position.append(kwargs.pop(dim_name, 0.0))
    #                     position = torch.tensor(position, device=self.device)
                        
    #                     val = np.array([kwargs.get('amp', 1.0)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_amp = kwargs.pop('amp', 1.0)[i]
    #                     else:
    #                         temp_amp = kwargs.pop('amp', 1.0)
    #                     val = np.array([kwargs.get('entry_angle', 0.0)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_entry_angle = kwargs.pop('entry_angle', 0.0)[i]
    #                     else:
    #                         temp_entry_angle = kwargs.pop('entry_angle', 0.0)
    #                     val = np.array([kwargs.get('phi', 0.0)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_phi = kwargs.pop('phi', 0.0)[i]
    #                     else:
    #                         temp_phi = kwargs.pop('phi', 0.0)
    #                     val = np.array([kwargs.get('theta', np.pi/2)]).squeeze()
    #                     if np.ndim(val) > 0:
    #                         temp_theta = kwargs.pop('theta', np.pi/2)[i]
    #                     else:
    #                         temp_theta = kwargs.pop('theta', np.pi/2)
                        
    #                     # Create the surrogate function with the specific parameters
    #                     if self.dim == 2:
    #                         surrogate_func = self.__call__(
    #                             1, amp=temp_amp, 
    #                             position=position, 
    #                             entry_angle=temp_entry_angle,
    #                             # sigma_front=kwargs.pop('sigma_front', 0.2),
    #                             # sigma_back=kwargs.pop('sigma_back', 0.05),
    #                             # sigma_perp=kwargs.pop('sigma_perp', 0.1)
    #                         )
    #                     else:  # 3D
    #                         surrogate_func = self.__call__(
    #                             1, amp=temp_amp, 
    #                             position=position, 
    #                             phi=temp_phi, 
    #                             theta=temp_theta,
    #                             # sigma_front=kwargs.pop('sigma_front', 0.2),
    #                             # sigma_back=kwargs.pop('sigma_back', 0.05),
    #                             # sigma_perp=kwargs.pop('sigma_perp', 0.1)
    #                         )
                        
    #                     surrogate_funcs.append(surrogate_func)
            
    #         # Store parameter values for later visualization
    #         self.param_values = param_values
            
    #         return surrogate_funcs

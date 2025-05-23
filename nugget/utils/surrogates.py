import torch
import numpy as np

class Surrogate:
    
    def __init__(self, device=None, dim=3, domain_size=2):
        """Base class for surrogate models."""
        self.device = device if device is not None else torch.device("cpu")
        self.dim = dim
        self.domain_size = domain_size
        self.half_domain = domain_size / 2
     
    def __call__(self, num_funcs, **kwargs):
        """
        Call the surrogate model with keyword arguments.
        
        Parameters:
        -----------
        kwargs : dict
            Keyword arguments for the surrogate model
            
        Returns:
        --------
        torch.Tensor
            Output of the surrogate model
        """
        raise NotImplementedError("Surrogate model not implemented.")
     
     
    def sample_power_law(self, E_min=0.8, E_max=1, gamma=2.7):
        """
        Sample from a power law distribution.
        
        Parameters:
        -----------
        E_min : float
            Minimum value
        E_max : float
            Maximum value
        gamma : float
            Power law index
            
        Returns:
        --------
        torch.Tensor
            Sampled value
        """
        r = torch.rand(1, device=self.device)
        exponent = 1 - gamma
        return ((E_max**exponent - E_min**exponent) * r + E_min**exponent) ** (1 / exponent)
    

    def generate_surrogate_functions(self, 
                                   num_funcs = 1, 
                                   func_type = 'background',
                                   parameter_ranges = None, 
                                   position_ranges = None,
                                   optimize_params = None,
                                   parameter_grid_size = 10,
                                   batch_num = 0,
                                   keep_param_const = None,
                                   randomly_sample_grid= False,
                                   random_other_grid = False,
                                #    surrogate_model = None
                                   ):
        """
        Generate surrogate functions for optimization.
        
        Parameters:
        -----------
        num_funcs : int
            Number of functions to generate
        func_type : str
            Type of functions to generate ('background' or 'signal')
        parameter_ranges : dict or None
            Dictionary of parameter ranges for surrogate functions
            Example: {'amp': [0.8, 1.0], 'phi': [0, 2*pi]}
        position_ranges : dict or None
            Dictionary of position ranges for surrogate functions
            Example: {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [-0.5, 0.5]}
        optimize_params : list or None
            List of parameters to optimize over (for signal functions)
        parameter_grid_size : int
            Size of the parameter grid for signal functions
        batch_num : int
            Batch number for signal functions
        keep_param_const : dict or None
            Dictionary of parameters to keep constant and their values
        randomly_sample_grid : bool
            Whether to randomly sample points from the optimization parameter grid
        random_other_grid : bool
            Whether to randomly sample points from the non-optimized parameters for each optimization parameter in the grid
        Returns:
        --------
        list
            List of generated surrogate functions
        """
        # Set default parameter ranges if not provided
        if parameter_ranges is None:
            parameter_ranges = {
                'amp': [0.8, 1.0],
                'entry_angle': [0, 2 * np.pi] if self.dim == 2 else None,
                'phi': [0, 2 * np.pi] if self.dim == 3 else None,
                'theta': [0, np.pi] if self.dim == 3 else None,
                # 'sigma_front': [0.1, 0.3],
                # 'sigma_back': [0.02, 0.1],
                # 'sigma_perp': [0.03, 0.15]
            }
            # Remove None entries
            parameter_ranges = {k: v for k, v in parameter_ranges.items() if v is not None}
        
        # Set default position ranges if not provided
        if position_ranges is None:
            position_ranges = {
                'x': [-self.half_domain, self.half_domain],
                'y': [-self.half_domain, self.half_domain]
            }
            if self.dim == 3:
                position_ranges['z'] = [-self.half_domain, self.half_domain]
                
        # Apply constant parameters if provided
        if keep_param_const is not None:
            for param, value in keep_param_const.items():
                if len(list(value) ) > 1:
                    value = value[batch_num % len(value)]
                if param in parameter_ranges:
                    parameter_ranges[param] = [value, value]
                if param in position_ranges:
                    position_ranges[param] = [value, value]
        
        # Different generation strategies for background vs. signal functions
        if func_type.lower() == 'background' or optimize_params is None or len(optimize_params) == 0:
            # For background functions or when not optimizing specific parameters,
            # generate random functions across the entire parameter space
            surrogate_funcs = []
            for _ in range(num_funcs):
                # Generate random parameters
                amp = np.random.uniform(*parameter_ranges['amp'])
                
                # Generate random position
                position = []
                for dim_name in ['x', 'y', 'z'][:self.dim]:
                    if dim_name in position_ranges:
                        position.append(np.random.uniform(*position_ranges[dim_name]))
                    else:
                        position.append(0.0)
                position = torch.tensor(position, device=self.device)
                
                # Other parameters depend on dimension
                if self.dim == 2:
                    entry_angle = np.random.uniform(*parameter_ranges['entry_angle'])
                    surrogate_func = self.__call__(
                        1, amp=amp, position=position, entry_angle=entry_angle,
                        # sigma_front=np.random.uniform(*parameter_ranges['sigma_front']),
                        # sigma_back=np.random.uniform(*parameter_ranges['sigma_back']),
                        # sigma_perp=np.random.uniform(*parameter_ranges['sigma_perp'])
                    )
                else:  # 3D
                    phi = np.random.uniform(*parameter_ranges['phi'])
                    theta = np.random.uniform(*parameter_ranges['theta'])
                    surrogate_func = self.__call__(
                        1, amp=amp, position=position, phi=phi, theta=theta,
                        # sigma_front=np.random.uniform(*parameter_ranges['sigma_front']),
                        # sigma_back=np.random.uniform(*parameter_ranges['sigma_back']),
                        # sigma_perp=np.random.uniform(*parameter_ranges['sigma_perp'])
                    )
                
                surrogate_funcs.append(surrogate_func)
                
            return surrogate_funcs
            
        else:  # Signal functions with parameter optimization
            # For signal functions when optimizing specific parameters,
            # create a grid over the parameter space
            
            # Store parameter values for visualization
            param_values = {}
            surrogate_funcs = []
            
            # First, generate a single random set of values for non-optimized parameters
            non_optimized_params = {}
            
            # Handle non-optimized parameters from parameter_ranges
            for param_name, bounds in parameter_ranges.items():
                if param_name not in optimize_params:
                    if not random_other_grid and not randomly_sample_grid:    
                        non_optimized_params[param_name] = np.random.uniform(*bounds)
                    else:
                        # Randomly sample values within the range
                        non_optimized_params[param_name] = np.random.uniform(*bounds, parameter_grid_size**len(optimize_params))
            
            # Handle position parameters - one random value for each non-optimized position
            for pos_param, bounds in position_ranges.items():
                if pos_param not in optimize_params:
                    non_optimized_params[pos_param] = np.random.uniform(*bounds)
            
            # Generate a grid of values for the optimized parameters
            if len(optimize_params) == 1:
                param_name = optimize_params[0]
                if param_name in parameter_ranges or param_name in position_ranges:
                    bounds = parameter_ranges.get(param_name, position_ranges.get(param_name))
                    
                    if randomly_sample_grid:
                        # Randomly sample values within the range
                        param_vals = np.random.uniform(bounds[0], bounds[1], parameter_grid_size)
                    else:
                        # Create evenly spaced grid
                        param_vals = np.linspace(bounds[0], bounds[1], parameter_grid_size)
                        # Take a subset for this batch
                        # indices = np.arange(batch_num * num_funcs, min((batch_num + 1) * num_funcs, parameter_grid_size))
                        # param_vals = param_vals[indices % len(param_vals)]
                    
                    param_values[param_name] = torch.tensor(param_vals, device=self.device)
                    
                    # Create functions for each parameter value
                    for i, val in enumerate(param_vals):
                        kwargs = {**non_optimized_params, param_name: val}
                        
                        # Extract position and other parameters
                        position = []
                        
                        for dim_name in ['x', 'y', 'z'][:self.dim]:
                            val = np.array([kwargs.get(dim_name, 0.0)]).squeeze()
                            if np.ndim(val) > 0:
                                position.append(kwargs.pop(dim_name, 0.0)[i])
                            else:
                                position.append(kwargs.pop(dim_name, 0.0))
                        val = np.array([kwargs.get('amp', 1.0)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_amp = kwargs.pop('amp', 1.0)[i]
                        else:
                            temp_amp = kwargs.pop('amp', 1.0)
                        val = np.array([kwargs.get('entry_angle', 0.0)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_entry_angle = kwargs.pop('entry_angle', 0.0)[i]
                        else:
                            temp_entry_angle = kwargs.pop('entry_angle', 0.0)
                        val = np.array([kwargs.get('phi', 0.0)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_phi = kwargs.pop('phi', 0.0)[i]
                        else:
                            temp_phi = kwargs.pop('phi', 0.0)
                        val = np.array([kwargs.get('theta', np.pi/2)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_theta = kwargs.pop('theta', np.pi/2)[i]
                        else:
                            temp_theta = kwargs.pop('theta', np.pi/2)
                            
                        
                        position = torch.tensor(position, device=self.device)
                        
                        # Create the surrogate function with the specific parameters
                        if self.dim == 2:
                            surrogate_func = self.__call__(
                                1, amp=temp_amp, 
                                position=position, 
                                entry_angle=temp_entry_angle,
                                # sigma_front=kwargs.pop('sigma_front', 0.2),
                                # sigma_back=kwargs.pop('sigma_back', 0.05),
                                # sigma_perp=kwargs.pop('sigma_perp', 0.1)
                            )
                        else:  # 3D
                            surrogate_func = self.__call__(
                                1, amp=temp_amp, 
                                position=position, 
                                phi=temp_phi, 
                                theta=temp_theta,
                                # sigma_front=kwargs.pop('sigma_front', 0.2),
                                # sigma_back=kwargs.pop('sigma_back', 0.05),
                                # sigma_perp=kwargs.pop('sigma_perp', 0.1)
                            )
                        
                        surrogate_funcs.append(surrogate_func)
                
            elif len(optimize_params) == 2:
                param1, param2 = optimize_params
                if (param1 in parameter_ranges or param1 in position_ranges) and \
                   (param2 in parameter_ranges or param2 in position_ranges):
                    
                    bounds1 = parameter_ranges.get(param1, position_ranges.get(param1))
                    bounds2 = parameter_ranges.get(param2, position_ranges.get(param2))
                    
                    if randomly_sample_grid:
                        # Randomly sample parameter combinations
                        param1_vals = np.random.uniform(bounds1[0], bounds1[1], num_funcs)
                        param2_vals = np.random.uniform(bounds2[0], bounds2[1], num_funcs)
                    else:
                        # Create a 2D grid of parameter values
                        grid_size1 = parameter_grid_size
                        grid_size2 = parameter_grid_size 
                        
                        param1_vals_full = np.linspace(bounds1[0], bounds1[1], grid_size1)
                        param2_vals_full = np.linspace(bounds2[0], bounds2[1], grid_size2)
                        
                        P1, P2 = np.meshgrid(param1_vals_full, param2_vals_full)
                        param1_vals = P1.flatten()
                        param2_vals = P2.flatten()
                        
                        # Take a subset for this batch
                        # start_idx = batch_num * num_funcs
                        # end_idx = min((batch_num + 1) * num_funcs, len(param1_vals))
                        
                        # if start_idx < len(param1_vals):
                        #     param1_vals = param1_vals[start_idx:end_idx]
                        #     param2_vals = param2_vals[start_idx:end_idx]
                        # else:
                        #     # Wrap around if we've exhausted the grid
                        #     indices = np.arange(start_idx, end_idx) % len(param1_vals)
                        #     param1_vals = param1_vals[indices]
                        #     param2_vals = param2_vals[indices]
                    
                    param_values[param1] = torch.tensor(param1_vals, device=self.device)
                    param_values[param2] = torch.tensor(param2_vals, device=self.device)
                    
                    # Create functions for each parameter combination
                    for i in range(len(param1_vals)):
                        kwargs = {**non_optimized_params, param1: param1_vals[i], param2: param2_vals[i]}
                        
                        # Extract position and other parameters
                        position = []
                        for dim_name in ['x', 'y', 'z'][:self.dim]:
                            val = np.array([kwargs.get(dim_name, 0.0)]).squeeze()
                            if np.ndim(val) > 0:
                                position.append(kwargs.pop(dim_name, 0.0)[i])
                            else:
                                position.append(kwargs.pop(dim_name, 0.0))
                        position = torch.tensor(position, device=self.device)
                        
                        val = np.array([kwargs.get('amp', 1.0)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_amp = kwargs.pop('amp', 1.0)[i]
                        else:
                            temp_amp = kwargs.pop('amp', 1.0)
                        val = np.array([kwargs.get('entry_angle', 0.0)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_entry_angle = kwargs.pop('entry_angle', 0.0)[i]
                        else:
                            temp_entry_angle = kwargs.pop('entry_angle', 0.0)
                        val = np.array([kwargs.get('phi', 0.0)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_phi = kwargs.pop('phi', 0.0)[i]
                        else:
                            temp_phi = kwargs.pop('phi', 0.0)
                        val = np.array([kwargs.get('theta', np.pi/2)]).squeeze()
                        if np.ndim(val) > 0:
                            temp_theta = kwargs.pop('theta', np.pi/2)[i]
                        else:
                            temp_theta = kwargs.pop('theta', np.pi/2)
                        
                        # Create the surrogate function with the specific parameters
                        if self.dim == 2:
                            surrogate_func = self.__call__(
                                1, amp=temp_amp, 
                                position=position, 
                                entry_angle=temp_entry_angle,
                                # sigma_front=kwargs.pop('sigma_front', 0.2),
                                # sigma_back=kwargs.pop('sigma_back', 0.05),
                                # sigma_perp=kwargs.pop('sigma_perp', 0.1)
                            )
                        else:  # 3D
                            surrogate_func = self.__call__(
                                1, amp=temp_amp, 
                                position=position, 
                                phi=temp_phi, 
                                theta=temp_theta,
                                # sigma_front=kwargs.pop('sigma_front', 0.2),
                                # sigma_back=kwargs.pop('sigma_back', 0.05),
                                # sigma_perp=kwargs.pop('sigma_perp', 0.1)
                            )
                        
                        surrogate_funcs.append(surrogate_func)
            
            # Store parameter values for later visualization
            self.param_values = param_values
            
            return surrogate_funcs

class SkewedGaussian(Surrogate):
    
    def __init__(self, device=None, dim=3, domain_size=2):
        """
        Initialize the Skewed Gaussian surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (2D or 3D)
        domain_size : int
            length of the domain 
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
    
    
    def skewed_anisotropic_gaussian(self, points, center, motion_dir, sigma_front, sigma_back, sigma_perp, amp):
        """
        Compute skewed anisotropic Gaussian function values.
        
        Parameters:
        -----------
        points : torch.Tensor
            Points to evaluate at (N, dim)
        center : torch.Tensor
            Center of the Gaussian (dim,)
        motion_dir : torch.Tensor
            Direction of motion (dim,)
        sigma_front : float
            Standard deviation in the forward direction
        sigma_back : float
            Standard deviation in the backward direction
        sigma_perp : float
            Standard deviation in the perpendicular direction
        amp : float
            Amplitude
            
        Returns:
        --------
        torch.Tensor
            Function values
        """
        # Ensure inputs have correct dimensions
        if points.dim() == 1:
            points = points.unsqueeze(0)  # Add batch dimension
            
        diff = points - center  # (N, dim)

        # Normalize motion_dir to ensure it's a unit vector
        motion_dir = motion_dir / torch.norm(motion_dir)
        
        # Projection onto motion direction
        d_motion = torch.sum(diff * motion_dir.unsqueeze(0), dim=1)  # (N,)
        
        if self.dim == 2:
            # Create perpendicular direction vector in 2D
            perp_dir = torch.tensor([-motion_dir[1], motion_dir[0]], device=self.device)
            
            # Project onto perpendicular direction
            d_perp = torch.sum(diff * perp_dir.unsqueeze(0), dim=1)  # (N,)
            
            # Piecewise std in motion direction
            sigmas_motion = torch.where(
                d_motion >= 0,
                torch.tensor(sigma_front, device=self.device),
                torch.tensor(sigma_back, device=self.device)
            )
            
            # Gaussian in motion direction
            gauss_motion = torch.exp(-0.5 * (d_motion / sigmas_motion)**2)
            
            # Gaussian in perpendicular direction
            gauss_perp = torch.exp(-0.5 * (d_perp / sigma_perp)**2)
            
            return amp * gauss_motion * gauss_perp
        
        elif self.dim == 3:
            # For 3D, create a perpendicular plane
            # First perpendicular vector (using cross product with a reference vector)
            ref_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=motion_dir.dtype)
            if torch.allclose(motion_dir, ref_vec, atol=1e-6) or torch.allclose(motion_dir, -ref_vec, atol=1e-6):
                ref_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=motion_dir.dtype)
                
            perp_dir1 = torch.cross(motion_dir, ref_vec)
            perp_dir1 = perp_dir1 / torch.norm(perp_dir1)
            
            # Second perpendicular vector
            perp_dir2 = torch.cross(motion_dir, perp_dir1)
            
            # Project onto perpendicular directions
            d_perp1 = torch.sum(diff * perp_dir1.unsqueeze(0), dim=1)  # (N,)
            d_perp2 = torch.sum(diff * perp_dir2.unsqueeze(0), dim=1)  # (N,)
            
            # Piecewise std in motion direction
            sigmas_motion = torch.where(
                d_motion >= 0,
                torch.tensor(sigma_front, device=self.device),
                torch.tensor(sigma_back, device=self.device)
            )
            
            # Gaussians in each direction
            gauss_motion = torch.exp(-0.5 * (d_motion / sigmas_motion)**2)
            gauss_perp1 = torch.exp(-0.5 * (d_perp1 / sigma_perp)**2)
            gauss_perp2 = torch.exp(-0.5 * (d_perp2 / sigma_perp)**2)
            
            return amp * gauss_motion * gauss_perp1 * gauss_perp2
        
    
    def __call__(self, nu_num, test_points=None, amp=None, position=None, entry_angle=None, 
                   phi=None, theta=None, sigma_front=None, sigma_back=None, sigma_perp=None):
        """
        Generate a set of test functions.
        
        Parameters:
        -----------
        nu_num : int
            Number of test functions to generate
        test_points : torch.Tensor (optional)
            Points to evaluate the test functions at
        amp : float or torch.Tensor or None
            Amplitude (energy) for the functions. If None, random values are generated.
        position : torch.Tensor or None
            Center position for the functions. If None, random positions are generated.
            Should be a tensor of shape (dim,) for a single position or (nu_num, dim) for multiple positions.
        entry_angle : float or torch.Tensor or None
            Entry angle in radians for 2D functions. If None, random angles are generated.
        phi : float or torch.Tensor or None
            Azimuthal angle in radians for 3D functions. If None, random angles are generated.
        theta : float or torch.Tensor or None
            Polar angle in radians for 3D functions. If None, random angles are generated.
        sigma_front : float or torch.Tensor or None
            Standard deviation in the forward direction. If None, calculated based on amplitude.
        sigma_back : float or torch.Tensor or None
            Standard deviation in the backward direction. If None, calculated based on amplitude.
        sigma_perp : float or torch.Tensor or None
            Standard deviation in the perpendicular direction. If None, calculated based on amplitude.
            
        Returns:
        --------
        tuple of lists
            (true_functions, true_function_values (if test_points is provided))
        """
        true_functions = []
        test_funcs = []
        
        for k in range(nu_num):
            # Generate or use provided amplitude (energy)
            if amp is None:
                # Random amplitude from power law distribution
                amps = self.sample_power_law()
            else:
                if isinstance(amp, torch.Tensor) and amp.numel() > 1 and nu_num > 1:
                    amps = amp[k]
                else:
                    amps = torch.tensor(amp, device=self.device) if not isinstance(amp, torch.Tensor) else amp
            
            # Generate or use provided center position
            if position is None:
                # Random center point
                center = torch.rand(self.dim, device=self.device) * self.domain_size - self.domain_size / 2
            else:
                if isinstance(position, torch.Tensor) and position.dim() > 1 and nu_num > 1:
                    center = position[k]
                else:
                    center = position.clone() if isinstance(position, torch.Tensor) else torch.tensor(position, device=self.device)
            
            # Generate or use provided direction vector
            if self.dim == 2:
                if entry_angle is None:
                    # Random angle
                    angle = torch.rand(1, device=self.device) * 2 * np.pi
                else:
                    if isinstance(entry_angle, torch.Tensor) and entry_angle.numel() > 1 and nu_num > 1:
                        angle = entry_angle[k]
                    else:
                        angle = torch.tensor(entry_angle, device=self.device) if not isinstance(entry_angle, torch.Tensor) else entry_angle
                
                motion_dir = torch.tensor([torch.cos(angle), torch.sin(angle)], device=self.device)
            else:  # 3D
                if phi is None or theta is None:
                    # Random direction
                    phi_val = torch.rand(1, device=self.device) * 2 * np.pi
                    theta_val = torch.rand(1, device=self.device) * np.pi
                else:
                    # Use provided angles
                    if isinstance(phi, torch.Tensor) and phi.numel() > 1 and nu_num > 1:
                        phi_val = phi[k]
                    else:
                        phi_val = torch.tensor(phi, device=self.device) if not isinstance(phi, torch.Tensor) else phi
                    
                    if isinstance(theta, torch.Tensor) and theta.numel() > 1 and nu_num > 1:
                        theta_val = theta[k]
                    else:
                        theta_val = torch.tensor(theta, device=self.device) if not isinstance(theta, torch.Tensor) else theta
                
                motion_dir = torch.tensor([
                    torch.sin(theta_val) * torch.cos(phi_val),
                    torch.sin(theta_val) * torch.sin(phi_val),
                    torch.cos(theta_val)
                ], device=self.device)
                
            motion_dir = motion_dir / torch.norm(motion_dir)
            
            # Calculate sigmas based on amplitude if not provided
            if sigma_front is None:
                sigma_front_val = 1/amps
            else:
                if isinstance(sigma_front, torch.Tensor) and sigma_front.numel() > 1 and nu_num > 1:
                    sigma_front_val = sigma_front[k]
                else:
                    sigma_front_val = torch.tensor(sigma_front, device=self.device) if not isinstance(sigma_front, torch.Tensor) else sigma_front
            
            if sigma_back is None:
                sigma_back_val = 1/amps / 5
            else:
                if isinstance(sigma_back, torch.Tensor) and sigma_back.numel() > 1 and nu_num > 1:
                    sigma_back_val = sigma_back[k]
                else:
                    sigma_back_val = torch.tensor(sigma_back, device=self.device) if not isinstance(sigma_back, torch.Tensor) else sigma_back
            
            if sigma_perp is None:
                sigma_perp_val = 1/amps / 3
            else:
                if isinstance(sigma_perp, torch.Tensor) and sigma_perp.numel() > 1 and nu_num > 1:
                    sigma_perp_val = sigma_perp[k]
                else:
                    sigma_perp_val = torch.tensor(sigma_perp, device=self.device) if not isinstance(sigma_perp, torch.Tensor) else sigma_perp
            
            # Create closure for this function
            def make_true_function(
                amps=amps.clone(),
                center=center.clone(),
                sigma_front=sigma_front_val.clone(),
                sigma_back=sigma_back_val.clone(),
                sigma_perp=sigma_perp_val.clone(),
                motion_dir=motion_dir.clone(),
                num=1
            ):
                def true_function_closure(xy):
                    return self.skewed_anisotropic_gaussian(
                        xy, center, motion_dir, sigma_front, sigma_back, sigma_perp, amps
                    )
                return true_function_closure
            
            true_function = make_true_function()
            if test_points is not None:
                f_test = true_function(test_points)
            
            if nu_num > 1:
                true_functions.append(true_function)
                if test_points is not None:
                    test_funcs.append(f_test)
            else:
                if test_points is not None:
                    return true_function, f_test
                else:
                    return true_function
        if test_points is not None:       
            return true_functions, test_funcs
        else:
            return true_functions
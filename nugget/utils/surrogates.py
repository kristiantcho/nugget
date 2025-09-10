import torch
import numpy as np
import math


class FourierFeatures(torch.nn.Module):
    """
    Multiscale Fourier feature mapping for neural networks.
    
    This module applies a Fourier feature transformation to input coordinates
    using multiple frequency scales. The transformation maps input coordinates
    to a higher-dimensional space using sine and cosine functions.
    """
    
    def __init__(self, input_dim, num_frequencies=64, frequency_scale=1.0, learnable=False):
        """
        Initialize Fourier feature mapping.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input coordinates
        num_frequencies : int
            Number of frequency components (output will be 2 * num_frequencies)
        frequency_scale : float
            Scale factor for frequency sampling
        learnable : bool
            If True, frequencies are learnable parameters. If False, they are fixed.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.frequency_scale = frequency_scale
        self.output_dim = 2 * num_frequencies
        
        # Generate frequency matrix
        # Each row corresponds to one frequency component for all input dimensions
        frequencies = torch.randn(num_frequencies, input_dim) * frequency_scale
        
        if learnable:
            self.frequencies = torch.nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)
    
    def forward(self, x):
        """
        Apply Fourier feature mapping.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input coordinates of shape (..., input_dim)
            
        Returns:
        --------
        torch.Tensor
            Fourier features of shape (..., 2 * num_frequencies)
        """
        # x shape: (..., input_dim)
        # frequencies shape: (num_frequencies, input_dim)
        
        # Compute 2π * frequencies * x for all frequency components
        # This does a dot product between each input vector and each frequency vector
        # Shape: (..., num_frequencies)
        projected = torch.matmul(x, self.frequencies.T) * 2 * math.pi
        
        # Apply sine and cosine to get Fourier features
        # Shape: (..., num_frequencies)
        sin_features = torch.sin(projected)
        cos_features = torch.cos(projected)
        
        # Concatenate sine and cosine features
        # Shape: (..., 2 * num_frequencies)
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        
        return fourier_features


class ResidualBlock(torch.nn.Module):
    """A residual block with optional dimension matching."""
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main path
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Skip connection - add projection if dimensions don't match
        if input_dim != output_dim:
            self.skip_projection = torch.nn.Linear(input_dim, output_dim)
        else:
            self.skip_projection = None
    
    def forward(self, x):
        # Main path
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Skip connection
        if self.skip_projection is not None:
            skip = self.skip_projection(x)
        else:
            skip = x
        
        # Add residual connection
        return out + skip
    
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
     
     
    def sample_power_law(self, E_min=0.8, E_max=1, gamma=2.7, n_samples=1):
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
        
        r = torch.rand(n_samples, device=self.device)
        exponent = 1 - gamma
        return ((E_max**exponent - E_min**exponent) * r + E_min**exponent) ** (1 / exponent)
    

    def sample_background_zenith(self, a=1.5, n_samples=1):
        """
        Sample zenith angles for atmospheric background neutrinos.
        Biases toward the horizon using a simple analytical form.
        Returns angles in radians.
        
        The larger `a` is, the stronger the bias toward horizontal.
        """
        def pdf(cos_theta):
            return 1 + a * (1 - np.abs(cos_theta))

        cos_theta_vals = []
        n_accepted = 0
        n_trials = 0
        while n_accepted < n_samples:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(0, 1 + a)  # max of PDF = 1 + a
            if y <= pdf(x):
                cos_theta_vals.append(x)
                n_accepted += 1
            n_trials += 1
            if n_trials > 20 * n_samples:
                raise RuntimeError("Rejection sampling failed to converge")

        cos_theta = np.array(cos_theta_vals)
        theta = np.arccos(cos_theta)
        return torch.tensor(theta, device=self.device)
    
    
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
                # else:
                #     # Extract scalar value from single-element list
                #     value = value[0] if isinstance(value, (list, tuple)) else value
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
    
    def __init__(self, device=None, dim=3, domain_size=2, background=False):
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
        self.background = background    
    
    
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
        # if points.dim() == 1:
        #     points = points.unsqueeze(0)  # Add batch dimension
            
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
                sigma_front * torch.ones_like(d_motion, device=self.device),
                sigma_back * torch.ones_like(d_motion, device=self.device)
            )
            
            # Gaussians in each direction
            gauss_motion = torch.exp(-0.5 * (d_motion / sigmas_motion)**2)
            gauss_perp1 = torch.exp(-0.5 * (d_perp1 / sigma_perp)**2)
            gauss_perp2 = torch.exp(-0.5 * (d_perp2 / sigma_perp)**2)
            
            return amp * gauss_motion * gauss_perp1 * gauss_perp2
        
    
    def __call__(self, nu_num=1, test_points=None, amp=None, position=None, entry_angle=None, 
                   phi=None, theta=None, sigma_front=None, sigma_back=None, sigma_perp=None, sigma_factor=1):
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
        
    
        # Generate or use provided amplitude (energy)
        if amp is None:
            # Random amplitude from power law distribution
            amps = self.sample_power_law(gamma=3.7 if self.background else 2.7)
        else:
            # Preserve gradients if amp is already a tensor
            if isinstance(amp, torch.Tensor):
                amps = amp
            else:
                amps = torch.tensor(amp, device=self.device)
        
        # Generate or use provided center position
        if position is None:
            # Random center point
            center = torch.rand(self.dim, device=self.device) * self.domain_size - self.domain_size / 2
        else:
            # Preserve gradients if position is already a tensor with gradients
            if isinstance(position, torch.Tensor):
                center = position
            else:
                center = torch.tensor(position, device=self.device)
        
        # Generate or use provided direction vector
        if self.dim == 2:
            if entry_angle is None:
                # Random angle
                angle = torch.rand(1, device=self.device) * 2 * np.pi
            else:
                # Preserve gradients if entry_angle is already a tensor with gradients
                if isinstance(entry_angle, torch.Tensor):
                    angle = entry_angle
                else:
                    angle = torch.tensor(entry_angle, device=self.device)
            
            motion_dir = torch.stack([torch.cos(angle), torch.sin(angle)])
        else:  # 3D
            if phi is None or theta is None:
                # Random direction
                phi_val = torch.rand(1, device=self.device) * 2 * np.pi
                theta_val = torch.rand(1, device=self.device) * np.pi
                if self.background:
                    theta_val = self.sample_background_zenith()
            else:
                # Use provided angles - preserve gradients
                if isinstance(phi, torch.Tensor):
                    phi_val = phi
                else:
                    phi_val = torch.tensor(phi, device=self.device)
                
                if isinstance(theta, torch.Tensor):
                    theta_val = theta
                else:
                    theta_val = torch.tensor(theta, device=self.device)
            
            motion_dir = torch.stack([
                torch.sin(theta_val) * torch.cos(phi_val),
                torch.sin(theta_val) * torch.sin(phi_val),
                torch.cos(theta_val)
            ])
            
        motion_dir = motion_dir / torch.norm(motion_dir)
        
        # Calculate sigmas based on amplitude if not provided
        if sigma_front is None:
            sigma_front_val = sigma_factor / amps
        else:
            # Preserve gradients if sigma_front is already a tensor with gradients
            if isinstance(sigma_front, torch.Tensor):
                sigma_front_val = sigma_front
            else:
                sigma_front_val = torch.tensor(sigma_front, device=self.device)
        
        if sigma_back is None:
            sigma_back_val = sigma_factor / amps / 5
        else:
            # Preserve gradients if sigma_back is already a tensor with gradients
            if isinstance(sigma_back, torch.Tensor):
                sigma_back_val = sigma_back
            else:
                sigma_back_val = torch.tensor(sigma_back, device=self.device)
        
        if sigma_perp is None:
            sigma_perp_val = sigma_factor / amps / 3
        else:
            # Preserve gradients if sigma_perp is already a tensor with gradients
            if isinstance(sigma_perp, torch.Tensor):
                sigma_perp_val = sigma_perp
            else:
                sigma_perp_val = torch.tensor(sigma_perp, device=self.device)
        
        # Create closure for this function
        def make_true_function(
            amps=amps,  # Don't clone - preserve gradients
            center=center,  # Don't clone - preserve gradients
            sigma_front=sigma_front_val,  # Don't clone - preserve gradients
            sigma_back=sigma_back_val,  # Don't clone - preserve gradients
            sigma_perp=sigma_perp_val,  # Don't clone - preserve gradients
            motion_dir=motion_dir,  # Don't clone - preserve gradients
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
            test_funcs.append(f_test)
            return true_function, test_funcs
        else:
            return true_function
        
class SymbolicReg(Surrogate):
    """
    Symbolic Regression surrogate model based on Julia implementation.
    
    This class implements an analytic amplitude surrogate that produces a 16-entry vector
    output based on symbolic regression from the Julia code. It calculates amplitude
    predictions using multiple rotation matrices and the symbolic regression formula.
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, num_rotations=16):
        """
        Initialize the Symbolic Regression surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (must be 3D for this model)
        domain_size : int
            Length of the domain 
        num_rotations : int
            Number of rotation matrices to use (determines output vector size)
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        if dim != 3:
            raise ValueError("SymbolicReg model only supports 3D input space")
        
        self.num_rotations = num_rotations
        
        # Generate random rotation matrices
        self.rotation_matrices = self._generate_rotation_matrices(num_rotations)
        
        # Model constants from Julia code
        self.constants = {
            'c1': 7.2352023283837395e-6,
            'exp1': 0.03443865311003615,
            'c2': 12.39167568068273,
            'c3': 0.38839803112959387,
            'c4': 0.1738077225915957,
            'c5': 3.00545648970963,
            'exp2': 0.5743625057140219,
            'c6': 9.341833951320876
        }
    
    def _generate_rotation_matrices(self, num_rotations):
        """Generate rotation matrices that map uniformly distributed directions to [0,0,1]."""
        rotation_matrices = []
        
        # Generate uniformly distributed points on unit sphere using hexagonal packing
        directions = self._generate_uniform_sphere_directions(num_rotations)
        
        # Target direction [0, 0, 1]
        target = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
        for direction in directions:
            # Create rotation matrix that maps direction to [0,0,1]
            R = self._rotation_matrix_from_vectors(direction, target)
            rotation_matrices.append(R)
        
        return rotation_matrices
    
    def _generate_uniform_sphere_directions(self, n_points):
        """Generate uniformly distributed directions on unit sphere using Fibonacci spiral."""
        directions = []
        
        # Use Fibonacci spiral for uniform distribution on sphere
        golden_ratio = (1 + 5**0.5) / 2
        
        if n_points == 1:
            # Special case for single point
            directions.append(torch.tensor([0.0, 0.0, 1.0], device=self.device))
        else:
            for i in range(n_points):
                # Fibonacci spiral parameters
                theta = 2 * torch.pi * i / golden_ratio
                phi = torch.acos(1 - 2 * i / (n_points - 1))
                
                # Convert to Cartesian coordinates
                x = torch.sin(phi) * torch.cos(theta)
                y = torch.sin(phi) * torch.sin(theta)
                z = torch.cos(phi)
                
                direction = torch.tensor([x, y, z], device=self.device)
                directions.append(direction)
        
        return directions
    
    def _rotation_matrix_from_vectors(self, vec1, vec2):
        """
        Create rotation matrix that rotates vec1 to vec2.
        
        Parameters:
        -----------
        vec1 : torch.Tensor
            Source vector (3D)
        vec2 : torch.Tensor
            Target vector (3D)
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        # Normalize vectors
        v1 = vec1 / torch.norm(vec1)
        v2 = vec2 / torch.norm(vec2)
        
        # Check if vectors are already aligned
        if torch.allclose(v1, v2, atol=1e-6):
            return torch.eye(3, device=self.device)
        
        # Check if vectors are opposite
        if torch.allclose(v1, -v2, atol=1e-6):
            # Find an orthogonal vector
            if abs(v1[0]) < 0.9:
                orthogonal = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                orthogonal = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            
            # Create 180-degree rotation around orthogonal axis
            axis = torch.cross(v1, orthogonal)
            axis = axis / torch.norm(axis)
            return self._rodrigues_rotation_matrix(axis, torch.pi)
        
        # General case: use Rodrigues' rotation formula
        axis = torch.cross(v1, v2)
        axis = axis / torch.norm(axis)
        
        # Calculate rotation angle
        cos_angle = torch.dot(v1, v2)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        
        return self._rodrigues_rotation_matrix(axis, angle)
    
    def _rodrigues_rotation_matrix(self, axis, angle):
        """
        Create rotation matrix using Rodrigues' rotation formula.
        
        Parameters:
        -----------
        axis : torch.Tensor
            Rotation axis (unit vector)
        angle : torch.Tensor
            Rotation angle in radians
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Cross product matrix for axis
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]], 
            [-axis[1], axis[0], 0]
        ], device=self.device)
        
        # Rodrigues' formula
        R = torch.eye(3, device=self.device) + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
        
        return R
    
    def _cart_to_sph(self, xyz):
        """
        Convert Cartesian coordinates to spherical coordinates.
        
        Parameters:
        -----------
        xyz : torch.Tensor
            Cartesian coordinates (x, y, z)
            
        Returns:
        --------
        tuple : (theta, phi) spherical angles
        """
        x, y, z = xyz[0], xyz[1], xyz[2]
        
        # Ensure we don't divide by zero
        r = torch.norm(xyz)
        if r < 1e-10:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))  # polar angle
        phi = torch.atan2(y, x)  # azimuthal angle
        
        return theta, phi
    
    def _sr_amplitude(self, dist, energy, pos_ct, dir_ct, dphi):
        """
        Symbolic regression amplitude function from Julia code.
        
        Parameters:
        -----------
        dist : torch.Tensor
            Distance
        energy : torch.Tensor
            Energy
        pos_ct : torch.Tensor
            cos(theta) for position
        dir_ct : torch.Tensor
            cos(theta) for direction
        dphi : torch.Tensor
            Difference in phi angles
            
        Returns:
        --------
        torch.Tensor
            Amplitude value
        """
        c = self.constants
        
        # Calculate intermediate terms
        denominator_term1 = c['c1'] * (dist ** c['exp1'])
        denominator_term2 = c['c1'] * torch.cos(dphi)
        denominator = denominator_term1 + denominator_term2
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-10)
        
        log_term = torch.log(energy / denominator)
        
        # Calculate the complex denominator for the fraction
        complex_denom_term1 = c['exp1'] * (c['c2'] ** pos_ct) * (energy ** c['exp1'])
        complex_denom_term2 = c['exp1'] * dist
        complex_denom_term3 = (dir_ct - c['c3']) * (pos_ct - c['c4'])
        complex_denom = complex_denom_term1 + complex_denom_term2 + complex_denom_term3 + c['c5']
        
        # Avoid negative or zero values for power operation
        complex_denom = torch.clamp(complex_denom, min=1e-10)
        
        fraction = log_term / (complex_denom ** c['exp2'])
        
        exponent = pos_ct + fraction - c['c6']
        
        # Calculate 10^exponent
        amplitude = 10.0 ** exponent
        
        return amplitude
    
    def _create_model_input(self, position, direction, energy, target_position):
        """
        Create model input by applying rotation matrices and calculating features.
        
        This method transforms physical neutrino parameters into the standardized 5-dimensional
        feature vectors required by the symbolic regression model. The transformation involves:
        
        1. Computing the relative position vector from source to target
        2. Applying multiple rotation matrices to achieve rotational invariance
        3. Converting to spherical coordinates for each rotation
        4. Extracting the key geometric and physical features
        
        Parameters:
        -----------
        position : torch.Tensor
            Source position vector (3D Cartesian coordinates) where the neutrino originated
        direction : torch.Tensor  
            Direction vector (3D Cartesian, unit vector) of neutrino propagation
        energy : torch.Tensor
            Energy value of the neutrino event (scalar)
        target_position : torch.Tensor
            Target position vector (3D Cartesian coordinates) where we evaluate the model
            
        Returns:
        --------
        torch.Tensor
            Input buffer with shape (num_rotations, 5) where each row contains the 5 model features:
            
            dist: Euclidean distance from source to target position
                      This provides spatial scale information for the light yield prediction
                      
            energy: Neutrino energy (same for all rotations)
                      Fundamental physical parameter affecting light production
                      
            pos_ct: cos(theta_position) where theta_position is the polar angle 
                      of the rotated relative position vector in spherical coordinates
                      Range: [-1, 1], represents the z-component of the normalized relative position
                      
            dir_ct: cos(theta_direction) where theta_direction is the polar angle
                      of the rotated direction vector in spherical coordinates  
                      Range: [-1, 1], represents the z-component of the normalized direction
                      
            dphi: Difference in azimuthal angles (phi_position - phi_direction)
                      Range: [-π, π], captures the relative angular orientation between
                      the position and direction vectors in the rotated coordinate system
                      
        """
        # Calculate relative position
        rel_pos = position - target_position
        dist = torch.norm(rel_pos)
        
        # Normalize relative position
        if dist > 1e-10:
            rel_pos_norm = rel_pos / dist
        else:
            rel_pos_norm = torch.zeros_like(rel_pos)
        
        # Initialize input buffer
        input_buffer = torch.zeros(self.num_rotations, 5, device=self.device)
        
        for i, R in enumerate(self.rotation_matrices):
            # Apply rotation to position and direction
            pos_rot = torch.matmul(R, rel_pos_norm)
            dir_rot = torch.matmul(R, direction)
            
            # Normalize direction
            dir_rot = dir_rot / torch.norm(dir_rot)
            
            # Convert to spherical coordinates
            pos_rot_th, pos_rot_phi = self._cart_to_sph(pos_rot)
            dir_rot_th, dir_rot_phi = self._cart_to_sph(dir_rot)
            
            # Calculate dphi
            dphi = pos_rot_phi - dir_rot_phi
            
            # Store in input buffer: (dist, energy, cos(pos_rot_th), cos(dir_rot_th), dphi)
            input_buffer[i] = torch.tensor([
                dist,
                energy,
                torch.cos(pos_rot_th),
                torch.cos(dir_rot_th),
                dphi
            ], device=self.device)
        
        return input_buffer
    
    def __call__(self, test_points=None, position=None, direction=None, 
                 energy=None, theta=None, phi=None, **kwargs):
        """
        Generate symbolic regression model output.
        
        Parameters:
        -----------
        nu_num : int
            Number of functions to generate (not used in this implementation)
        test_points : torch.Tensor (optional)
            Points to evaluate at (if provided, used as target positions)
        position : torch.Tensor or None
            Source position. If None, random position is generated.
        direction : torch.Tensor or None
            Direction vector (3D Cartesian). If None, random direction is generated.
            Takes precedence over theta/phi if both are provided.
        energy : float or torch.Tensor or None
            Energy value. If None, random energy is generated.
        theta : float or torch.Tensor or None
            Polar angle in radians (0 to π). Used if direction is None.
        phi : float or torch.Tensor or None
            Azimuthal angle in radians (0 to 2π). Used if direction is None.
            
        Returns:
        --------
        function or tuple
            Function that evaluates the symbolic regression model at given points,
            or tuple with function and test values if test_points provided
        """
        # Generate random parameters if not provided
        if position is None:
            position = torch.rand(3, device=self.device) * self.domain_size - self.half_domain
        else:
            position = torch.tensor(position, device=self.device) if not isinstance(position, torch.Tensor) else position
        
        # Handle direction specification - Cartesian takes precedence over spherical
        if direction is None:
            if theta is not None and phi is not None:
                # Convert spherical coordinates to Cartesian
                theta_val = torch.tensor(theta, device=self.device) if not isinstance(theta, torch.Tensor) else theta
                phi_val = torch.tensor(phi, device=self.device) if not isinstance(phi, torch.Tensor) else phi
                
                # Convert to Cartesian coordinates
                direction = torch.tensor([
                    torch.sin(theta_val) * torch.cos(phi_val),
                    torch.sin(theta_val) * torch.sin(phi_val),
                    torch.cos(theta_val)
                ], device=self.device)
            else:
                # Random direction on unit sphere
                direction = torch.randn(3, device=self.device)
                direction = direction / torch.norm(direction)
        else:
            # Use provided Cartesian direction
            direction = torch.tensor(direction, device=self.device) if not isinstance(direction, torch.Tensor) else direction
            direction = direction / torch.norm(direction)
        
        if energy is None:
            # Random energy from power law distribution
            energy = self.sample_power_law(E_min=0.1, E_max=10.0, gamma=2.7)
        else:
            energy = torch.tensor(energy, device=self.device) if not isinstance(energy, torch.Tensor) else energy
        
        # Create closure function
        def symbolic_reg_function(target_points):
            """
            Evaluate symbolic regression model at target points.
            
            Parameters:
            -----------
            target_points : torch.Tensor
                Target points to evaluate at, shape (N, 3)
                
            Returns:
            --------
            torch.Tensor
                Model output with shape (N, num_rotations)
            """
            if target_points.dim() == 1:
                target_points = target_points.unsqueeze(0)
            
            n_points = target_points.shape[0]
            output = torch.zeros(n_points, self.num_rotations, device=self.device)
            
            for i, target_point in enumerate(target_points):
                # Create model input for this target point
                model_input = self._create_model_input(position, direction, energy, target_point)
                
                # Calculate amplitude for each rotation
                for j in range(self.num_rotations):
                    dist, eng, pos_ct, dir_ct, dphi = model_input[j]
                    amplitude = self._sr_amplitude(dist, eng, pos_ct, dir_ct, dphi)
                    output[i, j] = amplitude
            
            return output
        
        if test_points is not None:
            test_output = symbolic_reg_function(test_points)
            return symbolic_reg_function, [test_output]
        else:
            return symbolic_reg_function
        
class LLRnet(Surrogate):
    """
    Log-Likelihood Ratio network for training an MLP classifier to estimate LLR.
    
    This network is trained as a binary classifier but uses the sigmoid trick to compute
    Log-Likelihood Ratios. The network outputs probabilities through a sigmoid activation,
    and the LLR is computed as log(p/(1-p)) where p is the output probability.
    
    The network supports parallel Fourier mapping layers with corresponding MLPs that
    process different frequency scales simultaneously. This allows the network to capture
    patterns at multiple scales and combine them for improved performance.
    
    Architecture:
    - Multiple parallel branches, each with:
      * Optional Fourier feature mapping at different frequency scales
      * Either separate MLPs per branch OR a single shared MLP (when shared_mlp=True)
    - Final MLP that concatenates all branch outputs and applies sigmoid
    
    When shared_mlp=True, each Fourier branch output is separately fed to the same
    shared MLP and the outputs are concatenated before the final layer. If branches
    have different Fourier output dimensions, smaller inputs are zero-padded to match
    the maximum dimension.
    
    Example Usage:
    --------------
    
    # Single branch (traditional architecture)
    model = LLRnet(dim=3, num_parallel_branches=1, frequency_scale=1.0)
    
    # Multiple branches with different frequency scales and shared MLP
    model = LLRnet(
        dim=3, 
        num_parallel_branches=3,
        frequency_scales=[0.5, 2.0, 8.0],
        num_frequencies_per_branch=[32, 64, 32],
        shared_mlp=True  # Use single MLP for all branches
    )
    
    # Train with event data
    history = model.train_with_event_data(
        signal_event_params=signal_params,
        background_event_params=background_params,
        signal_surrogate_func=signal_func,
        background_surrogate_func=background_func,
        epochs=100
    )
    
    Can work with provided background/signal functions or raw neutrino event data.
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, hidden_dims=[128, 64, 32], 
                 dropout_rate=0.1, learning_rate=1e-3, use_fourier_features=True,
                 num_frequencies=64, frequency_scale=1.0, learnable_frequencies=False,
                 num_parallel_branches=1, frequency_scales=None, num_frequencies_per_branch=None,
                 shared_mlp=False, use_residual_connections=False):
        """
        Initialize the LLRnet surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (2D or 3D)
        domain_size : int
            Length of the domain 
        hidden_dims : list
            List of hidden layer dimensions for the MLP
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for training
        use_fourier_features : bool
            Whether to use Fourier feature mapping at the input
        num_frequencies : int
            Number of frequency components for Fourier features (used if single branch)
        frequency_scale : float
            Scale factor for frequency sampling in Fourier features (used if single branch)
        learnable_frequencies : bool
            If True, Fourier frequencies are learnable parameters
        num_parallel_branches : int
            Number of parallel Fourier+MLP branches to use
        frequency_scales : list or None
            List of frequency scales for each branch. If None, uses geometric progression
        num_frequencies_per_branch : list or None
            List of number of frequencies for each branch. If None, uses num_frequencies for all
        shared_mlp : bool
            If True, uses a single shared MLP for all Fourier branches instead of separate MLPs
            This reduces parameters while still allowing multiple frequency scales
        use_residual_connections : bool
            If True, uses residual connections in the MLP layers for better gradient flow
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_fourier_features = use_fourier_features
        self.num_frequencies = num_frequencies
        self.frequency_scale = frequency_scale
        self.learnable_frequencies = learnable_frequencies
        self.num_parallel_branches = num_parallel_branches
        self.shared_mlp = shared_mlp
        self.use_residual_connections = use_residual_connections
        
        # Handle multiple branch configurations
        if num_parallel_branches > 1:
            # Set up frequency scales for each branch
            if frequency_scales is None:
                # Use geometric progression of frequency scales
                self.frequency_scales = [frequency_scale * (2.0 ** i) for i in range(num_parallel_branches)]
            else:
                if len(frequency_scales) != num_parallel_branches:
                    raise ValueError(f"frequency_scales must have length {num_parallel_branches}")
                self.frequency_scales = frequency_scales
            
            # Set up number of frequencies for each branch
            if num_frequencies_per_branch is None:
                self.num_frequencies_per_branch = [num_frequencies] * num_parallel_branches
            else:
                if len(num_frequencies_per_branch) != num_parallel_branches:
                    raise ValueError(f"num_frequencies_per_branch must have length {num_parallel_branches}")
                self.num_frequencies_per_branch = num_frequencies_per_branch
        else:
            # Single branch case
            self.frequency_scales = [frequency_scale]
            self.num_frequencies_per_branch = [num_frequencies]
        
        # Initialize network architecture
        self.fourier_features_list = None
        self.mlp_branches = None
        self.shared_branch_mlp = None  # Single shared MLP for all branches
        self.final_mlp = None
        self.optimizer = None
        self.loss_fn = torch.nn.BCELoss()  # Changed from BCEWithLogitsLoss
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_trained = False
        
    def _build_network(self, input_dim):
        """Build the parallel MLP network architecture with multiple Fourier feature mappings."""
        
        print(f"Building parallel network architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Number of parallel branches: {self.num_parallel_branches}")
        print(f"  Shared MLP: {self.shared_mlp}")
        
        if self.use_fourier_features:
            # Create multiple Fourier feature layers
            self.fourier_features_list = torch.nn.ModuleList()
            fourier_output_dims = []
            
            for i in range(self.num_parallel_branches):
                fourier_layer = FourierFeatures(
                    input_dim=input_dim,
                    num_frequencies=self.num_frequencies_per_branch[i],
                    frequency_scale=self.frequency_scales[i],
                    learnable=self.learnable_frequencies
                ).to(self.device)
                
                self.fourier_features_list.append(fourier_layer)
                fourier_output_dims.append(fourier_layer.output_dim)
                
                print(f"  Branch {i}: {self.num_frequencies_per_branch[i]} frequencies, "
                      f"scale {self.frequency_scales[i]:.3f}, output dim {fourier_layer.output_dim}")
        else:
            # No Fourier features, use raw input for all branches
            self.fourier_features_list = None
            fourier_output_dims = [input_dim] * self.num_parallel_branches
        
        # Create MLP branches - either shared or separate
        if self.shared_mlp:
            # For shared MLP, create a single MLP that can handle different input dimensions
            # We'll use the maximum Fourier output dimension or pad smaller inputs
            max_fourier_dim = max(fourier_output_dims)
            
            # Create shared MLP layers with optional residual connections
            shared_layers = []
            current_dim = max_fourier_dim
            
            # Input layer
            shared_layers.append(torch.nn.Linear(current_dim, self.hidden_dims[0]))
            shared_layers.append(torch.nn.ReLU())
            shared_layers.append(torch.nn.Dropout(self.dropout_rate))
            current_dim = self.hidden_dims[0]
            
            # Hidden layers with optional residual connections
            if self.use_residual_connections and len(self.hidden_dims) > 1:
                # Use residual blocks for hidden layers
                for j in range(len(self.hidden_dims) - 1):
                    shared_layers.append(ResidualBlock(current_dim, self.hidden_dims[j + 1], self.dropout_rate))
                    current_dim = self.hidden_dims[j + 1]
            else:
                # Regular linear layers
                for j in range(len(self.hidden_dims) - 1):
                    shared_layers.append(torch.nn.Linear(self.hidden_dims[j], self.hidden_dims[j + 1]))
                    shared_layers.append(torch.nn.ReLU())
                    shared_layers.append(torch.nn.Dropout(self.dropout_rate))
            
            # Output layer for shared MLP (no activation yet)
            branch_output_dim = self.hidden_dims[-1]
            shared_layers.append(torch.nn.Linear(current_dim, branch_output_dim))
            
            self.shared_branch_mlp = torch.nn.Sequential(*shared_layers).to(self.device)
            self.mlp_branches = None  # Not used in shared mode
            
            # Store input dimensions for padding logic
            self.fourier_output_dims = fourier_output_dims
            self.max_fourier_dim = max_fourier_dim
            
            # All branches will have same output dimension
            branch_output_dims = [branch_output_dim] * self.num_parallel_branches
            
            residual_info = "with residual connections" if self.use_residual_connections else "without residual connections"
            print(f"  Shared MLP {residual_info}: {max_fourier_dim} -> {self.hidden_dims} -> {branch_output_dim}")
            print(f"  Fourier output dims: {fourier_output_dims}")
            
        else:
            # Create separate MLP branches (original behavior)
            self.mlp_branches = torch.nn.ModuleList()
            self.shared_branch_mlp = None  # Not used in separate mode
            self.shared_core_mlp = None
            branch_output_dims = []
            
            for i in range(self.num_parallel_branches):
                mlp_input_dim = fourier_output_dims[i]
                
                # Create MLP for this branch with optional residual connections
                branch_layers = []
                current_dim = mlp_input_dim
                
                # Input layer
                branch_layers.append(torch.nn.Linear(current_dim, self.hidden_dims[0]))
                branch_layers.append(torch.nn.ReLU())
                branch_layers.append(torch.nn.Dropout(self.dropout_rate))
                current_dim = self.hidden_dims[0]
                
                # Hidden layers with optional residual connections
                if self.use_residual_connections and len(self.hidden_dims) > 1:
                    # Use residual blocks for hidden layers
                    for j in range(len(self.hidden_dims) - 1):
                        branch_layers.append(ResidualBlock(current_dim, self.hidden_dims[j + 1], self.dropout_rate))
                        current_dim = self.hidden_dims[j + 1]
                else:
                    # Regular linear layers
                    for j in range(len(self.hidden_dims) - 1):
                        branch_layers.append(torch.nn.Linear(self.hidden_dims[j], self.hidden_dims[j + 1]))
                        branch_layers.append(torch.nn.ReLU())
                        branch_layers.append(torch.nn.Dropout(self.dropout_rate))
                
                # Output layer for this branch (no activation yet)
                branch_output_dim = self.hidden_dims[-1]
                branch_layers.append(torch.nn.Linear(current_dim, branch_output_dim))
                
                branch_mlp = torch.nn.Sequential(*branch_layers).to(self.device)
                self.mlp_branches.append(branch_mlp)
                branch_output_dims.append(branch_output_dim)
                
                residual_info = "with residual connections" if self.use_residual_connections else "without residual connections"
                print(f"  Branch {i} MLP {residual_info}: {mlp_input_dim} -> {self.hidden_dims} -> {branch_output_dim}")
        
        # Create final MLP that combines all branch outputs
        total_branch_output_dim = sum(branch_output_dims)
        final_layers = []
        
        # Optionally add more layers to process the concatenated features
        final_hidden_dim = min(64, total_branch_output_dim // 2)  # Adaptive sizing
        
        final_layers.append(torch.nn.Linear(total_branch_output_dim, final_hidden_dim))
        final_layers.append(torch.nn.ReLU())
        final_layers.append(torch.nn.Dropout(self.dropout_rate))
        
        # # Final output layer with sigmoid
        final_layers.append(torch.nn.Linear(final_hidden_dim, 1))
        final_layers.append(torch.nn.Sigmoid())
        
        self.final_mlp = torch.nn.Sequential(*final_layers).to(self.device)
        
        print(f"  Final MLP: {total_branch_output_dim} -> {final_hidden_dim} -> 1")
        # print(f"  Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
        # Create optimizer for all trainable parameters
        all_params = []
        
        # Add MLP branch parameters
        if self.shared_mlp:
            # Add shared MLP parameters
            all_params.extend(self.shared_branch_mlp.parameters())
        else:
            for branch in self.mlp_branches:
                all_params.extend(branch.parameters())
        
        # Add final MLP parameters
        all_params.extend(self.final_mlp.parameters())
        
        # Add Fourier feature parameters if learnable
        if self.fourier_features_list is not None and self.learnable_frequencies:
            for fourier_layer in self.fourier_features_list:
                all_params.extend(fourier_layer.parameters())
        
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
    def prepare_data_from_functions(self, points, signal_funcs, background_funcs, 
                                  n_samples_per_func=1000, signal_ratio=0.5):
        """
        Prepare training data from provided signal and background functions.
        
        Parameters:
        -----------
        points : torch.Tensor
            Points to evaluate functions at (N, dim)
        signal_funcs : list
            List of signal functions
        background_funcs : list
            List of background functions
        n_samples_per_func : int
            Number of samples to generate per function
        signal_ratio : float
            Ratio of signal to total samples
            
        Returns:
        --------
        tuple : (features, labels)
            features: torch.Tensor of shape (total_samples, feature_dim)
            labels: torch.Tensor of shape (total_samples,)
        """
        all_features = []
        all_labels = []
        
        n_signal_samples = int(n_samples_per_func * signal_ratio)
        n_background_samples = n_samples_per_func - n_signal_samples
        
        # Generate signal samples
        for signal_func in signal_funcs:
            # Sample random points
            sample_indices = torch.randint(0, len(points), (n_signal_samples,))
            sample_points = points[sample_indices]
            
            # Evaluate signal function
            signal_values = signal_func(sample_points)
            
            # Create features (points + signal values + additional features)
            features = torch.cat([
                sample_points,
                signal_values.unsqueeze(-1) if signal_values.dim() == 1 else signal_values
            ], dim=1)
            
            all_features.append(features)
            all_labels.append(torch.ones(n_signal_samples, device=self.device))
        
        # Generate background samples
        for background_func in background_funcs:
            # Sample random points
            sample_indices = torch.randint(0, len(points), (n_background_samples,))
            sample_points = points[sample_indices]
            
            # Evaluate background function
            background_values = background_func(sample_points)
            
            # Create features (points + background values + additional features)
            features = torch.cat([
                sample_points,
                background_values.unsqueeze(-1) if background_values.dim() == 1 else background_values
            ], dim=1)
            
            all_features.append(features)
            all_labels.append(torch.zeros(n_background_samples, device=self.device))
        
        # Combine all data
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Shuffle the data
        perm = torch.randperm(len(features))
        features = features[perm]
        labels = labels[perm]
        
        return features, labels
    
    def prepare_data_from_raw(self, event_data, labels, additional_features=None):
        """
        Prepare training data from raw neutrino event data.
        
        Parameters:
        -----------
        event_data : torch.Tensor or dict
            Raw event data. Can be:
            - Tensor of shape (N, feature_dim) for direct feature input
            - Dict with keys like 'positions', 'light_yield', 'energy', etc.
        labels : torch.Tensor
            Binary labels (0 for background, 1 for signal)
        additional_features : torch.Tensor or None
            Additional engineered features to include
            
        Returns:
        --------
        tuple : (features, labels)
            features: torch.Tensor of shape (N, feature_dim)
            labels: torch.Tensor of shape (N,)
        """
        if isinstance(event_data, dict):
            # Extract features from dictionary
            feature_list = []
            
            # Add position features if available
            if 'positions' in event_data:
                positions = event_data['positions']
                if not isinstance(positions, torch.Tensor):
                    positions = torch.tensor(positions, device=self.device, dtype=torch.float32)
                feature_list.append(positions)
            
            # Add light yield features if available
            if 'light_yield' in event_data:
                light_yield = event_data['light_yield']
                if not isinstance(light_yield, torch.Tensor):
                    light_yield = torch.tensor(light_yield, device=self.device, dtype=torch.float32)
                if light_yield.dim() == 1:
                    light_yield = light_yield.unsqueeze(-1)
                feature_list.append(light_yield)
            
            # # Add energy features if available
            # if 'energy' in event_data:
            #     energy = event_data['energy']
            #     if not isinstance(energy, torch.Tensor):
            #         energy = torch.tensor(energy, device=self.device, dtype=torch.float32)
            #     if energy.dim() == 1:
            #         energy = energy.unsqueeze(-1)
            #     feature_list.append(energy)
            
            # # Add timing features if available
            # if 'timing' in event_data:
            #     timing = event_data['timing']
            #     if not isinstance(timing, torch.Tensor):
            #         timing = torch.tensor(timing, device=self.device, dtype=torch.float32)
            #     if timing.dim() == 1:
            #         timing = timing.unsqueeze(-1)
            #     feature_list.append(timing)
            
            # Combine all features
            features = torch.cat(feature_list, dim=1)
            
        else:
            # Direct tensor input
            features = event_data
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Add additional engineered features if provided
        if additional_features is not None:
            if not isinstance(additional_features, torch.Tensor):
                additional_features = torch.tensor(additional_features, device=self.device, dtype=torch.float32)
            features = torch.cat([features, additional_features], dim=1)
        
        # Ensure labels are tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device, dtype=torch.float32)
        
        return features, labels
    
    def train(self, features, labels, epochs=100, batch_size=256, validation_split=0.2, 
              verbose=True, early_stopping_patience=10):
        """
        Train the LLR network.
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features of shape (N, feature_dim)
        labels : torch.Tensor
            Binary labels of shape (N,)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : bool
            Whether to print training progress
        early_stopping_patience : int
            Number of epochs to wait for improvement before early stopping
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        # Build network if not already built
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            self._build_network(features.shape[1])
        
        # Split data into train and validation
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle data
        perm = torch.randperm(n_samples)
        features = features[perm]
        labels = labels[perm]
        
        train_features = features[:n_train]
        train_labels = labels[:n_train]
        val_features = features[n_train:]
        val_labels = labels[n_train:]
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            if self.shared_mlp:
                self.shared_branch_mlp.train()
            else:
                for branch in self.mlp_branches:
                    branch.train()
            self.final_mlp.train()
            if self.fourier_features_list is not None:
                for fourier_layer in self.fourier_features_list:
                    fourier_layer.train()
                
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_train, batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self._forward_pass(batch_features)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation phase
            if self.shared_mlp:
                self.shared_branch_mlp.eval()
            else:
                for branch in self.mlp_branches:
                    branch.eval()
            self.final_mlp.eval()
            if self.fourier_features_list is not None:
                for fourier_layer in self.fourier_features_list:
                    fourier_layer.eval()
                
            with torch.no_grad():
                val_outputs = self._forward_pass(val_features)
                val_loss = self.loss_fn(val_outputs, val_labels).item()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state_dict = {
                    'mlp_branches': [branch.state_dict().copy() for branch in self.mlp_branches] if not self.shared_mlp else None,
                    'shared_branch_mlp': self.shared_branch_mlp.state_dict().copy() if self.shared_mlp else None,
                    'final_mlp': self.final_mlp.state_dict().copy(),
                    'fourier_features_list': [fourier_layer.state_dict().copy() for fourier_layer in self.fourier_features_list] if self.fourier_features_list is not None else None
                }
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                if self.shared_mlp:
                    self.shared_branch_mlp.load_state_dict(self.best_state_dict['shared_branch_mlp'])
                else:
                    for i, branch in enumerate(self.mlp_branches):
                        branch.load_state_dict(self.best_state_dict['mlp_branches'][i])
                self.final_mlp.load_state_dict(self.best_state_dict['final_mlp'])
                if self.fourier_features_list is not None and self.best_state_dict['fourier_features_list'] is not None:
                    for i, fourier_layer in enumerate(self.fourier_features_list):
                        fourier_layer.load_state_dict(self.best_state_dict['fourier_features_list'][i])
                break
        
        self.is_trained = True
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
    def _forward_pass(self, points):
        """
        Internal method for forward pass through the parallel network.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features
            
        Returns:
        --------
        torch.Tensor
            Network output (probabilities)
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=self.device, dtype=torch.float32)
        else:
            points = points.float().to(self.device)  # Ensure float32 and correct device
        
        # Process through each parallel branch
        branch_outputs = []
        
        if self.shared_mlp:
            # Use shared MLP for all branches - process each branch separately
            for i in range(self.num_parallel_branches):
                branch_input = points
                
                # Apply Fourier feature mapping if enabled
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](branch_input)
                
                # Pad input to max dimension if necessary
                current_dim = branch_input.size(-1)
                if current_dim < self.max_fourier_dim:
                    padding_size = self.max_fourier_dim - current_dim
                    padding = torch.zeros(branch_input.size(0), padding_size, device=self.device, dtype=branch_input.dtype)
                    branch_input = torch.cat([branch_input, padding], dim=-1)
                
                # Forward pass through the shared MLP for this branch
                branch_output = self.shared_branch_mlp(branch_input)
                branch_outputs.append(branch_output)
        else:
            # Use separate MLPs for each branch (original behavior)
            for i in range(self.num_parallel_branches):
                branch_input = points
                
                # Apply Fourier feature mapping if enabled
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](branch_input)
                
                # Forward pass through branch MLP
                branch_output = self.mlp_branches[i](branch_input)
                branch_outputs.append(branch_output)
        
        # Concatenate all branch outputs
        concatenated_features = torch.cat(branch_outputs, dim=-1)
        
        # Final MLP with sigmoid activation
        final_output = self.final_mlp(concatenated_features)
        
        return final_output.squeeze()
    
    def __call__(self, points, return_probabilities=True):
        """
        Evaluate the trained LLR network on input points.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
        return_probabilities : bool
            If True, return probabilities (default behavior since network outputs probabilities)
            If False, return LLR values
            
        Returns:
        --------
        torch.Tensor
            Probabilities or LLR values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling. Use .train() method first.")
        
        if self.shared_mlp:
            self.shared_branch_mlp.eval()
        else:
            for branch in self.mlp_branches:
                branch.eval()
        self.final_mlp.eval()
        if self.fourier_features_list is not None:
            for fourier_layer in self.fourier_features_list:
                fourier_layer.eval()
            
        with torch.no_grad():
            probabilities = self._forward_pass(points)
            
            if return_probabilities:
                return probabilities
            else:
                # Convert probabilities to LLR: log(p/(1-p))
                epsilon = 1e-7  # Small value to prevent log(0)
                prob_clamped = torch.clamp(probabilities, epsilon, 1 - epsilon)
                return torch.log(prob_clamped / (1 - prob_clamped))
    
    def predict_log_likelihood_ratio(self, points):
        """
        Compute the Log-Likelihood Ratio using the sigmoid trick.
        
        This method computes log(p/(1-p)) where p is the output probability from the network.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
            
        Returns:
        --------
        torch.Tensor
            Log-likelihood ratios
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling. Use .train() method first.")
        
        if self.shared_mlp:
            self.shared_branch_mlp.eval()
        else:
            for branch in self.mlp_branches:
                branch.eval()
        self.final_mlp.eval()
        if self.fourier_features_list is not None:
            for fourier_layer in self.fourier_features_list:
                fourier_layer.eval()
            
        with torch.no_grad():
            # Get probabilities from the network (already has sigmoid)
            probabilities = self._forward_pass(points)
            
            # Compute LLR: log(p/(1-p))
            # epsilon = 1e-7  # Small value to prevent log(0)
            # prob_clamped = torch.clamp(probabilities, epsilon, 1 - epsilon)
            # llr = torch.log(prob_clamped / (1 - prob_clamped))
            llr = torch.log(probabilities + 1e-10) - torch.log(1 - (probabilities + 1e-10))
            return llr
    
    def predict_likelihood_ratio(self, points):
        """
        Compute the Likelihood Ratio (not log).
        
        This method computes p/(1-p) where p is the sigmoid of the network output.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
            
        Returns:
        --------
        torch.Tensor
            Likelihood ratios
        """
        log_ratios = self.predict_log_likelihood_ratio(points)
        return torch.exp(log_ratios)
    
    def predict_proba(self, points):
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features
            
        Returns:
        --------
        torch.Tensor
            Probabilities of being signal (class 1)
        """
        return self.__call__(points, return_probabilities=True)
    
    def predict(self, points, threshold=0.5):
        """
        Get binary predictions.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features
        threshold : float
            Decision threshold for classification
            
        Returns:
        --------
        torch.Tensor
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(points)
        return (probabilities > threshold).float()
    
    def evaluate(self, features, labels, metrics=['accuracy', 'precision', 'recall', 'f1']):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        features : torch.Tensor
            Test features
        labels : torch.Tensor
            True labels
        metrics : list
            List of metrics to compute
            
        Returns:
        --------
        dict
            Dictionary of computed metrics
        """
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        results = {}
        
        # Convert to numpy for sklearn metrics
        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities.cpu().numpy()
        
        if 'accuracy' in metrics:
            from sklearn.metrics import accuracy_score
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in metrics:
            from sklearn.metrics import precision_score
            results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        if 'recall' in metrics:
            from sklearn.metrics import recall_score
            results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        if 'f1' in metrics:
            from sklearn.metrics import f1_score
            results['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if 'auc' in metrics:
            from sklearn.metrics import roc_auc_score
            results['auc'] = roc_auc_score(y_true, y_prob)
        
        return results
    
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LLRnet Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")
        
        save_dict = {
            'mlp_branches_state_dict': [branch.state_dict() for branch in self.mlp_branches] if not self.shared_mlp else None,
            'shared_branch_mlp_state_dict': self.shared_branch_mlp.state_dict() if self.shared_mlp else None,
            'final_mlp_state_dict': self.final_mlp.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'device': self.device,
            'dim': self.dim,
            'domain_size': self.domain_size,
            'use_fourier_features': self.use_fourier_features,
            'num_frequencies': self.num_frequencies,
            'frequency_scale': self.frequency_scale,
            'learnable_frequencies': self.learnable_frequencies,
            'num_parallel_branches': self.num_parallel_branches,
            'frequency_scales': self.frequency_scales,
            'num_frequencies_per_branch': self.num_frequencies_per_branch,
            'shared_mlp': self.shared_mlp
        }
        
        if self.fourier_features_list is not None:
            save_dict['fourier_features_list_state_dict'] = [fourier.state_dict() for fourier in self.fourier_features_list]
        else:
            save_dict['fourier_features_list_state_dict'] = None
            
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update model parameters
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # Load Fourier feature parameters if available (for backward compatibility)
        self.use_fourier_features = checkpoint.get('use_fourier_features', False)
        self.num_frequencies = checkpoint.get('num_frequencies', 64)
        self.frequency_scale = checkpoint.get('frequency_scale', 1.0)
        self.learnable_frequencies = checkpoint.get('learnable_frequencies', False)
        
        # Load parallel branch parameters (new format)
        self.num_parallel_branches = checkpoint.get('num_parallel_branches', 1)
        self.frequency_scales = checkpoint.get('frequency_scales', [self.frequency_scale])
        self.num_frequencies_per_branch = checkpoint.get('num_frequencies_per_branch', [self.num_frequencies])
        self.shared_mlp = checkpoint.get('shared_mlp', False)
        
        # Determine if this is old format (single MLP) or new format (parallel branches)
        is_old_format = 'model_state_dict' in checkpoint
        
        if is_old_format:
            # Convert old format to new format for backward compatibility
            print("Loading old format model - converting to single branch architecture")
            self.num_parallel_branches = 1
            self.frequency_scales = [self.frequency_scale]
            self.num_frequencies_per_branch = [self.num_frequencies]
            
            # Infer input_dim from old format
            if self.use_fourier_features and 'fourier_features_state_dict' in checkpoint:
                fourier_state = checkpoint['fourier_features_state_dict']
                if fourier_state is not None:
                    frequencies_shape = fourier_state['frequencies'].shape
                    input_dim = frequencies_shape[1]
                else:
                    first_layer_weight = checkpoint['model_state_dict']['0.weight']
                    fourier_output_dim = first_layer_weight.shape[1]
                    input_dim = fourier_output_dim // (2 * self.num_frequencies)
            else:
                first_layer_weight = checkpoint['model_state_dict']['0.weight']
                input_dim = first_layer_weight.shape[1]
            
            # Build network
            self._build_network(input_dim)
            
            # Load old states into first branch
            self.mlp_branches[0].load_state_dict(checkpoint['model_state_dict'])
            
            if self.fourier_features_list is not None and 'fourier_features_state_dict' in checkpoint:
                fourier_state = checkpoint['fourier_features_state_dict']
                if fourier_state is not None:
                    self.fourier_features_list[0].load_state_dict(fourier_state)
                    
        else:
            # New format with parallel branches
            # Infer input_dim from first branch's Fourier features
            if self.use_fourier_features and 'fourier_features_list_state_dict' in checkpoint:
                fourier_states = checkpoint['fourier_features_list_state_dict']
                if fourier_states is not None and len(fourier_states) > 0:
                    frequencies_shape = fourier_states[0]['frequencies'].shape
                    input_dim = frequencies_shape[1]
                else:
                    # Fallback: infer from first MLP branch
                    first_branch_state = checkpoint['mlp_branches_state_dict'][0]
                    first_layer_key = next(iter(first_branch_state.keys()))
                    if '0.weight' in first_layer_key:
                        input_dim = first_branch_state['0.weight'].shape[1]
                    else:
                        # Try different naming convention
                        for key in first_branch_state.keys():
                            if 'weight' in key and len(first_branch_state[key].shape) == 2:
                                input_dim = first_branch_state[key].shape[1]
                                break
            else:
                # No Fourier features, get input dim from first MLP branch
                first_branch_state = checkpoint['mlp_branches_state_dict'][0]
                first_layer_key = next(iter(first_branch_state.keys()))
                if '0.weight' in first_layer_key:
                    input_dim = first_branch_state['0.weight'].shape[1]
                else:
                    for key in first_branch_state.keys():
                        if 'weight' in key and len(first_branch_state[key].shape) == 2:
                            input_dim = first_branch_state[key].shape[1]
                            break
            
            # Build network
            self._build_network(input_dim)
            
            # Load parallel branch states
            if self.shared_mlp and 'shared_branch_mlp_state_dict' in checkpoint:
                # Load shared MLP state
                self.shared_branch_mlp.load_state_dict(checkpoint['shared_branch_mlp_state_dict'])
            elif not self.shared_mlp and 'mlp_branches_state_dict' in checkpoint:
                # Load separate branch states
                for i, branch_state in enumerate(checkpoint['mlp_branches_state_dict']):
                    self.mlp_branches[i].load_state_dict(branch_state)
            
            # Load final MLP state
            self.final_mlp.load_state_dict(checkpoint['final_mlp_state_dict'])
            
            # Load Fourier features if available
            if self.fourier_features_list is not None and 'fourier_features_list_state_dict' in checkpoint:
                fourier_states = checkpoint['fourier_features_list_state_dict']
                if fourier_states is not None:
                    for i, fourier_state in enumerate(fourier_states):
                        self.fourier_features_list[i].load_state_dict(fourier_state)
        
        self.is_trained = True
    
    def generate_batch_data_from_events(self, event_indices=None, optimization_points=None, signal_surrogate_func=None, 
                                       background_surrogate_func=None, signal_event_params=None,
                                       background_event_params=None, batch_size=256, 
                                       signal_ratio=0.5, add_noise=True, sig_noise_scale=0.1,
                                       bkg_noise_scale = 0.2, random_seed=None, 
                                       samples_per_epoch=10, keep_opt_point=False, sr_mode=False, output_denoised=False, 
                                       return_event_params=False):
        """
        Generate training data on-the-fly from neutrino event parameters for batch training.
        
        Features include neutrino event data remapped relative to optimization points:
        - Relative position coordinates (event center - optimization point)
        - Relative incoming angles (zenith, azimuth)
        - Light yield values computed from surrogate functions (with optional noise)
        - Other event parameters
        
        The optimization point coordinates themselves are NOT included in features.
        
        Parameters:
        -----------
        event_indices : list or torch.Tensor or None
            Indices of events to use. If None, uses all available events.
        optimization_points : torch.Tensor or None
            Points at which to evaluate/optimize, shape (N, dim). If None, 
            random points will be sampled from the domain for each batch.
        signal_surrogate_func : callable or None
            Function that takes event parameters and optimization point, returns light yield
            Signature: light_yield = func(event_params, opt_point)
            where event_params is a dict with keys like 'position', 'zenith', 'azimuth', 'energy', etc.
        background_surrogate_func : callable or None
            Similar function for background events
        signal_event_params : dict or None
            Dictionary containing signal event parameters:
            {
                'positions': torch.Tensor, shape (n_signal_events, dim) - event centers
                'zenith_angles': torch.Tensor, shape (n_signal_events,) - incoming zenith angles
                'azimuth_angles': torch.Tensor, shape (n_signal_events,) - incoming azimuth angles  
                'energies': torch.Tensor, shape (n_signal_events,) - event energies (optional)
                'other_features': torch.Tensor, shape (n_signal_events, n_features) - additional features (optional)
            }
        background_event_params : dict or None
            Similar dictionary for background events
        batch_size : int
            Size of each training batch
        signal_ratio : float
            Fraction of signal events in each batch (0 to 1)
        add_noise : bool
            Whether to add noise to light yield values
        sig_noise_scale : float
            Standard deviation of Gaussian noise to add to signal light yields
        bkg_noise_scale : float
            Standard deviation of Gaussian noise to add to background light yields
        random_seed : int or None
            Random seed for reproducibility
        samples_per_event : int
            Number of optimization points to sample per event in each batch
        return_event_params : bool
            Whether to return the original event parameters for selected events
            
        Returns:
        --------
        torch.utils.data.DataLoader
            DataLoader yielding batches of:
            - If return_event_params=False and output_denoised=False: (features, labels)
            - If return_event_params=False and output_denoised=True: (features, labels, denoised_yield)
            - If return_event_params=True and output_denoised=False: (features, labels, event_params_batch)
            - If return_event_params=True and output_denoised=True: (features, labels, denoised_yield, event_params_batch)
            
            Features shape: (batch_size, feature_dim)
            Labels shape: (batch_size,) - binary (0=background, 1=signal)
            event_params_batch: list of dicts with original event parameters corresponding to each sample
                               - Each dict contains the original event parameters for the selected event
                               - The order corresponds directly to the labels (label[i] corresponds to event_params_batch[i])
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Validate inputs - need either surrogate functions or direct light yields
        if signal_surrogate_func is None and background_surrogate_func is None:
            if signal_event_params is None and background_event_params is None:
                raise ValueError("Must provide either surrogate functions or event parameters with light yields")
            # Check if light yields are provided in event params
            if signal_event_params and 'light_yields' not in signal_event_params:
                raise ValueError("signal_event_params must contain 'light_yields' if no signal_surrogate_func provided")
            if background_event_params and 'light_yields' not in background_event_params:
                raise ValueError("background_event_params must contain 'light_yields' if no background_surrogate_func provided")
        
        if signal_event_params is None and background_event_params is None:
            raise ValueError("At least one of signal_event_params or background_event_params must be provided")
        
        # Convert optimization points to tensor if provided and ensure float32
        # if optimization_points is not None:
        #     if not isinstance(optimization_points, torch.Tensor):
        #         optimization_points = torch.tensor(optimization_points, device=self.device, dtype=torch.float32)
        #     else:
        #         optimization_points = optimization_points.float().to(self.device)
        
        # Handle event indices - determine which events to use
        total_signal_events = len(signal_event_params['position']) if signal_event_params else 0
        total_background_events = len(background_event_params['position']) if background_event_params else 0
        
        if event_indices is None:
            # Use all events
            signal_indices = list(range(total_signal_events)) if total_signal_events > 0 else []
            background_indices = list(range(total_background_events)) if total_background_events > 0 else []
        else:
            # Filter provided indices by signal/background
            if isinstance(event_indices, torch.Tensor):
                event_indices = event_indices.tolist()
            
            # For simplicity, assume event_indices refers to combined pool
            # In practice, you might want to pass separate signal_indices and background_indices
            # total_events = total_signal_events + total_background_events
            # signal_indices = [i for i in event_indices if i < total_signal_events]
            # background_indices = [i - total_signal_events for i in event_indices 
            #                     if i >= total_signal_events and i < total_events]
        
        class EventDataset(torch.utils.data.Dataset):
            def __init__(self, sig_indices, bg_indices, opt_points, sig_func, bg_func, sig_params, bg_params, sig_ratio, add_noise, sig_noise_scale, bkg_noise_scale, device, dim, samples_per_epoch, domain_size, keep_opt_point=False, sr_mode=False, output_denoised=False, return_event_params=False):
                self.sig_indices = sig_indices
                self.bg_indices = bg_indices
                self.opt_points = opt_points
                self.sig_surrogate_func = sig_func
                self.bg_surrogate_func = bg_func
                self.sig_params = sig_params
                self.bg_params = bg_params
                self.sig_ratio = sig_ratio
                self.add_noise = add_noise
                self.sig_noise_scale = sig_noise_scale
                self.bkg_noise_scale = bkg_noise_scale
                self.device = device
                self.dim = dim
                self.samples_per_epoch = samples_per_epoch
                self.domain_size = domain_size
                self.keep_opt_point = keep_opt_point
                self.sr_mode = sr_mode
                self.return_event_params = return_event_params
                # Calculate total number of available event-optimization point combinations
                # self.total_combinations = (len(self.sig_indices) + len(self.bg_indices)) * self.samples_per_event
                self.total_combinations = samples_per_epoch
                self.output_denoised = output_denoised
                # Determine feature dimension
                self.feature_dim = self._calculate_feature_dim()
                
                
                
            def _calculate_feature_dim(self):
                """Calculate the total feature dimension."""
                # Base features: relative position (dim) + relative angles (2) + light_yield (1)
                if not self.sr_mode:
                    feature_dim = self.dim + 2 + 1
                    if self.keep_opt_point:
                        feature_dim += self.dim
                    
                    # Add energy if available
                    if (self.sig_params and 'energy' in self.sig_params) or \
                    (self.bg_params and 'energy' in self.bg_params):
                        feature_dim += 1
                else:
                    feature_dim = 6
                
                # Add other features if available
                if self.sig_params and 'other_features' in self.sig_params or \
                   self.bg_params and 'other_features' in self.bg_params:
                    feature_dim += self.sig_params['other_features'].shape[1]
                    
                
                return feature_dim
                
            def __len__(self):
                return self.total_combinations
                
            def __getitem__(self, idx):
                """Generate a training sample by sampling an event and optimization point."""
                # Calculate which event and which sample within that event
                # total_events = len(self.sig_indices) + len(self.bg_indices)
                # event_group_idx = idx // self.samples_per_event
                # sample_within_event = idx % self.samples_per_event
                # event_idx = idx // self.samples_per_event
                # opt_point_sample = idx % self.samples_per_event
                
                # Decide if this should be a signal or background sample based on ratio
                is_signal = torch.rand(1).item() < self.sig_ratio
                
                # Sample an event
                if is_signal and len(self.sig_indices) > 0:
                    event_idx = torch.randint(0, len(self.sig_indices), (1,)).item()
                    actual_event_idx = self.sig_indices[event_idx]
                    background = False
                elif not is_signal and len(self.bg_indices) > 0:
                    event_idx = torch.randint(0, len(self.bg_indices), (1,)).item()
                    actual_event_idx = self.bg_indices[event_idx]
                    background = True
                
                # Sample an optimization point (this is where samples_per_event matters)
                if self.opt_points is not None:
                    # Sample from provided optimization points
                    opt_idx = torch.randint(0, len(self.opt_points), (1,)).item()
                    opt_point = torch.tensor(self.opt_points[opt_idx], device=self.device, dtype=torch.float32)
                else:
                    # Generate random optimization point from domain
                    # Use sample_within_event as part of random seed for reproducibility within event
                    # torch.manual_seed(torch.initial_seed() + idx)
                    opt_point = torch.rand(self.dim, device=self.device, dtype=torch.float32) * self.domain_size - self.domain_size/2
                
                # Generate sample data
                if not self.output_denoised:
                    features, label = self._generate_sample(opt_point, actual_event_idx, background)
                    if not self.return_event_params:
                        return features, label
                    else:
                        # Extract event parameters for the selected event
                        if background:
                            selected_event_params = self._extract_event_params(self.bg_params, actual_event_idx)
                        else:
                            selected_event_params = self._extract_event_params(self.sig_params, actual_event_idx)
                        return features, label, selected_event_params
                else:
                    features, label, denoised = self._generate_sample(opt_point, actual_event_idx, background)
                    if not self.return_event_params:
                        return features, label, denoised
                    else:
                        # Extract event parameters for the selected event
                        if background:
                            selected_event_params = self._extract_event_params(self.bg_params, actual_event_idx)
                        else:
                            selected_event_params = self._extract_event_params(self.sig_params, actual_event_idx)
                        return features, label, denoised, selected_event_params
                    
            
                
            def _extract_event_params(self, event_params, event_idx):
                """Extract event parameters for a specific event index."""
                if event_params is None:
                    return {}
                
                extracted_params = {}
                for key, values in event_params.items():
                    if isinstance(values, torch.Tensor):
                        extracted_params[key] = values[event_idx].clone()
                    elif isinstance(values, (list, tuple)):
                        extracted_params[key] = values[event_idx]
                    else:
                        # For scalar values, just copy
                        extracted_params[key] = values
                        
                return extracted_params
                
            def _generate_sample(self, opt_point, event_idx, background=False):
                """Generate a surrogate sample relative to the optimization point."""
                # Randomly select a background event
                if background:
                    sample_event_params = self.bg_params
                    surrogate_func = self.bg_surrogate_func
                    noise_scale = self.bkg_noise_scale
                else:                    
                    sample_event_params = self.sig_params
                    surrogate_func = self.sig_surrogate_func
                    noise_scale = self.sig_noise_scale
                # n_events = len(sample_event_params['positions'])
                # event_idx = torch.randint(0, n_events, (1,)).item()
                denoised = None
                # Compute light yield using surrogate function or use provided value
                if surrogate_func is not None:
                    light_yield = surrogate_func(sample_event_params, opt_point, event_idx)
                    if not isinstance(light_yield, torch.Tensor):
                        light_yield = torch.tensor(light_yield, device=self.device, dtype=torch.float32)
                    else:
                        light_yield = light_yield.float()  # Ensure float32
                else:
                    # Use pre-computed light yield
                    light_yield = sample_event_params['light_yield'][event_idx].clone().float()
                
                # Add noise to light yield if requested
                if self.add_noise:
                    noise = torch.randn(1, device=self.device, dtype=torch.float32) * noise_scale
                    denoised = light_yield.clone()
                    light_yield = light_yield + light_yield*noise
                
                # Create relative features
                features = self._create_relative_features(
                    opt_point, light_yield,
                    sample_event_params, event_idx
                )
                if background:
                    if features.dim() > 1:
                        labels = torch.zeros(features.shape[0], device=self.device, dtype=torch.float32)  # Background label
                    else:
                        labels = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                else:
                    if features.dim() > 1:
                        labels = torch.ones(features.shape[0], device=self.device, dtype=torch.float32)  # Signal label
                    else:
                        labels = torch.tensor(1.0, device=self.device, dtype=torch.float32)

                if not self.output_denoised:
                    return features, labels
                else:
                    # Return features, labels, and denoised light yield
                    return features, labels, denoised

            def _create_relative_features(self, opt_point, light_yield, event_params, event_idx):
                """Create feature vector with relative positioning and angles.
                Order or features:
                1. Relative position (event center - optimization point)
                1.5. Optionally include optimization point itself
                2. Relative zenith and azimuth angles
                3. Energy (if available)
                4. Light yield (with optional noise)
                5. Other features (if available)"""
                feature_list = []
                
                # Ensure all inputs are float32
                event_pos = event_params['position'][event_idx].float()
                zenith = event_params['zenith'][event_idx].float()
                azimuth = event_params['azimuth'][event_idx].float()
                opt_point = opt_point.float()
                light_yield = light_yield.float()
                energy = event_params.get('energy', None)
                if energy is not None:
                    energy = event_params['energy'][event_idx].float()
                
                # Relative position: event_center - optimization_point
                if not self.sr_mode:
                    relative_pos = event_pos - opt_point
                    feature_list.append(relative_pos)
                    if self.keep_opt_point:
                        # Optionally include the optimization point itself
                        feature_list.append(opt_point)
                    
                    # Relative angles (zenith and azimuth)
                    # Note: These are already in the event coordinate system
                    # Could be modified to be relative to optimization point if needed
                    feature_list.append(torch.tensor([zenith, azimuth], device=self.device, dtype=torch.float32))
                    if energy is not None:
                        energy_feature = energy.unsqueeze(0) if energy.dim() == 0 else energy
                        feature_list.append(energy_feature.float())
                else:
                    sr_model = SymbolicReg(num_rotations=1)
                    
                    # dist = torch.norm(event_pos - opt_point)   
                    direction = torch.tensor([
                        torch.sin(zenith) * torch.cos(azimuth),
                        torch.sin(zenith) * torch.sin(azimuth),
                        torch.cos(zenith)
                    ], device=self.device)
                    input_buffer = sr_model._create_model_input(event_pos, direction, energy, opt_point)
                    for feature in input_buffer:
                        feature_list.append(feature.float())
                # Light yield (possibly with noise)
                light_yield_feature = light_yield.unsqueeze(0) if light_yield.dim() == 0 else light_yield
                feature_list.append(light_yield_feature.float())
                
                # Add energy if available
                # if energy is not None:
                #     feature_list.append(energy_feature.float())

                # Add other features if available
                if 'other_features' in event_params:
                    other_feats = event_params['other_features'][event_idx].float()
                    feature_list.append(other_feats)
                
                # Concatenate all features - all should now be float32
                features = torch.cat(feature_list, dim=0)
                
                return features
        
        # Create dataset
        dataset = EventDataset(
            signal_indices, background_indices, optimization_points, 
            signal_surrogate_func, background_surrogate_func,
            signal_event_params, background_event_params,
            signal_ratio, add_noise, sig_noise_scale, bkg_noise_scale, 
            self.device, self.dim, samples_per_epoch, self.domain_size, keep_opt_point, 
            sr_mode, output_denoised, return_event_params
        )
        
        # Create DataLoader with appropriate collate function based on return structure
        def collate_batch(batch):
            """Custom collate function to handle different return structures."""
            if not return_event_params and not output_denoised:
                # Standard: (features, labels)
                return (
                    torch.stack([item[0] for item in batch]),
                    torch.stack([item[1] for item in batch])
                )
            elif not return_event_params and output_denoised:
                # With denoised: (features, labels, denoised)
                return (
                    torch.stack([item[0] for item in batch]),
                    torch.stack([item[1] for item in batch]),
                    torch.stack([item[2] for item in batch])
                )
            elif return_event_params and not output_denoised:
                # With event params: (features, labels, event_params)
                features = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
                
                # Collect event params - each item[2] is a single event parameter dict
                event_params_batch = [item[2] for item in batch]
                
                return features, labels, event_params_batch
            else:
                # With both denoised and event params: (features, labels, denoised, event_params)
                features = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
                denoised = torch.stack([item[2] for item in batch])
                
                # Collect event params - each item[3] is a single event parameter dict
                event_params_batch = [item[3] for item in batch]
                
                return features, labels, denoised, event_params_batch
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_batch
        )
        
        return dataloader
        
        # Example usage with return_event_params=True:
        # dataloader = generate_batch_data_from_events(..., return_event_params=True)
        # for batch_data in dataloader:
        #     if output_denoised:
        #         features, labels, denoised, event_params_batch = batch_data
        #     else:
        #         features, labels, event_params_batch = batch_data
        #     # event_params_batch is a list of dicts with length equal to batch_size
        #     # Each dict contains the original event parameters for the corresponding sample
        #     # The order corresponds directly to labels: event_params_batch[i] goes with labels[i]
    
    def train_with_event_data(self, optimization_points=None, signal_surrogate_func=None,
                            background_surrogate_func=None, signal_event_params=None,
                            background_event_params=None, epochs=100, batch_size=256,
                            signal_ratio=0.5, add_noise=True, signal_noise_scale=0.1,
                            background_noise_scale=0.2,
                            validation_split=0.2, verbose=True, early_stopping_patience=10,
                            random_seed=None, early_stopping=True, samples_per_epoch=10, keep_opt_point=False, sr_mode=False):
        """
        Train the LLR network using on-the-fly generated data from neutrino events.
        
        This version splits the neutrino events for train/validation rather than 
        optimization points. Each batch samples random optimization points and 
        random events to create training samples.
        
        Parameters:
        -----------
        optimization_points : torch.Tensor or None
            Points at which to evaluate/optimize, shape (N, dim). If None,
            random points will be sampled from the domain for each batch.
        signal_surrogate_func : callable or None
            Function that takes event parameters and optimization point, returns light yield for signal events
        background_surrogate_func : callable or None
            Function that takes event parameters and optimization point, returns light yield for background events
        signal_event_params : dict or None
            Dictionary containing signal event parameters (see generate_batch_data_from_events)
        background_event_params : dict or None
            Dictionary containing background event parameters
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        signal_ratio : float
            Fraction of signal events in each batch
        add_noise : bool
            Whether to add noise to light yield values
        signal_noise_scale : float
            Standard deviation of noise to add to signal light yields
        background_noise_scale : float
            Standard deviation of noise to add to background light yields
        validation_split : float
            Fraction of neutrino events to use for validation
        verbose : bool
            Whether to print training progress
        early_stopping_patience : int
            Epochs to wait for improvement before early stopping
        random_seed : int or None
            Random seed for reproducibility
        early_stopping : bool
            Whether to use early stopping based on validation loss
        samples_per_event : int
            Number of optimization points to sample per event in each batch
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Convert optimization points to tensor if provided and ensure float32
        if optimization_points is not None:
            if not isinstance(optimization_points, torch.Tensor):
                optimization_points = torch.tensor(optimization_points, device=self.device, dtype=torch.float32)
            else:
                optimization_points = optimization_points.float().to(self.device)
        
        # Split neutrino events into train and validation
        # Determine total number of signal and background events
        n_signal_events = len(signal_event_params['position']) if signal_event_params else 0
        n_background_events = len(background_event_params['position']) if background_event_params else 0
        
        # Split signal events
        if n_signal_events > 0:
            n_val_signal = int(n_signal_events * validation_split)
            signal_perm = torch.randperm(n_signal_events)
            train_signal_indices = signal_perm[:-n_val_signal].tolist() if n_val_signal > 0 else signal_perm.tolist()
            val_signal_indices = signal_perm[-n_val_signal:].tolist() if n_val_signal > 0 else []
        else:
            train_signal_indices = []
            val_signal_indices = []
        
        # Split background events
        if n_background_events > 0:
            n_val_background = int(n_background_events * validation_split)
            background_perm = torch.randperm(n_background_events)
            train_background_indices = background_perm[:-n_val_background].tolist() if n_val_background > 0 else background_perm.tolist()
            val_background_indices = background_perm[-n_val_background:].tolist() if n_val_background > 0 else []
        else:
            train_background_indices = []
            val_background_indices = []
        
        # Create filtered event parameters for train and validation
        def filter_event_params(event_params, indices):
            if event_params is None or len(indices) == 0:
                return None
            
            filtered_params = {}
            for key, values in event_params.items():
                if isinstance(values, torch.Tensor):
                    filtered_params[key] = values[indices]
                elif isinstance(values, (list, np.ndarray)):
                    filtered_params[key] = torch.tensor([values[i] for i in indices], 
                                                      device=self.device, dtype=torch.float32)
            return filtered_params
        
        train_signal_params = filter_event_params(signal_event_params, train_signal_indices)
        val_signal_params = filter_event_params(signal_event_params, val_signal_indices)
        train_background_params = filter_event_params(background_event_params, train_background_indices)
        val_background_params = filter_event_params(background_event_params, val_background_indices)
        
        # Create training event indices (just range since we've already filtered the params)
        train_sig_indices = list(range(len(train_signal_indices))) if train_signal_params else []
        train_bg_indices = list(range(len(train_background_indices))) if train_background_params else []
        val_sig_indices = list(range(len(val_signal_indices))) if val_signal_params else []
        val_bg_indices = list(range(len(val_background_indices))) if val_background_params else []
        
        # Create data loaders
        train_loader = self.generate_batch_data_from_events(
            event_indices=None,  # Use all filtered events
            optimization_points=optimization_points,
            signal_surrogate_func=signal_surrogate_func, 
            background_surrogate_func=background_surrogate_func,
            signal_event_params=train_signal_params, 
            background_event_params=train_background_params,
            batch_size=batch_size, signal_ratio=signal_ratio, add_noise=add_noise, 
            sig_noise_scale=signal_noise_scale, bkg_noise_scale=background_noise_scale, 
            random_seed=random_seed, samples_per_epoch=samples_per_epoch, keep_opt_point=keep_opt_point, sr_mode=sr_mode
        )
        
        val_loader = self.generate_batch_data_from_events(
            event_indices=None,  # Use all filtered events
            optimization_points=optimization_points,
            signal_surrogate_func=signal_surrogate_func, 
            background_surrogate_func=background_surrogate_func,
            signal_event_params=val_signal_params, 
            background_event_params=val_background_params,
            batch_size=min(batch_size, len(val_sig_indices) + len(val_bg_indices)) if (val_sig_indices or val_bg_indices) else batch_size, 
            signal_ratio=signal_ratio, add_noise=add_noise, 
            sig_noise_scale=signal_noise_scale, bkg_noise_scale=background_noise_scale, 
            random_seed=random_seed, samples_per_epoch=samples_per_epoch, keep_opt_point=keep_opt_point, sr_mode=sr_mode
        )
        
        # Get a sample batch to determine feature dimension
        sample_batch = next(iter(train_loader))
        raw_feature_dim = sample_batch[0].shape[1]
        
        # Build network if not already built
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            self._build_network(raw_feature_dim)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            if self.shared_mlp:
                self.shared_branch_mlp.train()
            else:
                for branch in self.mlp_branches:
                    branch.train()
            self.final_mlp.train()
            if self.fourier_features_list is not None:
                for fourier_layer in self.fourier_features_list:
                    fourier_layer.train()
                
            train_loss = 0.0
            n_batches = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self._forward_pass(batch_features)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation phase
            if self.shared_mlp:
                self.shared_branch_mlp.eval()
            else:
                for branch in self.mlp_branches:
                    branch.eval()
            self.final_mlp.eval()
            if self.fourier_features_list is not None:
                for fourier_layer in self.fourier_features_list:
                    fourier_layer.eval()
                
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    val_outputs = self._forward_pass(batch_features)
                    loss = self.loss_fn(val_outputs, batch_labels)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            val_loss /= n_val_batches if n_val_batches > 0 else 1
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_state_dict = {
                        'mlp_branches': [branch.state_dict().copy() for branch in self.mlp_branches] if not self.shared_mlp else None,
                        'shared_branch_mlp': self.shared_branch_mlp.state_dict().copy() if self.shared_mlp else None,
                        'final_mlp': self.final_mlp.state_dict().copy(),
                        'fourier_features_list': [fourier_layer.state_dict().copy() for fourier_layer in self.fourier_features_list] if self.fourier_features_list is not None else None
                    }
                else:
                    patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                if self.shared_mlp:
                    self.shared_branch_mlp.load_state_dict(self.best_state_dict['shared_branch_mlp'])
                else:
                    for i, branch in enumerate(self.mlp_branches):
                        branch.load_state_dict(self.best_state_dict['mlp_branches'][i])
                self.final_mlp.load_state_dict(self.best_state_dict['final_mlp'])
                if self.fourier_features_list is not None and self.best_state_dict['fourier_features_list'] is not None:
                    for i, fourier_layer in enumerate(self.fourier_features_list):
                        fourier_layer.load_state_dict(self.best_state_dict['fourier_features_list'][i])
                break
        
        self.is_trained = True
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }


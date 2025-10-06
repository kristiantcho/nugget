from nugget.surrogates.base_surrogate import Surrogate
import torch
import numpy as np

class SkewedGaussian(Surrogate):
    
    def __init__(self, device=None, dim=3, domain_size=2, background=False, **kwargs):
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
        self.kwargs = kwargs
    
    
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
        motion_dir = motion_dir / torch.norm(motion_dir, dim=-1, keepdim=True)
        
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
            # Ensure motion_dir is 1D for cross product operations
            if motion_dir.dim() > 1:
                motion_dir = motion_dir.squeeze()
                
            # First perpendicular vector (using cross product with a reference vector)
            ref_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=motion_dir.dtype)
            if torch.allclose(motion_dir, ref_vec, atol=1e-6) or torch.allclose(motion_dir, -ref_vec, atol=1e-6):
                ref_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=motion_dir.dtype)
                
            perp_dir1 = torch.linalg.cross(motion_dir, ref_vec)
            perp_dir1 = perp_dir1 / torch.norm(perp_dir1)
            
            # Second perpendicular vector
            perp_dir2 = torch.linalg.cross(motion_dir, perp_dir1)
            perp_dir2 = perp_dir2 / torch.norm(perp_dir2)
            perp_dir2 = perp_dir2 / torch.norm(perp_dir2, dim=-1, keepdim=True)
            
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
                center = position.squeeze()  # Remove batch dimensions to get shape [3]
            else:
                center = torch.tensor(position, device=self.device).squeeze()
        
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
                    phi_val = phi.squeeze()  # Ensure scalar
                else:
                    phi_val = torch.tensor(phi, device=self.device)
                
                if isinstance(theta, torch.Tensor):
                    theta_val = theta.squeeze()  # Ensure scalar
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
    
    def light_yield_surrogate(self, **kwargs):
        """
        Surrogate function that computes light yield using SkewedGaussian.
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'zenith', 'azimuth', 'energy', etc.
        opt_point : torch.Tensor
            Optimization point where light yield is evaluated
            
        Returns:
        --------
        torch.Tensor
            Light yield value at the optimization point
        """
        # Extract event parameters
        # event_center = event_params['position']
        # zenith = event_params['zenith']
        # azimuth = event_params['azimuth'] 
        
        # energy = event_params.get('energy', torch.tensor(1.0))
        
            # Handle gradient case - preserve gradients
        opt_point = kwargs.get('opt_point', None)
        event_params = kwargs.get('event_params', None)
        event_center = event_params.get('position', None)
        zenith = event_params.get('zenith', None)
        azimuth = event_params.get('azimuth', None)
        energy = event_params.get('energy', None)
        theta = zenith  # polar angle from z-axis
        phi = azimuth   # azimuthal angle
        # sigma_front = 0.1 / torch.sqrt(energy + 0.1)
        # sigma_back = 0.02 / torch.sqrt(energy + 0.1) 
        # sigma_perp = 0.05 / torch.sqrt(energy + 0.1)
        
        # Generate the SkewedGaussian function for this event
        event_function = self.__call__(
            amp=energy,
            position=event_center,
            phi=phi,
            theta=theta,
            sigma_factor=self.kwargs.get('sigma_factor', 10)
            # sigma_front=sigma_front,
            # sigma_back=sigma_back,
            # sigma_perp=sigma_perp
        )
        
        # Evaluate the function at the optimization point
        # For gradient computation, avoid operations that break gradient flow
        
        # Ensure opt_point has the right shape without breaking gradients
       
        opt_point_input = opt_point
        light_yield = event_function(opt_point_input)
        return light_yield
      
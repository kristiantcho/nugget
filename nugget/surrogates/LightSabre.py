from nugget.surrogates.base_surrogate import Surrogate
import torch
import numpy as np


class LightSabre(Surrogate):
    """
    LightSabre light yield surrogate model based on the neutrino-effective-area implementation.
    
    This model calculates Cherenkov light yield from muon tracks using a physically-motivated
    parametrization that accounts for distance-dependent attenuation and angular effects.
    
    Reference: https://github.com/PLEnuM-group/neutrino-effective-area
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, 
                 effective_photocathode_area=84e-4, lambda_p=25.0, lambda_mu=3.0, **kwargs):
        """
        Initialize the LightSabre surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (must be 3D for this model)
        domain_size : int
            Length of the domain
        effective_photocathode_area : float
            Effective photocathode area in m^2 (default: 84e-4 m^2)
        lambda_p : float
            Photon absorption length in meters (default: 25 m)
        lambda_mu : float
            Muon scattering length in meters (default: 3 m)
        """
        if dim != 3:
            raise ValueError("LightSabre model only supports 3D geometry")
        
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        self.effective_photocathode_area = effective_photocathode_area
        self.lambda_p = lambda_p
        self.lambda_mu = lambda_mu
        self.kwargs = kwargs
        
        # Polynomial coefficients for photons per meter calculation (reversed order for polyval)
        self.photons_per_m_coeffs = torch.tensor([4.9489616410707695,
            -2.4858046180252362,
            1.1885034976827853,
            -0.2015848374875856,
            0.01626011917463439,
            -0.0004947317263294414]
        , device=self.device)
        
        self.min_energy = 1e2  # 100 GeV
        self.max_energy = 1e8  # 100 PeV
    
    def lightsabre_photons_per_m(self, energy):
        """
        Calculate Cherenkov photon yield per unit length using Lightsabre model.
        
        Parameters:
        -----------
        energy : torch.Tensor
            Muon energy in GeV
            
        Returns:
        --------
        torch.Tensor
            Photon yield in photons/m (300-800 nm wavelength range)
        """
        # Clamp energy to valid range
        e_sane = torch.clamp(energy, self.min_energy, self.max_energy)
        
        # Log10 of energy
        log_e = torch.log10(e_sane)
        
        # Polynomial evaluation (manual implementation since torch doesn't have polyval)
        # Coefficients are in order: c0, c1, c2, ..., c5 for c0 + c1*x + c2*x^2 + ...
        poly_result = self.photons_per_m_coeffs[0]
        for i in range(1, len(self.photons_per_m_coeffs)):
            poly_result = poly_result + self.photons_per_m_coeffs[i] * (log_e ** i)
        
        lightyield = 10.0 ** poly_result
        
        # Bare Cherenkov photons per meter (300-800 nm wavelength range)
        photons_per_m = 40528.49371849151
        
        # Energy-dependent correction factors
        lamb = 0.1879
        kappa = 0.02055
        
        ladd = lamb + kappa * torch.log(e_sane)
        lightyield_bare = (1 + ladd) * photons_per_m
        
        # Total light yield: polynomial fit + bare Cherenkov yield
        total_lightyield = lightyield + lightyield_bare
        
        return total_lightyield
    
    def distance_to_line(self, points, track_pos, track_dir):
        """
        Calculate perpendicular distance from points to a line (muon track).
        
        Parameters:
        -----------
        points : torch.Tensor
            Points as array of shape (N, 3) or (3,)
        track_pos : torch.Tensor
            Point on line as array of shape (3,)
        track_dir : torch.Tensor
            Line direction vector as array of shape (3,)
            
        Returns:
        --------
        torch.Tensor
            Perpendicular distances as scalar or array of shape (N,)
        """
        # Ensure points is 2D
        if points.dim() == 1:
            points = points.unsqueeze(0)
        
        # Normalize direction vector
        track_dir = track_dir / torch.norm(track_dir, dim=-1, keepdim=True)
        
        # Calculate cross product: (p - x0) × d
        diff = points - track_pos.unsqueeze(0)
        cross_product = torch.cross(diff, track_dir.unsqueeze(0).expand(diff.shape[0], -1), dim=1)
        
        # Calculate norm of cross product
        distances = torch.norm(cross_product, dim=1)
        
        # Return scalar if input was 1D
        if distances.shape[0] == 1:
            return distances.squeeze(0)
        
        return distances
    
    def lightyield_for_distance(self, distance, energy):
        """
        Calculate expected photon count based on perpendicular distance from track.
        
        This implements the LightSabre parametrization with exponential attenuation
        and geometric factors.
        
        Parameters:
        -----------
        distance : torch.Tensor
            Perpendicular distance from track to optical module in meters
        energy : torch.Tensor
            Muon energy in GeV
            
        Returns:
        --------
        torch.Tensor
            Expected number of detected photons
        """
        # Get photons per meter for this energy (l0 in the original code)
        l0 = self.lightsabre_photons_per_m(energy)
        
        # Cherenkov angle for ice (n=1.33)
        theta_c = torch.acos(torch.tensor(1.0/1.33, device=self.device))
        sin_theta_c = torch.sin(theta_c)
        
        # Optical parameters for ice
        lambda_abs = 125.0  # Absorption length in meters
        lambda_sca = 33.0   # Scattering length in meters
        
        # Calculate effective photon propagation length
        lambda_p = torch.sqrt(torch.tensor(lambda_abs * lambda_sca / 3.0, device=self.device))
        
        # Scattering parameter
        zeta = torch.exp(torch.tensor(-lambda_sca / lambda_abs, device=self.device))
        lambda_c = lambda_sca / (3.0 * zeta)
        
        # Muon scattering effective length
        lambda_mu = (lambda_c / sin_theta_c**2 * 2.0 / (np.pi * lambda_p))
        
        # Avoid division by zero
        distance_safe = torch.clamp(distance, min=1e-6)
        
        # LightSabre formula with attenuation and geometric factors
        numerator = l0 * self.effective_photocathode_area * (1.0 / (2.0 * np.pi * sin_theta_c))
        numerator = numerator * torch.exp(-distance_safe / lambda_p)
        
        denominator = (torch.sqrt(lambda_mu * distance_safe) * 
                      torch.tanh(torch.sqrt(distance_safe / lambda_mu)))
        
        light_yield = numerator / denominator
        
        return light_yield
    
    def __call__(self, track_pos=None, track_dir=None, track_energy=None, om_positions=None,
                 test_points=None, **kwargs):
        """
        Calculate light yield at optical module positions from a muon track.
        
        Parameters:
        -----------
        track_pos : torch.Tensor
            Track position as array of shape (3,)
        track_dir : torch.Tensor
            Track direction as array of shape (3,)
        track_energy : torch.Tensor or float
            Track energy in GeV
        om_positions : torch.Tensor
            Optical module positions as array of shape (N, 3)
        test_points : torch.Tensor (optional, alias for om_positions)
            Alternative parameter name for optical module positions
            
        Returns:
        --------
        torch.Tensor
            Expected photon counts at each optical module position
        """
        # Handle test_points as alias for om_positions
        if om_positions is None and test_points is not None:
            om_positions = test_points
        
        # Validate inputs
        if track_pos is None or track_dir is None or track_energy is None or om_positions is None:
            raise ValueError("track_pos, track_dir, track_energy, and om_positions must be provided")
        
        # Ensure tensors are on the correct device
        if isinstance(track_pos, torch.Tensor):
            track_pos = track_pos.to(self.device)
        else:
            track_pos = torch.tensor(track_pos, device=self.device)
        
        if isinstance(track_dir, torch.Tensor):
            track_dir = track_dir.to(self.device)
        else:
            track_dir = torch.tensor(track_dir, device=self.device)
        
        if isinstance(track_energy, torch.Tensor):
            track_energy = track_energy.to(self.device)
        else:
            track_energy = torch.tensor(track_energy, device=self.device)
        
        if isinstance(om_positions, torch.Tensor):
            om_positions = om_positions.to(self.device)
        else:
            om_positions = torch.tensor(om_positions, device=self.device)
        
        # Squeeze to remove batch dimensions
        track_pos = track_pos.squeeze()
        track_dir = track_dir.squeeze()
        if track_energy.dim() > 0:
            track_energy = track_energy.squeeze()
        
        # Calculate perpendicular distances from track to optical modules
        distances = self.distance_to_line(om_positions, track_pos, track_dir)
        
        # Calculate light yield at each distance
        light_yield = self.lightyield_for_distance(distances, track_energy)
        
        return light_yield
    
    def light_yield_surrogate(self, **kwargs):
        """
        Surrogate function that computes light yield using LightSabre model.
        
        This method provides a consistent interface with other surrogate models
        and can be used in optimization workflows.
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'zenith', 'azimuth', 'energy'
        opt_point : torch.Tensor
            Optimization point where light yield is evaluated (single point or array)
            
        Returns:
        --------
        torch.Tensor
            Light yield value(s) at the optimization point(s)
        """
        # Extract parameters
        opt_point = kwargs.get('opt_point', None)
        event_params = kwargs.get('event_params', None)
        
        if event_params is None:
            raise ValueError("event_params must be provided")
        
        if opt_point is None:
            raise ValueError("opt_point must be provided")
        
        # Extract event parameters
        track_pos = event_params.get('position', None)
        zenith = event_params.get('zenith', None)
        azimuth = event_params.get('azimuth', None)
        energy = event_params.get('energy', None)
        
        if track_pos is None or zenith is None or azimuth is None or energy is None:
            raise ValueError("event_params must contain 'position', 'zenith', 'azimuth', and 'energy'")
        
        # Convert spherical angles to Cartesian direction
        if isinstance(zenith, torch.Tensor):
            theta = zenith.squeeze()
            phi = azimuth.squeeze()
        else:
            theta = torch.tensor(zenith, device=self.device).squeeze()
            phi = torch.tensor(azimuth, device=self.device).squeeze()
        
        # Ensure theta and phi are scalars (0-dimensional tensors)
        if theta.dim() > 0:
            theta = theta.squeeze()
        if phi.dim() > 0:
            phi = phi.squeeze()
        
        track_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ])
        
        # Call the main function
        light_yield = self.__call__(
            track_pos=track_pos,
            track_dir=track_dir,
            track_energy=energy,
            om_positions=opt_point
        )
        
        return light_yield

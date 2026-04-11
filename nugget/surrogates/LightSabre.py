from nugget.surrogates.base_surrogate import Surrogate
import torch
import numpy as np
from nugget.surrogates.pandel import Pandel, CPandel
# from nugget.surrogates.cpandel import cpandel_gen as CPandel


class LightSabre(Surrogate):
    """
    LightSabre light yield surrogate model based on the neutrino-effective-area implementation.
    
    This model calculates Cherenkov light yield from muon tracks using a physically-motivated
    parametrization that accounts for distance-dependent attenuation and angular effects.
    
    Reference: https://github.com/PLEnuM-group/neutrino-effective-area (muons/lightsabre)
    Reference: https://git.ecap.work/tu98saji/master-thesis-code (electron/cascade)
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, 
                 effective_photocathode_area=84e-4, **kwargs):
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
        self.refractive_index = self.kwargs.get('n_refraction', 1.3)
        self.beta = self.kwargs.get('beta', 3e-6) # cm^2/g
        self.alpha = self.kwargs.get('alpha', 2e-3) # GeV cm^2/g
        self.end_energy = self.kwargs.get('end_energy', 10) # GeV
        self.med_density = self.kwargs.get('med_density', 0.92) # g/cm^3 (default ice, 1.03 for water)
        self.use_max_energy_dist = self.kwargs.get('use_max_energy_dist', True)
        self.n0A = self.kwargs.get('n0A', 63894457.33843762) # fit-value from photon_yield-notebook
        self.particle_mode = self.kwargs.get('particle_mode', 'track') # 'track' or 'cascade'
        self.scattering_tau = self.kwargs.get('scattering_tau', 0.924) # from ANTARES water measurements, tau =  <cos(theta_sca)>, where theta_sca is the scattering angle
        self.poisson_rate_cap = float(self.kwargs.get('poisson_rate_cap', 1e8))

    def _sanitize_rate_for_poisson(self, rate):
        """Ensure Poisson rate is finite and non-negative to avoid CUDA kernel faults."""
        rate = torch.nan_to_num(rate, nan=0.0, posinf=self.poisson_rate_cap, neginf=0.0)
        return torch.clamp(rate, min=0.0, max=self.poisson_rate_cap)
    
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
    
    def lightyield_for_distance_cascade(self, r, rand_e):
        """calculates photon yield for cascade event with energy rand_e at distance r"""
        # constants for photon yield function
        lamda_a = self.kwargs.get('lambda_abs', 44.7) # absorption length in m at 450 nm
        lamda_e = self.kwargs.get('lambda_sca', 57.4)/(1-self.scattering_tau) # effective scattering length in m, considering isotropic scattering where lambda_s = effective scattering length, 
        Zeta = np.exp(-lamda_e/lamda_a)
        lamda_p = np.sqrt(lamda_a*lamda_e/3)/1.07 # characteristic propagation length with correction 1.07
        lamda_c = lamda_e/(3*Zeta)
        r_safe = torch.clamp(r, min=1e-6)
        photon_yield = self.n0A * 1/(4*np.pi) * torch.exp(-r_safe/lamda_p) * 1/(lamda_c * r_safe * torch.tanh(r_safe/lamda_c))
        photon_yield_resized = photon_yield * rand_e/3e5 # divide by 300TeV = 3e5 GeV
        photon_yield_resized = torch.nan_to_num(photon_yield_resized, nan=0.0, posinf=self.poisson_rate_cap, neginf=0.0)
        photon_yield_resized = torch.clamp(photon_yield_resized, min=0.0)
        
        return photon_yield_resized
    
    def get_max_energy_dist(self, energy):
        """
        Calculate the maximum distance at which a given muon with intial energy can still produce detectable light.
        
        Parameters:
        -----------
        energy : torch.Tensor
            Muon energy in GeV
            
        Returns:
        --------
        torch.Tensor
            Maximum distance in meters
        """
        epsilon = self.alpha/self.beta
        
        return (1/self.beta)*torch.log((energy + epsilon)/(self.end_energy + epsilon))/(100*self.med_density)
    
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
        dir_norm = torch.norm(track_dir, dim=-1, keepdim=True).clamp_min(1e-12)
        track_dir = track_dir / dir_norm
        track_dir = torch.nan_to_num(track_dir, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        theta_c = torch.acos(torch.tensor(1.0/self.refractive_index))
        sin_theta_c = torch.sin(theta_c)
        
        # Optical parameters for waterz
        lambda_abs = self.kwargs.get('lambda_abs', 44.7)  # Absorption length in meters
        lambda_sca = self.kwargs.get('lambda_sca', 57.4)/(1-self.scattering_tau)   # Scattering length in meters
        
        # Calculate effective photon propagation length
        lambda_p = torch.sqrt(torch.tensor(lambda_abs * lambda_sca / 3.0, device=self.device))
        
        # Scattering parameter
        zeta = torch.exp(torch.tensor(-lambda_sca / lambda_abs, device=self.device)).clamp_min(1e-12)
        lambda_c = lambda_sca / (3.0 * zeta)
        
        # Muon scattering effective length
        lambda_mu = (lambda_c / sin_theta_c**2 * 2.0 / (np.pi * lambda_p))
        lambda_mu = torch.clamp(lambda_mu, min=1e-12)
        
        # Avoid division by zero
        distance_safe = torch.clamp(distance, min=1e-6)
        
        # LightSabre formula with attenuation and geometric factors
        numerator = l0 * self.effective_photocathode_area * (1.0 / (2.0 * np.pi * sin_theta_c))
        numerator = numerator * torch.exp(-distance_safe / lambda_p)
        
        denominator = (torch.sqrt(lambda_mu * distance_safe) * 
                      torch.tanh(torch.sqrt(distance_safe / lambda_mu)))
        denominator = torch.clamp(denominator, min=1e-12)
        
        light_yield = numerator / denominator
        light_yield = torch.nan_to_num(light_yield, nan=0.0, posinf=self.poisson_rate_cap, neginf=0.0)
        light_yield = torch.clamp(light_yield, min=0.0)
        
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
        if self.particle_mode == 'track':
            distances = self.distance_to_line(om_positions, track_pos, track_dir)
            
            # Calculate light yield at each distance
            light_yield = self.lightyield_for_distance(distances, track_energy)
        else:
            # For cascade mode, calculate distance from cascade vertex and use cascade light yield
            distances = torch.norm(om_positions - track_pos.unsqueeze(0), dim=1)
            light_yield = self.lightyield_for_distance_cascade(distances, track_energy)
        light_yield = torch.nan_to_num(light_yield, nan=0.0, posinf=self.poisson_rate_cap, neginf=0.0)
        light_yield = torch.clamp(light_yield, min=0.0)
        return light_yield
    
    def light_yield_surrogate(self, **kwargs):
        """
        Surrogate function that computes light yield using LightSabre model.
        
        This method provides a consistent interface with other surrogate models
        and can be used in optimization workflows.
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'energy', and either:
            - 'direction': Cartesian direction vector of shape (3,). If present,
              this is used directly and 'zenith'/'azimuth' are ignored.
            - 'zenith' and 'azimuth': spherical angles in radians, used only
              when 'direction' is not in event_params.
        opt_point : torch.Tensor
            Optimization point where light yield is evaluated (single point or array)
        gradient_mode : bool
            If True, enables gradient tracking (requires_grad=True) on all event
            parameter tensors (position, energy, direction or zenith/azimuth).
            Can also be set via the 'gradient_mode' constructor kwarg. Default: False.
            
        Returns:
        --------
        torch.Tensor
            Light yield value(s) at the optimization point(s)
        """
        # Extract parameters
        opt_point = kwargs.get('opt_point', None)
        event_params = kwargs.get('event_params', None)
        gradient_mode = kwargs.get('gradient_mode', self.kwargs.get('gradient_mode', False))
        
        if event_params is None:
            raise ValueError("event_params must be provided")
        
        if opt_point is None:
            raise ValueError("opt_point must be provided")
        
        # Extract event parameters
        track_pos = event_params.get('position', None)
        energy = event_params.get('energy', None)
        angular_dir = event_params.get('direction', None)
        
        if track_pos is None or energy is None:
            raise ValueError("event_params must contain 'position' and 'energy'")
        
        # Convert position to tensor and optionally enable gradients
        if not isinstance(track_pos, torch.Tensor):
            track_pos = torch.tensor(track_pos, dtype=torch.float32, device=self.device)
        else:
            track_pos = track_pos.to(self.device)
        if gradient_mode:
            track_pos = track_pos.requires_grad_(True)
        
        # Convert energy to tensor and optionally enable gradients
        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, dtype=torch.float32, device=self.device)
        else:
            energy = energy.to(self.device)
        if gradient_mode:
            energy = energy.requires_grad_(True)
        
        # Build track direction: prefer 'direction', fall back to zenith/azimuth
        if angular_dir is not None:
            if not isinstance(angular_dir, torch.Tensor):
                track_dir = torch.tensor(angular_dir, dtype=torch.float32, device=self.device).squeeze()
            else:
                track_dir = angular_dir.to(self.device).squeeze()
            if gradient_mode:
                track_dir = track_dir.requires_grad_(True)
        else:
            zenith = event_params.get('zenith', None)
            azimuth = event_params.get('azimuth', None)
            if zenith is None or azimuth is None:
                raise ValueError(
                    "event_params must contain either 'direction' or both 'zenith' and 'azimuth'"
                )
            if not isinstance(zenith, torch.Tensor):
                theta = torch.tensor(zenith, dtype=torch.float32, device=self.device).squeeze()
            else:
                theta = zenith.to(self.device).squeeze()
            if not isinstance(azimuth, torch.Tensor):
                phi = torch.tensor(azimuth, dtype=torch.float32, device=self.device).squeeze()
            else:
                phi = azimuth.to(self.device).squeeze()
            if gradient_mode:
                theta = theta.requires_grad_(True)
                phi = phi.requires_grad_(True)
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
        if self.kwargs.get('use_poisson', False):
            light_yield = torch.poisson(self._sanitize_rate_for_poisson(light_yield))
        
        return light_yield
    
    
class LightSabrePATD(LightSabre):
    
    """
    LightSabrePATD extends LightSabre to generate photon arrival time distributions
    using the CPandel model.
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, **kwargs):
        """
        Initialize the LightSabrePATD surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (must be 3D for this model)
        domain_size : int
            Length of the domain
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size, **kwargs)    
    
    
    
    def light_yield_surrogate(self, **kwargs):
        """
        Generate photon arrival time distribution using CPandel model.
        
        This method samples photon hit times by:
        1. Using light_yield_surrogate to get expected photon count N
        2. Sampling N points along the track weighted by light yield
        3. Computing geometric time and residual time using CPandel
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'zenith', 'azimuth', 'energy'
        opt_point : torch.Tensor
            Detector position (single point)
        n_refraction : float
            Refractive index (default: 1.3)
        v_mu : float
            Muon velocity in m/ns (default: speed of light c)
        num_track_points : int
            Number of points to sample along track (default: 1000)
        cpandel_params : dict
            Parameters for CPandel (tau, lambda_s, lambda_a, v, s)
        max_photons : int or None
            Maximum number of photons to sample. If provided, samples min(N, max_photons)
            where N is the expected light yield. This speeds up computation when only
            a limited number of photons are needed.
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'hit_times': Array of photon hit times
            - 'num_photons': Number of photons actually sampled
            - 'expected_photons': Expected light yield N (before limiting)
            - 'residual_times': CPandel residual times
        """
        # Extract parameters
        opt_point = kwargs.get('opt_point', None)
        event_params = kwargs.get('event_params', None)
        max_photons = kwargs.get('max_photons', None)
        get_patd_probs = kwargs.get('get_patd_probs', False)
        
        c = 0.299792458  # speed of light in m/ns
        v_mu = self.kwargs.get('v_mu', c)
        num_track_points = max(int(self.kwargs.get('num_track_points', 1000)), 1)
        cpandel_params = self.kwargs.get('cpandel_params', {})
        
        if event_params is None or opt_point is None:
            raise ValueError("event_params and opt_point must be provided")
        
        # Extract event parameters
        track_pos = event_params.get('position', None)
        zenith = event_params.get('zenith', None)
        azimuth = event_params.get('azimuth', None)
        energy = event_params.get('energy', None)
        angular_dir = event_params.get('direction', None)
        
        # Convert to tensors
        if isinstance(track_pos, torch.Tensor):
            track_pos = track_pos.to(self.device).squeeze()
        else:
            track_pos = torch.tensor(track_pos, device=self.device).squeeze()
        
        if isinstance(opt_point, torch.Tensor):
            detector_pos = opt_point.to(self.device).squeeze()
        else:
            detector_pos = torch.tensor(opt_point, device=self.device).squeeze()
        
        # Convert spherical angles to Cartesian direction
        if isinstance(zenith, torch.Tensor):
            theta = zenith.squeeze()
            phi = azimuth.squeeze()
        else:
            theta = torch.tensor(zenith, device=self.device).squeeze()
            phi = torch.tensor(azimuth, device=self.device).squeeze()
        if angular_dir is not None:
            if isinstance(angular_dir, torch.Tensor):
                track_dir = angular_dir.to(self.device).squeeze()
            else:
                track_dir = torch.tensor(angular_dir, device=self.device).squeeze()
        else:
            track_dir = torch.stack([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta)
            ])
            track_dir = track_dir / torch.norm(track_dir)  # Normalize
        
        # Get expected number of photons at detector
        if self.kwargs.get('input_photons', None) is None:
            light_yield = self.__call__(
                track_pos=track_pos,
                track_dir=track_dir,
                track_energy=energy,
                om_positions=opt_point
            )
            if self.kwargs.get('use_poisson', False):
                light_yield = torch.poisson(light_yield)
        else:
            light_yield = torch.tensor(self.kwargs.get('input_photons', None), device=self.device)   
        
        expected_N = torch.round(light_yield).int().item()
  
        
        # Limit sampling if max_photons is provided
        if max_photons is not None and max_photons < expected_N:
            N = max_photons
        else:
            N = expected_N
        
        if N <= 0:
            return {'hit_times': torch.tensor([], device=self.device), 'num_photons': 0, 'expected_photons': expected_N, 'residual_times': torch.tensor([], device=self.device)}
        # Find the foot of the perpendicular from detector to track
        # Foot point is: track_pos + t_foot * track_dir where t_foot = (detector_pos - track_pos) · track_dir
        to_detector = detector_pos - track_pos
        t_foot = torch.dot(to_detector, track_dir) #distance along track to foot point
        foot_length = torch.norm(torch.linalg.cross(to_detector, track_dir))/torch.norm(track_dir)
        if not self.use_max_energy_dist:
        # Get the maximum distance S along track from foot point in both directions
            S = self.kwargs.get('track_segment_length', 200.0)  # Default 200m
            
            # Determine the range along the track:
            # - Backwards from foot: [t_foot - S, t_foot]
            # - Forwards from foot: [t_foot, t_foot + S]
            # But must not extend backwards past the interaction vertex (t=0)
            
            t_min = torch.max(torch.tensor(0.0, device=self.device), t_foot - S)  # Don't go past vertex
            t_max = t_foot + S
        
        else:
            t_min = 0
            t_max = self.get_max_energy_dist(energy).squeeze()
        
        # Sample points along the track from t_min to t_max
        t_vals = torch.linspace(t_min, t_max, num_track_points, device=self.device)
        track_points = track_pos.unsqueeze(0) + t_vals.unsqueeze(1) * track_dir.unsqueeze(0)
        
        # Calculate distances from each track point to detector
        distances = torch.norm(track_points - detector_pos.unsqueeze(0), dim=1)
        # print(distances[::50])
        
        # Cherenkov angle for water
        theta_c = torch.acos(torch.tensor(1.0/self.refractive_index, device=self.device))
        sin_theta_c = torch.sin(theta_c)
        
        # Optical parameters for water from Tobias K.
        lambda_abs = 44.7
        lambda_sca = 57.4
        lambda_p = torch.sqrt(torch.tensor(lambda_abs * lambda_sca / 3.0, device=self.device))
        zeta = torch.exp(torch.tensor(-lambda_sca / lambda_abs, device=self.device))
        lambda_c = lambda_sca / (3.0 * zeta)
        lambda_mu = (lambda_c / sin_theta_c**2 * 2.0 / (np.pi * lambda_p))
        
        # Avoid singularities when detector lies on/very near the sampled track.
        distances_safe = torch.clamp(distances, min=1e-6)
        # Calculate weights (unnormalized probabilities)
        numerator = (1.0 / (2.0 * np.pi * sin_theta_c))
        numerator = numerator * torch.exp(-distances_safe / lambda_p)
        
        denominator = (torch.sqrt(lambda_mu * distances_safe) * 
                      torch.tanh(torch.sqrt(distances_safe / lambda_mu)))
        
        weights = numerator / denominator
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)

        # Normalize weights to create probability distribution.
        # If the formula degenerates numerically, fall back to uniform sampling.
        weight_sum = torch.sum(weights)
        if (not torch.isfinite(weight_sum)) or (weight_sum <= 0):
            weights = torch.full_like(weights, 1.0 / weights.numel())
        else:
            weights = weights / weight_sum
        
        # Sample N points along the track according to weights
        # Convert N to int if it's a tensor, otherwise it's already a Python int
        num_samples = int(N.item()) if isinstance(N, torch.Tensor) else int(N)
        sampled_indices = torch.multinomial(weights, num_samples, replacement=True)
        # sampled_track_points = track_points[sampled_indices]
        sampled_t_vals = t_vals[sampled_indices]
        sampled_track_points = track_pos.unsqueeze(0) + sampled_t_vals.unsqueeze(1) * track_dir.unsqueeze(0)
        # Calculate geometric distances for sampled points
        d_geom = torch.norm(sampled_track_points - detector_pos.unsqueeze(0), dim=1)
        
        # Calculate distance along track from vertex (s)
        # sampled_t_vals represents distance from vertex along track direction
        s = sampled_t_vals
        
        # Calculate geometric time: t_geom = d/(c/n) + s/v_mu
        t_geom = d_geom / (c / self.refractive_index) + s / v_mu
        # t_geom_foot = t_foot / (c / self.refractive_index) + foot_length / v_mu
        # t_geom_track = torch.norm(to_detector) / (c / self.refractive_index)
        # t_geom_min = min(t_geom_foot, t_geom_track)
        if v_mu != c/self.refractive_index:
            short_track = t_foot - ((c / self.refractive_index) * foot_length)/torch.sqrt(torch.tensor(v_mu**2 - (c / self.refractive_index)**2, device=self.device))
            t_geom_min = (short_track / v_mu)  +  torch.sqrt((short_track-t_foot)**2 + foot_length**2) / (c / self.refractive_index)
        else:
            t_geom_min = torch.norm(to_detector) / (c / self.refractive_index)
        # Initialize CPandel model
        cpandel = CPandel(
            tau=cpandel_params.get('tau', 557.),
            lambda_s=cpandel_params.get('lambda_s', 33.3),
            lambda_a=cpandel_params.get('lambda_a', 98.),
            v=cpandel_params.get('v', 0.3/1.33),
            s=cpandel_params.get('s', 5.0)
        )
        
        # Sample residual times using CPandel for all geometric distances at once (vectorized)
        # When d is already an array of N distances, don't pass size parameter
        # to avoid creating an N×N matrix
        d_geom_numpy = d_geom.detach().cpu().numpy()
        t_residual_output = cpandel.rvs(d=d_geom_numpy, size=None)
        t_residual_probs = None
        if get_patd_probs:
            t_residual_probs = cpandel.pdf(t_residual_output, d=d_geom_numpy)
            
        # Handle both torch tensor and numpy array outputs
        if isinstance(t_residual_output, torch.Tensor):
            t_residual = t_residual_output.float().to(self.device)
        else:
            t_residual = torch.from_numpy(t_residual_output).float().to(self.device)
        
        # Total hit time = geometric time + residual time
        hit_times = t_geom + t_residual
        
        # need to think of a better return structure here
        
        return {'hit_times': hit_times, 'num_photons': N, 'expected_photons': expected_N, 'residual_times': t_residual, 'geometric_times': t_geom, 't_geom_min': t_geom_min, 'd_geom': d_geom, 'patd_probs': t_residual_probs}

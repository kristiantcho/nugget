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

    def call_batched(self, track_pos, track_dir, track_energy, om_positions):
        """
        Compute light yield for a batch of events and all OM positions in one pass.

        All heavy math (distance, attenuation, polynomial) is vectorised over the
        (n_events, n_points) grid with no Python loop.

        Parameters
        ----------
        track_pos : torch.Tensor, shape (n_events, 3)
        track_dir : torch.Tensor, shape (n_events, 3)  — need not be unit vectors
        track_energy : torch.Tensor, shape (n_events,)
        om_positions : torch.Tensor, shape (n_points, 3)

        Returns
        -------
        torch.Tensor, shape (n_events, n_points)
        """
        track_pos    = track_pos.to(self.device)
        track_dir    = track_dir.to(self.device)
        track_energy = track_energy.to(self.device)
        om_positions = om_positions.to(self.device)

        # Normalise directions: (n_events, 3)
        dir_norm = track_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)
        track_dir = track_dir / dir_norm

        if self.particle_mode == 'track':
            # diff[e, p] = om_positions[p] - track_pos[e]  →  (n_events, n_points, 3)
            diff = om_positions.unsqueeze(0) - track_pos.unsqueeze(1)

            # cross[e, p] = diff[e,p] × track_dir[e]  →  (n_events, n_points, 3)
            cross = torch.linalg.cross(
                diff,
                track_dir.unsqueeze(1).expand_as(diff),
            )
            distances = cross.norm(dim=2)  # (n_events, n_points)

            # lightyield_for_distance lifted to (n_events, n_points)
            l0 = self.lightsabre_photons_per_m(track_energy)  # (n_events,)

            theta_c    = torch.acos(torch.tensor(1.0 / self.refractive_index, device=self.device))
            sin_theta_c = torch.sin(theta_c)
            lambda_abs  = self.kwargs.get('lambda_abs', 44.7)
            lambda_sca  = self.kwargs.get('lambda_sca', 57.4) / (1 - self.scattering_tau)
            lambda_p    = torch.sqrt(torch.tensor(lambda_abs * lambda_sca / 3.0, device=self.device))
            zeta        = torch.exp(torch.tensor(-lambda_sca / lambda_abs, device=self.device)).clamp_min(1e-12)
            lambda_c    = lambda_sca / (3.0 * zeta)
            lambda_mu   = (lambda_c / sin_theta_c ** 2 * 2.0 / (np.pi * lambda_p)).clamp_min(1e-12)

            d_safe = distances.clamp(min=1e-6)  # (n_events, n_points)

            # numerator: (n_events, 1) broadcast over points
            numerator = (l0 * self.effective_photocathode_area / (2.0 * np.pi * sin_theta_c)).unsqueeze(1)
            numerator = numerator * torch.exp(-d_safe / lambda_p)

            denominator = (torch.sqrt(lambda_mu * d_safe) *
                           torch.tanh(torch.sqrt(d_safe / lambda_mu))).clamp_min(1e-12)

            light_yield = numerator / denominator
        else:
            # Cascade: distance from vertex to each OM
            diff = om_positions.unsqueeze(0) - track_pos.unsqueeze(1)  # (n_events, n_points, 3)
            distances = diff.norm(dim=2)                                 # (n_events, n_points)

            lamda_a = self.kwargs.get('lambda_abs', 44.7)
            lamda_e = self.kwargs.get('lambda_sca', 57.4) / (1 - self.scattering_tau)
            lamda_p = np.sqrt(lamda_a * lamda_e / 3) / 1.07
            Zeta    = np.exp(-lamda_e / lamda_a)
            lamda_c = lamda_e / (3 * Zeta)

            r_safe = distances.clamp(min=1e-6)
            photon_yield = (self.n0A / (4 * np.pi) *
                            torch.exp(-r_safe / lamda_p) /
                            (lamda_c * r_safe * torch.tanh(r_safe / lamda_c)))
            # scale by energy: (n_events, 1) broadcast
            light_yield = photon_yield * (track_energy / 3e5).unsqueeze(1)

        light_yield = torch.nan_to_num(light_yield, nan=0.0, posinf=self.poisson_rate_cap, neginf=0.0)
        light_yield = light_yield.clamp(min=0.0)
        return light_yield

    def light_yield_surrogate_batched(self, om_positions, event_params_list):
        """
        Compute light yields for a list of events and all OM positions in one GPU call.

        Parameters
        ----------
        om_positions : torch.Tensor, shape (n_points, 3)
        event_params_list : list of dict
            Each dict must contain 'position', 'energy', and either 'direction' or
            'zenith'/'azimuth'.

        Returns
        -------
        torch.Tensor, shape (n_events, n_points)
        """
        positions, directions, energies = [], [], []
        for ep in event_params_list:
            pos = ep['position']
            if not isinstance(pos, torch.Tensor):
                pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
            positions.append(pos.to(self.device).reshape(3))

            energy = ep['energy']
            if not isinstance(energy, torch.Tensor):
                energy = torch.tensor(energy, dtype=torch.float32, device=self.device)
            energies.append(energy.to(self.device).reshape(()))

            if 'direction' in ep:
                d = ep['direction']
                if not isinstance(d, torch.Tensor):
                    d = torch.tensor(d, dtype=torch.float32, device=self.device)
                directions.append(d.to(self.device).reshape(3))
            else:
                theta = ep['zenith']
                phi   = ep['azimuth']
                if not isinstance(theta, torch.Tensor):
                    theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
                if not isinstance(phi, torch.Tensor):
                    phi = torch.tensor(phi, dtype=torch.float32, device=self.device)
                theta, phi = theta.to(self.device).squeeze(), phi.to(self.device).squeeze()
                d = torch.stack([
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(theta),
                ])
                directions.append(d)

        track_pos    = torch.stack(positions)   # (n_events, 3)
        track_dir    = torch.stack(directions)  # (n_events, 3)
        track_energy = torch.stack(energies)    # (n_events,)

        return self.call_batched(track_pos, track_dir, track_energy, om_positions)
    
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

    def _parse_event_params(self, event_params):
        """Return (track_pos, track_dir, energy) as normalized device tensors."""
        track_pos = event_params.get('position', None)
        energy = event_params.get('energy', None)
        angular_dir = event_params.get('direction', None)

        if track_pos is None or energy is None:
            raise ValueError("event_params must contain 'position' and 'energy'")

        if isinstance(track_pos, torch.Tensor):
            track_pos = track_pos.to(self.device).squeeze()
        else:
            track_pos = torch.tensor(track_pos, dtype=torch.float32, device=self.device).squeeze()

        if isinstance(energy, torch.Tensor):
            energy = energy.to(self.device).squeeze()
        else:
            energy = torch.tensor(energy, dtype=torch.float32, device=self.device).squeeze()

        if angular_dir is not None:
            if isinstance(angular_dir, torch.Tensor):
                track_dir = angular_dir.to(self.device).squeeze()
            else:
                track_dir = torch.tensor(angular_dir, dtype=torch.float32, device=self.device).squeeze()
        else:
            zenith = event_params.get('zenith', None)
            azimuth = event_params.get('azimuth', None)
            if zenith is None or azimuth is None:
                if self.particle_mode == 'cascade':
                    # Direction is irrelevant for cascade PATD geometry; keep a sane placeholder.
                    track_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
                else:
                    raise ValueError("event_params must contain 'direction' or both 'zenith' and 'azimuth'")
            else:
                theta = zenith.to(self.device).squeeze() if isinstance(zenith, torch.Tensor) else torch.tensor(zenith, dtype=torch.float32, device=self.device).squeeze()
                phi = azimuth.to(self.device).squeeze() if isinstance(azimuth, torch.Tensor) else torch.tensor(azimuth, dtype=torch.float32, device=self.device).squeeze()
                track_dir = torch.stack([
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(theta),
                ])

        track_dir = track_dir / torch.norm(track_dir).clamp_min(1e-12)
        return track_pos, track_dir, energy

    def _empty_patd_dict(self, expected_N=0):
        return {
            'hit_times': torch.tensor([], device=self.device),
            'num_photons': 0,
            'expected_photons': expected_N,
            'residual_times': torch.tensor([], device=self.device),
            'geometric_times': torch.tensor([], device=self.device),
            'vertex_times': torch.tensor([], device=self.device),
            'emission_points': torch.empty((0, 3), device=self.device),
            't_geom_min': torch.tensor(1e-6, device=self.device),
            'd_geom': torch.tensor([], device=self.device),
            'patd_probs': None,
        }

    def _sample_cpandel(self, cpandel, d_geom_i, t_geom_i, s_i, track_pos, track_dir, v_mu, get_patd_probs, cascade_mode=False):
        """Run CPandel sampling for one detector and return hit-level tensors."""
        d_geom_numpy = d_geom_i.detach().cpu().numpy()
        t_residual_output = cpandel.rvs(d=d_geom_numpy, size=None)
        t_residual_probs = None
        if get_patd_probs:
            t_residual_probs = cpandel.pdf(t_residual_output, d=d_geom_numpy)
            if isinstance(t_residual_probs, np.ndarray):
                t_residual_probs = torch.from_numpy(t_residual_probs).float().to(self.device)

        if isinstance(t_residual_output, torch.Tensor):
            t_residual = t_residual_output.float().to(self.device)
        else:
            t_residual = torch.from_numpy(t_residual_output).float().to(self.device)

        hit_times = t_geom_i + t_residual
        if cascade_mode:
            emission_points = track_pos.unsqueeze(0).expand(s_i.shape[0], -1)
            vertex_times = torch.zeros_like(s_i)
        else:
            emission_points = track_pos.unsqueeze(0) + s_i.unsqueeze(1) * track_dir.unsqueeze(0)
            vertex_times = s_i / v_mu
        return hit_times, t_residual, t_geom_i, vertex_times, emission_points, t_residual_probs

    def _compute_track_weights(self, t_min, t_max, track_pos, track_dir, detector_pos):
        """
        Compute per-track-point emission weights for a single detector position.
        Returns (t_vals, sampled_t_vals, d_geom) after multinomial sampling.
        """
        num_track_points = max(int(self.kwargs.get('num_track_points', 1000)), 1)
        t_vals = torch.linspace(float(t_min), float(t_max), num_track_points, device=self.device)
        track_points = track_pos.unsqueeze(0) + t_vals.unsqueeze(1) * track_dir.unsqueeze(0)

        distances = torch.norm(track_points - detector_pos.unsqueeze(0), dim=1)

        theta_c = torch.acos(torch.tensor(1.0 / self.refractive_index, device=self.device))
        sin_theta_c = torch.sin(theta_c)
        lambda_abs, lambda_sca = 44.7, 57.4/(1-self.scattering_tau)
        lambda_p = torch.sqrt(torch.tensor(lambda_abs * lambda_sca / 3.0, device=self.device))
        zeta = torch.exp(torch.tensor(-lambda_sca / lambda_abs, device=self.device))
        lambda_c = lambda_sca / (3.0 * zeta)
        lambda_mu = (lambda_c / sin_theta_c ** 2 * 2.0 / (np.pi * lambda_p)).clamp_min(1e-12)

        d_safe = distances.clamp(min=1e-6)
        w = (torch.exp(-d_safe / lambda_p) / (2.0 * np.pi * sin_theta_c)) / \
            (torch.sqrt(lambda_mu * d_safe) * torch.tanh(torch.sqrt(d_safe / lambda_mu))).clamp_min(1e-12)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        w_sum = w.sum()
        if not torch.isfinite(w_sum) or w_sum <= 0:
            w = torch.full_like(w, 1.0 / w.numel())
        else:
            w = w / w_sum
        return t_vals, w, distances

    def _compute_track_weights_batch(self, t_min, t_max, track_pos, track_dir, detector_positions):
        """
        Compute emission weights for all detectors simultaneously.
        Returns t_vals (T,), weights (T, n_pts), distances (T, n_pts).
        """
        num_track_points = max(int(self.kwargs.get('num_track_points', 1000)), 1)
        t_vals = torch.linspace(float(t_min), float(t_max), num_track_points, device=self.device)  # (T,)
        track_points = track_pos.unsqueeze(0) + t_vals.unsqueeze(1) * track_dir.unsqueeze(0)       # (T, 3)

        # distances[i, j] = distance from track point i to detector j
        dists = (track_points.unsqueeze(1) - detector_positions.unsqueeze(0)).norm(dim=2)           # (T, n_pts)

        theta_c = torch.acos(torch.tensor(1.0 / self.refractive_index, device=self.device))
        sin_theta_c = torch.sin(theta_c)
        lambda_abs, lambda_sca = 44.7, 57.4/(1-self.scattering_tau)
        lambda_p = torch.sqrt(torch.tensor(lambda_abs * lambda_sca / 3.0, device=self.device))
        zeta = torch.exp(torch.tensor(-lambda_sca / lambda_abs, device=self.device))
        lambda_c = lambda_sca / (3.0 * zeta)
        lambda_mu = (lambda_c / sin_theta_c ** 2 * 2.0 / (np.pi * lambda_p)).clamp_min(1e-12)

        d_safe = dists.clamp(min=1e-6)
        w = (torch.exp(-d_safe / lambda_p) / (2.0 * np.pi * sin_theta_c)) / \
            (torch.sqrt(lambda_mu * d_safe) * torch.tanh(torch.sqrt(d_safe / lambda_mu))).clamp_min(1e-12)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)   # (T, n_pts)

        w_sums = w.sum(dim=0, keepdim=True)  # (1, n_pts)
        bad = ~torch.isfinite(w_sums) | (w_sums <= 0)
        w = torch.where(bad.expand_as(w), torch.full_like(w, 1.0 / num_track_points), w / w_sums.clamp_min(1e-12))
        return t_vals, w, dists

    def light_yield_surrogate(self, **kwargs):
        """
        Generate photon arrival time distribution(s) using the CPandel model.

        When opt_point is a single detector position (shape (3,) or (1, 3)), returns a
        single dict.  When opt_point is a batch of positions (shape (n_pts, 3) with
        n_pts > 1), returns a list of n_pts dicts, with all geometry parallelised across
        detectors before the per-detector CPandel sampling loop.

        Parameters
        ----------
        event_params : dict
            Contains 'position', 'energy', and either 'direction' or 'zenith'/'azimuth'.
        opt_point : Tensor, shape (3,) | (1, 3) | (n_pts, 3)
            Detector position(s).
        max_photons : int or None
            Cap on photons sampled per detector.
        get_patd_probs : bool
            If True, also return CPandel PDF values for each residual time.
        use_perpendicular_distance_only : bool
            Use only foot-point geometry (no along-track resampling).

        Returns
        -------
        dict  — single detector input
        list[dict]  — multi-detector input (one dict per detector, same keys as single)
        """
        opt_point = kwargs.get('opt_point', None)
        event_params = kwargs.get('event_params', None)

        if event_params is None or opt_point is None:
            raise ValueError("event_params and opt_point must be provided")

        # Normalise opt_point to a float32 device tensor
        if not isinstance(opt_point, torch.Tensor):
            opt_point = torch.tensor(opt_point, dtype=torch.float32, device=self.device)
        else:
            opt_point = opt_point.to(self.device).float()

        # Strip already-extracted keys so they aren't double-passed as kwargs
        rest = {k: v for k, v in kwargs.items() if k not in ('opt_point', 'event_params')}

        # Dispatch: single point → dict, multiple points → list[dict]
        squeezed = opt_point.squeeze()
        if squeezed.dim() == 1:
            return self._patd_single(squeezed, event_params, **rest)
        else:
            pts = squeezed if squeezed.dim() == 2 else opt_point.view(-1, 3)
            return self._patd_batch(pts, event_params, **rest)

    # ------------------------------------------------------------------
    # Single-detector path (original logic, unchanged behaviour)
    # ------------------------------------------------------------------
    def _patd_single(self, detector_pos, event_params, **kwargs):
        max_photons = kwargs.get('max_photons', None)
        get_patd_probs = kwargs.get('get_patd_probs', False)
        use_perpendicular_distance_only = kwargs.get(
            'use_perpendicular_distance_only',
            self.kwargs.get('use_perpendicular_distance_only', False)
        )
        c = 0.299792458
        v_mu = self.kwargs.get('v_mu', c)
        cpandel_params = self.kwargs.get('cpandel_params', {})
        is_cascade = self.particle_mode == 'cascade'

        track_pos, track_dir, energy = self._parse_event_params(event_params)

        if self.kwargs.get('input_photons', None) is None:
            light_yield = self.__call__(
                track_pos=track_pos, track_dir=track_dir,
                track_energy=energy, om_positions=detector_pos.unsqueeze(0)
            )
            if self.kwargs.get('use_poisson', False):
                light_yield = torch.poisson(self._sanitize_rate_for_poisson(light_yield))
        else:
            light_yield = torch.tensor(self.kwargs.get('input_photons'), device=self.device)

        expected_N = torch.round(light_yield).int().detach().cpu().item()
        N = min(expected_N, max_photons) if (max_photons is not None and max_photons < expected_N) else expected_N

        if N <= 0:
            return self._empty_patd_dict(expected_N)

        to_detector = detector_pos - track_pos
        t_foot = torch.dot(to_detector, track_dir)
        foot_length = torch.norm(torch.linalg.cross(to_detector, track_dir)) / torch.norm(track_dir)
        d_vertex = torch.norm(to_detector)
        num_samples = int(N.item()) if isinstance(N, torch.Tensor) else int(N)

        if is_cascade:
            s = torch.zeros((num_samples,), device=self.device)
            d_geom = torch.full((num_samples,), d_vertex.clamp(min=1e-6).item(), device=self.device)
            t_geom_min = d_vertex / (c / self.refractive_index)
        elif use_perpendicular_distance_only:
            if t_foot < 0:
                return self._empty_patd_dict(expected_N)
            s = torch.full((num_samples,), t_foot.item(), device=self.device)
            d_geom = torch.full((num_samples,), foot_length.clamp(min=1e-6).item(), device=self.device)
            t_geom_min = foot_length / (c / self.refractive_index) + t_foot / v_mu
        else:
            if t_foot < 0:
                return self._empty_patd_dict(expected_N)
            if not self.use_max_energy_dist:
                S = self.kwargs.get('track_segment_length', 200.0)
                t_min = torch.max(torch.tensor(0.0, device=self.device), t_foot - S)
                t_max = t_foot + S
            else:
                t_min = 0
                t_max = self.get_max_energy_dist(energy).squeeze()

            t_vals, weights, distances = self._compute_track_weights(t_min, t_max, track_pos, track_dir, detector_pos)
            sampled_indices = torch.multinomial(weights, num_samples, replacement=True)
            s = t_vals[sampled_indices]
            d_geom = distances[sampled_indices]

            if v_mu != c / self.refractive_index:
                short_track = t_foot - ((c / self.refractive_index) * foot_length) / \
                    torch.sqrt(torch.tensor(v_mu ** 2 - (c / self.refractive_index) ** 2, device=self.device))
                t_geom_min = short_track / v_mu + \
                    torch.sqrt((short_track - t_foot) ** 2 + foot_length ** 2) / (c / self.refractive_index)
            else:
                t_geom_min = torch.norm(to_detector) / (c / self.refractive_index)

        t_geom = d_geom / (c / self.refractive_index) + s / v_mu

        cpandel = CPandel(
            tau=cpandel_params.get('tau', 557.), lambda_s=cpandel_params.get('lambda_s', 57.4),
            lambda_a=cpandel_params.get('lambda_a', 44.7), v=cpandel_params.get('v', 0.3 / 1.33),
            s=cpandel_params.get('s', 5.0)
        )
        hit_times, t_residual, t_geom, vertex_times, emission_points, t_residual_probs = \
            self._sample_cpandel(
                cpandel, d_geom, t_geom, s, track_pos, track_dir, v_mu, get_patd_probs,
                cascade_mode=is_cascade
            )

        return {
            'hit_times': hit_times,
            'num_photons': N,
            'expected_photons': light_yield,
            'residual_times': t_residual,
            'geometric_times': t_geom,
            'vertex_times': vertex_times,
            'emission_points': emission_points,
            't_geom_min': t_geom_min,
            'd_geom': d_geom,
            'patd_probs': t_residual_probs,
        }

    # ------------------------------------------------------------------
    # Multi-detector path: geometry parallelised, CPandel loop per det.
    # ------------------------------------------------------------------
    def _patd_batch(self, detector_positions, event_params, **kwargs):
        """
        Compute PATD for n_pts detector positions simultaneously.

        All geometry (foot lengths, t_geom_min, light yield, Poisson sampling, and
        the full (T × n_pts) track-weight matrix for the non-perp mode) is computed
        in parallel.  Only the CPandel rvs call loops over detectors because each
        detector has a different number of photons N_i.

        Returns list[dict] of length n_pts.
        """
        max_photons = kwargs.get('max_photons', None)
        get_patd_probs = kwargs.get('get_patd_probs', False)
        use_perpendicular_distance_only = kwargs.get(
            'use_perpendicular_distance_only',
            self.kwargs.get('use_perpendicular_distance_only', False)
        )
        c = 0.299792458
        v_mu = self.kwargs.get('v_mu', c)
        cpandel_params = self.kwargs.get('cpandel_params', {})
        n_pts = detector_positions.shape[0]
        is_cascade = self.particle_mode == 'cascade'

        track_pos, track_dir, energy = self._parse_event_params(event_params)

        # ---- Vectorised geometry ----------------------------------------
        to_detector = detector_positions - track_pos.unsqueeze(0)                      # (n_pts, 3)
        t_foot = (to_detector * track_dir.unsqueeze(0)).sum(dim=1)                     # (n_pts,)
        cross = torch.linalg.cross(
            to_detector, track_dir.unsqueeze(0).expand(n_pts, 3)
        )
        foot_length = cross.norm(dim=1) / track_dir.norm().clamp_min(1e-12)            # (n_pts,)
        valid_mask = torch.ones_like(t_foot, dtype=torch.bool) if is_cascade else (t_foot >= 0)  # (n_pts,)
        d_vertex = to_detector.norm(dim=1)                                              # (n_pts,)

        # ---- Vectorised light yield + Poisson ---------------------------
        if self.kwargs.get('input_photons', None) is None:
            light_yield = self.__call__(
                track_pos=track_pos, track_dir=track_dir,
                track_energy=energy, om_positions=detector_positions
            )                                                                           # (n_pts,)
            if self.kwargs.get('use_poisson', False):
                light_yield = torch.poisson(self._sanitize_rate_for_poisson(light_yield))
        else:
            light_yield = torch.full((n_pts,), float(self.kwargs.get('input_photons')), device=self.device)

        N_samples = torch.round(light_yield).int()                                     # (n_pts,)
        if max_photons is not None:
            N_samples = N_samples.clamp(max=int(max_photons))

        # ---- Vectorised t_geom_min --------------------------------------
        fl_safe = foot_length.clamp(min=1e-6)
        if is_cascade:
            t_geom_min_batch = d_vertex.clamp(min=1e-6) / (c / self.refractive_index)  # (n_pts,)
        elif use_perpendicular_distance_only:
            t_geom_min_batch = fl_safe / (c / self.refractive_index) + t_foot / v_mu  # (n_pts,)
        else:
            if v_mu != c / self.refractive_index:
                denom = torch.sqrt(
                    torch.tensor(v_mu ** 2 - (c / self.refractive_index) ** 2, device=self.device)
                )
                short_track = t_foot - (c / self.refractive_index) * fl_safe / denom
                t_geom_min_batch = short_track / v_mu + \
                    torch.sqrt((short_track - t_foot) ** 2 + fl_safe ** 2) / (c / self.refractive_index)
            else:
                t_geom_min_batch = to_detector.norm(dim=1) / (c / self.refractive_index)

        # ---- Track-weight matrix (non-perp mode, computed once) ---------
        if not is_cascade and not use_perpendicular_distance_only:
            if not self.use_max_energy_dist:
                S = self.kwargs.get('track_segment_length', 200.0)
                t_min = max(0.0, float(t_foot.min().item()) - S)
                t_max = float(t_foot.max().item()) + S
            else:
                t_min = 0.0
                t_max = float(self.get_max_energy_dist(energy).squeeze().item())

            t_vals, weights_matrix, dists_matrix = self._compute_track_weights_batch(
                t_min, t_max, track_pos, track_dir, detector_positions
            )                                          # (T,), (T, n_pts), (T, n_pts)

        # ---- Build CPandel once -----------------------------------------
        cpandel = CPandel(
            tau=cpandel_params.get('tau', 557.), lambda_s=cpandel_params.get('lambda_s', 57.4),
            lambda_a=cpandel_params.get('lambda_a', 44.7), v=cpandel_params.get('v', 0.3 / 1.33),
            s=cpandel_params.get('s', 5.0)
        )

        # ---- Per-detector CPandel loop (all geometry already done) ------
        results = []
        for i in range(n_pts):
            if not valid_mask[i].item():
                results.append(self._empty_patd_dict(light_yield[i].item()))
                continue

            N_i = int(N_samples[i].item())
            if N_i <= 0:
                results.append(self._empty_patd_dict(light_yield[i].item()))
                continue

            if is_cascade:
                s_i = torch.zeros((N_i,), device=self.device)
                d_geom_i = torch.full((N_i,), d_vertex[i].clamp(min=1e-6).item(), device=self.device)
            elif use_perpendicular_distance_only:
                s_i = torch.full((N_i,), t_foot[i].item(), device=self.device)
                d_geom_i = torch.full((N_i,), fl_safe[i].item(), device=self.device)
            else:
                sampled_idx = torch.multinomial(weights_matrix[:, i], N_i, replacement=True)
                s_i = t_vals[sampled_idx]
                d_geom_i = dists_matrix[sampled_idx, i]

            t_geom_i = d_geom_i / (c / self.refractive_index) + s_i / v_mu

            hit_times, t_residual, t_geom_i, vertex_times, emission_points, t_residual_probs = \
                self._sample_cpandel(
                    cpandel, d_geom_i, t_geom_i, s_i, track_pos, track_dir, v_mu, get_patd_probs,
                    cascade_mode=is_cascade
                )

            results.append({
                'hit_times': hit_times,
                'num_photons': N_i,
                'expected_photons': light_yield[i],
                'residual_times': t_residual,
                'geometric_times': t_geom_i,
                'vertex_times': vertex_times,
                'emission_points': emission_points,
                't_geom_min': t_geom_min_batch[i],
                'd_geom': d_geom_i,
                'patd_probs': t_residual_probs,
            })

        return results

    def eval_patd_log_probs(self, t_residuals_fixed, opt_point, event_params):
        """
        Re-evaluate CPandel log-pdf at pre-sampled residual times with gradient-enabled geometry.

        Computes foot_length (perpendicular distance from track to detector) from event_params
        with full gradient support. Exact for use_perpendicular_distance_only=True; a
        differentiable perpendicular-distance approximation for the general mode.

        Parameters
        ----------
        t_residuals_fixed : Tensor, shape (N_hits,)
            Pre-sampled residual times, treated as fixed constants for grad computation.
        opt_point : Tensor, shape (3,)
            Detector position.
        event_params : dict
            Must contain 'position' and 'direction' (or 'zenith'/'azimuth'). Parameters
            may carry requires_grad=True for Fisher info Jacobian computation.

        Returns
        -------
        Tensor, shape (N_hits,)
            log(CPandel.pdf(t_residuals, d=foot_length(theta))).
        """
        track_pos, track_dir, _ = self._parse_event_params(event_params)

        if not isinstance(opt_point, torch.Tensor):
            opt_point = torch.tensor(opt_point, dtype=torch.float32, device=self.device)
        opt_point = opt_point.float().to(self.device)

        if not isinstance(t_residuals_fixed, torch.Tensor):
            t_residuals_fixed = torch.tensor(t_residuals_fixed, dtype=torch.float32, device=self.device)
        t_residuals_fixed = t_residuals_fixed.float().to(self.device)

        to_detector = opt_point - track_pos
        if self.particle_mode == 'cascade':
            d_geom = to_detector.norm().clamp_min(1e-6).expand(t_residuals_fixed.shape[0])
        else:
            cross = torch.linalg.cross(to_detector, track_dir)
            foot_length = cross.norm() / track_dir.norm().clamp_min(1e-12)
            d_geom = foot_length.clamp(min=1e-6).expand(t_residuals_fixed.shape[0])

        cpandel_params = self.kwargs.get('cpandel_params', {})
        cpandel = CPandel(
            tau=cpandel_params.get('tau', 557.),
            lambda_s=cpandel_params.get('lambda_s', 57.4),
            lambda_a=cpandel_params.get('lambda_a', 44.7),
            v=cpandel_params.get('v', 0.3 / 1.33),
            s=cpandel_params.get('s', 5.0),
        )

        probs = cpandel.pdf(t_residuals_fixed, d=d_geom)
        return torch.log(probs.clamp(min=1e-40))

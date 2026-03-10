import os
import pickle
from nugget.losses.base_loss import LossFunction
from scipy.interpolate import UnivariateSpline
import numpy as np
import h5py
from pathlib import Path
from nugget.losses.trigger import TriggerLoss
import torch
from nugget.samplers.cyl_sampler import CylinderSampler

# Default packaged data directory (nugget/assets/data)
# data_dir = Path(__file__).resolve().parents[1] / "assets" / "data"
data_dir=''
def muon_range(energy):
    """
    Approximate muon range
    
    Values taken from MMC paper
    """
    a = 0.212/1.2
    b = 0.251e-3/1.2
    return np.log(1 + energy * b/a) / b

def average_chord_length(cos_theta, cyl_radius, cyl_height):
    """Calculate average chord length of cylinder for a given direction"""
    # Convert to torch tensor if not already, preserve gradient
    if not isinstance(cos_theta, torch.Tensor):
        cos_theta = torch.as_tensor(cos_theta, dtype=torch.float32)
    
    # Ensure it's at least 1D for indexing
    is_scalar = cos_theta.dim() == 0
    if is_scalar:
        cos_theta = cos_theta.unsqueeze(0)
    
    # Create mask for cos_theta == 0
    mask = (cos_theta == 0)
    
    # Initialize result
    result = torch.ones_like(cos_theta)
    
    # Handle cos_theta == 0 case
    if torch.any(mask):
        result = torch.where(mask, np.pi * cyl_radius / 2, result)
    
    # Handle cos_theta != 0 case
    if torch.any(~mask):
        # Use where to maintain differentiability
        ct_masked = torch.abs(cos_theta)
        chord_nonzero = cyl_radius / ((cyl_radius / (cyl_height / ct_masked)) + 2/np.pi*torch.sqrt(1 - cos_theta * cos_theta))
        result = torch.where(~mask, chord_nonzero, result)
    
    # Return scalar if input was scalar
    if is_scalar:
        result = result.squeeze(0)
    
    return result

def projected_area(cos_theta, cyl_radius, cyl_height):
    """Calculate projected area of cylinder for a given direction"""
    # Ensure cos_theta is a torch tensor
    if not isinstance(cos_theta, torch.Tensor):
        cos_theta = torch.as_tensor(cos_theta, dtype=torch.float32)
    
    cap = np.pi*cyl_radius**2
    sides = 2*cyl_radius*cyl_height
    return cap*torch.abs(cos_theta) + sides*torch.sqrt(1 - cos_theta**2)

def cyl_volume(cyl_radius, cyl_height):
    """Calculate cylinder volume"""
    return np.pi*cyl_radius**2*cyl_height


def interaction_prob(cos_theta, energy, cyl_radius, cyl_height, xsec, which="CC_nu", extend_range=False):
    """
    Calculate neutrini interaction probability.
    
    Parameters:
    -----------
    - cos_theta
    - energy
    - cyl_radius
    
    """    
    number_density = 33.3679E21 # 1/cm^-3
    nucleon_density = 18 * number_density
    
    chord_len = average_chord_length(cos_theta, cyl_radius, cyl_height) * 1E2 # cm

    if extend_range:
        #extend by muon range
        #get muon energy
        mean_e_mu = xsec.mean_e_lep(energy, which=which)
    
        chord_len += muon_range(mean_e_mu) * 1E2

    lamb = 1 / (xsec.total_xsec(energy, which) * nucleon_density)

    return 1- torch.exp(-chord_len / lamb)

def neutrino_effective_area(cos_theta, energy, cyl_radius, cyl_height, xsec, transmission, flavor="numu", average_nu_nubar=True):
    """Calculate neutrino effective area"""

    tprob = transmission(cos_theta, energy, flavor=flavor)
    # Convert transmission probability to torch tensor if needed
    if not isinstance(tprob, torch.Tensor):
        tprob = torch.as_tensor(tprob, dtype=torch.float32)

    if average_nu_nubar:
        factor = 0.5
    else:
        factor = 1
    
    int_prob_CC = factor * (
        interaction_prob(cos_theta, energy, cyl_radius, cyl_height, xsec, which="CC_nu", extend_range=flavor=="numu") +
        interaction_prob(cos_theta, energy, cyl_radius, cyl_height, xsec, which="CC_nubar", extend_range=flavor=="numu")
    )
    int_prob_NC = factor * (
        interaction_prob(cos_theta, energy, cyl_radius, cyl_height, xsec, which="NC_nu") +
        interaction_prob(cos_theta, energy, cyl_radius, cyl_height, xsec, which="NC_nubar")
    )
    int_prob = int_prob_CC + int_prob_NC
    proj_a = projected_area(cos_theta, cyl_radius, cyl_height)

    return tprob * int_prob * proj_a

class CrossSection:
    """
    DIS Cross Section

    Reads a cross-section table in nuSQuIDS format
    """

    def __init__(self, filename=None):
        """Initialize cross section interpolators from nuSQuIDS format HDF5 file.
        
        Args:
            filename: Path to HDF5 file containing cross section tables
        """
        keys = ["CC_nu", "CC_nubar", "NC_nu", "NC_nubar"]
        self.total_xsec_splines = {}
        self.diff_xsec_splines = {}

        self.y_sampling_splines = {}

        if filename is None:
            filename = data_dir / "csms_square.h5"

        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(
                f"Cross section table not found: {filename}. "
                "Pass 'filename=' explicitly or ensure nugget/assets/data is present."
            )

        with h5py.File(filename, "r") as hdl:
            self._logenergies = hdl["energies"][:] -9 # GeV

            zs = hdl["zs"][:]
            for key in keys:
                logsigma = hdl["s_"+key][:] # log10(cm^2)
                self.total_xsec_splines[key] = UnivariateSpline(self._logenergies, logsigma, s=0)

                splines = {}
                for e_ix, _ in enumerate(self._logenergies):
                    diff_xs = hdl["dsdy_"+key][:, e_ix]

                    cumu = np.cumsum(diff_xs)
                    cumu = cumu / cumu[-1]

                    inverted_cumu = UnivariateSpline(cumu,  zs, s=0, ext=3)
                    splines[e_ix] = inverted_cumu
                self.y_sampling_splines[key] = splines
        

        self.min_e = 10**np.min(self._logenergies)
        self.max_e = 10**np.max(self._logenergies)

       
        # Calculate mean outgoing lepton energies

        loge_grid = np.linspace(2, 9, 300)
        self.mean_elep_splines = {}
        for key in keys:
            mean_elep = np.log10([np.average(self.sample_e_lep(10**le, size=50000, which=key)) for le in loge_grid])
            self.mean_elep_splines[key] = UnivariateSpline(loge_grid, mean_elep, s=0)
   
    def z(self, e_lep, e_nu):
        """Calculate Bjorken-z scaling variable from lepton and neutrino energies."""
        return (e_lep - self.min_e)/(e_nu - self.min_e)

    def e_lep(self, z, e_nu):
        """Calculate lepton energy from Bjorken-z and neutrino energy."""
        return z*(e_nu - self.min_e) + self.min_e


    def total_xsec(self, energy, which="CC_nu"):
        """Get total neutrino-nucleon cross section.
        
        Args:
            energy: Neutrino energy in GeV
            which: Interaction type ("CC_nu", "CC_nubar", "NC_nu", "NC_nubar")
            
        Returns:
            Cross section in cm²
        """
        return 10 ** self.total_xsec_splines[which](np.log10(energy))
        

    def _get_y_sampling_spline(self, energy, which):
        """Get interpolator for sampling outgoing lepton energies.
        
        Args:
            energy: Neutrino energy in GeV
            which: Interaction type
            
        Returns:
            Spline interpolator for inverse cumulative distribution
        """
        if (energy < 1E2) or (energy > 1E9):
            raise RuntimeError("Energy outside range")
        lookup = np.searchsorted(self._logenergies, np.log10(energy))
        if lookup < 0:
            raise RuntimeError("Energy below minimum energy")
    
        if lookup >= len(self._logenergies):
            raise RuntimeError("Energy above minimum energy")
        
        
        spline = self.y_sampling_splines[which][lookup]
        return spline


    def sample_e_lep(self, energy, size=None, which="CC_nu"):
        """Sample outgoing lepton energies from differential cross section.
        
        Args:
            energy: Neutrino energy in GeV
            size: Number of samples (None for single value)
            which: Interaction type
            
        Returns:
            Sampled lepton energy/energies in GeV
        """
        spline = self._get_y_sampling_spline(energy, which)
        z = spline(np.random.uniform(size=size))

        return self.e_lep(z, energy)
    
    def mean_e_lep(self, energy, which="CC_nu"):
        """Get mean outgoing lepton energy for given neutrino energy.
        
        Args:
            energy: Neutrino energy in GeV
            which: Interaction type
            
        Returns:
            Mean lepton energy in GeV
        """
        return 10**self.mean_elep_splines[which](np.log10(energy))


class TransmissionProb:
    """Neutrino transmission probability through Earth."""
    
    def __init__(self, filename=None
                 ):
        """Initialize transmission probability interpolators.
        
        Args:
            filename: Path to pickle file containing transmission splines
        """
        if filename is None:
            filename = data_dir / "transm_inter_splines.pickle"

        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(
                f"Transmission spline file not found: {filename}. "
                "Pass 'filename=' explicitly or ensure nugget/assets/data is present."
            )

        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.splines = data["transmission_prob"]

    def __call__(self, cos_theta, energy, flavor="numu"):
        """Calculate transmission probability for given trajectory and energy.
        
        Args:
            cos_theta: Cosine of zenith angle
            energy: Neutrino energy in GeV
            flavor: Neutrino flavor ("numu", "nue", "nutau")
            
        Returns:
            Transmission probability [0, 1]
        """
        return self.splines[flavor](cos_theta, np.log10(energy), grid=False)
    
# def get_bounding_cylinder(positions):
#     """Get bounding cylinder for a set of 3D points."""

#     x_min, y_min = torch.amin(positions[:, :2], axis=0)

#     x_max, y_max = torch.amax(positions[:, :2], axis=0)
#     z_min = torch.amin(positions[:, 2])
#     z_max = torch.amax(positions[:, 2])

#     center_x = (x_min + x_max) / 2
#     center_y = (y_min + y_max) / 2
#     center_z = (z_min + z_max) / 2

#     radius = torch.sqrt(((x_max - x_min) / 2)**2 + ((y_max - y_min) / 2)**2)

#     height = z_max - z_min

#     return torch.tensor([center_x, center_y, center_z]), radius, height

def get_bounding_cylinder(positions):
    """Get tightest bounding cylinder for a set of 3D points."""
    
    xy_positions = positions[:, :2]
    
    # Start with centroid
    center_xy = torch.mean(xy_positions, dim=0)
    
    # Get radius from centroid
    distances = torch.norm(xy_positions - center_xy, dim=1)
    radius = torch.max(distances)
    
    # Optional: Refine by checking if we can reduce radius
    # by slightly moving the center
    max_dist_idx = torch.argmax(distances)
    farthest_point = xy_positions[max_dist_idx]
    
    # Z-bounds
    z_min = torch.amin(positions[:, 2])
    z_max = torch.amax(positions[:, 2])
    center_z = (z_min + z_max) / 2
    height = z_max - z_min
    
    center = torch.tensor([center_xy[0], center_xy[1], center_z], device=positions.device)
    return center, radius, height

def get_weighted_bounding_cylinder(positions, point_weights=None, temperature=1.0):
    """Get weighted bounding cylinder using differentiable centroid-based approach.
    
    This version uses weighted centroids and softmax-based operations to compute
    the bounding cylinder properties in a fully differentiable manner.
    
    Parameters:
    -----------
    positions : torch.Tensor
        3D positions of points, shape (n_points, 3)
    point_weights : torch.Tensor, optional
        Weights for each point, shape (n_points,). If None, uniform weights are used.
    temperature : float
        Temperature parameter for softmax operations. Lower values make the 
        bounds tighter but less differentiable. Default: 1.0
        
    Returns:
    --------
    tuple
        (center, radius, height) where center is [cx, cy, cz], radius is the 
        weighted radius in XY plane, and height is weighted Z extent
    """
    device = positions.device
    xy_positions = positions[:, :2]
    z_positions = positions[:, 2]
    
    # Handle point weights
    if point_weights is None:
        point_weights = torch.ones(len(positions), device=device)
    
    # Normalize weights to sum to 1
    weights_normalized = point_weights #/ torch.sum(point_weights)
    
    # Weighted centroid in XY
    center_xy = torch.sum(weights_normalized.unsqueeze(1) * xy_positions, dim=0)
    
    # Weighted centroid in Z
    center_z = torch.sum(weights_normalized * z_positions, dim=0)
    
    # Calculate distances from weighted centroid
    distances_xy = torch.norm(xy_positions - center_xy.unsqueeze(0), dim=1)
    
    # Soft-maximum for radius using weighted softmax
    # This is differentiable and approximates max(distances)
    softmax_weights = torch.softmax(point_weights * distances_xy / temperature, dim=0)
    radius = torch.sum(softmax_weights * distances_xy)
    
    # Soft bounds for height using weighted softmax
    # Soft minimum for z
    z_min_weights = torch.softmax(-point_weights * z_positions / temperature, dim=0)
    z_min = torch.sum(z_min_weights * z_positions)
    
    # Soft maximum for z
    z_max_weights = torch.softmax(point_weights * z_positions / temperature, dim=0)
    z_max = torch.sum(z_max_weights * z_positions)
    
    # Recalculate center_z as midpoint of soft bounds
    center_z = (z_min + z_max) / 2
    height = z_max - z_min
    
    center = torch.stack([center_xy[0], center_xy[1], center_z])
    return center, radius, height

class EffectiveAreaLoss(LossFunction):
    """Loss class to maximize neutrino effective area."""
    
    def __init__(
        self,
        xsec=None,
        transmission=None,
        flavor="numu",
        average_nu_nubar=True,
        trigger=None,
        device=None,
        domain_size=2500,
    ):
        """Initialize effective area loss function.
        
        Args:
            xsec: CrossSection object
            transmission: TransmissionProb object
            flavor: Neutrino flavor ("numu", "nue", "nutau")
            average_nu_nubar: Whether to average over neutrinos and antineutrinos
        """
        super().__init__(device)
        self.xsec = CrossSection() if xsec is None else xsec
        self.transmission = TransmissionProb() if transmission is None else transmission
        self.flavor = flavor
        self.average_nu_nubar = average_nu_nubar
        self.trigger = TriggerLoss(device=self.device) if trigger is None else trigger
        self.domain_size = domain_size

    def map_string_weights_to_points(self, points_3d, string_xy, string_weights):
        """
        Map string weights to point weights based on xy positions.
        
        Points with the same xy coordinates (belonging to the same string) 
        will receive the same weight. Weights are passed through sigmoid.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            All detector points, shape (n_points, 3)
        string_xy : torch.Tensor
            String xy positions, shape (n_strings, 2)
        string_weights : torch.Tensor
            Weights for each string, shape (n_strings,)
            
        Returns:
        --------
        torch.Tensor
            Weights for each point (after sigmoid), shape (n_points,)
        """
        n_points = len(points_3d)
        n_strings = len(string_xy)
        
        # Extract xy coordinates from points
        points_xy = points_3d[:, :2]  # Shape: (n_points, 2)
        
        # Initialize point weights
        point_weights = torch.zeros(n_points, device=self.device)
        
        # For each point, find the closest string and assign its weight
        for i, point_xy in enumerate(points_xy):
            # Calculate distance to all strings
            distances = torch.norm(string_xy - point_xy.unsqueeze(0), dim=1)
            
            # Find the closest string
            closest_string_idx = torch.argmin(distances)
            
            # Assign the string's weight to this point
            point_weights[i] = string_weights[closest_string_idx]
        
        # Apply sigmoid to weights (not controlled by temperature)
        point_weights_sigmoid = torch.sigmoid(self.weight_sigmoid_sharpness * point_weights)
        
        return point_weights_sigmoid
    
    
    def __call__(self, geom_dict, **kwargs):
        
        
        points_3d = geom_dict.get("points_3d")
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        # event_params_list = kwargs.get('signal_event_params', None)
        # precomputed_light_yield = kwargs.get('precomputed_light_yield_per_point_per_event', None)
        # signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events_per_bin', 100)
        num_energy_bins = kwargs.get('num_energy_bins', 30)
        num_zenith_bins = kwargs.get('num_zenith_bins', 30)
        energy_range = kwargs.get('energy_range', (1e2, 1e8))
        zenith_range = kwargs.get('zenith_range', (-1, 1))
        temperature = kwargs.get('bounding_cylinder_temperature', 0.1)
        cylinder_kwargs = kwargs.get('cylinder_sampler_kwargs', {})
        perfect_trigger = kwargs.get('perfect_efficiency', False)
        pc_ly_per_event_per_point_per_e_per_ct = kwargs.get('pc_ly_per_point_per_event_per_e_per_ct', None)
        
        
        # signal_sampler = CylinderSampler(event_type='signal', domain_size=2500, E_min=energy_range, E_max=1e8, energy_dist='log_uniform')
    
        point_weights = None
        if string_weights is not None:
            if string_xy is None:
                # Get unique xy coordinates in order of first appearance
                seen = set()
                unique_indices = []
                for i, xy in enumerate(points_3d[:, :2]):
                    xy_tuple = tuple(xy.tolist())
                    if xy_tuple not in seen:
                        seen.add(xy_tuple)
                        unique_indices.append(i)
                
                string_xy = points_3d[unique_indices, :2]
            
            point_weights = self.map_string_weights_to_points(points_3d, string_xy, string_weights)
        else:
            point_weights = torch.ones(len(points_3d), device=self.device)
        
        center, cyl_radius, cyl_height = get_weighted_bounding_cylinder(points_3d, point_weights=point_weights, temperature=temperature)
        
        energy_bins = torch.linspace(np.log10(energy_range[0]), np.log10(energy_range[1]), num_energy_bins+1, device=self.device)
        zenith_bins = torch.linspace(zenith_range[0], zenith_range[1], num_zenith_bins+1, device=self.device)
        energy_centers = 10**((energy_bins[:-1] + energy_bins[1:]) / 2)
        zenith_centers = (zenith_bins[:-1] + zenith_bins[1:]) / 2
        
        effective_areas = torch.zeros((num_zenith_bins, num_energy_bins), device=self.device)
        for e_ind, e in enumerate(energy_centers):
            for ct_ind, ct in enumerate(zenith_centers):
                if perfect_trigger:
                    detector_efficiency = 1.0
                else:
                    sampled_events = CylinderSampler(domain_size=self.domain_size, E_min=e, E_max=e, cos_range=(ct, ct), event_type='signal', **cylinder_kwargs).sample_events(num_events)
                    light_yield_per_event_per_point = torch.zeros((len(sampled_events), len(points_3d)), device=self.device)
                    for i, event_params in enumerate(sampled_events):
                        light_yield_per_event_per_point[i] = surrogate_func(opt_point=points_3d, event_params=event_params)
                    detector_efficiency = torch.mean(self.trigger.compute_trigger_probability_batch_events(points_3d=points_3d, precomputed_light_yield=light_yield_per_event_per_point,
                                                                            string_weights=point_weights, surrogate_func=surrogate_func, event_params_list=sampled_events)) 
                eff_area = neutrino_effective_area(ct, e, cyl_radius, cyl_height, self.xsec, self.transmission, flavor=self.flavor, average_nu_nubar=self.average_nu_nubar)
                effective_areas[ct_ind, e_ind] = eff_area * detector_efficiency

        effective_area_loss = 1/torch.mean(effective_areas)
        
        # if precomputed_light_yield is None:
        #     precomputed_light_yield = torch.zeros((len(event_params_list), len(points_3d)), device=self.device)
        #     for i, event_params in enumerate(event_params_list):
        #         precomputed_light_yield[i] = surrogate_func(opt_point=points_3d, event_params=event_params)
        
        return {
            "effective_area_loss": effective_area_loss,
            "bounding_cylinder_radius": cyl_radius,
            "bounding_cylinder_height": cyl_height,
            "effective_area_matrix": effective_areas,
            "energy_centers": energy_centers,
            "zenith_centers": zenith_centers
        }      
                
                
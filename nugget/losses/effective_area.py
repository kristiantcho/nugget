import os
import pickle
import random
from nugget.losses.base_loss import LossFunction
from scipy.interpolate import UnivariateSpline
import numpy as np
import h5py
from pathlib import Path
from nugget.losses.trigger import TriggerLoss
import torch
from nugget.samplers.cyl_sampler import CylinderSampler

# Default packaged data directory (nugget/assets/data)
data_dir = Path(__file__).resolve().parents[1] / "assets" / "data"
# data_dir=''


def _to_numpy_no_grad(x):
    """Convert tensor-like input to numpy for table/spline evaluation."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _extract_energy_and_cos_zenith_batch(event_params_batch, device, dtype):
    """Extract batched energy and cos_zenith from dict or list-of-dicts event params."""
    if isinstance(event_params_batch, dict):
        if "energy" not in event_params_batch:
            raise ValueError("event_params_batch must include 'energy'")
        energy = torch.as_tensor(event_params_batch["energy"], device=device, dtype=dtype).reshape(-1)

        if "cos_zenith" in event_params_batch:
            cos_zenith = torch.as_tensor(event_params_batch["cos_zenith"], device=device, dtype=dtype).reshape(-1)
        elif "zenith" in event_params_batch:
            zenith = torch.as_tensor(event_params_batch["zenith"], device=device, dtype=dtype).reshape(-1)
            cos_zenith = torch.cos(zenith)
        else:
            raise ValueError("event_params_batch must include 'cos_zenith' or 'zenith'")

        return energy, cos_zenith

    if isinstance(event_params_batch, (list, tuple)):
        energies = []
        cos_zeniths = []
        for event_params in event_params_batch:
            if "energy" not in event_params:
                raise ValueError("Each event must include 'energy'")
            energies.append(torch.as_tensor(event_params["energy"], device=device, dtype=dtype).reshape(-1)[0])

            if "cos_zenith" in event_params:
                cos_zeniths.append(torch.as_tensor(event_params["cos_zenith"], device=device, dtype=dtype).reshape(-1)[0])
            elif "zenith" in event_params:
                zen = torch.as_tensor(event_params["zenith"], device=device, dtype=dtype).reshape(-1)[0]
                cos_zeniths.append(torch.cos(zen))
            else:
                raise ValueError("Each event must include 'cos_zenith' or 'zenith'")

        return torch.stack(energies, dim=0), torch.stack(cos_zeniths, dim=0)

    raise TypeError("event_params_batch must be a dict or a list/tuple of dicts")


def neutrino_effective_area_many(
    cos_theta,
    energy,
    cyl_radius,
    cyl_height,
    xsec,
    transmission,
    flavor="numu",
    average_nu_nubar=True,
    *,
    include_projected_area: bool = True,
):
    """Calculate neutrino effective area for many events in a single batched call.

    If ``include_projected_area`` is False, the projected-area factor is omitted
    (treated as 1) and only ``transmission * interaction_prob`` is returned.
    """
    if not isinstance(cos_theta, torch.Tensor):
        cos_theta = torch.as_tensor(cos_theta)
    if not isinstance(energy, torch.Tensor):
        energy = torch.as_tensor(energy, device=cos_theta.device, dtype=cos_theta.dtype)
    else:
        energy = energy.to(device=cos_theta.device, dtype=cos_theta.dtype)

    tprob = transmission(cos_theta, energy, flavor=flavor)
    if not isinstance(tprob, torch.Tensor):
        tprob = torch.as_tensor(tprob, device=cos_theta.device, dtype=cos_theta.dtype)

    number_density = 33.3679E21  # 1/cm^-3
    nucleon_density = 18 * number_density

    chord_len = average_chord_length(cos_theta, cyl_radius, cyl_height) * 1e2

    def _int_prob(which, extend_range=False):
        chord_len_local = chord_len
        if extend_range:
            mean_e_mu = xsec.mean_e_lep(energy, which=which)
            mu_ext = torch.as_tensor(muon_range(mean_e_mu), device=chord_len_local.device, dtype=chord_len_local.dtype)
            chord_len_local = chord_len_local + mu_ext * 1e2

        lamb = 1 / (xsec.total_xsec(energy, which) * nucleon_density)
        lamb = torch.as_tensor(lamb, device=chord_len_local.device, dtype=chord_len_local.dtype)
        return 1 - torch.exp(-chord_len_local / lamb)

    factor = 0.5 if average_nu_nubar else 1.0
    int_prob_cc = factor * (
        _int_prob("CC_nu", extend_range=flavor == "numu")
        + _int_prob("CC_nubar", extend_range=flavor == "numu")
    )
    int_prob_nc = factor * (_int_prob("NC_nu") + _int_prob("NC_nubar"))
    int_prob = int_prob_cc + int_prob_nc

    if not include_projected_area:
        proj_a = 1.0
    else:
        proj_a = projected_area(cos_theta, cyl_radius, cyl_height)

    return tprob * int_prob * proj_a

def cyl_volume(cyl_radius, cyl_height):
    """Calculate cylinder volume"""
    return np.pi*cyl_radius**2*cyl_height


def interaction_prob(
    cos_theta,
    energy,
    cyl_radius,
    cyl_height,
    xsec,
    which="CC_nu",
    extend_range=False,
):
    """
    Calculate neutrino interaction probability.

    Parameters:
    -----------
    - cos_theta
    - energy
    - cyl_radius

    """
    number_density = 33.3679E21 # 1/cm^-3
    nucleon_density = 18 * number_density

    chord_len = average_chord_length(cos_theta, cyl_radius, cyl_height) * 1e2  # cm

    if extend_range:
        #extend by muon range
        #get muon energy
        mean_e_mu = xsec.mean_e_lep(energy, which=which)

        mu_ext = torch.as_tensor(muon_range(mean_e_mu), device=chord_len.device, dtype=chord_len.dtype)
        chord_len += mu_ext * 1E2

    lamb = 1 / (xsec.total_xsec(energy, which) * nucleon_density)
    lamb = torch.as_tensor(lamb, device=chord_len.device, dtype=chord_len.dtype)

    return 1- torch.exp(-chord_len / lamb)

def neutrino_effective_area(
    cos_theta,
    energy,
    cyl_radius,
    cyl_height,
    xsec,
    transmission,
    flavor="numu",
    average_nu_nubar=True,
    *,
    include_projected_area: bool = True,
):
    """Calculate neutrino effective area.

    If ``include_projected_area`` is False, the projected-area factor is omitted
    (treated as 1) and only ``transmission * interaction_prob`` is returned.
    """

    tprob = transmission(cos_theta, energy, flavor=flavor)
    # Convert transmission probability to torch tensor if needed
    if not isinstance(tprob, torch.Tensor):
        if isinstance(cos_theta, torch.Tensor):
            tprob = torch.as_tensor(tprob, device=cos_theta.device, dtype=cos_theta.dtype)
        else:
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
    if not include_projected_area:
        proj_a = 1.0
    else:
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
        energy_np = _to_numpy_no_grad(energy)
        return 10 ** self.total_xsec_splines[which](np.log10(energy_np))
        

    def _get_y_sampling_spline(self, energy, which):
        """Get interpolator for sampling outgoing lepton energies.
        
        Args:
            energy: Neutrino energy in GeV
            which: Interaction type
            
        Returns:
            Spline interpolator for inverse cumulative distribution
        """
        energy_np = _to_numpy_no_grad(energy)
        if np.any((energy_np < 1E2) | (energy_np > 1E9)):
            raise RuntimeError("Energy outside range")
        lookup = np.searchsorted(self._logenergies, np.log10(energy_np))
        if np.ndim(lookup) != 0:
            lookup = int(np.ravel(lookup)[0])
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
        energy_np = _to_numpy_no_grad(energy)
        return 10**self.mean_elep_splines[which](np.log10(energy_np))


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
        cos_theta_np = _to_numpy_no_grad(cos_theta)
        energy_np = _to_numpy_no_grad(energy)
        return self.splines[flavor](cos_theta_np, np.log10(energy_np), grid=False)
    
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
    dtype = positions.dtype

    xy_positions = positions[:, :2]
    z_positions = positions[:, 2]

    if point_weights is None:
        point_weights = torch.ones(len(positions), device=device, dtype=dtype)
    else:
        point_weights = point_weights.to(device=device, dtype=dtype)

    w = point_weights

    # Weighted xy center.
    center_xy = torch.sum(w.unsqueeze(1) * xy_positions, dim=0) / torch.sum(w)
    # distances_xy = torch.sqrt(torch.sum((xy_positions - center_xy.unsqueeze(0)) ** 2, dim=1))
    # center_xy = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    distances_xy = torch.sqrt(torch.sum((xy_positions - center_xy.unsqueeze(0)) ** 2, dim=1))
    # log_w = torch.log(w)

    # weighted_terms = (distances_xy + log_w) / temperature
    # d_ref = torch.max(weighted_terms).detach()
    # radius = temperature * (d_ref + torch.logsumexp(weighted_terms - d_ref, dim=0))
    # radius = torch.sqrt(torch.sum(w * torch.sum((xy_positions - center_xy.unsqueeze(0)) ** 2, dim=1)) / torch.sum(w))

    # radius = torch.sum(distances_xy * w * torch.exp(temperature * distances_xy) /
    #                    torch.sum(w * torch.exp(temperature * distances_xy)))

    weighted_terms = temperature * distances_xy * w
    radius = torch.logsumexp(weighted_terms, dim=0) / temperature
    # Smooth z-bounds via weighted log-sum-exp (stable references).
    z_max = torch.max(z_positions)
    # z_max = z_ref_max + tau * torch.logsumexp((z_positions*log_w - z_ref_max) / tau, dim=0)

    z_min = torch.min(z_positions)
    # z_min = z_ref_min - tau * torch.logsumexp((z_ref_min - z_positions*log_w) / tau, dim=0)

    center_z = 0.5 * (z_min + z_max)
    height = z_max - z_min

    center = torch.stack([center_xy[0], center_xy[1], center_z])
    return center, radius, height


def get_weighted_min_enclosing_circle(
    string_xy,
    string_weights=None,
    temperature=1,
    n_directions=360,
    angular_kappa=5.0,
    downweight_untriggerable=False,
    trigger_neighbor_distance=550.0,
    trigger_min_neighbors=30.0,
    trigger_distance_sharpness=0.05,
    trigger_count_sharpness=1.0,
):
    """Differentiably estimate a weighted enclosing-circle radius around a fixed origin.

    Parameters
    ----------
    string_xy : torch.Tensor
        XY positions of strings, shape (n_strings, 2).
    string_weights : torch.Tensor, optional
        Continuous importance in [0, 1] per string, shape (n_strings,). If
        None, all strings are treated as fully active (weight 1).
    temperature : float
        Softmax temperature for the per-direction outer-distance soft-max.
        Lower values sharpen it toward the true (hard) max distance among
        effectively-active, direction-aligned strings.
    n_directions : int
        Number of directions sampled evenly around the origin.
    angular_kappa : float
        Sharpness of the angular gate. Higher values restrict each direction's
        soft-max to strings more tightly aligned with that direction.
    downweight_untriggerable : bool
        If True, heavily downweight strings that cannot be part of a trigger,
        i.e. strings that do not have at least ``trigger_min_neighbors`` other
        (weighted) strings within ``trigger_neighbor_distance`` in the XY plane.
        This prevents a lone, far-flung string from inflating the radius: an
        isolated string is not "triggerable", so it should not enlarge the
        sampling cylinder. Fully differentiable (soft neighbor count + soft gate).
    trigger_neighbor_distance : float
        XY neighborhood radius (same units as string_xy) used to count
        neighbors for the triggerability gate. Mirrors the trigger's sliding-bar
        length. Default 550.0.
    trigger_min_neighbors : float
        Minimum (weighted) neighbor count for a string to count as triggerable.
        Mirrors the trigger's ``min_points_threshold``. Default 30.0.
    trigger_distance_sharpness : float
        Sharpness of the soft "is neighbor within trigger_neighbor_distance"
        sigmoid (higher = sharper cutoff at the neighbor distance).
    trigger_count_sharpness : float
        Sharpness of the soft "neighbor count >= trigger_min_neighbors" gate
        (higher = sharper transition around the threshold).

    Returns
    -------
    tuple
        (center_xy, radius) where center_xy is a fixed zero vector, shape (2,),
        and radius is a 0-d tensor, differentiable w.r.t. string_xy and
        string_weights.
    """
    device = string_xy.device
    dtype = string_xy.dtype
    n_strings = string_xy.shape[0]

    if string_weights is None:
        w = torch.ones(n_strings, device=device, dtype=dtype)
    else:
        w = string_weights.to(device=device, dtype=dtype)

    if downweight_untriggerable:
        # Soft count of weighted neighbors within trigger_neighbor_distance (excluding self),
        # then a soft gate at trigger_min_neighbors. A string that fails the gate is
        # multiplied down toward ~0 so it no longer contributes to (inflates) the radius.
        pairwise = torch.cdist(string_xy, string_xy)  # (n_strings, n_strings)
        within = torch.sigmoid(trigger_distance_sharpness * (trigger_neighbor_distance - pairwise))
        # Exclude self from the neighbor count.
        within = within - torch.diag(torch.diagonal(within))
        neighbor_count = within @ w  # (n_strings,) weighted neighbor count per string
        triggerable = torch.sigmoid(trigger_count_sharpness * (neighbor_count - trigger_min_neighbors))
        w = w * triggerable

    center_xy = torch.zeros(2, device=device, dtype=dtype)

    distances = torch.sqrt(torch.sum(string_xy ** 2, dim=1) + 1e-12)  # (n_strings,)

    # Unit direction of each string from the origin.
    string_dir = string_xy / distances.unsqueeze(1)  # (n_strings, 2)

    angles = torch.linspace(0.0, 2 * torch.pi, n_directions + 1, device=device, dtype=dtype)[:-1]
    u = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (n_directions, 2)

    # Cosine alignment of every string with every sampled direction, in [-1, 1].
    align = string_dir @ u.T  # (n_strings, n_directions) 
    align = torch.nn.functional.softplus(align, beta=angular_kappa)  # (n_strings, n_directions)
    # Per-(string, direction) score used for the outer-distance soft-max: raw
    # distance, discounted by string weight and angular misalignment together.
    combined = distances.unsqueeze(1) * w.unsqueeze(1) * align / temperature
    per_direction_radius = temperature * torch.logsumexp(combined, dim=0)

    radius = torch.mean(per_direction_radius)
    return center_xy, radius


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

    def map_string_weights_to_points(self, points_3d, string_xy, string_weights, assignment_temperature=1.0):
        """
        Map string weights to point weights with differentiable soft assignment.

        Each point receives a weighted combination of all string weights based on
        XY proximity. This preserves gradient flow to both string weights and
        string XY coordinates.
        
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
        point_weights_sigmoid = torch.sigmoid(point_weights)
        
        return point_weights_sigmoid
    
    
    def __call__(self, geom_dict, **kwargs):
        
        
        points_3d = geom_dict.get("points_3d")
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        # event_params_list = kwargs.get('signal_event_params', None)
        # precomputed_light_yield = kwargs.get('precomputed_light_yield_per_point_per_event', None)
        # signal_sampler = kwargs.get('signal_sampler', None)
        num_events_per_bin = kwargs.get('num_events_per_bin', 100)
        num_energy_bins = kwargs.get('num_energy_bins', 30)
        num_zenith_bins = kwargs.get('num_zenith_bins', 30)
        energy_range = kwargs.get('energy_range', (1e2, 1e8))
        zenith_range = kwargs.get('cos_zenith_range', (-1, 1))
        temperature = kwargs.get('bounding_cylinder_temperature', 0.1)
        downweight_untriggerable = kwargs.get('downweight_untriggerable', False)
        trigger_neighbor_distance = kwargs.get('trigger_neighbor_distance', 550.0)
        trigger_min_neighbors = kwargs.get('trigger_min_neighbors', 30.0)
        trigger_distance_sharpness = kwargs.get('trigger_distance_sharpness', 0.05)
        trigger_count_sharpness = kwargs.get('trigger_count_sharpness', 1.0)
        include_projected_area = kwargs.get('include_projected_area', True)
        # If True, use the signal_sampler's own generation cylinder
        # (radius/height/center) for the effective-area cylinder instead of the
        # geometry-derived bounding cylinder. This makes A_eff use a fixed
        # generation volume (chord length, interaction prob, and projected area
        # all), decoupling it from the string spread.
        use_sampler_cylinder = kwargs.get('use_sampler_cyl_for_volume', False)
        use_batched_effective_area = kwargs.get('use_batched_effective_area', False)
        cylinder_kwargs = kwargs.get('cylinder_sampler_kwargs', {})
        pc_ly_per_event_per_point_per_e_per_ct = kwargs.get('pc_ly_per_point_per_event_per_e_per_ct', None)
        event_params_list = kwargs.get('signal_event_params', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        precomputed_light_yield_per_point_per_event = kwargs.get('precomputed_light_yield_per_point_per_event', None)
        num_events_for_per_event = kwargs.get('num_events', None)
        # If True and events are provided, compute per-event effective area and
        # set loss to reciprocal mean effective area over events.
        per_event_effective_area_loss = kwargs.get('per_event_effective_area_loss', True)
        # If True, sample all bin events at once and bin results in post-processing
        # instead of running a separate trigger+surrogate call per (E, cos_zenith) bin.
        use_batched_binned_trigger = kwargs.get('use_batched_binned_trigger', False)

        if precomputed_light_yield_per_point_per_event is None:
            precomputed_light_yield_per_point_per_event = pc_ly_per_event_per_point_per_e_per_ct

        # Per-event mode can sample directly when events are not passed explicitly.
        if per_event_effective_area_loss and event_params_list is None and signal_sampler is not None:
            sample_count = num_events_for_per_event if num_events_for_per_event is not None else num_events_per_bin
            event_params_list = signal_sampler.sample_events(sample_count)

        # Optional: choose trigger computation mode
        # - None: auto (batched if precomputed yields provided, else single loop)
        # - True: force batched
        # - False: force single-event loop
        use_batched_trigger = kwargs.get('use_batched_trigger', None)


        # signal_sampler = CylinderSampler(event_type='signal', domain_size=2500, E_min=energy_range, E_max=1e8, energy_dist='log_uniform')

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

        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None

        

        if use_sampler_cylinder:
            if signal_sampler is None or not hasattr(signal_sampler, 'cylinder'):
                raise ValueError(
                    "use_sampler_cylinder=True requires a signal_sampler with a 'cylinder' attribute"
                )
            sampler_cyl = signal_sampler.cylinder
            cyl_radius = torch.as_tensor(sampler_cyl.radius, device=self.device, dtype=points_3d.dtype)
            cyl_height = torch.as_tensor(sampler_cyl.height, device=self.device, dtype=points_3d.dtype)
            center = torch.as_tensor(sampler_cyl.center, device=self.device, dtype=points_3d.dtype).reshape(-1)
        else:
            center_xy, cyl_radius = get_weighted_min_enclosing_circle(
                        string_xy, string_weights=string_probs, temperature=temperature,
                        downweight_untriggerable=downweight_untriggerable,
                        trigger_neighbor_distance=trigger_neighbor_distance,
                        trigger_min_neighbors=trigger_min_neighbors,
                        trigger_distance_sharpness=trigger_distance_sharpness,
                        trigger_count_sharpness=trigger_count_sharpness,
                    )
            z_positions = points_3d[:, 2]
            z_max = torch.max(z_positions)
            z_min = torch.min(z_positions)
            cyl_height = z_max - z_min
            center = torch.stack([center_xy[0], center_xy[1], 0.5 * (z_min + z_max)])
        # geom_dict for the trigger MUST carry string_xy/string_weights so the trigger
        # weights its points by the current (continuous) string activation. Passing only
        # points_3d would silently give every point weight 1, making detector efficiency
        # independent of string_weights.
        trigger_geom_dict = {'points_3d': points_3d}
        if string_xy is not None:
            trigger_geom_dict['string_xy'] = string_xy
        if string_weights is not None:
            trigger_geom_dict['string_weights'] = string_weights
        def _extract_energy_and_cos_zenith(event_params):
            if 'energy' not in event_params:
                raise ValueError("Each event must provide 'energy' for per-event effective area mode")

            energy = torch.as_tensor(event_params['energy'], device=self.device, dtype=points_3d.dtype).reshape(-1)[0]

            if 'cos_zenith' in event_params:
                cos_zenith = torch.as_tensor(event_params['cos_zenith'], device=self.device, dtype=points_3d.dtype).reshape(-1)[0]
            elif 'zenith' in event_params:
                zenith = torch.as_tensor(event_params['zenith'], device=self.device, dtype=points_3d.dtype).reshape(-1)[0]
                cos_zenith = torch.cos(zenith)
            else:
                raise ValueError("Each event must provide either 'cos_zenith' or 'zenith' for per-event effective area mode")

            return energy, cos_zenith

        if per_event_effective_area_loss and event_params_list is not None:
            if num_events_for_per_event is not None and len(event_params_list) > num_events_for_per_event:
                selected_indices = random.sample(range(len(event_params_list)), int(num_events_for_per_event))
                event_params_list = [event_params_list[i] for i in selected_indices]

                if precomputed_light_yield_per_point_per_event is not None:
                    precomputed_light_yield_per_point_per_event = precomputed_light_yield_per_point_per_event[selected_indices]

            trigger_out = self.trigger(
                trigger_geom_dict,
                **{
                    **kwargs,
                    'signal_surrogate_func': surrogate_func,
                    'signal_event_params': event_params_list,
                    'precomputed_light_yield_per_point_per_event': precomputed_light_yield_per_point_per_event,
                    'use_batched_trigger': use_batched_trigger,
                },
            )
            per_event_trigger = trigger_out['t_per_event']

            if use_batched_effective_area:
                per_event_energies, per_event_cos_zenith = _extract_energy_and_cos_zenith_batch(
                    event_params_list,
                    device=self.device,
                    dtype=points_3d.dtype,
                )
                per_event_effective_areas = neutrino_effective_area_many(
                    per_event_cos_zenith,
                    per_event_energies,
                    cyl_radius,
                    cyl_height,
                    self.xsec,
                    self.transmission,
                    flavor=self.flavor,
                    average_nu_nubar=self.average_nu_nubar,
                    include_projected_area=include_projected_area,
                )
                per_event_effective_areas = per_event_effective_areas * per_event_trigger
            else:
                per_event_effective_areas = torch.zeros(len(event_params_list), device=self.device, dtype=points_3d.dtype)
                per_event_energies = torch.zeros(len(event_params_list), device=self.device, dtype=points_3d.dtype)
                per_event_cos_zenith = torch.zeros(len(event_params_list), device=self.device, dtype=points_3d.dtype)

                for i, event_params in enumerate(event_params_list):
                    energy_i, cos_zenith_i = _extract_energy_and_cos_zenith(event_params)
                    per_event_energies[i] = energy_i
                    per_event_cos_zenith[i] = cos_zenith_i

                    eff_area_i = neutrino_effective_area(
                        cos_zenith_i,
                        energy_i,
                        cyl_radius,
                        cyl_height,
                        self.xsec,
                        self.transmission,
                        flavor=self.flavor,
                        average_nu_nubar=self.average_nu_nubar,
                        include_projected_area=include_projected_area,
                    )
                    per_event_effective_areas[i] = eff_area_i * per_event_trigger[i]

            effective_area_loss = 1 / torch.mean(per_event_effective_areas)

            return {
                "effective_area_loss": effective_area_loss,
                "weighted_bounding_cylinder_center": center,
                "weighted_bounding_cylinder_radius": cyl_radius,
                "weighted_bounding_cylinder_height": cyl_height,
                "bounding_cylinder_radius": cyl_radius,
                "bounding_cylinder_height": cyl_height,
                "bounding_cylinder_center": center,
                "effective_area_matrix": per_event_effective_areas,
                "effective_area_params":event_params_list,
                "detector_efficiencies": per_event_trigger,
                "effective_area_per_event": per_event_effective_areas,
            }
        
        energy_bins = torch.linspace(np.log10(energy_range[0]), np.log10(energy_range[1]), num_energy_bins+1, device=self.device)
        zenith_bins = torch.linspace(zenith_range[0], zenith_range[1], num_zenith_bins+1, device=self.device)
        energy_centers = 10**((energy_bins[:-1] + energy_bins[1:]) / 2)
        zenith_centers = (zenith_bins[:-1] + zenith_bins[1:]) / 2

        effective_areas = torch.zeros((num_zenith_bins, num_energy_bins), device=self.device)
        detector_efficiencies = torch.zeros((num_zenith_bins, num_energy_bins), device=self.device)

        if use_batched_binned_trigger:
            # Sample all events for all bins at once, run a single batched trigger+surrogate
            # pass, then bin results in post-processing. Avoids num_energy_bins*num_zenith_bins
            # separate trigger calls and CylinderSampler constructions.
            # Use binned_trigger_batch_size to cap memory per chunk (None = one big batch).
            binned_trigger_batch_size = kwargs.get('binned_trigger_batch_size', None)
            total_events = num_energy_bins * num_zenith_bins * num_events_per_bin
            all_events = CylinderSampler(
                domain_size=self.domain_size,
                E_min=energy_range[0],
                E_max=energy_range[1],
                cos_range=zenith_range,
                event_type='signal',
                energy_dist='log_uniform',
                uniform_zenith_sampling=True,
                **cylinder_kwargs,
            ).sample_events(total_events)

            # Extract energies and cos_zenith for all events upfront (cheap, no surrogate)
            all_energies = torch.stack([
                torch.as_tensor(ep['energy'], device=self.device, dtype=points_3d.dtype).reshape(-1)[0]
                for ep in all_events
            ])
            all_cos_zenith = torch.stack([
                torch.cos(torch.as_tensor(ep['zenith'], device=self.device, dtype=points_3d.dtype).reshape(-1)[0])
                if 'cos_zenith' not in ep else
                torch.as_tensor(ep['cos_zenith'], device=self.device, dtype=points_3d.dtype).reshape(-1)[0]
                for ep in all_events
            ])

            counts = torch.zeros((num_zenith_bins, num_energy_bins), device=self.device)

            trigger_out = self.trigger(
                trigger_geom_dict,
                **{
                    **kwargs,
                    'signal_surrogate_func': surrogate_func,
                    'signal_event_params': all_events,
                    'precomputed_light_yield_per_point_per_event': None,
                    'use_batched_trigger': use_batched_trigger,
                    'binned_trigger_batch_size': binned_trigger_batch_size,
                },
            )
            all_trigger = trigger_out['t_per_event']
            if kwargs.get('detach_trigger', False):
                all_trigger = all_trigger.detach()

            log_e = torch.log10(all_energies)
            e_idx = torch.bucketize(log_e, energy_bins[1:-1])
            ct_idx = torch.bucketize(all_cos_zenith, zenith_bins[1:-1])
            for k in range(total_events):
                ei, ci = e_idx[k].item(), ct_idx[k].item()
                if 0 <= ei < num_energy_bins and 0 <= ci < num_zenith_bins:
                    detector_efficiencies[ci, ei] += all_trigger[k]
                    counts[ci, ei] += 1

            nonzero = counts > 0
            detector_efficiencies[nonzero] /= counts[nonzero]

            for e_ind, e in enumerate(energy_centers):
                for ct_ind, ct in enumerate(zenith_centers):
                    detector_efficiency = detector_efficiencies[ct_ind, e_ind]
                    eff_area = neutrino_effective_area(
                        ct, e, cyl_radius, cyl_height,
                        self.xsec, self.transmission,
                        flavor=self.flavor,
                        average_nu_nubar=self.average_nu_nubar,
                        include_projected_area=include_projected_area,
                    )
                    effective_areas[ct_ind, e_ind] = eff_area * detector_efficiency

        else:
            for e_ind, e in enumerate(energy_centers):
                for ct_ind, ct in enumerate(zenith_centers):
                    sampled_events = CylinderSampler(domain_size=self.domain_size, E_min=10**energy_bins[e_ind], E_max=10**energy_bins[e_ind + 1], cos_range=(zenith_bins[ct_ind], zenith_bins[ct_ind + 1]),
                                                     event_type='signal', energy_dist='log_uniform', uniform_zenith_sampling=True, **cylinder_kwargs).sample_events(num_events_per_bin)

                    trigger_out = self.trigger(
                        trigger_geom_dict,
                        **{
                            **kwargs,
                            'signal_surrogate_func': surrogate_func,
                            'signal_event_params': sampled_events,
                            'precomputed_light_yield_per_point_per_event': None,
                            'use_batched_trigger': use_batched_trigger,
                        },
                    )
                    detector_efficiency = trigger_out['detector_efficiency']
                    detector_efficiencies[ct_ind, e_ind] = detector_efficiency
                    eff_area = neutrino_effective_area(
                        ct,
                        e,
                        cyl_radius,
                        cyl_height,
                        self.xsec,
                        self.transmission,
                        flavor=self.flavor,
                        average_nu_nubar=self.average_nu_nubar,
                        include_projected_area=include_projected_area,
                    )
                    effective_areas[ct_ind, e_ind] = eff_area * detector_efficiency

        effective_area_loss = 1/torch.mean(effective_areas)
        
        # if precomputed_light_yield is None:
        #     precomputed_light_yield = torch.zeros((len(event_params_list), len(points_3d)), device=self.device)
        #     for i, event_params in enumerate(event_params_list):
        #         precomputed_light_yield[i] = surrogate_func(opt_point=points_3d, event_params=event_params)
        
        return {
            "effective_area_loss": effective_area_loss,
            "weighted_bounding_cylinder_center": center,
            "weighted_bounding_cylinder_radius": cyl_radius,
            "weighted_bounding_cylinder_height": cyl_height,
            "bounding_cylinder_radius": cyl_radius,
            "bounding_cylinder_height": cyl_height,
            "bounding_cylinder_center": center,
            "effective_area_matrix": effective_areas,
            "energy_centers": energy_centers,
            "zenith_centers": zenith_centers,
            "detector_efficiencies": detector_efficiencies
        }      
                
                
# sample_uniform_rays.py
from dataclasses import dataclass
import math
import torch
from typing import Tuple, Optional, List
from nugget.samplers.base_sampler import Sampler
import numpy as np

# -------------------------
# Geometry dataclass
# -------------------------
# @dataclass
class CylinderSurface():
   def __init__(self, center, height, radius, device=None):
        if isinstance(center, torch.Tensor):
            self.center = center.to(device) if device is not None else center
        else:
            self.center = torch.tensor(center, device=device)
        if isinstance(height, torch.Tensor):
            self.height = height.to(device) if device is not None else height
        else:
            self.height = torch.tensor(height, device=device)
        if isinstance(radius, torch.Tensor):
            self.radius = radius.to(device) if device is not None else radius
        else:
            self.radius = torch.tensor(radius, device=device)
# -------------------------
# Spherical <-> Cartesian
# -------------------------
def sph_to_cart(theta, phi):
    """
    theta: polar angle in [0, pi] (0 = +z)
    phi: azimuth in [0, 2pi)
    returns unit vector (3,)
    """
    st = torch.sin(theta)
    return torch.tensor([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)])

def cart_to_sph(vec):
    """
    vec: torch tensor (3,)
    returns (theta, phi, r)
    """
    x, y, z = vec[0], vec[1], vec[2]
    r = torch.sqrt(x*x + y*y + z*z)
    if r == 0:
        return 0.0, 0.0, 0.0
    theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))
    phi = torch.atan2(y, x)
    return theta, phi, r

# -------------------------
# Projected area helpers
# -------------------------
def projected_area(cyl, cos_theta):
    """
    projected_area(c, cos_theta) = cap_area * |cosθ| + sides * sqrt(1 - cosθ^2)
    where cap_area = π r^2 and sides = 2 r h
    """
    cap = torch.pi * cyl.radius**2
    sides = 2.0 * cyl.radius * cyl.height
   
    ct = cos_theta
    s = torch.sqrt(1.0 - ct*ct)
    return cap * torch.abs(ct) + sides * s

def maximum_proj_area(cyl):
    """
    Uses the same formula as the Julia code:
    maximum_proj_area(c) = projected_area(c, cos(atan(2*h/(π*r))))
    """
    if cyl.radius == 0:
        return 0.0
    phi0 = torch.atan(2.0 * cyl.height / (torch.pi * cyl.radius))
    return projected_area(cyl, torch.cos(phi0))


def _parse_zenith_cos_range(cos_range):
    if isinstance(cos_range, str):
        normalized = cos_range.strip().lower()
        if "horizontal" in normalized:
            return "horizontal", None, None
        if "vertical" in normalized:
            return "vertical", None, None
        raise ValueError(
            "cos_range must be a numeric pair or one of the zenith selectors 'horizontal'/'vertical'"
        )

    if isinstance(cos_range, torch.Tensor):
        cos_range = cos_range.detach().cpu().tolist()

    if not isinstance(cos_range, (tuple, list)) or len(cos_range) != 2:
        raise TypeError(
            "cos_range must be a length-2 sequence, a torch tensor with two values, or a zenith selector string"
        )

    cos_min = cos_range[0]
    cos_max = cos_range[1]
    return "numeric", cos_min, cos_max

# -------------------------
# Ray - cylinder intersection
# -------------------------
def get_intersection_cylinder(cyl, position, direction):
    """
    Compute interval of t (t_enter, t_exit) such that
      pos(t) = position + t * direction
    is inside the finite cylinder centered at cyl.center with height and radius.
    Returns (t_enter, t_exit) with t_enter <= t_exit, or (None, None) if no intersection.
    This is a robust method using quadratic solve for side intersections and plane solves for caps.
    """
    # transform into cylinder local coordinates (centered)
    pos = position - cyl.center
    x, y, z = pos[0], pos[1], pos[2]
    dx, dy, dz = direction[0], direction[1], direction[2]

    r = cyl.radius
    h2 = cyl.height / 2.0

    # Solve side intersections (infinite cylinder): (x + t dx)^2 + (y + t dy)^2 = r^2
    A = dx*dx + dy*dy
    B = 2.0 * (x*dx + y*dy)
    C = x*x + y*y - r*r

    t_side_candidates = []
    if torch.abs(A) > 1e-15:
        disc = B*B - 4.0*A*C
        if disc >= 0.0:
            sqrt_disc = torch.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2.0*A)
            t2 = (-B + sqrt_disc) / (2.0*A)
            t_side_candidates = [min(t1,t2), max(t1,t2)]
    # else A == 0 => direction parallel to cylinder axis (dx=dy≈0) => no side intersections unless x^2+y^2==r^2 (grazing)

    # Solve cap intersections (planes z = ±h/2)
    t_caps = []
    if torch.abs(dz) > 1e-15:
        t_top = ( h2 - z) / dz
        t_bot = (-h2 - z) / dz
        # For each t, check if (x + t dx)^2 + (y + t dy)^2 <= r^2
        for t in (t_top, t_bot):
            xx = x + t*dx
            yy = y + t*dy
            if xx*xx + yy*yy <= r*r + 1e-12:
                t_caps.append(t)
    # Combine candidate intervals: we need intersection of the span where sides hold and caps hold.
    # Approach: find intervals where inside side and inside caps and intersect them.
    intervals = []

    # If we have valid side interval, add as an interval
    if len(t_side_candidates) == 2:
        intervals.append((t_side_candidates[0], t_side_candidates[1]))

    # Caps can create small intervals: if both caps t exist, then the region between them (if inside circle) is candidate
    if len(t_caps) == 2:
        t_caps_sorted = sorted(t_caps)
        intervals.append((t_caps_sorted[0], t_caps_sorted[1]))
    elif len(t_caps) == 1:
        # one cap intersection only: may still produce intersection if starting inside cylinder side region
        # We won't append an interval for a single cap alone; side interval handling below will intersect
        pass

    # If no side interval but caps exist (rare), consider caps interval
    if not intervals and len(t_caps) == 2:
        intervals.append(tuple(sorted(t_caps)))

    # If we have no intervals at this point, attempt a different robust approach:
    # Build a candidate set of t values from side roots and cap ts and examine contiguous segments where the point is inside
    candidate_t = []
    if len(t_side_candidates) == 2:
        candidate_t += t_side_candidates
    candidate_t += t_caps
    # Add 0 as a reference (current position)
    candidate_t += [0.0]
    candidate_t = sorted(set(candidate_t))

    # Examine midpoints between consecutive candidate_t to see if inside cylinder; build intervals
    final_intervals = []
    for i in range(len(candidate_t)-1):
        a = candidate_t[i]
        b = candidate_t[i+1]
        mid = 0.5*(a+b)
        px = x + mid*dx
        py = y + mid*dy
        pz = z + mid*dz
        if (px*px + py*py <= r*r + 1e-12) and (pz >= -h2 - 1e-12) and (pz <= h2 + 1e-12):
            final_intervals.append((a,b))
    # Merge final_intervals if any
    if final_intervals:
        # take union
        t_min = min(iv[0] for iv in final_intervals)
        t_max = max(iv[1] for iv in final_intervals)
        return t_min, t_max

    # If previously computed intervals from sides/caps exist, intersect them
    if intervals:
        # Intersection of all intervals
        cur_min = -1e300
        cur_max = 1e300
        for (lo, hi) in intervals:
            cur_min = max(cur_min, lo)
            cur_max = min(cur_max, hi)
        if cur_min <= cur_max:
            return cur_min, cur_max

    # If we got here, no intersection
    return None, None

def get_intersection_box(center, domain_size, position, direction):
    """
    Compute interval of t (t_enter, t_exit) such that
      pos(t) = position + t * direction
    is inside the cubic domain centered at 'center' with size 'domain_size'.
    Returns (t_enter, t_exit) with t_enter <= t_exit, or (None, None) if no intersection.
    """
    # Compute box bounds
    half_size = domain_size / 2.0
    box_min = center - half_size
    box_max = center + half_size
    
    # Ray-box intersection using slab method
    t_min = -1e300
    t_max = 1e300
    
    for i in range(3):
        if torch.abs(direction[i]) > 1e-15:
            t1 = (box_min[i] - position[i]) / direction[i]
            t2 = (box_max[i] - position[i]) / direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray parallel to slab - check if position is within bounds
            if position[i] < box_min[i] or position[i] > box_max[i]:
                return None, None
    
    if t_min <= t_max:
        return t_min, t_max
    else:
        return None, None

# -------------------------
# Sampling rays (vectorized)
# -------------------------
def sample_uniform_ray(rng, cyl, cos_range = torch.tensor([-1.0, 1.0]), 
                       n_samples= 1, device = None, find_exact_intersection = False,
                       random_position_within_cylinder = False, 
                       random_position_within_cubic_domain = False, domain_size = 2.0,
                       uniform_zenith_sampling = False):
    """
    Sample multiple (position, direction) pairs in a vectorized manner.

    Parameters:
    -----------
    rng : torch.Generator, optional
        Random number generator for reproducible draws
    cyl : CylinderSurface
        Cylinder geometry
    cos_range : tuple
        (min, max) range for cos(theta), or the strings 'horizontal'/'vertical'
        to match the zenith windows used by event selection.
    n_samples : int
        Number of rays to sample
    device : torch.device, optional
        Device for tensor computations
    find_exact_intersection : bool
        If True, compute exact intersection with cylinder surface. If False, use initial sampled position.
    random_position_within_cylinder : bool
        If True, randomly sample a position along the ray within the cylinder domain.
    random_position_within_cubic_domain : bool
        If True, randomly sample a position along the ray within a cubic domain of size domain_size.
    domain_size : float
        Size of the cubic domain for random_position_within_cubic_domain option.
    uniform_zenith_sampling : bool
        If True, sample zenith angles uniformly over the requested range instead
        of using the projected-area rejection sampler.
        
    Returns:
    --------
    tuple
        (positions, directions) where:
        - positions: torch.Tensor of shape (n_samples, 3)
        - directions: torch.Tensor of shape (n_samples, 3)
    """
    if device is None:
        device = cyl.center.device
    
    dtype = torch.get_default_dtype()
    max_area = maximum_proj_area(cyl)
    cos_range_mode, cos_min_raw, cos_max_raw = _parse_zenith_cos_range(cos_range)
    if cos_range_mode == "numeric":
        cos_min = torch.as_tensor(cos_min_raw, device=device, dtype=dtype)
        cos_max = torch.as_tensor(cos_max_raw, device=device, dtype=dtype)
        fixed_costheta = (torch.abs(cos_min - cos_max) < 1e-15)
    else:
        cos_min = None
        cos_max = None
        fixed_costheta = False

    if uniform_zenith_sampling:
        if cos_range_mode == "horizontal":
            theta_min = torch.acos(torch.as_tensor(0.2, device=device, dtype=dtype))
            theta_max = torch.acos(torch.as_tensor(-0.2, device=device, dtype=dtype))
            if fixed_costheta:
                theta = torch.full((n_samples,), theta_min, dtype=dtype, device=device)
            else:
                u = torch.rand(n_samples, generator=rng, device=device, dtype=dtype)
                theta = theta_min + u * (theta_max - theta_min)
        elif cos_range_mode == "vertical":
            theta_min = torch.acos(torch.as_tensor(0.8, device=device, dtype=dtype))
            theta_mid = torch.acos(torch.as_tensor(-0.8, device=device, dtype=dtype))
            theta_max = torch.pi
            side_selector = torch.rand(n_samples, generator=rng, device=device, dtype=dtype) < 0.5
            u = torch.rand(n_samples, generator=rng, device=device, dtype=dtype)
            theta = torch.empty(n_samples, dtype=dtype, device=device)
            theta[side_selector] = u[side_selector] * theta_min
            theta[~side_selector] = theta_mid + u[~side_selector] * (theta_max - theta_mid)
        else:
            theta_min = torch.acos(torch.clamp(cos_max, -1.0, 1.0))
            theta_max = torch.acos(torch.clamp(cos_min, -1.0, 1.0))
            if fixed_costheta:
                theta = torch.full((n_samples,), theta_min, dtype=dtype, device=device)
            else:
                u = torch.rand(n_samples, generator=rng, device=device, dtype=dtype)
                theta = theta_min + u * (theta_max - theta_min)
        cos_theta = torch.cos(theta)
    else:
        # Rejection sample cos_theta for all samples
        cos_theta_list = []
        needed = n_samples
        while needed > 0:
            if cos_range_mode != "numeric":
                batch_size = needed * 3
                u = torch.rand(batch_size, generator=rng, device=device, dtype=dtype)
                cand = -1.0 + 2.0 * u
                if cos_range_mode == "horizontal":
                    range_mask = torch.abs(cand) < 0.2
                else:
                    range_mask = torch.abs(cand) > 0.8
                q = torch.rand(batch_size, generator=rng, device=device, dtype=dtype)
                proj_areas = torch.tensor([projected_area(cyl, c) for c in cand], 
                                          dtype=dtype, device=device)
                accepted = range_mask & (q * max_area <= proj_areas)
                accepted_cands = cand[accepted][:needed]
                
                cos_theta_list.append(accepted_cands)
                needed -= len(accepted_cands)
            elif fixed_costheta:
                cand = torch.full((needed,), cos_min, dtype=dtype, device=device)
                cos_theta_list.append(cand)
                break
            else:
                # Sample more than needed to reduce iterations
                batch_size = needed * 3
                u = torch.rand(batch_size, generator=rng, device=device, dtype=dtype)
                cand = cos_min + u * (cos_max - cos_min)
                q = torch.rand(batch_size, generator=rng, device=device, dtype=dtype)
                
                # Vectorized acceptance
                proj_areas = torch.tensor([projected_area(cyl, c) for c in cand], 
                                          dtype=dtype, device=device)
                accepted = q * max_area <= proj_areas
                accepted_cands = cand[accepted][:needed]
                
                cos_theta_list.append(accepted_cands)
                needed -= len(accepted_cands)
        
        cos_theta = torch.cat(cos_theta_list)[:n_samples]
    
    # Sample phi
    phi = torch.rand(n_samples, generator=rng, device=device, dtype=dtype) * 2.0 * torch.pi
    
    # Compute theta
    theta = torch.acos(cos_theta)
    
    # Compute direction vectors (vectorized)
    st = torch.sin(theta)
    ct = torch.cos(theta)
    cp = torch.cos(phi)
    sp = torch.sin(phi)
    
    directions = torch.stack([st * cp, st * sp, ct], dim=1)  # (n_samples, 3)
    
    # Compute projected ellipse parameters
    a = torch.sin(theta) * (cyl.height / 2.0)  # (n_samples,)
    b = torch.abs(ct) * cyl.radius  # (n_samples,)
    
    # Rejection sample x, y for each sample
    x_list = []
    y_list = []
    for i in range(n_samples):
        a_i = a[i].item()
        b_i = b[i].item()
        uni_x_min, uni_x_max = -cyl.radius, cyl.radius
        uni_y_min, uni_y_max = -(a_i + b_i), (a_i + b_i)
        
        while True:
            x_cand = uni_x_min + (uni_x_max - uni_x_min) * torch.rand(1, generator=rng, device=device, dtype=dtype)
            y_cand = uni_y_min + (uni_y_max - uni_y_min) * torch.rand(1, generator=rng, device=device, dtype=dtype)
            val = a_i + b_i * torch.sqrt(torch.clamp(1.0 - (x_cand**2) / (cyl.radius**2), min=0.0))
            if torch.abs(y_cand) <= val + 1e-12:
                x_list.append(x_cand)
                y_list.append(y_cand)
                break
    
    x_vals = torch.cat(x_list)  # (n_samples,)
    y_vals = torch.cat(y_list)  # (n_samples,)
    
    # Build local positions
    pos_local = torch.stack([y_vals, x_vals, torch.zeros(n_samples, dtype=dtype, device=device)], dim=1)  # (n_samples, 3)
    
    # Build rotation matrices for each sample (vectorized)
    # Ry(theta) for each sample
    zeros = torch.zeros(n_samples, dtype=dtype, device=device)
    ones = torch.ones(n_samples, dtype=dtype, device=device)
    
    Ry = torch.stack([
        torch.stack([ct, zeros, st], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-st, zeros, ct], dim=1)
    ], dim=1)  # (n_samples, 3, 3)
    
    # Rz(phi) for each sample
    Rz = torch.stack([
        torch.stack([cp, -sp, zeros], dim=1),
        torch.stack([sp, cp, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # (n_samples, 3, 3)
    
    # R = Rz @ Ry
    R = torch.bmm(Rz, Ry)  # (n_samples, 3, 3)
    
    # Apply rotation: pos_rot = R @ pos_local
    pos_rot = torch.bmm(R, pos_local.unsqueeze(-1)).squeeze(-1)  # (n_samples, 3)
    pos_world = pos_rot + cyl.center.unsqueeze(0)  # (n_samples, 3)
    
    # Optionally find exact intersections with cylinder
    if find_exact_intersection:
        positions_final = []
        directions_final = []
        
        for i in range(n_samples):
            pos_i = pos_world[i]
            dir_i = directions[i]
            
            t_enter, t_exit = get_intersection_cylinder(cyl, pos_i, dir_i)
            if t_enter is None:
                # Try opposite direction
                t_enter2, t_exit2 = get_intersection_cylinder(cyl, pos_i, -dir_i)
                if t_enter2 is None:
                    positions_final.append(pos_i)
                    directions_final.append(dir_i)
                else:
                    positions_final.append(pos_i + (-dir_i) * t_enter2)
                    directions_final.append(-dir_i)
            else:
                positions_final.append(pos_i + dir_i * t_enter)
                directions_final.append(dir_i)
        
        return torch.stack(positions_final), torch.stack(directions_final)
    elif random_position_within_cylinder:
        # Sample random positions along rays within the cylinder
        positions_final = []
        directions_final = []
        
        for i in range(n_samples):
            pos_i = pos_world[i]
            dir_i = directions[i]
            
            t_enter, t_exit = get_intersection_cylinder(cyl, pos_i, dir_i)
            if t_enter is None:
                # Try opposite direction
                t_enter2, t_exit2 = get_intersection_cylinder(cyl, pos_i, -dir_i)
                if t_enter2 is None:
                    # No intersection found, use initial position
                    positions_final.append(pos_i)
                    directions_final.append(dir_i)
                else:
                    # Randomly sample along ray in opposite direction
                    t_random = torch.rand(1, generator=rng, device=device, dtype=pos_i.dtype).item()
                    t_sample = t_enter2 + t_random * (t_exit2 - t_enter2)
                    positions_final.append(pos_i + (-dir_i) * t_sample)
                    directions_final.append(-dir_i)
            else:
                # Randomly sample along ray between entry and exit
                t_random = torch.rand(1, generator=rng, device=device, dtype=pos_i.dtype).item()
                t_sample = t_enter + t_random * (t_exit - t_enter)
                positions_final.append(pos_i + dir_i * t_sample)
                directions_final.append(dir_i)
        
        return torch.stack(positions_final), torch.stack(directions_final)
    elif random_position_within_cubic_domain:
        # Sample random positions along rays within the cubic domain
        positions_final = []
        directions_final = []
        
        for i in range(n_samples):
            pos_i = pos_world[i]
            dir_i = directions[i]
            
            t_enter, t_exit = get_intersection_box(cyl.center, domain_size, pos_i, dir_i)
            if t_enter is None:
                # Try opposite direction
                t_enter2, t_exit2 = get_intersection_box(cyl.center, domain_size, pos_i, -dir_i)
                if t_enter2 is None:
                    # No intersection found, use initial position
                    positions_final.append(pos_i)
                    directions_final.append(dir_i)
                else:
                    # Randomly sample along ray in opposite direction within cubic domain
                    t_random = torch.rand(1, generator=rng, device=device, dtype=pos_i.dtype).item()
                    t_sample = t_enter2 + t_random * (t_exit2 - t_enter2)
                    positions_final.append(pos_i + (-dir_i) * t_sample)
                    directions_final.append(-dir_i)
            else:
                # Randomly sample along ray between entry and exit of cubic domain
                t_random = torch.rand(1, generator=rng, device=device, dtype=pos_i.dtype).item()
                t_sample = t_enter + t_random * (t_exit - t_enter)
                positions_final.append(pos_i + dir_i * t_sample)
                directions_final.append(dir_i)
        
        return torch.stack(positions_final), torch.stack(directions_final)
    else:
        
        return pos_world, directions


# -------------------------
# CylinderSampler class
# -------------------------
class CylinderSampler(Sampler):
    
    def __init__(self, device=None, dim=3, domain_size=2, cylinder_center=None, cylinder_height=None, cylinder_radius=None, **kwargs):
        """
        Sampler for cylindrical detector geometry.
        
        Parameters:
        -----------
        device : torch.device, optional
            Device to use for computations
        dim : int
            Dimensionality (default: 3)
        domain_size : float
            Size of the domain (default: 2)
        cylinder_center : torch.Tensor or list, optional
            Center of the cylinder (default: [0, 0, 0])
        cylinder_height : float
            Height of the cylinder (default: 500.0)
        cylinder_radius : float
            Radius of the cylinder (default: 500.0)
        kwargs : dict
            Additional keyword arguments
        """
        super().__init__(device, dim, domain_size)
        self.kwargs = kwargs
        
        # Set up cylinder geometry
        if cylinder_center is None:
            cylinder_center = torch.zeros(3, device=self.device)
        elif not isinstance(cylinder_center, torch.Tensor):
            cylinder_center = torch.tensor(cylinder_center, device=self.device)
        else:
            cylinder_center = cylinder_center.to(self.device)
        if cylinder_height is None:
            cylinder_height = domain_size
        if cylinder_radius is None:
            cylinder_radius = domain_size / 2.0
        self.cylinder = CylinderSurface(
            center=cylinder_center,
            height=cylinder_height,
            radius=cylinder_radius,
            device=self.device
        )
        
        # Option to find exact intersection with cylinder
        self.find_exact_intersection = kwargs.get('find_exact_intersection', False)
        
        # Option to randomly sample position along ray within cylinder
        self.random_position_along_ray = kwargs.get('random_position_along_ray', False)
        
        # Option to randomly sample position along ray within cubic domain
        self.random_position_within_cubic_domain = kwargs.get('random_position_within_cubic_domain', False)

        # Option to sample zenith uniformly instead of using the projected-area sampler
        self.uniform_zenith_sampling = kwargs.get('uniform_zenith_sampling', False)
        
        # Option to force events to point towards cylinder center
        self.point_towards_center = kwargs.get('point_towards_center', False)
        
        # Event type for energy sampling
        self.event_type = kwargs.get('event_type', 'signal')
    
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
        n_samples : int
            Number of samples
            
        Returns:
        --------
        torch.Tensor
            Sampled values
        """
        if E_max - E_min <= 1e-12:
            return torch.full((n_samples,), E_min, device=self.device)
        else:
            r = torch.rand(n_samples, device=self.device)
            exponent = 1 - gamma
            return ((E_max**exponent - E_min**exponent) * r + E_min**exponent) ** (1 / exponent)
        
    def sample_uniform_logE(self, E_min=0.8, E_max=1.0, n_samples=1):
        """
        Sample uniformly in log-energy.
        
        Parameters:
        -----------
        E_min : float
            Minimum energy
        E_max : float
            Maximum energy
        n_samples : int
            Number of samples
            
        Returns:
        --------
        torch.Tensor
            Sampled energies
        """
        log_E_min = torch.log(torch.tensor(E_min, device=self.device))
        log_E_max = torch.log(torch.tensor(E_max, device=self.device))
        r = torch.rand(n_samples, device=self.device)
        log_energies = log_E_min + r * (log_E_max - log_E_min)
        return torch.exp(log_energies)
    
    def sample_events(self, num_events):
        """
        Adjusted to mirror ToySampler output:
        Keys: energy (1,), zenith (1,), azimuth (1,), position (1,3)
        Dtypes: float32 except background zenith (float64) to match ToySampler's current behavior.
        """
        E_min = self.kwargs.get('E_min', 0.8)
        E_max = self.kwargs.get('E_max', 1.0)
        if self.event_type == 'signal':
            gamma = self.kwargs.get('gamma', 2.7)
        else:
            gamma = self.kwargs.get('gamma', 3.7)
        # Optional position bias (same names as ToySampler)
        x_bias = self.kwargs.get('x_bias', 0.0)
        y_bias = self.kwargs.get('y_bias', 0.0)
        z_bias = self.kwargs.get('z_bias', 0.0)

        # Sample all energies
        if E_max - E_min > 1e-12:
            if self.kwargs.get('energy_dist', 'power_law') == 'log_uniform':
                energies = self.sample_uniform_logE(E_min=E_min, E_max=E_max, n_samples=num_events)
            else:
                energies = self.sample_power_law(E_min=E_min, E_max=E_max, gamma=gamma, n_samples=num_events)
        else:
            energies = torch.full((num_events,), E_min, device=self.device)
        # Reuse existing geometric sampler for positions (discard directions afterward)
        cos_range = self.kwargs.get('cos_range', torch.tensor([-1.0, 1.0]))
        seed = self.kwargs.get('seed', None)
        if seed is not None:
            rng = torch.Generator(device=self.device).manual_seed(seed)
        else:
            rng = None

        positions, directions = sample_uniform_ray(
            rng, self.cylinder, cos_range,
            n_samples=num_events,
            device=self.device,
            find_exact_intersection=self.find_exact_intersection,
            random_position_within_cylinder=self.random_position_along_ray,
            random_position_within_cubic_domain=self.random_position_within_cubic_domain,
            domain_size=self.domain_size,
            uniform_zenith_sampling=self.uniform_zenith_sampling
        )

        # Build bias tensor (float32)
        # bias = torch.tensor([x_bias, y_bias, z_bias], device=self.device, dtype=torch.float32) * 1.0

        # Override directions if point_towards_center is enabled
        if self.point_towards_center:
            for i in range(num_events):
                direction_to_center = self.cylinder.center - positions[i]
                direction_norm = torch.sqrt(torch.sum(direction_to_center**2))
                if direction_norm > 1e-15:
                    directions[i] = direction_to_center / direction_norm

        event_params_list = []
        for i in range(num_events):
            # Extract theta, phi from direction (only to get angles with correct shapes/dtypes)
            theta, phi, _ = cart_to_sph(directions[i])
            
            if phi < 0:
                phi += 2.0 * math.pi

            # Match ToySampler dtype behavior:
            
            zenith_tensor = torch.tensor([theta.item()], device=self.device, dtype=torch.float32)

            azimuth_tensor = torch.tensor([phi.item()], device=self.device, dtype=torch.float32)

            pos = positions[i].unsqueeze(0).to(torch.float32)  # (1,3) float32

            event_params = {
                'energy': energies[i:i+1],          # (1,) float32
                'zenith': zenith_tensor,             # (1,) float32 or float64 (background)
                'azimuth': azimuth_tensor,           # (1,) float32
                'position': pos,                     # (1,3) float32
                'direction': directions[i].to(torch.float32)  # (1,3) float32
                # 'direction' removed to match ToySampler
            }
            event_params_list.append(event_params)

        return event_params_list
    
    def sample_detector_points(self, num_points):
        """
        Sample points within the cylindrical detector volume.
        
        Parameters:
        -----------
        num_points : int
            Number of points to sample
            
        Returns:
        --------
        torch.Tensor
            Sampled points within the cylinder (num_points, 3)
        """
        points = []
        while len(points) < num_points:
            # Sample in bounding box
            r = torch.rand(1, device=self.device)* self.cylinder.radius
            phi = torch.rand(1, device=self.device)* 2 * math.pi
            z = (torch.rand(1, device=self.device) - 0.5) * self.cylinder.height
            
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            
            point = torch.tensor([x, y, z], device=self.device)
            point = point + self.cylinder.center
            points.append(point)
        
        return torch.stack(points)

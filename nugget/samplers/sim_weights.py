import math
import torch

# nugget.losses.effective_area is imported inside the functions that use it: importing it here
# loads nugget.losses, whose analysis_loss imports this module again before it has finished loading.

M2_TO_CM2 = 1e4


def event_arrays(events, travel_direction=True):
    """Stack event dicts into (N,) tensors of energy, zenith, azimuth, cos_zenith in the arrival convention."""
    def stack(key):
        return torch.stack([torch.as_tensor(e[key]).reshape(-1)[0] for e in events])

    energy, zenith, azimuth = stack('energy'), stack('zenith'), stack('azimuth')
    if travel_direction:
        # Samplers emit the travel direction; transmission and atmospheric fluxes expect where the neutrino comes from.
        zenith = math.pi - zenith
        azimuth = torch.remainder(azimuth + math.pi, 2 * math.pi)
    return {'energy': energy, 'zenith': zenith, 'azimuth': azimuth, 'cos_zenith': torch.cos(zenith)}


def generation_params(sampler):
    """Generation settings of a CylinderSampler, as keyword arguments for simulation_weights."""
    kw = sampler.kwargs
    return {
        'E_min': float(kw.get('E_min', 0.8)),
        'E_max': float(kw.get('E_max', 1.0)),
        'energy_dist': kw.get('energy_dist', 'power_law'),
        'gamma': float(kw.get('gamma', 2.7 if sampler.event_type == 'signal' else 3.7)),
        'cos_range': tuple(torch.as_tensor(kw.get('cos_range', (-1.0, 1.0))).tolist()),
        'direction_dist': 'uniform_zenith' if sampler.uniform_zenith_sampling else 'projected_area',
    }


def energy_pdf(energy, E_min, E_max, energy_dist='power_law', gamma=2.7):
    """Generation pdf q(E) in 1/GeV."""
    if energy_dist == 'log_uniform':
        return 1.0 / (energy * math.log(E_max / E_min))
    expo = 1.0 - gamma
    return expo * energy ** (-gamma) / (E_max ** expo - E_min ** expo)


def _proj_area_antiderivative(c, radius, height):
    # int A_proj(c) dc with A_proj = pi R^2 |c| + 2 R H sqrt(1 - c^2)
    return math.pi * radius ** 2 * c * c.abs() / 2 + radius * height * (c * torch.sqrt(1 - c ** 2) + torch.asin(c))


def direction_pdf(cos_zenith, cyl_radius, cyl_height, cos_range=(-1.0, 1.0), direction_dist='projected_area'):
    """Generation pdf p(Omega) in 1/sr. 'isotropic' is on the full sphere."""
    # All three are symmetric in cos, so cos_range may be given in the travel or arrival convention.
    c_lo, c_hi = torch.as_tensor(cos_range, dtype=cos_zenith.dtype, device=cos_zenith.device)
    if direction_dist == 'isotropic':
        return torch.ones_like(cos_zenith) / (2 * math.pi * (c_hi - c_lo))
    if direction_dist == 'uniform_zenith':
        d_theta = torch.acos(c_lo) - torch.acos(c_hi)
        return 1.0 / (2 * math.pi * d_theta * torch.sqrt(1 - cos_zenith ** 2))
    if direction_dist == 'projected_area':
        from nugget.losses.effective_area import projected_area
        norm = 2 * math.pi * (_proj_area_antiderivative(c_hi, cyl_radius, cyl_height)
                              - _proj_area_antiderivative(c_lo, cyl_radius, cyl_height))
        return projected_area(cos_zenith, cyl_radius, cyl_height) / norm
    raise ValueError(f"Unknown direction_dist {direction_dist!r}")


def get_simulation_weights(
    energy,
    cos_zenith,
    cyl_radius,
    cyl_height,
    xsec=None,
    transmission=None,
    *,
    E_min=0.8,
    E_max=1.0,
    energy_dist='power_law',
    gamma=2.7,
    cos_range=(-1.0, 1.0),
    direction_dist='projected_area',
    n_gen=None,
    flavor='numu',
    average_nu_nubar=True,
):
    """Fluxless weight A_eff(E, cos; R, H) / (N_gen q(E) p(Omega)), in GeV cm^2 sr.

    cos_zenith is in the arrival convention (see event_arrays). cyl_radius/cyl_height may carry gradients.
    returns a (N,) tensor of weights, with the same dtype and device as energy. n_gen is the number of generated events.
    """
    from nugget.losses.effective_area import neutrino_effective_area

    n_gen = energy.numel() if n_gen is None else n_gen
    a_eff = neutrino_effective_area(
        cos_zenith, energy, cyl_radius, cyl_height, xsec, transmission,
        flavor=flavor, average_nu_nubar=average_nu_nubar,
    )
    q_e = energy_pdf(energy, E_min, E_max, energy_dist, gamma)
    p_omega = direction_pdf(cos_zenith, cyl_radius, cyl_height, cos_range, direction_dist)
    return M2_TO_CM2 * a_eff / (n_gen * q_e * p_omega)

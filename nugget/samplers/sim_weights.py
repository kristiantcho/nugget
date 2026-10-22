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
    out = {'energy': energy, 'zenith': zenith, 'azimuth': azimuth, 'cos_zenith': torch.cos(zenith)}
    if 'column_depth' in events[0]:  # RangeSampler events, g/cm^2
        out['column_depth'] = stack('column_depth')
    return out


def generation_params(sampler):
    """Generation settings of a CylinderSampler or RangeSampler, as keyword arguments for get_simulation_weights."""
    if hasattr(sampler, 'injection_radius'):
        th_lo, th_hi = sampler.zenith_range
        return {
            'mode': 'ranged',
            'E_min': sampler.E_min,
            'E_max': sampler.E_max,
            'energy_dist': sampler.energy_dist,
            'gamma': sampler.gamma,
            # In the travel convention, as for CylinderSampler (the sampler's zenith_range is the arrival direction).
            'cos_range': (-math.cos(th_lo), -math.cos(th_hi)),
            'azimuth_range': tuple(sampler.azimuth_range),
            'injection_radius': sampler.injection_radius,
        }
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


def _effective_area_fn(batched, xsec, transmission):
    from nugget.losses import effective_area as ea

    if batched:
        # neutrino_effective_area_many computes the chord length once for all four interaction
        # channels, but unlike neutrino_effective_area it has no defaults for xsec/transmission.
        xsec = ea.CrossSection() if xsec is None else xsec
        transmission = ea.TransmissionProb() if transmission is None else transmission
        return ea.neutrino_effective_area_many, xsec, transmission
    return ea.neutrino_effective_area, xsec, transmission


def get_simulation_weights(
    energy,
    cos_zenith,
    cyl_radius=None,
    cyl_height=None,
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
    batched=False,
    mode='cylinder',
    column_depth=None,
    include_transmission=True,
    **ranged_kwargs,
):
    """Fluxless weight A_eff(E, cos; R, H) / (N_gen q(E) p(Omega)), in GeV cm^2 sr.

    cos_zenith is in the arrival convention (see event_arrays). cyl_radius/cyl_height may carry gradients.
    returns a (N,) tensor of weights, with the same dtype and device as energy. n_gen is the number of generated events.
    batched=True uses neutrino_effective_area_many, which gives the same values.
    column_depth (g/cm^2, per event) replaces the cylinder chord in the interaction probability.
    mode='ranged' gives get_ranged_simulation_weights (RangeSampler events), with ranged_kwargs passed on.
    """
    if mode == 'ranged':
        return get_ranged_simulation_weights(
            energy, cos_zenith, column_depth, xsec, transmission, E_min=E_min, E_max=E_max,
            energy_dist=energy_dist, gamma=gamma, cos_range=cos_range, n_gen=n_gen, flavor=flavor,
            average_nu_nubar=average_nu_nubar, batched=batched, include_transmission=include_transmission,
            **ranged_kwargs)
    if ranged_kwargs:
        raise TypeError(f"Unexpected arguments for mode='cylinder': {sorted(ranged_kwargs)}")

    n_gen = energy.numel() if n_gen is None else n_gen
    area_fn, xsec, transmission = _effective_area_fn(batched, xsec, transmission)
    a_eff = area_fn(
        cos_zenith, energy, cyl_radius, cyl_height, xsec, transmission,
        flavor=flavor, average_nu_nubar=average_nu_nubar,
        column_depth=column_depth, include_transmission=include_transmission,
    )
    q_e = energy_pdf(energy, E_min, E_max, energy_dist, gamma)
    p_omega = direction_pdf(cos_zenith, cyl_radius, cyl_height, cos_range, direction_dist)
    return M2_TO_CM2 * a_eff / (n_gen * q_e * p_omega)


def get_ranged_simulation_weights(
    energy,
    cos_zenith,
    column_depth,
    xsec=None,
    transmission=None,
    *,
    E_min=1e2,
    E_max=1e7,
    energy_dist='power_law',
    gamma=1.5,
    cos_range=(-1.0, 1.0),
    azimuth_range=(0.0, 2 * math.pi),
    injection_radius=1100.0,
    n_gen=None,
    flavor='numu',
    average_nu_nubar=True,
    batched=False,
    include_transmission=True,
):
    """LeptonInjector ranged-mode OneWeight T * P_int * pi R_inj^2 / (N_gen q(E) p(Omega)), in GeV cm^2 sr.

    P_int = 1 - exp(-N_A X sigma) with X the per-event total column depth (g/cm^2, RangeSampler's
    'column_depth'). Directions are uniform over cos_range x azimuth_range. include_transmission=False
    gives LeptonWeighter's OneWeight, for fluxes already at the detector; True matches the surface fluxes
    of flux_weights and the cylinder weights.
    """
    if column_depth is None:
        raise ValueError("Ranged weights need each event's column_depth (from RangeSampler).")
    n_gen = energy.numel() if n_gen is None else n_gen
    area_fn, xsec, transmission = _effective_area_fn(batched, xsec, transmission)
    t_p_int = area_fn(
        cos_zenith, energy, None, None, xsec, transmission,
        flavor=flavor, average_nu_nubar=average_nu_nubar, include_projected_area=False,
        column_depth=column_depth, include_transmission=include_transmission,
    )
    q_e = energy_pdf(energy, E_min, E_max, energy_dist, gamma)
    solid_angle = (cos_range[1] - cos_range[0]) * (azimuth_range[1] - azimuth_range[0])
    disk_area = math.pi * injection_radius ** 2  # m^2
    return M2_TO_CM2 * t_p_int * disk_area * solid_angle / (n_gen * q_e)

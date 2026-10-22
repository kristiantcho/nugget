import math
import torch
from nugget.samplers.base_sampler import Sampler

# LeptonInjector ranged-mode injection as used for the P-ONE production in K. Bas, MSc thesis (2026), Sec. 5.1.
# Defaults are the nu_mu CC column of its Table 5.1.

# Earth model of that production (thesis Fig. 5.2): (r_inner, r_outer) in m, density at r_inner and r_outer in g/cm^3.
# Layers quoted with two densities are interpolated linearly in r, with the density rising inward.
EARTH_LAYERS = (
    (0.0,         1_221_500.0, 13.09, 12.76),     # inner core
    (1_221_500.0, 3_480_000.0, 12.17, 9.90),      # outer core
    (3_480_000.0, 5_701_000.0, 5.57, 4.38),       # lower mantle
    (5_701_000.0, 5_771_000.0, 3.99, 3.98),       # lower transition zone
    (5_771_000.0, 5_971_000.0, 3.98, 3.72),       # middle transition zone
    (5_971_000.0, 6_151_000.0, 3.54, 3.44),       # upper transition zone
    (6_151_000.0, 6_346_600.0, 3.38, 3.36),       # uppermost mantle
    (6_346_600.0, 6_356_000.0, 2.9, 2.9),         # inner crust
    (6_356_000.0, 6_375_477.0, 2.65, 2.65),       # bedrock
    (6_375_477.0, 6_378_134.0, 0.997, 0.997),     # ocean water
    (6_378_134.0, 6_478_000.0, 0.000811, 0.000811),  # atmosphere; the model ends here
)
# Distance of the detector origin from the Earth's centre, 500 m above the seabed (thesis Fig. 5.4).
DETECTOR_RADIUS = 6_375_977.0

M_TO_CM = 100.0
MWE_TO_GCM2 = 100.0  # 1 m water equivalent = 100 g/cm^2


def muon_range_mwe(energy):
    """LeptonInjector's 99.9% muon range (CPC 266 (2021) 108018, Eq. 2.2), in m water equivalent, for energy in GeV."""
    # d_a in GeV/mwe, d_b in 1/mwe; the argument of the log has to be E d_b / d_a to be dimensionless.
    d_a, d_b = 0.212 / 1.2, 0.251e-3 / 1.2
    return torch.log1p(energy * d_b / d_a) / d_b


def tau_range_mwe(energy):
    """LeptonInjector's 99.9% tau range (same paper, Eq. 2.3), in m water equivalent, for energy in GeV."""
    return muon_range_mwe(energy) + 3.8e4 * torch.log1p(energy / 5.6e7)


class EarthModel:
    """Radially layered density, zero outside the outermost layer."""

    def __init__(self, layers=EARTH_LAYERS, device=None):
        table = torch.tensor(layers, dtype=torch.float64, device=device)
        self.r_in, self.r_out, self.rho_in, self.rho_out = table.unbind(1)
        self.r_max = float(self.r_out.max())

    def density(self, r):
        """g/cm^3 at radius r (m)."""
        idx = torch.searchsorted(self.r_out, r.contiguous())
        inside = idx < len(self.r_out)
        idx = idx.clamp(max=len(self.r_out) - 1)
        r0, r1 = self.r_in[idx], self.r_out[idx]
        f = ((r - r0) / (r1 - r0)).clamp(0.0, 1.0)
        rho = self.rho_in[idx] + f * (self.rho_out[idx] - self.rho_in[idx])
        return torch.where(inside, rho, torch.zeros_like(rho))


def _interp_grid(t_query, t, C):
    """C (n, T) on the shared grid t (T,), evaluated at t_query (n,)."""
    i = torch.searchsorted(t, t_query.contiguous()).clamp(1, len(t) - 1)
    f = ((t_query - t[i - 1]) / (t[i] - t[i - 1])).clamp(0.0, 1.0)
    c0 = C.gather(1, (i - 1).unsqueeze(1)).squeeze(1)
    c1 = C.gather(1, i.unsqueeze(1)).squeeze(1)
    return c0 + f * (c1 - c0)


def _invert_grid(X, t, C):
    """Per row of the non-decreasing C (n, T), the t where C reaches X (n,)."""
    i = torch.searchsorted(C.contiguous(), X.unsqueeze(1).contiguous()).squeeze(1).clamp(1, C.shape[1] - 1)
    c0 = C.gather(1, (i - 1).unsqueeze(1)).squeeze(1)
    c1 = C.gather(1, i.unsqueeze(1)).squeeze(1)
    f = ((X - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0.0, 1.0)
    return t[i - 1] + f * (t[i] - t[i - 1])


class RangeSampler(Sampler):
    """LeptonInjector ranged-mode event sampler.

    Energy ~ E^-gamma, arrival direction uniform in cos(zenith) and azimuth, point of closest approach
    uniform on a disk of injection_radius perpendicular to the track, and the vertex drawn uniformly in
    column depth over 2 * endcap_length about it, extended upstream by the lepton range.
    Events have the CylinderSampler keys (zenith/azimuth/direction are the travel direction), plus
    'column_depth', the total column depth X_total of the sampling interval in g/cm^2, which the
    OneWeight needs for the forced-interaction probability.

    Parameters
    ----------
    gamma, E_min, E_max : spectral index and energy range in GeV (default 1.5, 1e2, 1e7)
    energy_dist : 'power_law' (default, E^-gamma) or 'log_uniform' (uniform in log E, i.e. gamma = 1)
    zenith_range, azimuth_range : arrival-direction ranges in rad (default (0, pi), (0, 2 pi))
    injection_radius, endcap_length : R_inj and L_end in m (default 1100, 1100)
    range_mode : 'muon' (default), 'tau', 'none', or a callable E [GeV] -> range [m w.e.]
    detector_center : nugget coordinates of the injection origin (default (0, 0, 0))
    detector_radius, earth_layers : position in and layers of the Earth model (defaults above)
    column_depth_step : integration step near the detector in m (default 10)
    chunk_size : events per column-depth integration pass, to bound memory (default 1000)
    seed : seeds a generator created once, so successive calls draw different events
    """

    def __init__(self, device=None, dim=3, domain_size=2, **kwargs):
        super().__init__(device, dim, domain_size)
        self.kwargs = kwargs
        self.event_type = kwargs.get('event_type', 'signal')
        self.gamma = float(kwargs.get('gamma', 1.5))
        self.E_min = float(kwargs.get('E_min', 1e2))
        self.E_max = float(kwargs.get('E_max', 1e7))  # 1e6 for the other samples in the thesis
        self.energy_dist = kwargs.get('energy_dist', 'power_law')
        if self.energy_dist not in ('power_law', 'log_uniform'):
            raise ValueError(f"energy_dist must be 'power_law' or 'log_uniform', got {self.energy_dist!r}")
        self.zenith_range = tuple(kwargs.get('zenith_range', (0.0, math.pi)))
        self.azimuth_range = tuple(kwargs.get('azimuth_range', (0.0, 2 * math.pi)))
        self.injection_radius = float(kwargs.get('injection_radius', 1100.0))
        self.endcap_length = float(kwargs.get('endcap_length', 1100.0))
        self.range_mode = kwargs.get('range_mode', 'muon')
        self.detector_radius = float(kwargs.get('detector_radius', DETECTOR_RADIUS))
        self.detector_center = torch.as_tensor(kwargs.get('detector_center', (0.0, 0.0, 0.0)),
                                               dtype=torch.float64, device=self.device)
        self.earth = EarthModel(kwargs.get('earth_layers', EARTH_LAYERS), device=self.device)
        self.column_depth_step = float(kwargs.get('column_depth_step', 10.0))
        self.chunk_size = int(kwargs.get('chunk_size', 1000))
        seed = kwargs.get('seed', None)
        self.rng = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)

        # Path-length grid upstream of the far end of the fixed segment: fine where the vertex can lie in
        # water or rock, coarse beyond, out to past the longest atmospheric path inside the model.
        fine_end = 60_000.0
        fine = torch.arange(0.0, fine_end, self.column_depth_step, dtype=torch.float64, device=self.device)
        coarse = torch.arange(fine_end, 2_000_000.0 + 1.0, 1000.0, dtype=torch.float64, device=self.device)
        self.t_grid = torch.cat([fine, coarse])

    def _rand(self, n):
        return torch.rand(n, generator=self.rng, device=self.device, dtype=torch.float64)

    def sample_energy(self, n):
        """Power law E^-gamma, or uniform in log E, on [E_min, E_max]."""
        u = self._rand(n)
        if self.energy_dist == 'log_uniform' or abs(self.gamma - 1.0) < 1e-12:
            return self.E_min * (self.E_max / self.E_min) ** u
        k = 1.0 - self.gamma
        return (self.E_min ** k + u * (self.E_max ** k - self.E_min ** k)) ** (1.0 / k)

    def lepton_range_column_depth(self, energy):
        """Range extension of the sampling interval, in g/cm^2."""
        if self.range_mode == 'none':
            return torch.zeros_like(energy)
        if self.range_mode == 'muon':
            return muon_range_mwe(energy) * MWE_TO_GCM2
        if self.range_mode == 'tau':
            return tau_range_mwe(energy) * MWE_TO_GCM2
        return torch.as_tensor(self.range_mode(energy), dtype=torch.float64, device=self.device) * MWE_TO_GCM2

    def _column_depth(self, start, d):
        """(n, T) column depth in g/cm^2 from start (n, 3) upstream along -d, at the path lengths t_grid."""
        q0 = start - self.detector_center
        q0 = q0 + torch.tensor([0.0, 0.0, self.detector_radius], dtype=torch.float64, device=self.device)
        t = self.t_grid
        # |q0 - t d|^2 = |q0|^2 - 2 t (q0 . d) + t^2, for unit d
        r2 = (q0 * q0).sum(1, keepdim=True) - 2.0 * t * (q0 * d).sum(1, keepdim=True) + t ** 2
        rho = self.earth.density(torch.sqrt(r2.clamp_min(0.0)))
        steps = 0.5 * (rho[:, 1:] + rho[:, :-1]) * (t[1:] - t[:-1]) * M_TO_CM
        return torch.cat([torch.zeros_like(rho[:, :1]), torch.cumsum(steps, dim=1)], dim=1)

    def _sample_chunk(self, n):
        energy = self.sample_energy(n)

        th_lo, th_hi = self.zenith_range
        cos_zen = math.cos(th_hi) + self._rand(n) * (math.cos(th_lo) - math.cos(th_hi))
        zen = torch.acos(cos_zen.clamp(-1.0, 1.0))
        az = self.azimuth_range[0] + self._rand(n) * (self.azimuth_range[1] - self.azimuth_range[0])
        # The sampled angles are where the neutrino comes from; it travels the opposite way.
        arrival = torch.stack([torch.sin(zen) * torch.cos(az), torch.sin(zen) * torch.sin(az), cos_zen], dim=1)
        d = -arrival

        # Point of closest approach, uniform on the disk perpendicular to d.
        ref = torch.zeros_like(d)
        ref[:, 2] = 1.0
        ref[d[:, 2].abs() > 0.9] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=self.device)
        e1 = torch.linalg.cross(d, ref)
        e1 = e1 / e1.norm(dim=1, keepdim=True)
        e2 = torch.linalg.cross(d, e1)
        rad = self.injection_radius * torch.sqrt(self._rand(n))
        psi = 2 * math.pi * self._rand(n)
        pca = self.detector_center + rad.unsqueeze(1) * (torch.cos(psi).unsqueeze(1) * e1 + torch.sin(psi).unsqueeze(1) * e2)

        # Column depth measured upstream from the downstream end of the fixed segment.
        end = pca + self.endcap_length * d
        C = self._column_depth(end, d)
        x_fixed = _interp_grid(torch.full_like(energy, 2 * self.endcap_length), self.t_grid, C)
        # The range extension cannot reach past the edge of the Earth model.
        x_total = torch.minimum(x_fixed + self.lepton_range_column_depth(energy), C[:, -1])

        # X_trav is uniform in [0, X_total] from the upstream entry point, i.e. X_total - X_trav from the end.
        x_trav = self._rand(n) * x_total
        t_vertex = _invert_grid(x_total - x_trav, self.t_grid, C)
        vertex = end - t_vertex.unsqueeze(1) * d

        zen_travel = math.pi - zen
        az_travel = torch.remainder(az + math.pi, 2 * math.pi)
        return energy, zen_travel, az_travel, vertex, d, x_total

    def sample_events(self, num_events):
        """List of event dicts, as CylinderSampler.sample_events."""
        events = []
        for start in range(0, num_events, self.chunk_size):
            n = min(self.chunk_size, num_events - start)
            energy, zen, az, vertex, d, x_total = self._sample_chunk(n)
            for i in range(n):
                events.append({
                    'energy': energy[i:i + 1],
                    'zenith': zen[i:i + 1],
                    'azimuth': az[i:i + 1],
                    'position': vertex[i:i + 1],
                    'direction': d[i],
                    'column_depth': x_total[i:i + 1],
                })
        return events

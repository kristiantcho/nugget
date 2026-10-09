"""JAX port of :mod:`nugget.surrogates.LightSabre`.

Closed-form Cherenkov light yield, so this is a direct translation with no models
to convert. Numerically it should match `LightSabre.call_batched` to float64
round-off; the constants and the clamping are deliberately identical, including
the ``sqrt(sum(x**2) + 1e-12)`` in place of a norm (whose Jacobian is 0/0 for a
detector point sitting exactly on the track).

The work lives in module-level pure functions so they jit and vmap cleanly;
:class:`LightSabreJax` is a thin wrapper mirroring the torch API.

Requires ``jax_enable_x64``; see :func:`nugget.utils.jax_bridge.configure_jax`.
"""

from typing import NamedTuple

import numpy as np

__all__ = ["LightSabreConfig", "call_batched", "photons_per_m", "LightSabreJax"]

# Polynomial fit, photons per metre vs log10(E). Same order as the torch model:
# c0 + c1*x + ... + c5*x^5.
_POLY = (
    4.9489616410707695,
    -2.4858046180252362,
    1.1885034976827853,
    -0.2015848374875856,
    0.01626011917463439,
    -0.0004947317263294414,
)
_BARE_PHOTONS_PER_M = 40528.49371849151     # 300-800 nm
_LAMB = 0.1879
_KAPPA = 0.02055


class LightSabreConfig(NamedTuple):
    """Everything `LightSabre.__init__` reads off kwargs, as one static bundle."""
    effective_photocathode_area: float = 84e-4
    refractive_index: float = 1.3
    lambda_abs: float = 44.7
    lambda_sca: float = 57.4
    scattering_tau: float = 0.924
    n0A: float = 63894457.33843762
    min_energy: float = 1e2
    max_energy: float = 1e8
    poisson_rate_cap: float = 1e8
    particle_mode: str = "track"

    @classmethod
    def from_torch(cls, surrogate):
        """Mirror an existing torch `LightSabre`, so the two can be compared."""
        kw = getattr(surrogate, "kwargs", {}) or {}
        return cls(
            effective_photocathode_area=float(surrogate.effective_photocathode_area),
            refractive_index=float(surrogate.refractive_index),
            lambda_abs=float(kw.get("lambda_abs", 44.7)),
            lambda_sca=float(kw.get("lambda_sca", 57.4)),
            scattering_tau=float(surrogate.scattering_tau),
            n0A=float(surrogate.n0A),
            min_energy=float(surrogate.min_energy),
            max_energy=float(surrogate.max_energy),
            poisson_rate_cap=float(surrogate.poisson_rate_cap),
            particle_mode=str(surrogate.particle_mode),
        )


def _optical_lengths(cfg):
    """(sin_theta_c, lambda_p, lambda_mu) -- all constants, computed in numpy."""
    theta_c = np.arccos(1.0 / cfg.refractive_index)
    sin_theta_c = np.sin(theta_c)
    lambda_sca = cfg.lambda_sca / (1.0 - cfg.scattering_tau)
    lambda_p = np.sqrt(cfg.lambda_abs * lambda_sca / 3.0)
    zeta = max(np.exp(-lambda_sca / cfg.lambda_abs), 1e-12)
    lambda_c = lambda_sca / (3.0 * zeta)
    lambda_mu = max(lambda_c / sin_theta_c ** 2 * 2.0 / (np.pi * lambda_p), 1e-12)
    return float(sin_theta_c), float(lambda_p), float(lambda_mu)


def _cascade_lengths(cfg):
    lamda_a = cfg.lambda_abs
    lamda_e = cfg.lambda_sca / (1.0 - cfg.scattering_tau)
    lamda_p = np.sqrt(lamda_a * lamda_e / 3.0) / 1.07
    zeta = np.exp(-lamda_e / lamda_a)
    lamda_c = lamda_e / (3.0 * zeta)
    return float(lamda_p), float(lamda_c)


def photons_per_m(energy, cfg=LightSabreConfig()):
    """Cherenkov yield per metre, photons/m. `energy` in GeV, any shape.

    Note the clamp to [min_energy, max_energy] zeroes the energy derivative
    outside that window -- as it does in the torch model.
    """
    import jax.numpy as jnp

    e_sane = jnp.clip(energy, cfg.min_energy, cfg.max_energy)
    log_e = jnp.log10(e_sane)

    poly = jnp.asarray(_POLY[0], dtype=e_sane.dtype)
    for i in range(1, len(_POLY)):
        poly = poly + _POLY[i] * log_e ** i
    lightyield = 10.0 ** poly

    ladd = _LAMB + _KAPPA * jnp.log(e_sane)
    return lightyield + (1.0 + ladd) * _BARE_PHOTONS_PER_M


def call_batched(track_pos, track_dir, track_energy, om_positions,
                 cfg=LightSabreConfig()):
    """(n_events, n_points) expected photons.

    Shapes mirror the torch method: positions/directions (n_events, 3),
    energies (n_events,), om_positions (n_points, 3). Directions need not be
    unit vectors.
    """
    import jax.numpy as jnp

    dir_norm = jnp.maximum(jnp.linalg.norm(track_dir, axis=1, keepdims=True), 1e-12)
    track_dir = track_dir / dir_norm

    diff = om_positions[None, :, :] - track_pos[:, None, :]   # (E, P, 3)

    if cfg.particle_mode == "track":
        cross = jnp.cross(diff, jnp.broadcast_to(track_dir[:, None, :], diff.shape))
        # sqrt(sum^2 + eps), not norm(): norm's Jacobian is 0/0 on the track line.
        distances = jnp.sqrt(jnp.sum(cross ** 2, axis=2) + 1e-12)

        sin_theta_c, lambda_p, lambda_mu = _optical_lengths(cfg)
        l0 = photons_per_m(track_energy, cfg)                 # (E,)
        d_safe = jnp.maximum(distances, 1e-6)

        numerator = (l0 * cfg.effective_photocathode_area
                     / (2.0 * np.pi * sin_theta_c))[:, None]
        numerator = numerator * jnp.exp(-d_safe / lambda_p)
        denominator = jnp.maximum(
            jnp.sqrt(lambda_mu * d_safe) * jnp.tanh(jnp.sqrt(d_safe / lambda_mu)),
            1e-12,
        )
        light_yield = numerator / denominator
    else:
        distances = jnp.sqrt(jnp.sum(diff ** 2, axis=2) + 1e-12)
        lamda_p, lamda_c = _cascade_lengths(cfg)
        r_safe = jnp.maximum(distances, 1e-6)
        photon_yield = (cfg.n0A / (4.0 * np.pi) * jnp.exp(-r_safe / lamda_p)
                        / (lamda_c * r_safe * jnp.tanh(r_safe / lamda_c)))
        light_yield = photon_yield * (track_energy / 3e5)[:, None]

    light_yield = jnp.nan_to_num(light_yield, nan=0.0,
                                 posinf=cfg.poisson_rate_cap, neginf=0.0)
    return jnp.maximum(light_yield, 0.0)


def direction_from_angles(zenith, azimuth):
    """Spherical -> unit cartesian, matching the torch convention."""
    import jax.numpy as jnp
    st = jnp.sin(zenith)
    return jnp.stack([st * jnp.cos(azimuth), st * jnp.sin(azimuth),
                      jnp.cos(zenith)], axis=-1)


class LightSabreJax:
    """Thin object wrapper, for parity with the torch `LightSabre` API."""

    def __init__(self, domain_size=2, dim=3, **kwargs):
        if dim != 3:
            raise ValueError("LightSabre only supports 3D geometry")
        self.dim = dim
        self.domain_size = domain_size
        fields = LightSabreConfig._fields
        self.cfg = LightSabreConfig(**{k: v for k, v in kwargs.items() if k in fields})
        self.kwargs = kwargs

    @classmethod
    def from_torch(cls, surrogate):
        obj = cls(domain_size=getattr(surrogate, "domain_size", 2))
        obj.cfg = LightSabreConfig.from_torch(surrogate)
        return obj

    def call_batched(self, track_pos, track_dir, track_energy, om_positions):
        return call_batched(track_pos, track_dir, track_energy, om_positions, self.cfg)

    def light_yield_surrogate_batched(self, om_positions, event_params_list):
        import jax.numpy as jnp
        pos, dirs, energies = events_to_arrays(event_params_list)
        return call_batched(jnp.asarray(pos), jnp.asarray(dirs),
                            jnp.asarray(energies), om_positions, self.cfg)


def events_to_arrays(event_params_list):
    """nugget event dicts -> (positions (E,3), directions (E,3), energies (E,)).

    Returns numpy, so this can run outside any traced context. Accepts torch
    tensors, numpy arrays or scalars, and 'direction' or 'zenith'/'azimuth'.
    """
    def as_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu()
        return np.asarray(x, dtype=np.float64)

    positions, directions, energies = [], [], []
    for ep in event_params_list:
        positions.append(as_np(ep["position"]).reshape(3))
        energies.append(as_np(ep["energy"]).reshape(()))
        if "direction" in ep:
            directions.append(as_np(ep["direction"]).reshape(3))
        else:
            zen = as_np(ep["zenith"]).reshape(())
            azi = as_np(ep["azimuth"]).reshape(())
            st = np.sin(zen)
            directions.append(np.array([st * np.cos(azi), st * np.sin(azi),
                                        np.cos(zen)]))
    return (np.stack(positions), np.stack(directions), np.stack(energies))

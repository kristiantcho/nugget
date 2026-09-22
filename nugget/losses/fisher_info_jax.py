"""Pure-JAX Poisson-mean Fisher information and weighted resolution.

A jit-compiled port of the ``compute_fisher_info_poisson_batched_events`` path in
:mod:`nugget.losses.fisher_info_helpers`, plus the resolution reduction from
``WeightedResolutionLoss``, built on :mod:`nugget.surrogates.LightSabreJax`.

Per event ``e`` and point ``i``::

    F_{e,i} = (dlambda_{e,i}/dtheta)(dlambda_{e,i}/dtheta)^T / lambda_{e,i}

with lambda the LightSabre expectation -- no Poisson sampling and no
zero-response gate, so every point contributes. Points are summed into their
string and weighted by ``sigmoid(string_weights)``.

Nothing here touches torch: inputs and outputs are JAX arrays throughout.
:class:`nugget.utils.hybrid_optimizer.JaxLossAdapter` converts to and from torch
and splices the gradients back into the torch graph, which is what makes the
result differentiable with respect to both ``points_3d`` and ``string_weights``.

Requires ``jax_enable_x64``; see :func:`nugget.utils.jax_bridge.configure_jax`.
"""

import numpy as np

from nugget.surrogates.LightSabreJax import (
    LightSabreConfig, call_batched, events_to_arrays,
)

__all__ = ["JaxResolutionLoss", "directional_resolution", "build_value_and_grad"]

_PARAM_DIMS = {"energy": 1, "direction": 3, "position": 3}


def _as_np(x):
    """Duck-typed -> numpy, so this module never imports torch."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64)


def _theta_layout(fisher_info_params):
    """[(name, offset, size), ...] and the total flat width."""
    layout, offset = [], 0
    for name in fisher_info_params:
        size = _PARAM_DIMS[name]
        layout.append((name, offset, size))
        offset += size
    return layout, offset


def directional_resolution(F3, n):
    """68% containment angular resolution from the (N,3,3) direction block.

    Projects onto the tangent plane perpendicular to the track and uses the
    closed-form 2x2 eigenvalues -- ``eigvalsh``'s backward carries
    1/(lambda_i - lambda_j), which blows up at the degeneracy you hit whenever
    the Fisher matrix is near zero.
    """
    import jax.numpy as jnp

    n = n / jnp.linalg.norm(n, axis=1, keepdims=True)

    z_ref = jnp.array([0.0, 0.0, 1.0], dtype=n.dtype)
    x_ref = jnp.array([1.0, 0.0, 0.0], dtype=n.dtype)
    parallel = jnp.abs(n @ z_ref) > 0.9
    ref = jnp.where(parallel[:, None], x_ref[None, :], z_ref[None, :])

    b1 = jnp.cross(n, ref)
    b1 = b1 / jnp.linalg.norm(b1, axis=1, keepdims=True)
    b2 = jnp.cross(n, b1)
    b2 = b2 / jnp.linalg.norm(b2, axis=1, keepdims=True)
    B = jnp.stack([b1, b2], axis=2)                       # (N, 3, 2)

    F2 = jnp.einsum('nab,nbc,ncd->nad', jnp.swapaxes(B, 1, 2), F3, B)   # (N, 2, 2)
    Cov2 = jnp.linalg.inv(F2)

    c00, c11, c01 = Cov2[:, 0, 0], Cov2[:, 1, 1], Cov2[:, 0, 1]
    half_tr = 0.5 * (c00 + c11)
    # Sum of squares, so the sqrt never sees a negative argument the way the
    # tr^2 - 4det form can at degeneracy.
    half_gap = jnp.sqrt((0.5 * (c00 - c11)) ** 2 + c01 ** 2)
    eigvals = jnp.stack([half_tr - half_gap, half_tr + half_gap], axis=1)
    return 1.515 * jnp.sqrt(jnp.mean(eigvals, axis=1))


def build_value_and_grad(fisher_info_params, cfg, n_strings, resolution_type,
                         metric, use_relative_energy, uninformative_fisher_value):
    """Build the jitted ``value_and_grad`` core. Static config is closed over."""
    import jax
    import jax.numpy as jnp

    layout, total_dims = _theta_layout(fisher_info_params)

    def lam_of_theta(theta, pos, direction, energy, points):
        """lambda at every point for one event, as a function of flat theta."""
        p_val, d_val, e_val = pos, direction, energy
        for name, off, size in layout:
            if name == "energy":
                e_val = theta[off:off + 1].reshape(())
            elif name == "direction":
                d_val = theta[off:off + 3]
            else:
                p_val = theta[off:off + 3]
        return call_batched(p_val.reshape(1, 3), d_val.reshape(1, 3),
                            e_val.reshape(1), points, cfg).reshape(-1)

    def fisher_by_string(points, theta0, poss, dirs, energies, point_to_string,
                         valid_point):
        """(E, n_strings, D, D)."""
        def per_event(theta, pos, direction, energy):
            lam = lam_of_theta(theta, pos, direction, energy, points)            # (P,)
            J = jax.jacfwd(lam_of_theta)(theta, pos, direction, energy, points)  # (P, D)
            return lam, J

        lam, J = jax.vmap(per_event)(theta0, poss, dirs, energies)   # (E,P), (E,P,D)

        # Poisson-mean Fisher, no zero-response gate: every point contributes.
        outer = jnp.einsum('epi,epj->epij', J, J)
        outer = outer / jnp.maximum(lam, 1e-10)[:, :, None, None]
        outer = outer * valid_point[None, :, None, None]

        n_events = theta0.shape[0]
        zeros = jnp.zeros((n_events, n_strings, total_dims, total_dims),
                          dtype=outer.dtype)
        # Scatter-add: duplicate indices accumulate, and .at[].add is
        # differentiable, unlike an in-place index_add_.
        return zeros.at[:, point_to_string].add(outer)

    def resolution(F, dirs, energies):
        if resolution_type == "angular":
            _, off, _ = next(x for x in layout if x[0] == "direction")
            return directional_resolution(F[:, off:off + 3, off:off + 3], dirs)
        _, off, _ = next(x for x in layout if x[0] == "energy")
        eye = jnp.eye(total_dims, dtype=F.dtype)
        cov = jnp.linalg.inv(F + 1e-20 * eye)
        res = jnp.sqrt(cov[:, off, off])
        return res / energies if use_relative_energy else res

    def core(points, string_weights, theta0, poss, dirs, energies,
             point_to_string, valid_point, string_present):
        F_str = fisher_by_string(points, theta0, poss, dirs, energies,
                                 point_to_string, valid_point)

        # Strings with no points get an uninformative diagonal rather than zero,
        # and any non-finite block is replaced outright: downstream this is
        # combined as sum_s sigmoid(w_s) * F_s, and 0 * NaN = NaN, so driving a
        # weight to zero would not neutralise a bad string.
        eye = jnp.eye(total_dims, dtype=F_str.dtype)
        fill = (uninformative_fisher_value * eye)[None, None, :, :]
        F_str = jnp.where(string_present[None, :, None, None], F_str, fill)
        finite = jnp.all(jnp.isfinite(F_str), axis=(2, 3))[:, :, None, None]
        F_str = jnp.where(finite, jnp.nan_to_num(F_str, nan=0.0, posinf=0.0,
                                                 neginf=0.0), fill)

        w = jax.nn.sigmoid(string_weights)
        F = jnp.einsum('s,esij->eij', w, F_str)

        res = resolution(F, dirs, energies)

        ok = jnp.isfinite(res) & (res > 1e-15)
        clean = jnp.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
        safe = jnp.where(ok, jnp.clip(clean, 1e-15, None), 1e6)
        if metric == "fom":
            total = 1.0 / jnp.sqrt(jnp.mean(1.0 / safe ** 2))
        elif metric == "median":
            total = jnp.median(safe)
        else:
            total = jnp.mean(safe)
        total = jnp.where(jnp.any(ok), total, jnp.asarray(1.0, dtype=total.dtype))
        return total, (res, F_str)

    return jax.jit(jax.value_and_grad(core, argnums=(0, 1), has_aux=True))


class JaxResolutionLoss:
    """Angular / energy resolution from the LightSabre Poisson Fisher, in JAX.

    Implements the adapter protocol documented on
    :class:`nugget.utils.hybrid_optimizer.JaxLossAdapter`; wrap it in that to use
    it as a nugget loss.

    Parameters
    ----------
    fisher_info_params : sequence of {'energy', 'direction', 'position'}
        Parameters to differentiate lambda with respect to, in flat-theta order.
    resolution_type : {'angular', 'energy'}
        'angular' needs 'direction' present, 'energy' needs 'energy'.
    lightsabre : LightSabre, LightSabreJax, LightSabreConfig or None
        Source of the optical constants. A torch `LightSabre` is mirrored via
        `LightSabreConfig.from_torch`, which is what makes the two comparable.
    """

    diff_keys = ("points_3d", "string_weights")

    def __init__(self, fisher_info_params=("direction", "position"),
                 resolution_type="angular", lightsabre=None,
                 uninformative_fisher_value=1e-6):
        bad = [p for p in fisher_info_params if p not in _PARAM_DIMS]
        if bad:
            raise ValueError(f"fisher_info_params must be a subset of "
                             f"{sorted(_PARAM_DIMS)}, got {bad}")
        if resolution_type not in ("angular", "energy"):
            raise ValueError("resolution_type must be 'angular' or 'energy'")
        if resolution_type == "angular" and "direction" not in fisher_info_params:
            raise ValueError("angular resolution needs 'direction' in fisher_info_params")
        if resolution_type == "energy" and "energy" not in fisher_info_params:
            raise ValueError("energy resolution needs 'energy' in fisher_info_params")

        self.fisher_info_params = tuple(fisher_info_params)
        self.resolution_type = resolution_type
        self.uninformative_fisher_value = float(uninformative_fisher_value)

        if lightsabre is None:
            self.cfg = LightSabreConfig()
        elif isinstance(lightsabre, LightSabreConfig):
            self.cfg = lightsabre
        elif hasattr(lightsabre, "cfg"):
            self.cfg = lightsabre.cfg
        else:
            self.cfg = LightSabreConfig.from_torch(lightsabre)

        self._fns = {}
        # Defaults, so value_and_grad works standalone if prepare has not run.
        self._metric = "fom"
        self._use_rel_e = False

    # -- adapter protocol ----------------------------------------------

    @property
    def loss_key(self):
        return ("angular_resolution" if self.resolution_type == "angular"
                else "energy_resolution")

    @property
    def aux_keys(self):
        return (f"{self.loss_key}_per_event", "fisher_info_per_string_per_event")

    def default_input(self, key, geom_dict):
        """Value for a diff key the geometry does not provide.

        Only ``string_weights``: a geometry without weights contributes every
        string fully, and the core applies sigmoid, so feed it a large logit.
        """
        import jax.numpy as jnp
        if key == "string_weights":
            return jnp.full(len(geom_dict["string_xy"]), 30.0)
        raise KeyError(f"geom_dict has no '{key}' and no default is defined")

    def prepare(self, geom_dict, **loss_params):
        """-> (static jax kwargs for the core, plain extras to pass through)."""
        import jax.numpy as jnp

        string_xy = geom_dict.get("string_xy", None)
        points = geom_dict.get("points_3d", None)
        if points is None or string_xy is None:
            raise ValueError("geom_dict needs 'points_3d' and 'string_xy'")

        params = loss_params.get("signal_event_params", None)
        sampler = loss_params.get("signal_sampler", None)
        if params is None:
            if sampler is None:
                raise ValueError("provide signal_event_params or a signal_sampler")
            params = sampler.sample_events(int(loss_params.get("num_events", 100)))

        poss, dirs, energies = events_to_arrays(params)
        theta0 = self._theta0(poss, dirs, energies)
        p2s, valid_point, string_present = self._point_to_string(
            _as_np(points), string_xy)

        static = dict(
            theta0=jnp.asarray(theta0), poss=jnp.asarray(poss),
            dirs=jnp.asarray(dirs), energies=jnp.asarray(energies),
            point_to_string=jnp.asarray(p2s), valid_point=jnp.asarray(valid_point),
            string_present=jnp.asarray(string_present),
        )
        # Only the two Python-level switches need carrying over to the jit key;
        # n_strings is read back off the weights array so the traced shapes and
        # the compiled core can never disagree.
        self._metric = loss_params.get("fisher_res_metric", "fom")
        self._use_rel_e = bool(loss_params.get("use_relative_energy", False))
        return static, {"resolution_params": params}

    def value_and_grad(self, points, string_weights, **static):
        """((value, aux), grads) -- all JAX, grads parallel to `diff_keys`."""
        n_strings = int(string_weights.shape[0])
        key = (n_strings, self._metric, self._use_rel_e)
        if key not in self._fns:
            self._fns[key] = build_value_and_grad(
                self.fisher_info_params, self.cfg, n_strings,
                self.resolution_type, self._metric, self._use_rel_e,
                self.uninformative_fisher_value,
            )
        return self._fns[key](points, string_weights, **static)

    # -- helpers -------------------------------------------------------

    def _theta0(self, poss, dirs, energies):
        layout, total = _theta_layout(self.fisher_info_params)
        out = np.zeros((poss.shape[0], total), dtype=np.float64)
        for name, off, _ in layout:
            if name == "energy":
                out[:, off] = energies
            elif name == "direction":
                out[:, off:off + 3] = dirs
            else:
                out[:, off:off + 3] = poss
        return out

    @staticmethod
    def _point_to_string(points_np, string_xy):
        """Exact-(x, y) point -> string map, plus which points and strings hit.

        Matching by equality has no derivative, so it is done once in numpy and
        handed in as data. Gradients reach `string_xy` through `points_3d` and
        the geometry's own graph instead.
        """
        if hasattr(string_xy, "detach"):
            # One transfer. Iterating a CUDA tensor row by row would cost a
            # device sync per string.
            sxy = string_xy.detach().cpu().numpy().reshape(-1, 2)
        else:
            sxy = np.asarray([[float(s[0]), float(s[1])] for s in string_xy],
                             dtype=np.float64)
        matches = ((points_np[:, 0][None, :] == sxy[:, 0][:, None]) &
                   (points_np[:, 1][None, :] == sxy[:, 1][:, None]))  # (S, P)
        valid = matches.any(axis=0)
        p2s = np.where(valid, matches.argmax(axis=0), 0).astype(np.int32)
        return p2s, valid.astype(np.float64), matches.any(axis=1)

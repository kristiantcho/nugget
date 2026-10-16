"""Optimizer that accumulates gradients across repeated and cross-device losses.

`HybridOptimizer` behaves like :class:`nugget.utils.basic_optimizer.Optimizer`
(same ALM, ConFIG, sigmoid wrapping, NaN reverting, checkpointing, visualization)
and adds two things:

* **Repeated evaluation** -- a loss can be called several times per iteration
  with the gradients summed before a single step, each call backwarded and freed
  immediately, so an update can see more events than fit on the GPU at once.
* **Per-loss devices** -- a loss can run on its own GPU, with torch and JAX
  losses (via :mod:`nugget.utils.jax_bridge`) landing in the same ``.grad``.

:class:`JaxLossAdapter` turns a pure-JAX loss into a torch one, and
``HybridOptimizer`` applies it automatically to any entry of ``loss_func_dict``
implementing that protocol -- so a JAX loss can be registered directly and can
also be repeated or pinned to a device like any other.

Configuration lives in ``loss_params_dict``::

    loss_params = {
        'loss_repeats':   {'angular_resolution_loss': 8, 'fisher_analysis_loss': 4},
        'loss_devices':   {'angular_resolution_loss': 'cuda:1'},
        'loss_reduction': {'angular_resolution_loss': 'mean'},   # or 'sum'
        'loss_accumulators': {'fisher_analysis_loss': 'analysis'},  # default: 'base'
        'empty_cache_per_repeat': True,
    }

``loss_accumulators`` picks the accumulator class from
:mod:`nugget.utils.grad_accumulation` (a name in ``ACCUMULATORS`` or a class);
losses not listed use ``GradientAccumulator``. Losses configuring no repeats, device
or accumulator are passed through untouched, so a run that configures nothing is
identical to `basic_optimizer`.

Accumulation happens on the **raw** loss, before weighting, the sigmoid wrapper
and the ALM augmentation; the result is spliced back into the graph so torch
applies those itself. This matters because ``sigmoid(s.W.L) - 0.5`` is nonlinear
in ``L``: averaging per-repeat sigmoided values would not equal the sigmoid of
the averaged value, so a repeated run would quietly disagree with an unrepeated
one.
"""

import torch

# Aliased: a bare `Optimizer` here would make `hybrid_optimizer.Optimizer`
# resolve to the base class, which silently skips the JAX/accumulation wrapping.
from nugget.utils.basic_optimizer import Optimizer as _BaseOptimizer
from nugget.utils.jax_bridge import inject_gradient, to_torch
from nugget.utils.grad_accumulation import GradientAccumulator, resolve_accumulator, geom_dict_device

# Old name, kept so existing imports keep working.
AccumulatingLoss = GradientAccumulator

__all__ = ["AccumulatingLoss", "JaxLossAdapter", "HybridOptimizer"]


class JaxLossAdapter:
    """Convert a JAX loss's output to torch and splice its gradients in.

    The wrapped loss has the ordinary nugget signature, but returns JAX arrays
    and a ``'_jax_grads'`` entry mapping geom_dict keys to d(loss)/d(that key).
    Gradients land on the geometry tensors, not the optimized leaves, so torch
    carries them the rest of the way through ``update_points``. Each is a true
    partial, so injecting several at once sums rather than double counts.
    """

    GRADS_KEY = "_jax_grads"

    def __init__(self, jax_loss, loss_name, device=None):
        self.jax_loss = jax_loss
        self.loss_name = loss_name
        self.device = device

    def __call__(self, geom_dict, **loss_params):
        import jax

        out = dict(self.jax_loss(geom_dict, **loss_params))
        grads = out.pop(self.GRADS_KEY, None) or {}

        reference = geom_dict.get("points_3d", None)
        if reference is None:
            reference = next((v for v in geom_dict.values() if torch.is_tensor(v)), None)
        if reference is None:
            raise ValueError(f"JaxLossAdapter('{self.loss_name}'): geom_dict has no tensors")
        device = self.device if self.device is not None else reference.device
        dtype = reference.dtype

        # Only inject where there is something to differentiate.
        targets, live_grads = [], []
        for key, grad in grads.items():
            tensor = geom_dict.get(key, None)
            if tensor is None or not tensor.requires_grad:
                continue
            targets.append(tensor)
            live_grads.append(to_torch(grad, device=tensor.device, dtype=tensor.dtype))

        result = {}
        for key, value in out.items():
            result[key] = (to_torch(value, device=device, dtype=dtype)
                           if isinstance(value, jax.Array) else value)

        if self.loss_name not in out:
            raise KeyError(
                f"JaxLossAdapter('{self.loss_name}'): the loss returned "
                f"{sorted(out)} but not '{self.loss_name}'; the key must match "
                "its name in loss_func_dict"
            )
        value = float(out[self.loss_name])
        result[self.loss_name] = (
            inject_gradient(value, live_grads, targets, device=device, dtype=dtype)
            if targets else torch.as_tensor(value, device=device, dtype=dtype)
        )
        return result


class HybridOptimizer(_BaseOptimizer):
    """`Optimizer` with per-loss gradient accumulation and per-loss devices.

    Reads `loss_repeats` / `loss_devices` / `loss_reduction` / `loss_accumulators`
    from ``loss_params_dict`` (see the module docstring) and wraps the affected
    entries of ``loss_func_dict`` in the chosen accumulator.

    `configure_jax` calls :func:`nugget.utils.jax_bridge.configure_jax` at
    construction; it is off by default since it only works before JAX is
    imported, so prefer calling it yourself at the top of your script. All other
    arguments are those of `basic_optimizer.Optimizer`.
    """

    def __init__(self, *args, configure_jax=False, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = bool(verbose)
        self._wrapped_losses = {}
        if configure_jax:
            from nugget.utils.jax_bridge import configure_jax as _configure
            _configure()

    @staticmethod
    def _is_jax_loss(obj):
        """True for a loss that returns JAX arrays (marked by `backend`)."""
        return getattr(obj, "backend", None) == "jax"

    def _wrap_losses(self, loss_func_dict, loss_params_dict):
        """Adapt pure-JAX losses, then wrap whatever asked for repeats/a device."""
        repeats_map = loss_params_dict.get("loss_repeats", {}) or {}
        devices_map = loss_params_dict.get("loss_devices", {}) or {}
        reduction_map = loss_params_dict.get("loss_reduction", {}) or {}
        on_repeat_map = loss_params_dict.get("on_repeat", {}) or {}
        accumulator_map = loss_params_dict.get("loss_accumulators", {}) or {}
        empty_cache = loss_params_dict.get("empty_cache_per_repeat", True)

        unknown = (set(repeats_map) | set(devices_map) | set(accumulator_map)) - set(loss_func_dict)
        if unknown:
            raise KeyError(
                f"loss_repeats / loss_devices / loss_accumulators name losses that are not in "
                f"loss_func_dict: {sorted(unknown)}"
            )

        wrapped = {}
        self._wrapped_losses = {}
        for loss_name, loss_func in loss_func_dict.items():
            # A pure-JAX loss is adapted first, so everything downstream --
            # including AccumulatingLoss -- sees an ordinary torch loss.
            if self._is_jax_loss(loss_func):
                loss_func = JaxLossAdapter(loss_func, loss_name, device=self.device)

            repeats = int(repeats_map.get(loss_name, 1))
            device = devices_map.get(loss_name, None)
            if repeats <= 1 and device is None and loss_name not in accumulator_map:
                wrapped[loss_name] = loss_func
                continue

            hook = on_repeat_map.get(loss_name) if isinstance(on_repeat_map, dict) else on_repeat_map
            accumulator_cls = resolve_accumulator(accumulator_map.get(loss_name))
            accumulating = accumulator_cls(
                loss_func,
                loss_name,
                repeats=repeats,
                device=device,
                reduction=reduction_map.get(loss_name, "mean"),
                owner=self,
                on_repeat=hook,
                empty_cache=empty_cache,
            )
            wrapped[loss_name] = accumulating
            self._wrapped_losses[loss_name] = accumulating

        if self.verbose and self._wrapped_losses:
            print("HybridOptimizer: gradient accumulation plan")
            for loss_name, acc in self._wrapped_losses.items():
                where = acc.device if acc.device is not None else f"{self.device} (primary)"
                print(f"  {loss_name}: {type(acc).__name__}, {acc.repeats}x on {where} "
                      f"[{acc.reduction}]")
        return wrapped

    def optimize(self, loss_func_dict, loss_dict=None, uw_loss_dict=None,
                 loss_weights_dict=None, loss_params_dict=None, n_iter=100,
                 print_freq=10, vis_freq=None, vis_kwargs=None, gif_freq=None,
                 **kwargs):
        if loss_params_dict is None:
            loss_params_dict = {}
        wrapped = self._wrap_losses(loss_func_dict, loss_params_dict)
        return super().optimize(
            loss_func_dict=wrapped,
            loss_dict=loss_dict,
            uw_loss_dict=uw_loss_dict,
            loss_weights_dict=loss_weights_dict,
            loss_params_dict=loss_params_dict,
            n_iter=n_iter,
            print_freq=print_freq,
            vis_freq=vis_freq,
            vis_kwargs=vis_kwargs,
            gif_freq=gif_freq,
            **kwargs,
        )

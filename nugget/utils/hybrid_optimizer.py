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
        'loss_repeats':   {'angular_resolution_loss': 8},
        'loss_devices':   {'angular_resolution_loss': 'cuda:1'},
        'loss_reduction': {'angular_resolution_loss': 'mean'},   # or 'sum'
        'empty_cache_per_repeat': True,
    }

Losses configuring neither repeats nor a device are passed through untouched, so
a run that configures nothing is identical to `basic_optimizer`.

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


class AccumulatingLoss:
    """Wrap a loss so it is evaluated `repeats` times with its gradients summed.

    Each repeat is forwarded, differentiated and freed before the next begins, so
    peak memory is that of a single call.

    Parameters
    ----------
    loss_func : callable
        Any nugget loss: ``loss(geom_dict, **loss_params) -> dict``.
    loss_name : str
        Key the loss is registered under; its returned dict must contain it.
    repeats : int
        Evaluations per optimizer step.
    device : str, torch.device or None
        Evaluate here; None keeps it on the optimizer's device.
    reduction : {'mean', 'sum'}
        'mean' divides by `repeats`, matching the scale of a single call when the
        loss is already a mean over events.
    owner : Optimizer
        Optimizer holding ``geom_dict`` and the device. Any
        `basic_optimizer.Optimizer` works, so this is usable without
        `HybridOptimizer`.
    on_repeat : callable or None
        ``on_repeat(loss_params, k, repeats) -> loss_params``, to vary a seed or
        subsample per repeat. By default every repeat gets identical parameters
        and the loss is expected to redraw internally.
    empty_cache : bool
        Release cached device memory after each repeat.
    pool_extras : bool
        Concatenate the cheap per-event diagnostics across repeats so the
        visualizer plots the whole sample rather than just the last repeat. Only
        1-D ``*_per_event`` tensors and ``resolution_params`` are pooled; larger
        per-event tensors (``fisher_info_per_string_per_event`` and friends) keep
        last-repeat semantics, since pooling those would rebuild the very
        memory peak the repeats exist to avoid.

    Notes
    -----
    Repeats only help if the loss varies between calls. A loss handed fixed
    ``signal_event_params`` with no internal sampling returns the identical
    gradient `repeats` times at `repeats` times the cost.
    """

    def __init__(self, loss_func, loss_name, repeats=1, device=None,
                 reduction="mean", owner=None, on_repeat=None, empty_cache=True,
                 pool_extras=True):
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.loss_func = loss_func
        self.loss_name = loss_name
        self.repeats = max(int(repeats), 1)
        self.device = device
        self.reduction = reduction
        self.owner = owner
        self.on_repeat = on_repeat
        self.empty_cache = bool(empty_cache)
        self.pool_extras = bool(pool_extras)

    # -- helpers -------------------------------------------------------

    def _detached_replica(self, geom_dict, work_device):
        """-> (replica_geom, real_inputs, replica_inputs) on `work_device`.

        Every differentiable float entry becomes an independent leaf. Detaching
        does two jobs: ``autograd.grad(..., retain_graph=False)`` then frees the
        loss's own subgraph without touching the real geometry graph, and each
        gradient is a true *partial*, so summing ``partial_i * d(x_i)/d(leaf)``
        is exactly the total derivative. Differentiating the real tensors instead
        would double-count shared paths -- ``points_3d`` depends on
        ``string_weights`` and both are in geom_dict.

        Aliased keys share one detached copy (``old_string_weights`` is literally
        the same object as ``string_weights``), which preserves that aliasing and
        stops it being counted twice.

        Only the geometry moves; build the loss's own tensors on `work_device`.
        """
        replica = {}
        by_id = {}
        real_inputs = []
        replica_inputs = []

        for key, value in geom_dict.items():
            if not torch.is_tensor(value):
                replica[key] = value
                continue
            if not (value.is_floating_point() and value.requires_grad):
                # Constant geometry still has to sit on the loss's device. When
                # only string_weights is optimized, points_3d lands here -- it is
                # built from string_xy and z_values, so it carries no grad, but
                # the loss very much still reads it. (.to() is a no-op when the
                # device already matches.)
                replica[key] = value.to(work_device)
                continue
            existing = by_id.get(id(value))
            if existing is None:
                existing = value.detach().to(work_device).requires_grad_(True)
                by_id[id(value)] = existing
                real_inputs.append(value)
                replica_inputs.append(existing)
            replica[key] = existing

        return replica, real_inputs, replica_inputs

    def _free(self, device):
        if not self.empty_cache or not torch.cuda.is_available():
            return
        target = torch.device(device)
        if target.type != "cuda":
            return
        with torch.cuda.device(target):
            torch.cuda.empty_cache()

    def _unpack(self, loss_stuff):
        """Mirror `Optimizer.optimize`'s conventions for a loss's return value."""
        if isinstance(loss_stuff, dict):
            return loss_stuff.get(self.loss_name, None), loss_stuff
        if isinstance(loss_stuff, (tuple, list)):
            value = loss_stuff[0] if len(loss_stuff) > 0 else None
            return value, ({self.loss_name: value} if value is not None else {})
        return loss_stuff, ({self.loss_name: loss_stuff} if loss_stuff is not None else {})

    # -- the loss interface --------------------------------------------

    def __call__(self, geom_dict, **loss_params):
        owner_device = getattr(self.owner, "device", None)
        if owner_device is None:
            owner_device = geom_dict_device(geom_dict)
        owner_device = torch.device(owner_device)
        work_device = torch.device(self.device) if self.device is not None else owner_device

        geom, real_inputs, replica_inputs = self._detached_replica(geom_dict, work_device)
        if not replica_inputs:
            raise RuntimeError(
                f"AccumulatingLoss('{self.loss_name}'): no geometry tensor requires "
                "grad. Call init_geometry() before optimize()."
            )

        scale = 1.0 / self.repeats if self.reduction == "mean" else 1.0

        totals = [None] * len(replica_inputs)
        total_value = 0.0
        extras = {}
        pooled = {}
        evaluated = 0

        for k in range(self.repeats):
            params = loss_params
            if self.on_repeat is not None:
                params = self.on_repeat(dict(loss_params), k, self.repeats)

            loss_stuff = self.loss_func(geom, **params)
            value, chunk_extras = self._unpack(loss_stuff)

            if value is None:
                print(f"Warning: {self.loss_name} did not return a valid loss value.")
                del loss_stuff
                continue
            if not torch.is_tensor(value) or value.numel() != 1:
                raise TypeError(
                    f"AccumulatingLoss('{self.loss_name}') needs a scalar tensor, got "
                    f"{type(value).__name__} with "
                    f"{value.numel() if torch.is_tensor(value) else 'no'} elements."
                )

            # retain_graph=True here would keep the per-event intermediates alive
            # and defeat the point of repeating at all.
            grads = torch.autograd.grad(
                value, replica_inputs, retain_graph=False, allow_unused=True
            )
            for i, grad in enumerate(grads):
                if grad is None:
                    continue
                contribution = grad.detach() * scale
                totals[i] = contribution if totals[i] is None else totals[i] + contribution

            total_value += scale * float(value.detach().cpu().item())
            evaluated += 1

            # Detach: these dicts carry per-event / per-string tensors, and a
            # graph-connected copy would pin the graph we just released.
            extras = {
                key: (item.detach() if torch.is_tensor(item) else item)
                for key, item in chunk_extras.items()
                if key != self.loss_name
            }
            if self.pool_extras:
                for key, item in extras.items():
                    if key.endswith("_per_event") and torch.is_tensor(item) and item.dim() == 1:
                        pooled.setdefault(key, []).append(item)
                    elif key == "resolution_params" and isinstance(item, list):
                        pooled.setdefault(key, []).append(item)

            del loss_stuff, value, grads, chunk_extras
            self._free(work_device)

        if evaluated == 0:
            return {self.loss_name: None}

        # Otherwise a failed repeat leaves the mean divided by the intended count
        # rather than the achieved one, shrinking both value and gradient.
        if self.reduction == "mean" and evaluated != self.repeats:
            correction = self.repeats / evaluated
            total_value *= correction
            totals = [None if t is None else t * correction for t in totals]

        # Replace the last repeat's diagnostics with the pooled ones, so the
        # per-event plots show the whole sample the gradient was built from.
        for key, pieces in pooled.items():
            if len(pieces) < 2:
                continue
            if torch.is_tensor(pieces[0]):
                extras[key] = torch.cat(pieces, dim=0)
            else:
                extras[key] = [entry for piece in pieces for entry in piece]

        scalar = inject_gradient(
            total_value, totals, real_inputs,
            device=owner_device, dtype=real_inputs[0].dtype,
        )
        return {self.loss_name: scalar, **extras}


def geom_dict_device(geom_dict):
    """Device of the first tensor in a geom_dict, as a fallback."""
    for value in geom_dict.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


class HybridOptimizer(_BaseOptimizer):
    """`Optimizer` with per-loss gradient accumulation and per-loss devices.

    Reads `loss_repeats` / `loss_devices` / `loss_reduction` from
    ``loss_params_dict`` (see the module docstring) and wraps the affected
    entries of ``loss_func_dict`` in :class:`AccumulatingLoss`.

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
        empty_cache = loss_params_dict.get("empty_cache_per_repeat", True)

        unknown = (set(repeats_map) | set(devices_map)) - set(loss_func_dict)
        if unknown:
            raise KeyError(
                f"loss_repeats / loss_devices name losses that are not in "
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
            if repeats <= 1 and device is None:
                wrapped[loss_name] = loss_func
                continue

            hook = on_repeat_map.get(loss_name) if isinstance(on_repeat_map, dict) else on_repeat_map
            accumulating = AccumulatingLoss(
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
                print(f"  {loss_name}: {acc.repeats}x on {where} "
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

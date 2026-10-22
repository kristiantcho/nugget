"""Gradient accumulation for losses evaluated over more events than fit in memory at once.

`GradientAccumulator` sums the gradients of a loss evaluated `repeats` times. That is exact
only when the loss is itself a sum/mean over events. `AnalysisAccumulator` handles a loss
that is a nonlinear function of additive histograms (the forward-folding analysis loss):

    grad L = dL/dmu |_(mu = sum_k mu_k)  .  sum_k dmu_k/dTheta

i.e. the histograms are pooled over all batches first, the head is differentiated once on
the pooled histograms, and that gradient is pushed back through each batch in turn.
"""

import torch
from nugget.utils.jax_bridge import inject_gradient

__all__ = [
    "GradientAccumulator", "AnalysisAccumulator", "ACCUMULATORS",
    "resolve_accumulator", "geom_dict_device", "detached_replica", "disjoint_batches",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def geom_dict_device(geom_dict):
    """Device of the first tensor in a geom_dict, as a fallback."""
    for value in geom_dict.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def detached_replica(geom_dict, work_device):
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
            # Constant geometry (e.g. points_3d when only string_weights is optimized)
            # still has to sit on the loss's device.
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


def free_device_cache(device):
    """Release cached memory on `device` if it is a CUDA device."""
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return
    with torch.cuda.device(target):
        torch.cuda.empty_cache()


def unpack_loss(loss_stuff, loss_name):
    """-> (value, extras), mirroring `Optimizer.optimize`'s conventions for a loss's return value."""
    if isinstance(loss_stuff, dict):
        return loss_stuff.get(loss_name, None), loss_stuff
    if isinstance(loss_stuff, (tuple, list)):
        value = loss_stuff[0] if len(loss_stuff) > 0 else None
        return value, ({loss_name: value} if value is not None else {})
    return loss_stuff, ({loss_name: loss_stuff} if loss_stuff is not None else {})


def add_grads(totals, grads, scale=1.0):
    """In place: totals[i] += scale * grads[i], skipping None."""
    for i, grad in enumerate(grads):
        if grad is None:
            continue
        contribution = grad.detach() * scale
        totals[i] = contribution if totals[i] is None else totals[i] + contribution


def detach_extras(extras, skip=()):
    """Detached copy of a loss's diagnostics; a graph-connected copy would pin the graph just released."""
    return {key: (item.detach() if torch.is_tensor(item) else item)
            for key, item in extras.items() if key not in skip}


def collect_poolable(extras, pooled):
    """Append the cheap per-event diagnostics (1-D ``*_per_event`` tensors, ``resolution_params``) to `pooled`."""
    for key, item in extras.items():
        if key.endswith("_per_event") and torch.is_tensor(item) and item.dim() == 1:
            pooled.setdefault(key, []).append(item)
        elif key == "resolution_params" and isinstance(item, list):
            pooled.setdefault(key, []).append(item)


def merge_pooled(extras, pooled):
    """Replace the last repeat's diagnostics in `extras` with the pooled ones."""
    for key, pieces in pooled.items():
        if len(pieces) < 2:
            continue
        if torch.is_tensor(pieces[0]):
            extras[key] = torch.cat(pieces, dim=0)
        else:
            extras[key] = [entry for piece in pieces for entry in piece]
    return extras


def _take(value, idx):
    """Index a per-event entry (list, tensor, or dict of either) with an index tensor."""
    if isinstance(value, dict):
        return {key: _take(item, idx) for key, item in value.items()}
    if torch.is_tensor(value):
        return value[idx.to(value.device)]
    return [value[i] for i in idx.tolist()]


def disjoint_batches(per_event, batch_size, rescale_keys=(), generator=None):
    """on_repeat hook: repeat k gets slice k of a fresh permutation of the precomputed per-event entries.

    per_event maps loss-param names to full-length values (list, tensor, or dict of tensors).
    Entries in rescale_keys are multiplied by N / batch_size, for weights normalised to all N events.
    """
    lengths = {len(next(iter(v.values()))) if isinstance(v, dict) else len(v) for v in per_event.values()}
    if len(lengths) != 1:
        raise ValueError(f"per_event entries have different lengths: {sorted(lengths)}")
    n_total = lengths.pop()
    state = {}

    def hook(params, k, repeats):
        if k == 0:
            if repeats * batch_size > n_total:
                raise ValueError(f"{repeats} disjoint batches of {batch_size} need more than the {n_total} events available")
            # New permutation every optimizer step; batches are disjoint within a step.
            state['perm'] = torch.randperm(n_total, generator=generator)
        idx = state['perm'][k * batch_size:(k + 1) * batch_size]
        for key, value in per_event.items():
            params[key] = _take(value, idx)
            if key in rescale_keys:
                params[key] = params[key] * (n_total / batch_size)
        # So the loss does not subsample the batch again.
        params['num_events'] = batch_size
        return params

    return hook


# ----------------------------------------------------------------------
# accumulators
# ----------------------------------------------------------------------

class GradientAccumulator:
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
        Optimizer holding ``geom_dict`` and the device.
    on_repeat : callable or None
        ``on_repeat(loss_params, k, repeats) -> loss_params``, to vary a seed or
        subsample per repeat. By default every repeat gets identical parameters
        and the loss is expected to redraw internally.
    empty_cache : bool
        Release cached device memory after each repeat.
    pool_extras : bool
        Concatenate the cheap per-event diagnostics across repeats so the
        visualizer plots the whole sample rather than just the last repeat.

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

    # -- shared steps --------------------------------------------------

    def _prepare(self, geom_dict):
        """-> (owner_device, work_device, replica_geom, real_inputs, replica_inputs)."""
        owner_device = getattr(self.owner, "device", None)
        if owner_device is None:
            owner_device = geom_dict_device(geom_dict)
        owner_device = torch.device(owner_device)
        work_device = torch.device(self.device) if self.device is not None else owner_device

        geom, real_inputs, replica_inputs = detached_replica(geom_dict, work_device)
        if not replica_inputs:
            raise RuntimeError(
                f"{type(self).__name__}('{self.loss_name}'): no geometry tensor requires "
                "grad. Call init_geometry() before optimize()."
            )
        return owner_device, work_device, geom, real_inputs, replica_inputs

    def _scale(self):
        return 1.0 / self.repeats if self.reduction == "mean" else 1.0

    def _repeat_params(self, loss_params, k):
        if self.on_repeat is None:
            return loss_params
        return self.on_repeat(dict(loss_params), k, self.repeats)

    def _free(self, device):
        if self.empty_cache:
            free_device_cache(device)

    def _finish(self, value, totals, real_inputs, owner_device, extras):
        """Splice the accumulated (value, gradients) back into the real geometry graph."""
        scalar = inject_gradient(
            value, totals, real_inputs,
            device=owner_device, dtype=real_inputs[0].dtype,
        )
        return {self.loss_name: scalar, **extras}

    # -- the loss interface --------------------------------------------

    def __call__(self, geom_dict, **loss_params):
        owner_device, work_device, geom, real_inputs, replica_inputs = self._prepare(geom_dict)
        scale = self._scale()

        totals = [None] * len(replica_inputs)
        total_value = 0.0
        extras = {}
        pooled = {}
        evaluated = 0

        for k in range(self.repeats):
            loss_stuff = self.loss_func(geom, **self._repeat_params(loss_params, k))
            value, chunk_extras = unpack_loss(loss_stuff, self.loss_name)

            if value is None:
                print(f"Warning: {self.loss_name} did not return a valid loss value.")
                del loss_stuff
                continue
            if not torch.is_tensor(value) or value.numel() != 1:
                raise TypeError(
                    f"{type(self).__name__}('{self.loss_name}') needs a scalar tensor, got "
                    f"{type(value).__name__} with "
                    f"{value.numel() if torch.is_tensor(value) else 'no'} elements."
                )

            # retain_graph=True here would keep the per-event intermediates alive
            # and defeat the point of repeating at all.
            grads = torch.autograd.grad(value, replica_inputs, retain_graph=False, allow_unused=True)
            add_grads(totals, grads, scale)
            total_value += scale * float(value.detach().cpu().item())
            evaluated += 1

            extras = detach_extras(chunk_extras, skip=(self.loss_name,))
            if self.pool_extras:
                collect_poolable(extras, pooled)

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

        return self._finish(total_value, totals, real_inputs, owner_device, merge_pooled(extras, pooled))


class AnalysisAccumulator(GradientAccumulator):
    """Accumulate a loss that is a nonlinear head on additive histograms, as in the forward-folding analysis.

    The wrapped loss must provide ``histograms(geom_dict, **p)``, returning the additive
    `state_keys` tensors plus ``analysis_param_names`` and ``analysis_replay`` (the kwargs that
    reproduce that exact batch), and ``fisher_head(*state, param_names, **p)``. `AnalysisLoss` does.
    With reduction='mean' each batch is taken as an estimate of the full histogram, i.e. the
    pooled sample has N_gen = repeats * batch size.
    """

    state_keys = ("analysis_mu", "analysis_grad_mu")

    def __call__(self, geom_dict, **loss_params):
        owner_device, work_device, geom, real_inputs, replica_inputs = self._prepare(geom_dict)
        scale = self._scale()
        skip = (*self.state_keys, "analysis_replay")

        # Pass 1: pool the histograms without a geometry graph, keeping each batch's replay
        # kwargs so pass 2 sees exactly the same events (and does not recompute flux weights).
        frozen = {key: (value.detach() if torch.is_tensor(value) else value) for key, value in geom.items()}
        state = [None] * len(self.state_keys)
        batch_params = []
        param_names = None
        extras, pooled = {}, {}
        for k in range(self.repeats):
            params = self._repeat_params(loss_params, k)
            # Fresh per batch and step, replayed in pass 2 so stochastic parts (e.g. photon draws) match.
            params = {**params, 'accumulator_seed': int(torch.randint(0, 2 ** 31, (1,)).item())}
            out = self.loss_func.histograms(frozen, **params)
            if param_names is not None and out["analysis_param_names"] != param_names:
                raise ValueError(f"Flux parameters differ between batches: {param_names} vs {out['analysis_param_names']}")
            param_names = out["analysis_param_names"]
            add_grads(state, [out[key] for key in self.state_keys], scale)
            batch_params.append({**params, **out["analysis_replay"]})

            extras = detach_extras(out, skip=skip)
            if self.pool_extras:
                collect_poolable(extras, pooled)
            del out
            self._free(work_device)

        # Head: differentiated once, on the pooled histograms.
        state = [s.requires_grad_(True) for s in state]
        head = self.loss_func.fisher_head(*state, param_names, **loss_params)
        value = head[self.loss_name]
        d_state = torch.autograd.grad(value, state)

        # Pass 2: push dL/dstate back through one batch at a time.
        totals = [None] * len(replica_inputs)
        for params in batch_params:
            out = self.loss_func.histograms(geom, **params)
            pairs = [(out[key], d * scale) for key, d in zip(self.state_keys, d_state) if out[key].requires_grad]
            if pairs:
                outputs, grad_outputs = zip(*pairs)
                grads = torch.autograd.grad(outputs, replica_inputs, grad_outputs=grad_outputs, allow_unused=True)
                add_grads(totals, grads)
                del grads
            del out, pairs
            self._free(work_device)

        extras = merge_pooled(extras, pooled)
        extras.update(detach_extras(head, skip=(self.loss_name,)))
        for key, s in zip(self.state_keys, state):
            extras[key] = s.detach()
        return self._finish(float(value.detach().cpu().item()), totals, real_inputs, owner_device, extras)


ACCUMULATORS = {
    "base": GradientAccumulator,
    "analysis": AnalysisAccumulator,
}


def resolve_accumulator(spec):
    """Accumulator class from a name in ACCUMULATORS, a class, or None (-> GradientAccumulator)."""
    if spec is None:
        return GradientAccumulator
    if isinstance(spec, str):
        return ACCUMULATORS[spec]
    return spec

"""Run a JAX loss inside nugget's torch optimization loop.

A JAX loss is wrapped in a ``torch.autograd.Function`` that calls
``jax.value_and_grad`` and replays the result as a torch scalar. Everything
downstream (loss weighting, sigmoid wrapping, ALM, ConFIG, ``backward()``) then
works unchanged, and injecting at ``points_3d`` lets torch carry the gradient
through ``update_points`` down to whichever leaf is being optimized.

Call :func:`configure_jax` before anything imports JAX.
"""

import os

import torch

__all__ = [
    "configure_jax",
    "to_jax",
    "to_torch",
    "inject_gradient",
    "JaxLoss",
]


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------

def configure_jax(preallocate=False, mem_fraction=None, x64=True, platform=None,
                  device=None, verbose=True):
    """Make JAX safe to run alongside torch in one process.

    JAX preallocates ~75% of each visible GPU on first use, which OOMs a process
    that is also running torch; ``preallocate=False`` disables that. ``x64=True``
    matches nugget's global float64 default. The environment variables only take
    effect if set before JAX is first imported.

    Parameters
    ----------
    platform : {'cuda', 'cpu', 'tpu'} or None
        Backend, not a device. Use `device` to pick which GPU.
    device : int, str or None
        Pin JAX's default device, e.g. ``2`` or ``'cuda:2'``. Does not change
        what torch sees. To keep JAX off the other GPUs entirely -- it still
        opens a context on each visible one -- set ``CUDA_VISIBLE_DEVICES``
        before the process starts instead.
    """
    import sys

    if platform is not None and any(c.isdigit() or c == ":" for c in str(platform)):
        raise ValueError(
            f"platform={platform!r} is a device, not a backend. Use "
            f"platform='cuda' to pick the backend and device={str(platform).split(':')[-1]} "
            "to pick the GPU."
        )

    already = "jax" in sys.modules
    if not preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(mem_fraction))
    if platform is not None:
        os.environ["JAX_PLATFORMS"] = str(platform)

    if already:
        print("jax_bridge.configure_jax: JAX was already imported, so the memory "
              "environment variables will NOT take effect in this process. Call "
              "configure_jax() before importing any JAX code.")

    import jax
    if x64:
        jax.config.update("jax_enable_x64", True)

    if device is not None:
        target = _select_device(jax, device)
        jax.config.update("jax_default_device", target)
        if verbose:
            print(f"jax_bridge: JAX default device is {target}")
    elif verbose:
        print(f"jax_bridge: JAX devices {jax.devices()}")
    return jax


def _select_device(jax, device):
    """int | 'cuda:N' | jax.Device -> jax.Device."""
    if hasattr(device, "platform"):
        return device
    text = str(device)
    kind, _, index = text.rpartition(":")
    kind = kind or ("cpu" if text == "cpu" else "cuda")
    if text == "cpu":
        index = "0"
    if not index.isdigit():
        raise ValueError(f"device={device!r} should be an int or 'cuda:N'")

    devices = jax.devices(kind)
    if int(index) >= len(devices):
        raise ValueError(
            f"device={device!r} but JAX sees only {len(devices)} {kind} device(s): "
            f"{devices}. If you set CUDA_VISIBLE_DEVICES, the index is into the "
            "visible list, not the physical one."
        )
    return devices[int(index)]


def _jax():
    import jax
    return jax


# ----------------------------------------------------------------------
# array interchange
# ----------------------------------------------------------------------

def to_jax(tensor):
    """torch -> JAX, zero-copy via dlpack where possible, else via host.

    Detaches: gradients come back separately from the JAX side.
    """
    jax = _jax()
    tensor = tensor.detach()
    if tensor.is_cuda:
        try:
            return jax.dlpack.from_dlpack(tensor)
        except Exception:
            try:
                from torch.utils import dlpack as torch_dlpack
                return jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor))
            except Exception:
                pass
    import jax.numpy as jnp
    return jnp.asarray(tensor.cpu().numpy())


def to_torch(array, device=None, dtype=None):
    """JAX -> torch, zero-copy via dlpack where possible, else via host."""
    try:
        tensor = torch.from_dlpack(array)
    except Exception:
        try:
            from torch.utils import dlpack as torch_dlpack
            tensor = torch_dlpack.from_dlpack(array.__dlpack__())
        except Exception:
            import numpy as np
            tensor = torch.from_numpy(np.asarray(array))
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device if device is not None else tensor.device,
                           dtype=dtype if dtype is not None else tensor.dtype)
    return tensor


# ----------------------------------------------------------------------
# the graph splice
# ----------------------------------------------------------------------

class _InjectGrad(torch.autograd.Function):
    """A torch scalar carrying a gradient computed elsewhere.

    The tensor inputs are passed only so autograd knows where to route it.
    """

    @staticmethod
    def forward(ctx, value, grads, out_device, out_dtype, *inputs):
        ctx.injected_grads = grads
        return torch.tensor(float(value), device=out_device, dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # One entry per forward argument: value, grads, out_device, out_dtype, *inputs
        out = [None, None, None, None]
        for grad in ctx.injected_grads:
            if grad is None:
                out.append(None)
            else:
                out.append(grad_output.to(device=grad.device, dtype=grad.dtype) * grad)
        return tuple(out)


def inject_gradient(value, grads, inputs, device=None, dtype=None):
    """Splice an externally computed ``(value, grads)`` into the torch graph.

    `inputs` need not be leaves: routing into ``points_3d`` lets torch carry the
    gradient the rest of the way. `grads` may contain None for unused inputs.
    """
    inputs = list(inputs)
    grads = list(grads)
    if len(grads) != len(inputs):
        raise ValueError(f"got {len(grads)} gradients for {len(inputs)} inputs")
    reference = next((t for t in inputs if torch.is_tensor(t)), None)
    if reference is None:
        raise ValueError("inject_gradient needs at least one tensor input")

    # Autograd requires each gradient on its input's device/dtype. This is also
    # what carries a gradient computed on another GPU (or by JAX) home.
    aligned = tuple(
        None if g is None else g.to(device=t.device, dtype=t.dtype)
        for g, t in zip(grads, inputs)
    )
    device = reference.device if device is None else device
    dtype = reference.dtype if dtype is None else dtype
    return _InjectGrad.apply(value, aligned, device, dtype, *inputs)


# ----------------------------------------------------------------------
# the loss adapter
# ----------------------------------------------------------------------

class JaxLoss:
    """Present a JAX function as a nugget loss.

    `fn` is called as ``fn(*arrays, **static_kwargs)``, where `arrays` are the
    `geom_keys` entries converted to JAX in order, and must return a scalar (or
    ``(scalar, aux)`` with ``has_aux=True``). It is differentiated with respect
    to those arrays.

    `loss_name` must be the key the loss is registered under in
    ``loss_func_dict`` -- ``Optimizer.optimize`` looks the value up by name and
    skips the loss if it is missing.
    """

    def __init__(self, fn, loss_name, geom_keys=("points_3d",), static_kwargs=None,
                 pass_loss_params=False, has_aux=False, device=None):
        self.fn = fn
        self.loss_name = loss_name
        self.geom_keys = tuple(geom_keys)
        self.static_kwargs = dict(static_kwargs or {})
        self.pass_loss_params = bool(pass_loss_params)
        self.has_aux = bool(has_aux)
        self.device = device
        self._value_and_grad = None

    def _build(self):
        if self._value_and_grad is None:
            jax = _jax()
            argnums = tuple(range(len(self.geom_keys)))
            self._value_and_grad = jax.value_and_grad(
                self.fn, argnums=argnums, has_aux=self.has_aux
            )
        return self._value_and_grad

    def __call__(self, geom_dict, **kwargs):
        missing = [k for k in self.geom_keys if geom_dict.get(k) is None]
        if missing:
            raise KeyError(
                f"JaxLoss '{self.loss_name}' needs geom_dict entries {missing}; "
                f"available keys are {sorted(geom_dict)}"
            )

        inputs = [geom_dict[k] for k in self.geom_keys]
        jax_inputs = [to_jax(t) for t in inputs]

        static = dict(self.static_kwargs)
        if self.pass_loss_params:
            static.update(kwargs)

        value_and_grad = self._build()
        result, jax_grads = value_and_grad(*jax_inputs, **static)
        aux = None
        if self.has_aux:
            value, aux = result
        else:
            value = result

        grads = [
            to_torch(g, device=t.device, dtype=t.dtype)
            for g, t in zip(jax_grads, inputs)
        ]

        device = self.device if self.device is not None else inputs[0].device
        scalar = inject_gradient(float(value), grads, inputs,
                                 device=device, dtype=inputs[0].dtype)

        out = {self.loss_name: scalar}
        if aux is not None:
            out[f"{self.loss_name}_aux"] = aux
        return out

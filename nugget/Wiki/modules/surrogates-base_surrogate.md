---
type: module
status: draft
sources:
  - ../../nugget/surrogates/base_surrogate.py
updated: 2026-04-18
---

# base_surrogate.py

## Purpose

Abstract base class for all surrogate models in nugget. Provides a common interface and shared functionality for physics-based and learned surrogate models that estimate detector response to neutrino events.

## Architecture

[Surrogate class](../../nugget/surrogates/base_surrogate.py#L4): Abstract base with `__init__` and `__call__` methods.

### Constructor Signature
```python
def __init__(self, device=None, dim=3, domain_size=2):
    """Base class for surrogate models."""
    self.device = device if device is not None else torch.device("cpu")
    self.dim = dim
    self.domain_size = domain_size
```

### Key Methods

- [\_\_init\_\_](../../nugget/surrogates/base_surrogate.py#L6): Initialize device (CPU/GPU), dimensionality, and domain extent
- [\_\_call\_\_](../../nugget/surrogates/base_surrogate.py#L13): Interface method for evaluating surrogate (must override in subclass)

## Training/Inference API

**Interface (to be implemented by subclasses):**
```python
def __call__(self, opt_point=None, event_params=None):
    """
    Evaluate surrogate model.
    
    Parameters:
    - opt_point: torch.Tensor (detector position, shape dim)
    - event_params: dict (event properties like position, energy, zenith, azimuth)
    
    Returns:
    - torch.Tensor (detector response or other surrogate output)
    """
```

No direct training; subclasses implement their own training methods.

## Inputs/Outputs Tensors

**Inputs:**
- `opt_point`: torch.Tensor of shape (dim,) — detector coordinates
- `event_params`: dict with keys like 'position', 'energy', 'zenith', 'azimuth', 'direction'

**Outputs:**
- Model-dependent (light yield scalar, PATD dict, etc.)

## Dependencies

- `torch`: tensor operations and device management
- `numpy`: numerical operations

## Notes

- All concrete surrogate models inherit from this base class
- Enforces common interface across physics-based (LightSabre, Pandel) and learned (ChargeNet, LLRnet, HitFlow) surrogates
- Supports both 2D and 3D detector geometries via `dim` parameter

## See also

- [[surrogates]] — overview of all surrogate models
- [[surrogates-ChargeNet]] — learned light yield regressor
- [[surrogates-LightSabre]] — physics-based Cherenkov surrogate
- [[concepts-surrogate-modelling]] — cross-cutting surrogate concepts

---
type: module
status: draft
sources:
  - ../../nugget/samplers/base_sampler.py
updated: 2026-04-18
---

# base_sampler.py

Abstract base class defining the sampler interface.

## Purpose

`Sampler` sets the standard constructor (device, dim, domain size) and
declares the two methods every concrete sampler must implement.

## Key class

### `Sampler` — [base_sampler.py:3](../../nugget/samplers/base_sampler.py#L3)

Constructor — [base_sampler.py:5](../../nugget/samplers/base_sampler.py#L5)
```python
Sampler(device=None, dim=3, domain_size=2)
```

- `sample_events(num_events, **kwargs)` — [base_sampler.py:13](../../nugget/samplers/base_sampler.py#L13)
  Returns a list of dicts with keys `position`, `energy`, `zenith`,
  `azimuth`, `lepton`.
- `sample_detector_points(num_points, **kwargs)` — [base_sampler.py:38](../../nugget/samplers/base_sampler.py#L38)
  Returns tensor shape `(num_points, dim)`.

Both raise `NotImplementedError` on the base.

## Event dict format

| Key | Shape |
|-----|-------|
| `position` | (1, 3) |
| `energy`   | (1,) |
| `zenith`   | (1,) |
| `azimuth`  | (1,) |
| `lepton`   | optional |

## See also

- [samplers](samplers.md)
- [samplers-toy_sampler](samplers-toy_sampler.md)
- [samplers-cyl_sampler](samplers-cyl_sampler.md)

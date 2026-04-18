---
type: module
status: draft
sources:
  - ../../nugget/samplers/toy_sampler.py
updated: 2026-04-18
---

# toy_sampler.py

Lightweight toy event generator for prototyping and baselines.

## Purpose

- Rapid prototyping and debugging
- Baseline comparisons without full detector simulation
- Training-pipeline smoke tests

## Key class

### `ToySampler` — [toy_sampler.py:6](../../nugget/samplers/toy_sampler.py#L6)

Inherits from `Sampler`. Power-law energy, biased angular, uniform
position (with optional offset).

Constructor — [toy_sampler.py:8](../../nugget/samplers/toy_sampler.py#L8)

Kwargs: `event_type` (`'signal'`/`'background'`), `E_min`, `E_max`,
`gamma` (default 2.7 signal / 3.7 background), `a` (horizon-bias for
background zenith, default 1.5), `x_bias`, `y_bias`, `z_bias`.

### Methods

- `sample_power_law(E_min, E_max, gamma, n)` — [toy_sampler.py:14](../../nugget/samplers/toy_sampler.py#L14) — inverse-transform sampling.
- `sample_background_zenith(a, n)` — [toy_sampler.py:38](../../nugget/samplers/toy_sampler.py#L38) — rejection sampling with PDF `1 + a(1 − |cos θ|)`.
- `sample_events(n)` — [toy_sampler.py:66](../../nugget/samplers/toy_sampler.py#L66).
- `sample_detector_points(n)` — [toy_sampler.py:111](../../nugget/samplers/toy_sampler.py#L111) — uniform in domain.

## Distributions

| Quantity | Distribution |
|----------|--------------|
| Energy   | Power-law index γ |
| Zenith (signal) | Uniform on [0, π] |
| Zenith (bkg) | Rejection with `1 + a(1 − |cos θ|)` |
| Azimuth  | Uniform [0, 2π] |
| Position | Uniform [−domain/2, domain/2]³ + bias |

## Notes

- Background zenith sampling is NumPy, per-event — not vectorized.
- Position bias scales by `domain_size`.

## See also

- [samplers](samplers.md)
- [samplers-base_sampler](samplers-base_sampler.md)
- [samplers-cyl_sampler](samplers-cyl_sampler.md)

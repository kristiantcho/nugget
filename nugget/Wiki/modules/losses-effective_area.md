---
type: module
status: draft
sources:
  - ../../nugget/losses/effective_area.py
updated: 2026-04-18
---

# effective_area.py

Muon-track effective area via chord-length integration + energy-dependent range cuts.

## Key items

- `muon_range(E)` — [L25](../../nugget/losses/effective_area.py#L25) — MMC-parametrized range.
- `average_chord_length(...)` — [L35](../../nugget/losses/effective_area.py#L35) — differentiable chord through a cylinder; handles horizontal (`cosθ=0`) separately.
- `_softmax_max / _softmax_min` — [L70](../../nugget/losses/effective_area.py#L70) — smooth extrema via log-sum-exp.
- `_extract_track_from_event_params` — [L82](../../nugget/losses/effective_area.py#L82).
- `EffectiveAreaLoss` — ≳100 lines; main class.

## Physics

```
A_eff ≈ Σ_strings prob(track hits string) · chord · range_cutoff(E, d_edge)
```
Supports regular and `use_irregular_cylinder` geometries; batched events.

## Dependencies

- `trigger.TriggerLoss` used internally.
- `samplers.cyl_sampler.CylinderSampler` for test points.
- `scipy.interpolate.UnivariateSpline` (range tables).

## See also

- [losses](losses.md), [losses-trigger](losses-trigger.md), [losses-pointsource_fom](losses-pointsource_fom.md)

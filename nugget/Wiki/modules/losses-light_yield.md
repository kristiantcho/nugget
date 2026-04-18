---
type: module
status: draft
sources:
  - ../../nugget/losses/light_yield.py
updated: 2026-04-18
---

# light_yield.py

Total photon-yield maximization losses.

## Classes
- `LightYieldLoss` — [L4](../../nugget/losses/light_yield.py#L4) — full detector.
- `WeightedLightYieldLoss` — [L44](../../nugget/losses/light_yield.py#L44) — per-string.
  - `light_yield_per_string` — [L62](../../nugget/losses/light_yield.py#L62)

## Formulation

```
ly_i = surrogate(point_i, event)
total = Σ_events Σ_i ly_i
L = (n_points * n_events) / total
```
Optional multiplicative Gaussian noise: `ly *= 1 + noise_scale * randn()`.

## See also

- [losses](losses.md), [losses-SNR](losses-SNR.md)

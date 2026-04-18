---
type: module
status: draft
sources:
  - ../../nugget/losses/pointsource_fom.py
updated: 2026-04-18
---

# pointsource_fom.py

Combined effective-area × resolution figure of merit.

## `FoMLoss` — [L8](../../nugget/losses/pointsource_fom.py#L8)

Internally instantiates `ResolutionLoss`/`WeightedResolutionLoss` and
`EffectiveAreaLoss`; `_get_events` parses either event lists or a sampler.

## Math

```
term_i = A_eff_i / (4π σ_θ_i²)
FoM    = 1 / sqrt(Σ_i term_i)
```

Kwargs: `llr_net`, `fisher_info_params`, `use_weighted_resolution`,
nested kwargs forwarded to the composed losses.

## See also

- [losses-effective_area](losses-effective_area.md), [losses-fisher_info](losses-fisher_info.md)

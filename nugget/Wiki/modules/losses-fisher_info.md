---
type: module
status: draft
sources:
  - ../../nugget/losses/fisher_info.py
updated: 2026-04-18
---

# fisher_info.py

Fisher-information-based angular / parameter resolution.

## Helpers

- `_pos_norm_divisor_from_domain_size` — [L14](../../nugget/losses/fisher_info.py#L14)
- `_llr_mask_from_true_ly` — [L43](../../nugget/losses/fisher_info.py#L43)
- `_fisher_chunk_cleanup` — [L47](../../nugget/losses/fisher_info.py#L47)
- `_llr_out_single_point_all_iters` — [L59](../../nugget/losses/fisher_info.py#L59)
- `_fisher_one_point_jacrev` — [L89](../../nugget/losses/fisher_info.py#L89) — ∂LLR/∂θ via `torch.func.jacrev`.

## Classes
- `ResolutionLoss` — full-detector resolution.
- `WeightedResolutionLoss` — per-string, requires `string_xy`, `points_per_string_list`.

## Math

```
J[i,j] = ∂LLR / ∂θ_j
F[i,j] = E[ J J^T ]
σ_θ   = 1 / sqrt(F_diag)
L     = 1 / sqrt( Σ_events 1/σ_θ )
```

Kwargs: `llr_net`, `fisher_info_params` (e.g. `['zenith','azimuth']`),
`llr_iterations`, `skip_zero_response`, `resolve_per_string`.

## Notes

~800 lines; chunked to fit GPU memory. Uses `torch.func` (jacrev, vmap, linearize).

## See also

- [losses](losses.md), [losses-LLR](losses-LLR.md), [losses-pointsource_fom](losses-pointsource_fom.md)

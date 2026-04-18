---
type: module
status: draft
sources:
  - ../../nugget/geometries/EvanescentString.py
updated: 2026-04-18
---

# EvanescentString.py

All `n_strings` initialized; each carries a learnable weight. Sigmoid
threshold gates active strings — enables soft or hard pruning.

## `EvanescentString(Geometry)` — [L7](../../nugget/geometries/EvanescentString.py#L7)

Ctor kwargs: `n_strings=1000`, `points_per_string=5`,
`starting_weight=1.0`, `random_weights=False`, `custom_z_spacing`,
`custom_string_spacing`, `hex_type='hexagonal'`,
`active_weights_mode=False`, `hybrid_mix_init=0.5`.

### Methods
- `initialize_points(initial_geometry=None, **kw)` — [L53](../../nugget/geometries/EvanescentString.py#L53). Supports `active_weights_mode`, `weight_threshold=0.7`.
- `update_points(string_xy, z_values, string_weights, string_indices, old_string_weights=None, **kw)` — [L177](../../nugget/geometries/EvanescentString.py#L177). If `active_weights_mode`: binarise via sigmoid > threshold.

### Extra outputs
Adds `string_weights`, `old_string_weights`, `active_string_indices`,
`active_weights_mode`, `weight_threshold`.

## Notes

- Strings are never removed — only gated via weights.
- All strings share `points_per_string` (unlike `DynamicString`).

## See also

- [geometries](geometries.md), [geometries-DynamicString](geometries-DynamicString.md)

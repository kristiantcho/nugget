---
type: module
status: draft
sources:
  - ../../nugget/geometries/DynamicString.py
updated: 2026-04-18
---

# DynamicString.py

Discrete vertical strings with learnable z-values; points may be
unevenly distributed per string.

## `DynamicString(Geometry)` — [L5](../../nugget/geometries/DynamicString.py#L5)

Ctor kwargs: `total_points`, `n_strings=30`, `random_xy=False`,
`custom_z_spacing`, `points_per_string`, `custom_string_spacing`,
`hex_type='hexagonal'`, `hybrid_mix_init=0.5`.

### Methods
- `initialize_points(initial_geometry=None, **kw)` — [L53](../../nugget/geometries/DynamicString.py#L53) — supports loading + weight-based filtering (sigmoid thresh).
- `update_points(z_values, string_xy, points_per_string_list, string_indices, **kw)` — [L438](../../nugget/geometries/DynamicString.py#L438).
- `_make_z_segment(n_points)` — [L38](../../nugget/geometries/DynamicString.py#L38) — fixed-spacing or uniform linspace.
- `_sync_total_points_from_points_per_string` — [L48](../../nugget/geometries/DynamicString.py#L48).

## Notes

- Unlike `EvanescentString`, allows varying `points_per_string` per string.
- Can migrate from other geometries via `initial_geometry`.

## See also

- [geometries](geometries.md), [geometries-EvanescentString](geometries-EvanescentString.md), [geometries-SpaceString](geometries-SpaceString.md)

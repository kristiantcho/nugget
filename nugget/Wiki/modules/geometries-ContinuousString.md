---
type: module
status: draft
sources:
  - ../../nugget/geometries/ContinuousString.py
updated: 2026-04-18
---

# ContinuousString.py

1D path positions in [0,1] mapped to 3D via assignment to `n_strings`
vertical strings. Enables smooth path deformation during optimization.

## `ContinuousString(Geometry)` — [L6](../../nugget/geometries/ContinuousString.py#L6)

Ctor kwargs: `optimize_xy` (random vs. hex grid), `total_points=150`,
`n_strings=30`, `optimize_positions_only`.

### Methods
- `initialize_points(initial_geometry=None, **kw)` — [L24](../../nugget/geometries/ContinuousString.py#L24)
- `update_points(**kw)` — [L121](../../nugget/geometries/ContinuousString.py#L121)
- `map_path_to_3d(path_positions, string_xy)` — [L148](../../nugget/geometries/ContinuousString.py#L148) — string = `floor(p/segment_width)`; z from relative position.
- `convert_z_values_to_path_positions(z, string_indices, mask=None)` — [L193](../../nugget/geometries/ContinuousString.py#L193) — inverse mapping.

### Output
Adds `path_positions ∈ [0,1]` to the common dict.

## See also

- [geometries](geometries.md), [geometries-DynamicString](geometries-DynamicString.md), [geometries-EvanescentString](geometries-EvanescentString.md)

---
type: module
status: draft
sources:
  - ../../nugget/geometries/SpaceString.py
updated: 2026-04-18
---

# SpaceString.py

Fixed XY (hex/circular/sunflower/hybrid grid) with learnable z-spacing.

## `SpaceString(Geometry)` — [L7](../../nugget/geometries/SpaceString.py#L7)

Ctor kwargs: `n_strings=1000`, `points_per_string=5`,
`starting_spacing=0.1`, `starting_z_spacing=None`,
`hex_type='hexagonal'`, `optimize_z=False`, `hybrid_mix_init=0.5`,
`make_hybrid_iter=True`, `hybrid_iter_step=0.01`.

### Methods
- `initialize_points(initial_geometry=None)` — [L41](../../nugget/geometries/SpaceString.py#L41)
- `update_points(string_xy, z_values, string_indices, string_spacing, z_spacing, points_per_string_list, **kw)` — [L173](../../nugget/geometries/SpaceString.py#L173). Regenerates hex grid from `string_spacing` + `hybrid_mix`; if `optimize_z`, rebuilds z-values from `z_spacing`.

### Extra outputs
Adds `string_spacing`, `z_spacing`, `hybrid_mix`.

## Notes

- `string_xy` is not a free parameter — always redrawn from grid recipe.

## See also

- [geometries](geometries.md), [geometries-EvanescentString](geometries-EvanescentString.md), [geometries-DynamicString](geometries-DynamicString.md)

---
type: module
status: draft
sources:
  - ../../nugget/geometries/base_geometry.py
updated: 2026-04-18
---

# base_geometry.py

Abstract `Geometry` class + grid utilities + Hungarian comparison.

## `Geometry` — [L8](../../nugget/geometries/base_geometry.py#L8)

Constructor: `(device=None, dim=3, domain_size=2)`.

Subclasses must override:
- `initialize_points(**kw)` — [L25](../../nugget/geometries/base_geometry.py#L25)
- `update_points(**kw)` — [L36](../../nugget/geometries/base_geometry.py#L36)

Grid generators (return `(n, 2)` XY tensor):
- `create_uniform_hexagonal_grid` — [L47](../../nugget/geometries/base_geometry.py#L47) — concentric rings, `1+3r(r+1)` points at ring r.
- `create_circular_hexagonal_grid` — [L207](../../nugget/geometries/base_geometry.py#L207)
- `create_sunflower_grid` — [L325](../../nugget/geometries/base_geometry.py#L325) — golden-angle spiral `π(3−√5)`.
- `create_hybrid_hex_sunflower_grid` — [L373](../../nugget/geometries/base_geometry.py#L373) — Hungarian-matched blend.

Spacing is binary-searched to fit a target point count.

## `compare_geometries(g1, g2, ...)` — [L536](../../nugget/geometries/base_geometry.py#L536)

Hungarian optimal matching between two geometry dicts. Returns dict
with `average_distance`, `matched_average_distance`, `total_distance`,
`matches`, `distances`, `n_matched`, `n_unmatched`, `penalty_contribution`.

Penalty options: `'domain_diagonal'`, `'max_distance'`,
`'mean_distance'`, or a scalar.

## `_assign_string_weights_to_points` — [L703](../../nugget/geometries/base_geometry.py#L703)
Broadcasts string-level weights to per-point weights (for evanescent).

## See also

- [geometries](geometries.md), all sibling strategies.

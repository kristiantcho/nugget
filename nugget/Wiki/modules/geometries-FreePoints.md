---
type: module
status: draft
sources:
  - ../../nugget/geometries/FreePoints.py
updated: 2026-04-18
---

# FreePoints.py

Simplest geometry: unconstrained points in 3D. No strings, no grid.

## `FreePoints(Geometry)` — [L6](../../nugget/geometries/FreePoints.py#L6)

- `initialize_points(n_points=50, initial_geometry=None, **kw)` — [L12](../../nugget/geometries/FreePoints.py#L12). If `initial_geometry` provides `points_3d`, load them; if it also has `string_weights`, filter via sigmoid > `weight_threshold` (default 0.7); else uniform random in `[-domain/2, domain/2]^dim`.
- `update_points(points, **kw)` — [L89](../../nugget/geometries/FreePoints.py#L89). Wraps in `{'points_3d': points}`.

## Notes

- Fully free — useful as baseline or final-stage unconstrained pass.
- Can absorb any upstream geometry via `initial_geometry`.

## See also

- [geometries](geometries.md), [geometries-base_geometry](geometries-base_geometry.md)

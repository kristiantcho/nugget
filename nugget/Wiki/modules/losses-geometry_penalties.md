---
type: module
status: draft
sources:
  - ../../nugget/losses/geometry_penalties.py
updated: 2026-04-18
---

# geometry_penalties.py

Soft geometric constraints.

## Classes

- `BoundaryPenalty` — [L5](../../nugget/losses/geometry_penalties.py#L5) — points inside box: `mean(clamp(|x|-d/2, min=0)^2)`.
- `StringBoundaryPenaltySquare` — [L38](../../nugget/losses/geometry_penalties.py#L38).
- `StringBoundaryPenaltyCircle` — [L76](../../nugget/losses/geometry_penalties.py#L76) — sigmoid smoothing at `r = d/2`.
- `RepulsionPenalty` — [L114](../../nugget/losses/geometry_penalties.py#L114) — O(n²) pairwise `Σ 1/(d²+eps)`.
- `LocalRepulsionPenalty` — [L157](../../nugget/losses/geometry_penalties.py#L157) — gated by `max_radius`, sigmoid sharpness.

## Notes

Use local repulsion for large N. String weights broadcast via `sigmoid`.

## See also

- [losses](losses.md), [geometries](geometries.md)

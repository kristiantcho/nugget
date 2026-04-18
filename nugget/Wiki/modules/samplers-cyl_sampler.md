---
type: module
status: draft
sources:
  - ../../nugget/samplers/cyl_sampler.py
updated: 2026-04-18
---

# cyl_sampler.py

Cylindrical detector sampler with ray-based geometric intersection and
projected-area rejection sampling.

## Purpose

Samples events on a finite-cylinder surface (weighted by projected
cross-section), with optional repositioning along the ray inside the
cylinder or within a cubic domain.

## Key items

### `CylinderSurface` — [cyl_sampler.py:13](../../nugget/samplers/cyl_sampler.py#L13)
Geometry descriptor: `center` (3,), `height`, `radius`.

### Helpers

- `sph_to_cart(θ, φ)` — [cyl_sampler.py:30](../../nugget/samplers/cyl_sampler.py#L30)
- `cart_to_sph(vec)` — [cyl_sampler.py:39](../../nugget/samplers/cyl_sampler.py#L39)
- `projected_area(cyl, cosθ)` — [cyl_sampler.py:55](../../nugget/samplers/cyl_sampler.py#L55) — `A = πr²|cosθ| + 2rh √(1−cos²θ)`.
- `maximum_proj_area(cyl)` — [cyl_sampler.py:67](../../nugget/samplers/cyl_sampler.py#L67)
- `get_intersection_cylinder(cyl, pos, dir)` — [cyl_sampler.py:80](../../nugget/samplers/cyl_sampler.py#L80)
- `get_intersection_box(center, size, pos, dir)` — [cyl_sampler.py:185](../../nugget/samplers/cyl_sampler.py#L185)
- `sample_uniform_ray(...)` — [cyl_sampler.py:220](../../nugget/samplers/cyl_sampler.py#L220) — vectorized ray sampling with projected-area rejection.

### `CylinderSampler` — [cyl_sampler.py:450](../../nugget/samplers/cyl_sampler.py#L450)

Inherits `Sampler`. Constructor — [cyl_sampler.py:452](../../nugget/samplers/cyl_sampler.py#L452).

Kwargs: `cylinder_center/height/radius`, `E_min`, `E_max`, `gamma`,
`energy_dist` (`power_law`/`log_uniform`), `event_type`, `cos_range`,
`find_exact_intersection`, `random_position_along_ray`,
`random_position_within_cubic_domain`, `point_towards_center`, `seed`,
`x_bias`, `y_bias`, `z_bias`.

### Methods

- `sample_power_law` — [cyl_sampler.py:509](../../nugget/samplers/cyl_sampler.py#L509)
- `sample_uniform_logE` — [cyl_sampler.py:536](../../nugget/samplers/cyl_sampler.py#L536)
- `sample_events(n)` — [cyl_sampler.py:560](../../nugget/samplers/cyl_sampler.py#L560)
- `sample_detector_points(n)` — [cyl_sampler.py:642](../../nugget/samplers/cyl_sampler.py#L642) — uniform in cylinder volume (rejection).

## Math

Direction rejection probability ∝ `A(cosθ) / A_max`. Position along
accepted ray: cylinder surface by default; optionally uniform on ray
segment inside cylinder or cubic domain.

## Notes

- Ray sampling is vectorized; some rejection steps are per-sample.
- Robust quadratic + planar ray-cylinder intersection, handles grazing.
- Batch `Rz`/`Ry` rotations.
- Output event dict omits `direction` to match `ToySampler` API.

## See also

- [samplers](samplers.md)
- [samplers-base_sampler](samplers-base_sampler.md)
- [samplers-toy_sampler](samplers-toy_sampler.md)
- [geometries](geometries.md)

---
type: module
status: draft
sources:
  - ../../nugget/other/
updated: 2026-04-18
---

# other

Non-code assets: detector-geometry presets (CSV index lists + NPY
coordinate arrays), a pure-water optical-properties notebook, and the
Smith 1981 absorption reference dataset.

## Geometry presets (entities)

All 2-D XY layouts. `*_xy.npy` holds coordinates; matching `.csv`
stores integer indices into the base [340-point grid](../entities/geom-340grid.md).

- [geom-340grid](../entities/geom-340grid.md) — base hex grid, 341 pts.
- [geom-102geom](../entities/geom-102geom.md) — 102-pt subset.
- [geom-160geom](../entities/geom-160geom.md) — 160-pt subset.
- [geom-compact](../entities/geom-compact.md), [geom-default](../entities/geom-default.md), [geom-donut](../entities/geom-donut.md), [geom-donut2](../entities/geom-donut2.md), [geom-expanded](../entities/geom-expanded.md), [geom-large](../entities/geom-large.md), [geom-modified](../entities/geom-modified.md) — 75-pt variants.

## Notebooks
- [notebook-grid-maker-visualizer-csvwriter](../entities/notebook-grid-maker-visualizer-csvwriter.md) — generates all presets.
- [notebook-water-model](../entities/notebook-water-model.md) — fits water optical model; consumes `water_smith81.csv`.

## See also
- [geometries](geometries.md), [modules/examples](examples.md)

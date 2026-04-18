---
type: entity
status: draft
sources:
  - ../../nugget/other/water_model.ipynb
  - ../../nugget/other/water_smith81.csv
updated: 2026-04-18
---

# notebook-water-model

Builds a water optical-properties model for Cherenkov simulation:

- Loads Smith 1981 pure-water absorption (`water_smith81.csv`).
- Fits scattering via the Kokhanovsky model.
- Adds residual absorption as exponential.
- Integrates STRAW attenuation measurements.
- Produces a `WaterModel` medium compatible with the `theia`
  physics engine; Cherenkov yield ≈ 47 800 photons/m over 270–700 nm.

## See also
- [modules/other](../modules/other.md), [surrogates-LightSabre](../modules/surrogates-LightSabre.md)

---
type: entity
status: draft
sources:
  - ../../nugget/examples/loss_landscape_test.ipynb
updated: 2026-04-18
---

# loss_landscape_test.ipynb

## Purpose
Comprehensive analysis of physics loss landscape by systematically varying detector geometry (string spacing, Z spacing) and computing multiple physics metrics (light yield, resolution, ROV constraints, trigger efficiency).

## What it does
High-dimensional parameter sweep exploring detector performance:

1. **Grid search**: Tests 30x30 combinations of string spacing (50-500m) and Z spacing (50-500m)
2. **Physics metrics computed**: 
   - Signal light yield loss
   - Angular resolution (Fisher information)
   - Energy resolution
   - Local string repulsion penalty
   - ROV physical constraint violations
   - Trigger efficiency and detector efficiency
3. **Spatial visualization**: 3D light yield heatmaps showing signal distribution
4. **Heatmap analysis**: 9-panel figure showing all loss components, total combined loss, resolution maps
5. **Individual geometry analysis**: Detailed breakdown of signal event properties

## Key analysis cells

**Cell 2**: Load cascade LightSabre surrogate and CylinderSampler (1600m domain)

**Cell 3**: Geometry loop - tests 30x30 grid of SpaceString geometries with varying spacings

**Cell 3 (analysis)**: For each geometry computes:
- Signal yield loss (via WeightedLightYieldLoss)
- ROV penalty (via ROVPenalty)
- Local repulsion penalty
- Angular resolution loss (via WeightedResolutionLoss)
- Energy resolution loss
- Trigger efficiency

**Cell 6**: Save 50-event loss landscape arrays to disk

**Cell 9**: Visualize 3D light yield distribution as XY heatmap (integrating over Z)

**Cell 14**: Generate 3x3 panel figure comparing all loss components in log scale

**Cell 16**: Analyze signal event distributions - zenith, azimuth, energy, position histograms

**Cell 17**: Load and plot pre-computed loss landscapes from saved arrays

**Cell 21**: Scan likelihood over energy-zenith parameter space assuming Poisson photon statistics

## Inputs
- **Geometry grid**: 30x30 combinations of string_spacing (50-500m) and z_spacing (50-500m)
- **Geometries tested**: SpaceString with hexagonal layout, 70 strings, 20 OM per string
- **Events per geometry**: 100 signal events with 1e2-1e8 GeV energy
- **Event volume**: 600m radius, 1000m height cylindrical region
- **Domain**: 2500m x 2500m x 2500m cube

## Outputs
- **Loss arrays**: NumPy .npy files for each loss component
- **Loss landscapes**: 9-panel figure showing all metrics
- **Sample metadata**: Pickle files storing event parameters and geometry info
- **Light yield map**: XY heatmap showing spatial sensitivity to signals
- **Likelihood scans**: 2D energy-zenith parameter scan plots

## Related modules
- [LightSabre surrogate](../../nugget/surrogates/LightSabre.py): Cascade light yield model
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling
- [SpaceString geometry](../../nugget/geometries/SpaceString.py): Detector layout variant
- [Fisher information loss](../../nugget/losses/fisher_info.py): Angular/energy resolution
- [ROV penalty](../../nugget/losses/geometry_penalties.py): Physical constraints
- [Trigger loss](../../nugget/losses/trigger.py): Trigger efficiency

## See also
- [res_test.py](example-res_test.md): Precomputes Fisher info for specific geometries
- [test_evaluation.ipynb](example-test_evaluation.md): Evaluates optimized geometries
- [loss_landscape_test.py](example-loss_landscape_test.md): Batch version of analysis

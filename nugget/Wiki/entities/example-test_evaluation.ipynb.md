---
type: entity
status: draft
sources:
  - ../../nugget/examples/test_evaluation.ipynb
updated: 2026-04-18
---

# test_evaluation.ipynb

## Purpose
Comprehensive evaluation of optimized detector geometries using precomputed Fisher information, computing angular resolution and point-source figure-of-merit (FOM) metrics across energy and zenith angle.

## What it does
Multi-geometry physics performance evaluation:

1. **Setup**: Initialize Evaluator and load precomputed Fisher information from res_test.py
2. **Geometry loading**: Load optimized geometries from res_test_make_geoms.py runs
3. **Metrics computation**:
   - Angular resolution (FOM) vs. energy and zenith
   - ROV penalty evaluation
   - Point-source FOM (spatial localization capability)
   - Detector efficiency (trigger efficiency)
4. **Comparative analysis**: Compare across geometry variants (default, large, compact, modified, donut, optimized)
5. **Visualization**: Multi-panel figures showing resolution evolution and FOM dependence on parameters
6. **Statistical summary**: Per-geometry metrics and confidence intervals

## Key workflow sections

**Cell 3**: Evaluator initialization (EvanescentString, 1027 strings, 20 OM per string, 1600m domain)

**Cell 5**: Loss function setup with precomputed Fisher info:
- Angular resolution loss (WeightedResolutionLoss)
- Point-source FOM loss (FoMLoss)
- ROV penalty evaluation
- Trigger efficiency loss

**Cell 7**: Geometry loading and preprocessing:
- Loads optimized geometries from res_test_make_geoms.py output folders
- Loads precomputed Fisher info per string per event
- Handles geometry variants (340grid baseline with selective activation)

**Cell 8-9**: Batch evaluation across multiple geometries

**Cell 11**: Statistical analysis of active string counts and spacing

**Cell 13**: Per-geometry resolution statistics with histograms in log space

**Cell 15-16**: Event property distributions - zenith, azimuth, energy, XY positions

## Inputs
- **Precomputed data**: 
  - Fisher info: fisher_info_per_string_per_event_10000_*.pt
  - Light yield: light_yield_per_string_10000_*.pt
  - Cached signal events (10,000 samples)
- **Geometries**: 
  - '800main_full_hex' with/without ROV constraints
  - '340grid' baseline variants
- **Physics parameters**:
  - Energy range: 1e2-1e8 GeV (30 bins log-spaced)
  - Zenith range: -1 to 0 cos(zenith) (30 bins)
  - 10,000 signal events per evaluation

## Outputs
- **Resolution metrics**: Angular FOM, energy resolution, pointsource FOM per geometry
- **Resolution maps**: Heatmaps showing FOM evolution vs. energy and zenith
- **Comparative plots**: Side-by-side geometry comparisons with confidence bands
- **Detector statistics**: Active string counts, spacing, center position summaries
- **Distribution histograms**: Zenith, azimuth, energy distributions of test events

## Related modules
- [EvanescentString geometry](../../nugget/geometries/EvanescentString.py): Detector geometry
- [Fisher information loss](../../nugget/losses/fisher_info.py): Angular/energy resolution
- [Point-source FOM loss](../../nugget/losses/pointsource_fom.py): Spatial localization FOM
- [Evaluator](../../nugget/utils/basic_evaluator.py): Batch evaluation framework
- [Visualizer](../../nugget/utils/vis_tools.py): Multi-panel plotting
- [Trigger loss](../../nugget/losses/trigger.py): Detector efficiency

## See also
- [res_test.py](example-res_test.md): Computes Fisher info precomputed data
- [res_test_make_geoms.py](example-res_test_make_geoms.md): Produces optimized geometries
- [loss_landscape_test.ipynb](example-loss_landscape_test.ipynb.md): Loss landscape exploration
- [rov_evaluation.ipynb](example-rov_evaluation.ipynb.md): ROV constraint analysis

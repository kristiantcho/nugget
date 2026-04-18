---
type: entity
status: draft
sources:
  - ../../nugget/examples/rov_evaluation.ipynb
updated: 2026-04-18
---

# rov_evaluation.ipynb

## Purpose
Evaluate Remotely Operated Vehicle (ROV) constraints across multiple optimized detector geometries and visualize constraint violations in XY projections.

## What it does
Multi-geometry ROV penalty analysis workflow:

1. **Setup**: Initialize Evaluator with EvanescentString geometry
2. **Load geometries**: Load multiple optimized geometry configurations from pickle files
3. **ROV evaluation**: Compute ROV penalty for each geometry
4. **Constraint visualization**: Plot string XY positions with ROV penalty colorscale showing violations
5. **Geometry statistics**: Analyze active strings, center position, average radius for each variant
6. **Comparative analysis**: Create summary statistics tables across geometry variants

## Key workflow sections

**Cell 3**: Geometry initialization (EvanescentString, 70 strings, 20 OM, 1200m domain)

**Cell 5**: ROV penalty definition with standard ROV dimensions (width=229m, height=159.9m)

**Cell 5 (also)**: Evaluator setup for loss computation and visualization

**Cell 6**: Geometry loading - loads multiple variants from pickle files:
- 600hexagon (default EvanescentString)
- modified, expanded, large, compact (from saved XY arrays)

**Cell 7**: Individual evaluation - computes and visualizes ROV penalty for single geometry

**Cell 8**: Multi-evaluation - batch evaluates all loaded geometries

**Cell 9-14**: Detailed per-geometry ROV statistics:
- Number of active strings (sigmoid weight > 0.7)
- Average distance to 5 nearest neighbors
- Minimum inter-string distance
- Center of mass
- Average radius (all points vs. convex hull outer points)

## Inputs
- **Geometries**: Multiple EvanescentString variants (70 strings, 20 OM per string)
- **Event sampler**: CylinderSampler (600m radius, 1000m height)
- **Light yield**: LightSabre (non-Poisson)
- **ROV dimensions**: 229m width, 159.9m height, 159.9m triangle length
- **Domain**: 1200-1600m cylindrical regions

## Outputs
- **ROV penalty maps**: XY scatter plots colored by per-string ROV penalty
- **Geometry statistics**: Summary metrics for active string configurations
- **Comparative tables**: Center position, radius, and spacing statistics across variants
- **Visualizations**: Multi-panel figures comparing geometry layouts with ROV colorscale

## Related modules
- [EvanescentString geometry](../../nugget/geometries/EvanescentString.py): Detector layout
- [ROV penalty](../../nugget/losses/geometry_penalties.py): Physical constraint computation
- [Evaluator](../../nugget/utils/basic_evaluator.py): Batch evaluation framework
- [Visualizer](../../nugget/utils/vis_tools.py): Plotting utilities
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling

## See also
- [test_evaluation.ipynb](example-test_evaluation.md): Evaluates angular resolution and FOM
- [dynamic_strings_test.ipynb](example-dynamic_strings_test.ipynb.md): Optimizes with ROV constraints
- [example_notebook.ipynb](example-example_notebook.md): Basic optimization with ROV penalty

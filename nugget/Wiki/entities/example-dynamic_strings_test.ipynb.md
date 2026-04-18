---
type: entity
status: draft
sources:
  - ../../nugget/examples/dynamic_strings_test.ipynb
updated: 2026-04-18
---

# dynamic_strings_test.ipynb

## Purpose
Interactive exploration of DynamicString geometry optimization: optimize both string positions and detector weights to maximize angular resolution under physical constraints.

## What it does
Multi-phase geometry optimization workflow:

1. **Setup**: Initialize DynamicString geometry with 70 strings in hexagonal layout, random XY positions
2. **Physics setup**: Configure angular resolution loss, local string repulsion, ROV penalty constraints
3. **Precomputation**: Optional: precompute LLR network metrics and light yield per string for faster optimization
4. **First phase**: Optimize string XY positions (70x2 parameters) with learning rate 2
5. **Second phase**: Switch to DynamicString to fine-tune individual detector positions along strings
6. **Visualization**: Real-time loss tracking, 3D geometry visualization, resolution vs. energy/zenith plots

## Key workflow sections

**Cell 3**: DynamicString initialization - 70 strings, 20 OM per string, 1200m domain, hexagonal layout

**Cell 5**: Visualizer and Evaluator setup for real-time plotting

**Cell 6**: Define loss functions (angular resolution, local repulsion, ROV penalty, boundary)

**Cell 7**: Configure loss weights and visualization parameters

**Cell 10**: Compute pairwise string distances to verify geometry properties

**Cell 11**: Optional LLR precomputation for efficient optimization

**Cell 13**: Main optimization loop - 2000 iterations with visualization frequency control

**Cell 14**: Generate final optimization animation

**Cell 15**: Create interactive 3D visualization of optimized geometry

**Cell 16**: Initialize second-phase DynamicString for fine-tuning detector positions

**Cell 19**: Second-phase optimization - refines z-positions and string XY with alternate_freq switching

## Inputs
- **Geometry**: 70 DynamicString, 20 OM per string, 1200m domain
- **Event sampler**: CylinderSampler (600m radius, 1000m height, random positions)
- **Light yield**: LightSabre with Poisson statistics
- **Physics target**: Angular resolution (Fisher information on direction)
- **Constraints**: ROV penalty, string boundary, local repulsion, number penalty
- **Optimization**: 2 phases - first 2000 iterations for XY, second 100 for z-positions

## Outputs
- **Optimized geometry**: ds70_r600_50_any_ang_res_rov/ (checkpoint folder)
- **Animation**: ../gifs/opt_test_ds70_r600_50_any_ang_res_rov.gif (optimization progress)
- **Visualizations**: Loss components, angular resolution maps, 3D detector layout

## Related modules
- [DynamicString geometry](../../nugget/geometries/DynamicString.py): Dynamic string geometry
- [Fisher information loss](../../nugget/losses/fisher_info.py): Angular resolution metric
- [Optimizer](../../nugget/utils/basic_optimizer.py): Geometry optimization engine
- [Visualizer](../../nugget/utils/vis_tools.py): Real-time plotting tools
- [ROV penalty](../../nugget/losses/geometry_penalties.py): Physical constraints

## See also
- [dynamic_strings_test.py](example-dynamic_strings_test.md): Batch version of this workflow
- [example_notebook.ipynb](example-example_notebook.md): Basic optimization tutorial
- [test_evaluation.ipynb](example-test_evaluation.md): Evaluates optimized geometries

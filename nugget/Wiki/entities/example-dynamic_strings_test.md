---
type: entity
status: draft
sources:
  - ../../nugget/examples/dynamic_strings_test.py
updated: 2026-04-18
---

# dynamic_strings_test.py

## Purpose
Test and optimize DynamicString detector geometry by learning optimal string XY positions for maximizing angular resolution under physical constraints (ROV, boundary, local string repulsion).

## What it does
Runs geometry optimization on a DynamicString with 70 dynamically positioned detector strings:

- **DynamicString geometry**: 70 strings with 20 OMs per string, variable XY positions, 50m Z spacing
- **Optimization target**: Angular resolution (Fisher information) on particle direction estimation
- **Constraints**: ROV physical constraints, string boundary, local string repulsion, string number penalty
- **Loss functions**: Weighted combination of physics objectives and geometric penalties
- **Visualizer**: Real-time 3D plots and loss component tracking

## Key code references
- [DynamicString initialization](../../nugget/examples/dynamic_strings_test.py#L48-L57): 70 strings, hexagonal layout, 1200m domain, 50m Z spacing, random XY initialization
- [Optimizer setup](../../nugget/examples/dynamic_strings_test.py#L66-L84): Augmented Lagrangian Method with ALM params (gamma=1e-2, alpha=0.95)
- [Loss function dictionary](../../nugget/examples/dynamic_strings_test.py#L155-L160): ROV penalty, angular resolution, local repulsion, boundary penalty
- [Optimization call](../../nugget/examples/dynamic_strings_test.py#L244-L264): 1000 iterations, saves every 100, visualization every frequency control

## Inputs
- **Geometry**: 70 dynamic strings, 20 OM per string, domain 1200m
- **Event sampler**: CylinderSampler (600m radius, 1000m height)
- **Light yield surrogate**: LightSabre (Poisson, non-cascade)
- **Training**: 1000 iterations, per_effective_area_loss enabled
- **Learning rate**: 50 for string_xy positions (without sigmoid)

## Outputs
- **Optimized geometry folder**: ds70_r600_50_any_ang_res_rov/ (saves every 100 iterations)
- **Optimization GIF**: ../gifs/opt_test_ds70_r600_50_any_ang_res_rov.gif (visualization animation)

## Related modules
- [DynamicString geometry](../../nugget/geometries/DynamicString.py): Dynamic string class
- [Optimizer](../../nugget/utils/basic_optimizer.py): Optimization engine with ALM
- [Visualizer](../../nugget/utils/vis_tools.py): Real-time visualization
- [Fisher information loss](../../nugget/losses/fisher_info.py): Angular resolution computation

## See also
- [uniform_rov_alm_test.py](example-uniform_rov_alm_test.md): Similar test with toy sampler
- [dynamic_strings_test.ipynb](example-dynamic_strings_test.md): Interactive exploration version

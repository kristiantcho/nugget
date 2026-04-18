---
type: entity
status: draft
sources:
  - ../../nugget/examples/uniform_rov_alm_test.py
updated: 2026-04-18
---

# uniform_rov_alm_test.py

## Purpose
Test geometry optimization with Augmented Lagrangian Method (ALM) using a toy uniform light yield surrogate and simple ToySampler events to validate ALM constraints and optimization dynamics.

## What it does
Runs a simplified geometry optimization benchmark:

- **Uniform light yield**: Constant light yield (no spatial variation) for focused testing of geometry constraints
- **Toy sampler**: Simple uniform event distribution within domain for fast iteration
- **EvanescentString geometry**: 1000 strings with random initial weights to be optimized
- **ALM solver**: Augmented Lagrangian Method with adaptive penalty updates
- **Constraints**: ROV physical constraints, string boundary, local repulsion, string number penalty, binarization
- **Output**: Saves geometry checkpoints every 100 iterations to folder

## Key code references
- [Uniform surrogate](../../nugget/examples/uniform_rov_alm_test.py#L5): Constant light yield across space
- [Toy sampler setup](../../nugget/examples/uniform_rov_alm_test.py#L6-L7): Signal and background toy events
- [Geometry initialization](../../nugget/examples/uniform_rov_alm_test.py#L9-L21): EvanescentString, 1000 strings, 1 OM per string, domain 2500m
- [Optimizer setup](../../nugget/examples/uniform_rov_alm_test.py#L26-L39): ALM with gamma=1e-2, alpha=0.95
- [Loss function setup](../../nugget/examples/uniform_rov_alm_test.py#L88-L159): Multiple constraints including ROV, boundary, repulsion
- [Optimization call](../../pygget/examples/uniform_rov_alm_test.py#L188-L203): 1000 iterations, saves every 100

## Inputs
- **Events**: Single signal and background event per iteration
- **Light yield**: Uniform surrogate (constant yield)
- **Geometry**: 1000 EvanescentString with 1 OM per string, 2500m domain
- **Optimization**: 1000 iterations, learning rate 0.1 for string_weights
- **ALM params**: gamma=1e-2, alpha=0.95, epsilon=1e-8

## Outputs
- **Geometry folder**: rov_uniform_geom_zero_weights/
- **Checkpoint files**: Multiple geom_*.pkl files (every 100 iterations)

## Related modules
- [EvanescentString geometry](../../nugget/geometries/EvanescentString.py): Detector layout
- [Optimizer](../../nugget/utils/basic_optimizer.py): ALM-based optimizer
- [ToySampler](../../nugget/samplers/toy_sampler.py): Simple event sampling
- [Geometry penalties](../../nugget/losses/geometry_penalties.py): ROV, boundary, repulsion constraints

## See also
- [dynamic_strings_test.py](example-dynamic_strings_test.md): Similar test with DynamicString
- [example_notebook.ipynb](example-example_notebook.md): Interactive version with visualization
- [res_test_make_geoms.py](example-res_test_make_geoms.md): Production-scale optimization with physics metrics

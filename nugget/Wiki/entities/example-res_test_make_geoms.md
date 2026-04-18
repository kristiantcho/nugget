---
type: entity
status: draft
sources:
  - ../../nugget/examples/res_test_make_geoms.py
updated: 2026-04-18
---

# res_test_make_geoms.py

## Purpose
Run large-scale geometry optimization loops using precomputed Fisher information matrices and light yield distributions to find optimal detector configurations that maximize angular resolution under ROV constraints.

## What it does
Executes 15 independent geometry optimizations:

- **Precomputed data**: Uses Fisher info and light yield tensors from res_test.py for speed
- **Geometry type**: EvanescentString with 1027 strings (full hexagonal)
- **Optimization target**: Angular resolution (Fisher information) as primary objective
- **Constraints**: ROV physical constraints, string boundary, local repulsion, string number penalty
- **ALM solver**: Augmented Lagrangian Method for constraint handling
- **Output**: Saves best and last geometry per optimization run

## Key code references
- [Precomputed data loading](../../nugget/examples/res_test_make_geoms.py#L36-L44): Loads precomputed Fisher info and light yield tensors
- [Geometry initialization](../../nugget/examples/res_test_make_geoms.py#L119-L128): EvanescentString, 1027 strings, 20 OM per string, random weights init
- [Optimizer setup](../../nugget/examples/res_test_make_geoms.py#L130-L143): ALM with Augmented Lagrangian Method
- [Loss function config](../../nugget/examples/res_test_make_geoms.py#L100-L112): ROV penalty, string number penalty, binarization
- [Optimization loop](../../nugget/examples/res_test_make_geoms.py#L116-L157): 15 independent runs, 2000 iterations each

## Inputs
- **Precomputed Fisher info**: fisher_info_per_string_per_event_10000_800main_full_hex_r600_50_1.pt
- **Precomputed light yield**: light_yield_per_string_10000_800main_full_hex_r600_50_1.pt
- **Signal events**: 10,000 cached signal events (1e2-1e8 GeV)
- **Geometry**: 1027 EvanescentString, 20 OM per string, 1600m domain
- **Optimization**: 2000 iterations per run, ALM penalty update

## Outputs
- **Geometry folder**: res_test/opt_geoms_full_hex_10000_r600_50_e6_e8_rov_1/
- **Geometry checkpoints**: geom_0.pkl through geom_14.pkl (one per optimization run)
- **Last geometry**: Also saved per run for analysis

## Related modules
- [EvanescentString geometry](../../nugget/geometries/EvanescentString.py): Detector layout
- [Optimizer](../../nugget/utils/basic_optimizer.py): Optimization with ALM
- [Fisher information loss](../../nugget/losses/fisher_info.py): Angular resolution metric
- [ROV penalty](../../nugget/losses/geometry_penalties.py): Physical constraints

## See also
- [res_test.py](example-res_test.md): Computes Fisher info matrices
- [test_evaluation.ipynb](example-test_evaluation.md): Evaluates optimized geometries
- [rov_evaluation.ipynb](example-rov_evaluation.md): Visualizes ROV constraints on geometries

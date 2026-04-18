---
type: module
status: draft
sources:
  - ../../nugget/utils/basic_evaluator.py
updated: 2026-04-18
---

# basic_evaluator.py

## Purpose

Provides single-shot, no-gradient evaluation of loss functions on a fixed geometry configuration. The `Evaluator` class is the evaluation counterpart to `basic_optimizer.Optimizer.optimize()`, enabling efficient assessment of geometry performance without computing gradients.

## Key Classes/Functions

### Evaluator

[`Evaluator`](../../nugget/utils/basic_evaluator.py#L6) — Main class for loss evaluation.

**Methods:**

- [`__init__(device, geometry, visualizer)`](../../nugget/utils/basic_evaluator.py#L14) — Initialize evaluator with device, geometry handler, and optional visualizer.

- [`_unpack_loss_output(loss_name, loss_stuff)`](../../nugget/utils/basic_evaluator.py#L24) — Static method to normalize loss function outputs (dict, tuple/list, or scalar) into a consistent (loss_value, extra_kwargs) tuple. Matches conventions used by the optimizer.

- [`evaluate(geom_dict, loss_func_dict, loss_params_dict, print_result, visualize, make_gif, vis_kwargs, update_points, **kwargs)`](../../nugget/utils/basic_evaluator.py#L44) — Evaluate all losses once (no gradients) on a single geometry.
  - **Parameters**: geometry dict, mapping of loss name → callable, optional shared loss parameters, print/viz flags, vis kwargs.
  - **Returns**: dict with `geom_dict`, `losses` (name → float), `vis_kwargs`.
  - Runs each loss function inside `torch.no_grad()`.
  - Optionally updates geometry points and invokes visualizer.

- [`evaluate_multi(geom_dicts, loss_func_dicts, loss_func_dict, loss_params_dict, loss_params_dicts, ...)`](../../nugget/utils/basic_evaluator.py#L127) — Evaluate multiple geometries in one call (no gradients).
  - **Parameters**: mapping of geom_name → geom_dict; per-geometry or shared loss functions; per-geometry or shared loss params.
  - **Returns**: dict mapping geom_name → (geom_dict, losses, vis_kwargs).
  - Enforces that `plot_types` are shared across all geometries.
  - Detects and warns if per-geometry loss_params dicts reuse the same dict object (to avoid silent data reuse bugs).
  - Falls back to single-geometry rendering if visualizer lacks `visualize_multi_progress()`.

## Evaluation Pipeline

1. **Initialize geometry**: Convert input `geom_dict` to internal representation via `geometry.initialize_points()`.
2. **Compute losses**: Loop through `loss_func_dict`, calling each loss function with geometry and shared params.
3. **Unpack outputs**: Normalize variable output formats (dict/tuple/scalar) using `_unpack_loss_output()`.
4. **Accumulate results**: Store loss values and extra kwargs (e.g., precomputed metrics) for visualization.
5. **Update geometry** (optional): Call `geometry.update_points()` to refresh derived properties.
6. **Visualize** (optional): Pass accumulated geometry and metrics to visualizer.
7. **Return**: dict with final geometry, scalar losses, and visualization state.

**Key property**: All evaluation runs under `torch.no_grad()` — no gradient computation or history retention.

## Dependencies

- `torch` — tensor operations, gradient control.
- `typing` — type hints (Any, Dict, Optional, Tuple, Union).
- Implicit: `geometry` object (expects `initialize_points()`, `update_points()` methods).
- Implicit: `visualizer` object (expects `visualize_progress()` and optionally `visualize_multi_progress()`).

## Notes

- **Loss output flexibility**: Loss functions can return a scalar, tuple, list, or dict. The `_unpack_loss_output()` method standardizes these.
- **Multi-geometry validation**: `evaluate_multi()` enforces consistent `plot_types` and detects dict-reuse bugs.
- **Visualizer fallback**: If multi-geometry visualization is unavailable, individual geometries are rendered separately.
- **No-gradient semantics**: Evaluations do not build a computation graph; results are scalar floats suitable for logging, serialization, or post-hoc analysis.
- **ALM support**: `evaluate_multi()` respects ALM history kwargs if present in `vis_kwargs`.

## See also

- [[utils-basic_optimizer]] — paired optimizer for iterative refinement
- [[utils-schedulers]] — learning rate scheduling during optimization
- [[utils-vis_tools]] — visualization backend
- [[concepts-losses]] — loss function design
- [[concepts-geometry]] — geometry representation


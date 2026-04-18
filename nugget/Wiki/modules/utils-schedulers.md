---
type: module
status: draft
sources:
  - ../../nugget/utils/schedulers.py
updated: 2026-04-18
---

# schedulers.py

## Purpose

Provides learning rate schedulers for controlling optimizer step sizes during iterative geometry optimization. Implements common schedules (cosine annealing, step decay, exponential, linear) and a factory function for easy scheduler creation.

## Key Classes/Functions

### Scheduler

[`Scheduler`](../../nugget/utils/schedulers.py#L6) — Base class for learning rate schedulers.

**Methods:**

- [`__init__(optimizer, device)`](../../nugget/utils/schedulers.py#L9) — Initialize scheduler with a torch.optim.Optimizer.
  - Stores initial learning rates for all parameter groups.
  - Tracks current iteration counter.

- [`step()`](../../nugget/utils/schedulers.py#L25) — Increment iteration counter and return current learning rates.

- [`state_dict()`](../../nugget/utils/schedulers.py#L38) — Serialize scheduler state (initial_lr, current_iteration).

- [`load_state_dict(state_dict)`](../../nugget/utils/schedulers.py#L52) — Restore scheduler from checkpoint.

- [`get_lr()`](../../nugget/utils/schedulers.py#L64) — Return current learning rates for all parameter groups.

### CosineScheduler

[`CosineScheduler(Scheduler)`](../../nugget/utils/schedulers.py#L76) — Cosine annealing learning rate scheduler.

- [`__init__(optimizer, num_iterations, eta_min, device)`](../../nugget/utils/schedulers.py#L79) — Initialize with total iterations and minimum learning rate.
  - Formula: `lr = eta_min + (initial_lr - eta_min) * (1 + cos(π * t / T_max)) / 2`
  - Smoothly decays LR from initial to minimum over the full training horizon.

- [`step()`](../../nugget/utils/schedulers.py#L98) — Update learning rates using cosine annealing formula.

### StepScheduler

[`StepScheduler(Scheduler)`](../../rugget/utils/schedulers.py#L147) — Step-decay learning rate scheduler.

- [`__init__(optimizer, step_size, gamma, device)`](../../nugget/utils/schedulers.py#L150) — Initialize with decay period and decay factor.
  - Every `step_size` iterations, multiply current LR by `gamma`.
  - Default: decay by 0.1 every N/3 iterations.

- [`step()`](../../nugget/utils/schedulers.py#L169) — Decay learning rate every step_size iterations.

### ExponentialScheduler

[`ExponentialScheduler(Scheduler)`](../../nugget/utils/schedulers.py#L216) — Exponential decay learning rate scheduler.

- [`__init__(optimizer, gamma, device)`](../../nugget/utils/schedulers.py#L219) — Initialize with constant decay factor.
  - Formula: `lr = lr * gamma` (every iteration).
  - Useful for rapid, smooth decay without a fixed horizon.

- [`step()`](../../nugget/utils/schedulers.py#L235) — Decay learning rate by constant factor each iteration.

### LinearScheduler

[`LinearScheduler(Scheduler)`](../../nugget/utils/schedulers.py#L279) — Linear decay learning rate scheduler.

- [`__init__(optimizer, num_iterations, end_factor, device)`](../../nugget/utils/schedulers.py#L282) — Initialize with total iterations and final scale factor.
  - Formula: `lr_factor = 1 - (1 - end_factor) * (t / num_iterations)`.
  - Linearly interpolates from 1.0 to `end_factor` over the training horizon.

- [`step()`](../../nugget/utils/schedulers.py#L301) — Update learning rates using linear interpolation.

### create_scheduler

[`create_scheduler(optimizer, num_iterations, scheduler_type, scheduler_params)`](../../nugget/utils/schedulers.py#L349) — Factory function to instantiate a scheduler.

**Parameters:**

- **optimizer**: torch.optim.Optimizer to schedule.
- **num_iterations**: Total training iterations (required for cosine, step, linear).
- **scheduler_type**: One of 'cosine', 'step', 'exp', 'linear', or None (no scheduling).
- **scheduler_params**: Dict of type-specific kwargs:
  - **cosine**: `eta_min` (default 0.0).
  - **step**: `step_size`, `gamma` (defaults 1/3, 0.1).
  - **exp**: `gamma` (default 0.95).
  - **linear**: `end_factor` (default 0.01).

**Returns:** Scheduler instance or None if `scheduler_type` is None.

## Scheduler Comparison

| Scheduler | Formula | Use Case | Horizon-aware |
|-----------|---------|----------|---------------|
| Cosine | `η_min + (η₀ - η_min) * (1 + cos(πt/T)) / 2` | Smooth warm scheduling | Yes (T_max) |
| Step | Multiply by γ every step_size | Piecewise constant decay | Yes (step_size) |
| Exponential | `lr * γ` (per iteration) | Rapid, unbounded decay | No |
| Linear | `η₀ * (1 - (1 - f) * t/T)` | Simplicity; predictable fade | Yes (num_iterations) |

## Dependencies

- `torch` — optimizer parameter group access.
- `math` — cosine/pi for annealing formulas.
- `numpy` — optional for utilities.

## Notes

- **Multi-parameter-group support**: All schedulers handle optimizers with multiple parameter groups (e.g., different LRs per layer or geometry aspect).
- **State checkpointing**: `state_dict()` and `load_state_dict()` enable resumable training.
- **Device handling**: Schedulers accept an optional device parameter (currently for consistency; computation is on CPU).
- **No retain_graph needed**: Schedulers do not interfere with gradient graphs.
- **Integration with optimizer**: Call `scheduler.step()` after `optimizer.step()` in each iteration.

## See also

- [[utils-basic_optimizer]] — uses schedulers to control learning rates
- [[utils-basic_evaluator]] — no scheduling (evaluation-only)
- [[concepts-optimization]] — optimization algorithm design


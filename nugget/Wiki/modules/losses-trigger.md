---
type: module
status: draft
sources:
  - ../../nugget/losses/trigger.py
updated: 2026-04-18
---

# trigger.py

Sliding-bar trigger-efficiency model.

## `TriggerLoss` — [L5](../../nugget/losses/trigger.py#L5)

- `map_string_weights_to_points` — [L70](../../nugget/losses/trigger.py#L70).
- `__call__` returns `trigger_loss`, `trigger_per_event`, `per_string_triggers`.

## Algorithm

1. `t1_i = w_i · σ((ly_i − threshold)/T1)` — point detection.
2. Project points onto track direction; slide a bar of length
   `distance_bar_length=550 m` (step = `distance_bar_step` or per-point).
3. Score each bar window: sum of `t1` inside window.
4. Aggregate via softmax (soft) or max (`use_hard_cuts`).
5. Gate per-string via `sigmoid(string_weights)`.

Key kwargs: `light_yield_threshold`, `distance_bar_length`,
`distance_bar_step`, `min_points_threshold`, `t1_temperature`,
`t3_temperature`, `t_temperature`, `use_hard_cuts`.

## See also

- [losses](losses.md), [losses-effective_area](losses-effective_area.md), [losses-trigger_old](losses-trigger_old.md)

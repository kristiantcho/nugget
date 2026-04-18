---
type: module
status: draft
sources:
  - ../../nugget/losses/base_loss.py
updated: 2026-04-18
---

# base_loss.py

Abstract `LossFunction` defining the protocol for every loss.

## `LossFunction` — [L5](../../nugget/losses/base_loss.py#L5)

- `__init__(device=None)` — defaults to CUDA if available.
- `__call__(geom_dict, **kwargs)` — must return a dict; at least one
  key must end with `_loss` (scalar tensor). Raises `NotImplementedError`
  on the base.

## See also

- [losses](losses.md)

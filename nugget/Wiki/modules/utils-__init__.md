---
type: module
status: draft
sources:
  - ../../nugget/utils/__init__.py
updated: 2026-04-18
---

# __init__.py

## Purpose

Package initialization file for the `nugget.utils` module. Imports and exposes all public submodules for convenient access: `schedulers`, `vis_tools`, `basic_optimizer`, and `basic_evaluator`.

## Exported Modules

- [`schedulers`](../../nugget/utils/schedulers.py) — Learning rate scheduling strategies.
- [`vis_tools`](../../nugget/utils/vis_tools.py) — Visualization toolkit.
- [`basic_optimizer`](../../nugget/utils/basic_optimizer.py) — Gradient-based geometry optimizer.
- [`basic_evaluator`](../../nugget/utils/basic_evaluator.py) — No-gradient loss evaluator.

## Usage

Users can import from the utils module as:

```python
from nugget.utils import schedulers, vis_tools, basic_optimizer, basic_evaluator

# Or import specific classes:
from nugget.utils.basic_optimizer import Optimizer
from nugget.utils.basic_evaluator import Evaluator
from nugget.utils.vis_tools import Visualizer
from nugget.utils.schedulers import CosineScheduler, create_scheduler
```

## Notes

- All submodules are imported at package level for convenient access.
- Submodule organization enables clear separation of concerns (optimization, evaluation, visualization, scheduling).

## See also

- [[utils]] — utils module overview
- [[utils-basic_optimizer]] — optimizer implementation
- [[utils-basic_evaluator]] — evaluator implementation
- [[utils-vis_tools]] — visualization toolkit
- [[utils-schedulers]] — learning rate schedulers


---
type: source
status: draft
sources:
  - ../../requirements.txt
updated: 2026-04-18
---

# requirements.txt

Runtime dependency pins. See [requirements.txt](../../requirements.txt).

- `torch>=1.8.0`, `numpy>=1.19.0`, `matplotlib>=3.3.0`,
  `scipy>=1.5.0`, `tqdm>=4.45.0`, `jupyter>=1.0.0`,
  `conflictfree>=0.1.8`, `imageio>=2.37.0`

`conflictfree` is used by [utils-basic_optimizer](../modules/utils-basic_optimizer.md)
for multi-objective gradient descent.

## See also
- [setup-py](setup-py.md), [readme](readme.md)

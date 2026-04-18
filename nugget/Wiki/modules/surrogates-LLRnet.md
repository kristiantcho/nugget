---
type: module
status: draft
sources:
  - ../../nugget/surrogates/LLRnet.py
updated: 2026-04-18
---

# LLRnet.py

Binary classifier (signal vs. background) outputting log-likelihood
ratios for geometry optimization via [losses-LLR](losses-LLR.md) and
[losses-fisher_info](losses-fisher_info.md).

## Key items
- MLP with `FourierFeatures` front-end — see [L1](../../nugget/surrogates/LLRnet.py#L1).
- Helpers `prepare_data_from_raw`, `predict_log_likelihood_ratio`
  consumed by loss classes.

## See also
- [surrogates](surrogates.md), [losses-LLR](losses-LLR.md), [losses-fisher_info](losses-fisher_info.md), [surrogates-old_LLRnet](surrogates-old_LLRnet.md)

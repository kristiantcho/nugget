---
type: module
status: draft
sources:
  - ../../nugget/losses/RBF.py
updated: 2026-04-18
---

# RBF.py

RBF-interpolation fidelity loss for evaluating a surrogate captured
on detector points vs. a random test grid.

## `RBFInterpolationLoss` — [L5](../../nugget/losses/RBF.py#L5)
- `compute_rbf_interpolant` — [L37](../../nugget/losses/RBF.py#L37) — fits RBF + evaluates.
- `rbf(r)` — [L72](../../nugget/losses/RBF.py#L72) — Gaussian kernel `exp(-ε r²)`.

## Math

`K_{ij} = exp(-ε ||x_i − x_j||²) + 1e-8 I`; `w = K⁻¹ f`;
`s(x) = Σ_i w_i K(||x − x_i||)`; loss = MSE(f_test, s_test).

Key kwargs: `epsilon=30.0`, `loss_func=MSE`, `test_points`.

## See also

- [losses](losses.md), [surrogates](surrogates.md)

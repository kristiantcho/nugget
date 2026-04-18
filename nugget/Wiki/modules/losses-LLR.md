---
type: module
status: draft
sources:
  - ../../nugget/losses/LLR.py
updated: 2026-04-18
---

# LLR.py

Log-likelihood-ratio losses using a pre-trained `llr_net`.

## Classes

- `WeightedLLRLoss` — [L5](../../nugget/losses/LLR.py#L5) — per-string LLR with sigmoid string weights.
  - `compute_LLR_per_string` — [L27](../../nugget/losses/LLR.py#L27)
- `WeightedMeanDifLLRLoss` — [L103](../../nugget/losses/LLR.py#L103) — signal − background per-string LLR.
- `LLRLoss` — [L165](../../nugget/losses/LLR.py#L165) — per-point LLR.
  - `compute_LLR_per_point` — [L186](../../nugget/losses/LLR.py#L186)
- `MeanDifLLRLoss` — [L241](../../nugget/losses/LLR.py#L241) — per-point signal − background.

## Formulation

```
LLR_i = E_events[ llr_net(prepare_features(point_i, event)) ]
L     = sigmoid(-1/N * sharpness * Σ w_i LLR_i)
```
`w_i = sigmoid(string_weights[i])` if provided, else uniform.

## Key kwargs
`llr_net`, `signal_event_params`, `background_event_params`,
`signal_surrogate_func`, `background_surrogate_func`, `sharpness`,
`event_labels`.

Returns: `*_llr_loss`, `signal_llr_per_string/point`, `signal_total_llr`,
`background_total_llr`.

## See also

- [losses](losses.md), [losses-fisher_info](losses-fisher_info.md), [surrogates-LLRnet](surrogates-LLRnet.md)

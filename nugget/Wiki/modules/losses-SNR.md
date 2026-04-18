---
type: module
status: draft
sources:
  - ../../nugget/losses/SNR.py
updated: 2026-04-18
---

# SNR.py

Signal-to-noise-ratio losses.

## Classes

- `SNRloss` — [L5](../../nugget/losses/SNR.py#L5) — full-detector SNR.
- `WeightedSNRLoss` — [L140](../../nugget/losses/SNR.py#L140) — per-string SNR with string weights.
  - `compute_snr_per_string` — [L184](../../nugget/losses/SNR.py#L184)

## Formulation

```
avg_sig = Σ_events Σ_i surrogate_sig * scale_sig / n_events
avg_bkg = Σ_events Σ_i surrogate_bkg * scale_bkg / n_events
SNR = avg_sig / sqrt(avg_bkg + eps)
L = sigmoid(-SNR * sharpness / n_points)
```
`no_background=True` uses a constant `background_scale` in place of a bkg surrogate.

## See also

- [losses](losses.md), [losses-light_yield](losses-light_yield.md), [losses-effective_area](losses-effective_area.md)

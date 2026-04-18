---
type: entity
status: draft
sources:
  - ../../nugget/examples/train_signal_only_llr_patd.py
updated: 2026-04-18
---

# train_signal_only_llr_patd.py

## Purpose
Train LLRnet with photon arrival time distribution (PATD) features for improved signal/background discrimination using timing information from individual photons.

## What it does
Advanced LLRnet training incorporating per-photon features including arrival times. Uses:

- **PATD timing features**: Individual photon hit times (delta_time) relative to geometric minimum
- **Cascade events**: Neutrino cascades with Poisson light yield and time distributions
- **Dynamic hit selection**: Variable photons per event with shuffling for robustness
- **7-layer MLP**: Deep network [64x7] for complex timing patterns
- **Per-photon training**: Each photon gets full event context features

## Key code references
- [LLRnet PATD setup](../../nugget/examples/train_signal_only_llr_patd.py#L30-L59): use_patd=True, input_delta_time=True for timing features, 7-layer architecture
- [LightSabrePATD surrogate](../../nugget/examples/train_signal_only_llr_patd.py#L10-L16): Photon timing with max energy distribution
- [PATD DataLoader](../../nugget/examples/train_signal_only_llr_patd.py#L63-L72): Per-photon features with shuffling
- [Training loop](../../nugget/examples/train_signal_only_llr_patd.py#L87-L93): 1000 epochs, input_dim=12

## Inputs
- **Event sampler**: CylinderSampler (domain 2500m, random ray positions)
- **Light yield**: LightSabrePATD with photon timing
- **Per-photon features**: 11 event context + timing features, minimum 1 photon
- **Training**: 1000 epochs, 5000 samples/epoch, batch 16, learning rate 1e-4

## Outputs
- **Model**: best_hit_llr_model_v3 (PATD-trained network)
- **Training history**: hit_llr_v3_training_history.pkl

## Related modules
- [LLRnet class](../../nugget/surrogates/LLRnet.py): LLR with PATD support
- [LightSabrePATD](../../nugget/surrogates/LightSabre.py): Photon arrival times
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling

## See also
- [train_signal_only_llr.py](example-train_signal_only_llr.md): Without timing features
- [make_data_signal_only_llr_patd.py](example-make_data_signal_only_llr_patd.md): Pregenerate PATD datasets
- [test_NSF_patd.ipynb](example-test_NSF_patd.md): Tests PATD-based inference

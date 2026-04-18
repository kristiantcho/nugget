---
type: entity
status: draft
sources:
  - ../../nugget/examples/train_signal_only_llr.py
updated: 2026-04-18
---

# train_signal_only_llr.py

## Purpose
Train LLRnet for neutrino event classification using only signal events with cascade physics mode for photon arrival time based discrimination.

## What it does
Trains a neural network to predict log-likelihood ratios for signal vs. background discrimination in cascade events:

- **LightSabre surrogate**: Poisson-based light yield with cascade particle mode
- **CylinderSampler**: Signal events with random positions along neutrino ray (domain 2000m)
- **LLRnet architecture**: 6 hidden layers [64,64,64,64,64,64], no Fourier features
- **DataLoader**: 2048 samples per epoch, batch size 16, 4 parallel workers

## Key code references
- [LLRnet initialization](../../nugget/examples/train_signal_only_llr.py#L17-L39): 6-layer MLP without Fourier features
- [LightSabre setup](../../nugget/examples/train_signal_only_llr.py#L4): Cascade particle mode with Poisson
- [Signal-only dataloader](../../nugget/examples/train_signal_only_llr.py#L41-L54): Signal events only training data
- [Training](../../nugget/examples/train_signal_only_llr.py#L56-L61): 1000 epochs with LR scheduler

## Inputs
- **Event sampler**: CylinderSampler (cascade mode, domain 2000m)
- **Light yield**: LightSabre cascade with Poisson sampling
- **Training**: 1000 epochs, batch 16, learning rate 1e-3

## Outputs
- **Model**: best_cascade_charge_llr_model_v1 (trained network)
- **Training history**: cascade_charge_llr_v1_training_history.pkl

## Related modules
- [LLRnet class](../../nugget/surrogates/LLRnet.py): LLR network architecture
- [LightSabre](../../nugget/surrogates/LightSabre.py): Light yield surrogate
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling

## See also
- [train_signal_only_llr_patd.py](example-train_signal_only_llr_patd.md): Enhanced with PATD timing
- [loss_landscape_test.ipynb](example-loss_landscape_test.md): LLR performance evaluation

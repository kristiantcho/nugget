---
type: entity
status: draft
sources:
  - ../../nugget/examples/train_chargenet.py
updated: 2026-04-18
---

# train_chargenet.py

## Purpose
Train ChargeNet, a neural network surrogate that predicts detector response (charge/photon counts) from neutrino event parameters using Fourier feature embeddings and multi-branch architecture.

## What it does
Trains ChargeNet to learn the mapping from event parameters (position, energy, direction) to detector charge response. Uses:

- **LightSabre surrogate**: Non-Poisson light yield model (no statistical noise)
- **CylinderSampler**: Signal events with log-uniform energy distribution over 1e2-1e8 GeV
- **Multi-branch neural network**: 2 parallel branches with Fourier features (scales 0.1, 0.4) plus shared MLP for ensemble learning
- **PyTorch DataLoader**: Efficient batched training with 4 parallel workers

## Key code references
- [ChargeNet initialization](../../nugget/examples/train_chargenet.py#L10-L30): Multi-branch architecture with Fourier features and residual connections
- [LightSabre setup](../../nugget/examples/train_chargenet.py#L5-L7): Non-Poisson surrogate configuration
- [DataLoader creation](../../nugget/examples/train_chargenet.py#L32-L42): Training data pipeline with 2048 samples per epoch, batch size 8
- [Training](../../nugget/examples/train_chargenet.py#L44-L49): 500 epochs with LR scheduler on plateau

## Inputs
- **Event sampler**: CylinderSampler with log-uniform energy (1e2-1e8 GeV)
- **Light yield surrogate**: LightSabre (non-Poisson)
- **Training parameters**:
  - Epochs: 500
  - Samples per epoch: 2048
  - Batch size: 8
  - Learning rate: 1e-4
  - Architecture: [64, 64, 64, 64] hidden dims

## Outputs
- **Model checkpoint**: best_charge_net_model (trained weights)
- **Training history**: charge_net_training_history.pkl (loss curves)

## Related modules
- [ChargeNet class](../../nugget/surrogates/ChargeNet.py): Core architecture
- [LightSabre surrogates](../../nugget/surrogates/LightSabre.py): Light yield model
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling

## See also
- [train_hitflow.py](example-train_hitflow.md): Trains on photon timing distributions
- [example_notebook.ipynb](example-example_notebook.md): Interactive demonstration

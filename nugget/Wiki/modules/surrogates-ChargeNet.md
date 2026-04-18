---
type: module
status: draft
sources:
  - ../../nugget/surrogates/ChargeNet.py
updated: 2026-04-18
---

# ChargeNet.py

## Purpose

Neural network surrogate for continuous light yield prediction. ChargeNet trains as a regression model to directly predict detector response (photon count) at a given detector location, given event parameters. Complements binary classification approaches like LLRnet with continuous output and optional log-scale training for better convergence.

## Architecture

### Core Components

[FourierFeatures module](../../nugget/surrogates/ChargeNet.py#L26): Multiscale Fourier feature mapping with learnable/fixed frequencies.

[ResidualBlock module](../../nugget/surrogates/ChargeNet.py#L85): Residual layers with optional dimension projection and SiLU activation.

[ChargeNet class](../../nugget/surrogates/ChargeNet.py#L120): Main network with configurable parallel Fourier branches.

### Network Structure
- **Multiple parallel branches** with independent or shared MLPs
- **Fourier feature layers** at different frequency scales (e.g., 0.5×, 1.0×, 2.0× scale)
- **Branch MLPs** process Fourier outputs independently
- **Final MLP** concatenates branch outputs and produces scalar light yield prediction (linear output for regression)

### Key Constructor Parameters
- `num_parallel_branches` (int, default 1): Number of parallel Fourier+MLP branches
- `frequency_scales` (list): Scale factors for each branch (geometric progression by default)
- `shared_mlp` (bool): Whether all branches use single shared MLP vs. separate MLPs
- `use_residual_connections` (bool): Enable ResidualBlock layers
- `log_scale_ly` (bool): **Recommended: train to predict log10(light_yield)** for better convergence
- `add_relative_pos`, `add_distance_from_beam`: Feature engineering options
- `norm_pos` (bool): Normalize coordinates by domain_size/2

## Training/Inference API

### Training

[\_build_network](../../nugget/surrogates/ChargeNet.py#L274): Dynamically constructs architecture based on input dimension and branch configuration.

[train_with_dataloader](../../nugget/surrogates/ChargeNet.py#L538): Standard PyTorch training loop with optional validation and learning rate scheduling.

```python
history = model.train_with_dataloader(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=100,
    early_stopping_patience=10
)
```

### Data Preparation

[prepare_data_from_raw](../../nugget/surrogates/ChargeNet.py#L418): Convert raw event data to feature tensors.

[ChargeDataset](../../nugget/surrogates/ChargeNet.py#L942): Custom dataset with optional minimum light yield threshold and resampling.

[create_charge_dataloader](../../nugget/surrogates/ChargeNet.py#L1026): Factory method for DataLoaders with noise injection and resampling strategies.

### Inference

[\_forward_pass](../../nugget/surrogates/ChargeNet.py#L671): Internal forward pass through parallel branches and final MLP.

[\_\_call\_\_](../../nugget/surrogates/ChargeNet.py#L725): Evaluate trained model (sets eval mode automatically).

[predict](../../nugget/surrogates/ChargeNet.py#L757): Predict light yield and optionally convert from log scale.

[evaluate](../../nugget/surrogates/ChargeNet.py#L788): Compute regression metrics (MSE, MAE, R², RMSE).

### Model Persistence

[save_model](../../nugget/surrogates/ChargeNet.py#L830), [load_model](../../nugget/surrogates/ChargeNet.py#L872): Save/restore full training state.

## Inputs/Outputs Tensors

**Training Inputs:**
- `features`: torch.Tensor, shape (batch, feature_dim) — concatenated position, direction, energy, etc.
- `targets`: torch.Tensor, shape (batch,) — light yield (optionally log-scaled)

**Inference Inputs:**
- `opt_point`: torch.Tensor, shape (dim,) — detector position
- `event_params`: dict with 'position', 'zenith', 'azimuth', 'energy', optionally 'direction'

**Outputs:**
- torch.Tensor, shape (batch,) or scalar — predicted light yield (converted from log if `log_scale_ly=True`)

## Dependencies

- `torch`: neural networks, tensor ops
- `numpy`: numerical
- `torch.utils.data`: Dataset, DataLoader

## Notes

- MSE loss for regression (not classification)
- Optional ReduceLROnPlateau scheduler
- Log-scale training (`log_scale_ly=True`) strongly recommended for stability
- Wrapper class [ChargeNetSurrogate](../../nugget/surrogates/ChargeNet.py#L1083) provides physics-simulation interface compatible with sampling workflows
- Can load checkpoints via [from_checkpoint](../../nugget/surrogates/ChargeNet.py#L1268)

## See also

- [[surrogates-base_surrogate]] — parent class
- [[surrogates-LLRnet]] — binary classification variant
- [[surrogates-HitFlowNet]] — normalizing flow surrogate
- [[concepts-surrogate-modelling]] — surrogate theory
- [[entities-FourierFeatures]] — feature engineering module (shared across nets)

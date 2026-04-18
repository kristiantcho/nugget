---
type: entity
status: draft
sources:
  - ../../nugget/examples/train_hitflow.py
updated: 2026-04-18
---

# train_hitflow.py

## Purpose
Train a HitFlow surrogate model to predict photon arrival time distributions (PATD) for neutrino detector events using the LightSabrePATD light yield surrogate.

## What it does
This script trains a neural network-based flow model ([HitFlow](../../nugget/surrogates/HitFlow.py)) that learns to model the distribution of photon hit times in a detector given event parameters (position, energy, direction). The training uses:

- **LightSabrePATD surrogate**: Generates photon arrival times for signal events with Poisson statistics and energy-dependent distributions
- **CylinderSampler**: Samples signal events uniformly within a cylindrical detection volume (2500m domain, 1e2-1e8 GeV energy range)
- **PiecewiseRationalQuadraticCDF layers**: Flow architecture with 12 layers and spline transforms for flexible density estimation

## Key code references
- [HitFlow initialization](../../nugget/examples/train_hitflow.py#L24-L38): Configures neural network depth, hidden features, spline bins, and learning parameters
- [LightSabrePATD setup](../../nugget/examples/train_hitflow.py#L6-L12): Configures photon arrival time distribution with Poisson sampling
- [CylinderSampler initialization](../../nugget/examples/train_hitflow.py#L14-L21): Event sampling with log-uniform energy distribution
- [Model training call](../../nugget/examples/train_hitflow.py#L40-L53): Trains for 3000 iterations with 5000 events per epoch, learning rate 1e-4

## Inputs
- **Event sampler**: CylinderSampler generating signal events (neutrino interactions)
- **Light yield surrogate**: LightSabrePATD computing photon counts and timing
- **Training parameters**: 
  - 3000 iterations (training loops)
  - 5000 events per epoch
  - Batch size: 32
  - Learning rate: 1e-4
  - Minimum hits per event: 10
  - Maximum hits per event: 100

## Outputs
- **Model checkpoint**: `best_hitflow_model_v4/` (trained neural network weights)
- **Training history**: `hitflow_training_history_v4.npy` (loss/metric evolution)

## Related modules
- [HitFlow surrogate](../../nugget/surrogates/HitFlow.py): Core flow model class
- [LightSabre surrogates](../../nugget/surrogates/LightSabre.py): Light yield prediction
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event generation
- [train_hitflownet.py](example-train_hitflownet.md): Extended version creating HitFlowNet datasets

## See also
- [train_chargenet.py](example-train_chargenet.md): Trains ChargeNet for charge-only prediction
- [example_notebook.ipynb](example-example_notebook.md): Interactive workflow demonstration

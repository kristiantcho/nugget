---
type: entity
status: draft
sources:
  - ../../nugget/examples/train_hitflownet.py
updated: 2026-04-18
---

# train_hitflownet.py

## Purpose
Pregenerate and save a HitFlowNet dataset by training individual normalizing flows for each detector event, then prepare for downstream HitFlowNet model training on photon arrival time distributions.

## What it does
Creates a large training dataset for HitFlowNet by sampling signal events, training individual normalizing flows for each event's photon timing distribution, and saving flow parameters to disk.

The HitFlowNet architecture combines per-event flows with a meta-network that learns relationships between flows and event parameters.

## Key code references
- [HitFlowNet initialization](../../nugget/examples/train_hitflownet.py#L24-L48): 5-layer flow with PiecewiseRationalQuadraticCDF transforms, num_bins=4, tail_bound=6.0
- [LightSabrePATD surrogate](../../nugget/examples/train_hitflownet.py#L6-L12): Photon arrival time with Poisson
- [CylinderSampler](../../nugget/examples/train_hitflownet.py#L14-L21): Signal events in cylindrical volume
- [Flow dataset creation](../../nugget/examples/train_hitflownet.py#L50-L63): Per-event flow training with 2000 iterations, saves every 10 events

## Inputs
- **Event sampler**: CylinderSampler with 1e2-1e8 GeV energy
- **Light yield surrogate**: LightSabrePATD with Poisson sampling
- **Dataset**: 100 events, 5 photons minimum per event
- **Flow training**: 2000 iterations per event, learning rate 1e-3

## Outputs
- **Flow dataset directory**: hitflownet_flow_dataset_test/ (individual flow parameters)
- **Flow metadata**: Saved for quick loading

## Related modules
- [HitFlowNet class](../../nugget/surrogates/HitFlowNet.py): Meta-network architecture
- [LightSabre surrogates](../../nugget/surrogates/LightSabre.py): PATD generation

## See also
- [train_hitflow.py](example-train_hitflow.md): Single flow model alternative
- [test_NSF_patd.ipynb](example-test_NSF_patd.md): Tests flow training on photon timing

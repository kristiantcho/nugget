---
type: entity
status: draft
sources:
  - ../../nugget/examples/make_data_signal_only_llr_patd.py
updated: 2026-04-18
---

# make_data_signal_only_llr_patd.py

## Purpose
Pregenerate a large HDF5 dataset of signal events with photon arrival time distribution (PATD) features for efficient training of LLRnet with timing information.

## What it does
Generates and saves a 1 million event training dataset with per-photon timing features:

- **LightSabrePATD surrogate**: Generates photon arrival times with Poisson statistics and energy-dependent distributions
- **Event sampling**: Signal events with log-uniform energy (1e2-1e8 GeV) and random positions
- **Per-photon features**: 11 event parameters + individual photon hit times
- **HDF5 storage**: Efficient disk storage for sequential training with PyTorch DataLoader
- **Maximum photons**: 200 photons per event (truncated/padded)

## Key code references
- [LLRnet configuration](../../nugget/examples/make_data_signal_only_llr_patd.py#L7-L30): 3 parallel branches with Fourier features, domain 2500m, use_patd=True
- [LightSabrePATD setup](../../nugget/examples/make_data_signal_only_llr_patd.py#L3): PATD surrogate with 500 track points
- [Dataset generation](../../nugget/examples/make_data_signal_only_llr_patd.py#L32-L39): Pregenerates 1 million events, max 200 photons per event
- [Output file](../../nugget/examples/make_data_signal_only_llr_patd.py#L38): 1e6_200_patd_dataset.h5

## Inputs
- **Event sampler**: CylinderSampler with log-uniform energy (1e2-1e8 GeV)
- **Light yield**: LightSabrePATD with 500 track points
- **Data parameters**:
  - Number of events: 1,000,000
  - Maximum photons per event: 200
  - Event labels: position, energy, direction

## Outputs
- **HDF5 dataset**: 1e6_200_patd_dataset.h5 (1M events with timing features)
- **Generation stats**: Printed summary of dataset creation

## Related modules
- [LLRnet class](../../nugget/surrogates/LLRnet.py): Pregenerated dataloader support
- [LightSabrePATD](../../nugget/surrogates/LightSabre.py): PATD generation
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling

## See also
- [train_signal_only_llr_patd.py](example-train_signal_only_llr_patd.md): Uses pregenerated data for training
- [test_NSF_patd.ipynb](example-test_NSF_patd.md): Tests trained models on timing data

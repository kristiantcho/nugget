---
type: entity
status: draft
sources:
  - ../../nugget/examples/example_notebook.ipynb
updated: 2026-04-18
---

# example_notebook.ipynb

## Purpose
Interactive tutorial demonstrating the complete NUGGET workflow: from event sampling and surrogate model training to detector optimization with real-time visualization.

## What it does
Walks through a full geometry optimization cycle with detailed explanations:

1. **Surrogate models**: Creates a SkewedGaussian light yield model and initializes ToySampler events
2. **LLRnet training**: Trains a multi-branch LLR network to discriminate signal vs. background with Fourier features and residual connections
3. **Data preprocessing**: Generates 250 signal/background events and precomputes LLR and light yield metrics per string
4. **Geometry setup**: Initializes EvanescentString geometry with 1000 strings in a 4m domain (toy scale)
5. **Optimization loop**: Runs 100 iterations optimizing string weights with multiple physics-based and geometric penalties
6. **Visualization**: Creates loss component plots, weight distributions, signal/background contours
7. **Output**: Saves best geometry and generates optimization animation GIF

## Key workflow cells

**Cell 2**: Create light yield surrogate and samplers - SkewedGaussian with sigma_factor=6

**Cell 9**: LLRnet initialization - 256-128-64-32 hidden dims, 2 parallel branches with Fourier features, 64 frequencies per branch

**Cell 11**: Create dataloaders - separate training (10k samples), validation (1k), and test (10k) loaders

**Cell 13**: Train LLRnet - 800 epochs with early stopping patience 50 on validation loss

**Cell 18**: Initialize EvanescentString geometry and Optimizer with ALM

**Cell 20**: Precompute LLR metrics per string for efficient optimization

**Cell 21**: Precompute light yield per string

**Cell 22**: Precompute Fisher information for resolution calculations

**Cell 24**: Run main optimization - 100 iterations with visualization every 10 iterations

**Cell 25**: Create final animation GIF from accumulated frames

## Inputs
- **Toy domain**: 4m x 4m x 4m cube
- **Event sample sizes**: 250 signal + 250 background
- **LLRnet**: 256-128-64-32 architecture, Fourier features enabled
- **Geometry**: 1000 strings with 5 OM per string
- **Optimization**: 100 iterations, loss weights for yield (1e3), LLR (2.5), boundary (1000), penalties

## Outputs
- **Trained LLRnet**: best_toy_model (saved model weights)
- **Precomputed metrics**: .pt files for LLR, light yield, Fisher info
- **Best geometry**: best_geom.pkl (lowest combined loss)
- **Visualization**: Optimization GIF showing loss evolution and detector layout

## Related modules
- [SkewedGaussian surrogate](../../nugget/surrogates/SkewedGaussian.py): Toy light yield model
- [LLRnet](../../nugget/surrogates/LLRnet.py): LLR network with Fourier features
- [EvanescentString](../../nugget/geometries/EvanescentString.py): Detector geometry with weights
- [ToySampler](../../nugget/samplers/toy_sampler.py): Simple event distribution
- [Visualizer](../../nugget/utils/vis_tools.py): Real-time plotting
- [Optimizer](../../nugget/utils/basic_optimizer.py): Geometry optimization with ALM

## See also
- [dynamic_strings_test.ipynb](example-dynamic_strings_test.md): DynamicString variant
- [test_NSF_patd.ipynb](example-test_NSF_patd.md): PATD-based training and inference
- [loss_landscape_test.ipynb](example-loss_landscape_test.md): Comprehensive loss landscape analysis

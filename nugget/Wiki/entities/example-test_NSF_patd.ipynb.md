---
type: entity
status: draft
sources:
  - ../../nugget/examples/test_NSF_patd.ipynb
updated: 2026-04-18
---

# test_NSF_patd.ipynb

## Purpose
Test and develop normalizing flow models (NSF/HitFlow) for photon arrival time distribution (PATD) modeling with likelihood-based inference and parameter recovery.

## What it does
Advanced PATD density estimation and inference workflow:

1. **PATD surrogate**: Initialize LightSabrePATD for photon timing with cascades
2. **Flow training**: Train conditional normalizing flow (6 AR layers, custom transforms)
3. **Context features**: Per-photon feature vector including event and geometry-dependent timing
4. **Likelihood computation**: Compute conditional log-likelihood at observed hit times
5. **Parameter scanning**: 2D likelihood scan over energy-zenith parameter space
6. **Flow validation**: Compare trained flow PDF against true CPandel distributions
7. **HitFlow evaluation**: Test HitFlow surrogate as alternative to custom NSF

## Key workflow sections

**Cell 2**: PATD surrogate and CylinderSampler initialization

**Cell 3-5**: Sample high-multiplicity events and explore photon arrival time distributions

**Cell 6**: Custom flow architecture definition:
- min_geometric_time_transform: Subtract geometric timing minimum
- LogTransform: Log-scale time axis for better numerical stability
- 6 autoregressive RQS layers: Flexible nonlinear transforms
- Mixing transforms: PiecewiseRationalQuadraticCDF between AR layers

**Cell 7**: Training loop - 5000 iterations minimizing negative log-likelihood

**Cell 9**: Flow PDF validation against true CPandel PDF from LightSabre

**Cell 11**: 2D energy-zenith likelihood scan (60x60 grid)

**Cell 13**: Best-fit parameter recovery from likelihood surface

**Cell 15**: Interactive 3D scatter plot of detector response with light yield coloring

**Cell 16-17**: Load and evaluate HitFlow model as alternative to custom NSF

**Cell 18**: HitFlow 2D energy-zenith likelihood scan using evaluate_pdf method

## Inputs
- **Surrogate**: LightSabrePATD with 2000 track points, max energy distribution
- **Events**: Signal events with 1e2-1e8 GeV energy range
- **Flow architecture**: 6 AR layers with 128 hidden features, custom transforms
- **Training**: 5000 iterations, learning rate 0.5e-4, batch size up to 300 photons
- **HitFlow model**: Loaded from best_hitflow_model_v4 checkpoint

## Outputs
- **Trained flow**: Custom NSF model with optimized parameters
- **Likelihood surfaces**: 2D contour plots for energy-zenith (numpy arrays)
- **PDF comparisons**: Scatter plots comparing flow vs. true CPandel PDFs
- **Best-fit parameters**: MLE parameter estimates from likelihood scans
- **HitFlow evaluation**: Alternative PDF estimates and likelihood surfaces

## Related modules
- [LightSabrePATD](../../nugget/surrogates/LightSabre.py): Photon timing surrogate with CPandel
- [CPandel](../../nugget/surrogates/cpandel/cpandel.py): Pandel photon PDF model
- [HitFlow](../../nugget/surrogates/HitFlow.py): Pre-trained flow surrogate
- [nflows library](../../external/nflows/): Normalizing flow implementations
- [CylinderSampler](../../nugget/samplers/cyl_sampler.py): Event sampling

## See also
- [train_signal_only_llr_patd.py](example-train_signal_only_llr_patd.md): PATD-based LLR training
- [make_data_signal_only_llr_patd.py](example-make_data_signal_only_llr_patd.md): Pregenerate PATD datasets
- [train_hitflow.py](example-train_hitflow.md): HitFlow training script

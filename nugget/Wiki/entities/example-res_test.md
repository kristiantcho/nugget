---
type: entity
status: draft
sources:
  - ../../nugget/examples/res_test.py
updated: 2026-04-18
---

# res_test.py

## Purpose
Compute Fisher information (angular and energy resolution) matrices for multiple detector geometries and precompute light yield distributions for efficient geometry optimization.

## What it does
Evaluates detector physics performance (angular resolution, light yield) for different geometries:

- **Event sampling**: 100,000 signal events in specified cylindrical volume (600m radius, 1000m height)
- **Fisher information computation**: Per-string per-event Fisher information matrices for angular resolution assessment
- **Light yield computation**: Per-string light yield for efficiency calculations
- **Geometry variants**: Tests '800main_full_hex' and '340grid' layouts with 1027 and 1600 domain settings
- **LLRnet integration**: Uses trained LLRnet (best_charge_llr_model_v4) for Fisher info computation

## Key code references
- [Event sampling](../../nugget/examples/res_test.py#L33-L43): 100,000 events with specified center, radius, height
- [Geometry initialization](../../nugget/examples/res_test.py#L52-L72): EvanescentString with hexagonal layout, 20 OM per string
- [Fisher info computation](../../nugget/examples/res_test.py#L124-L137): Per-string per-event matrices with chunked computation
- [Light yield computation](../../nugget/examples/res_test.py#L139-L144): Per-string light yield using surrogate
- [Output saving](../../nugget/examples/res_test.py#L148-L149): PyTorch tensor files for reuse in optimization

## Inputs
- **Events**: 100,000 sampled signal events (1e2-1e8 GeV)
- **Geometries**: Two layouts ('800main_full_hex', '340grid') with 1027 or 1600 strings respectively
- **Light yield surrogate**: LightSabre with Poisson statistics
- **LLR network**: best_charge_llr_model_v4 (trained for Fisher info)
- **Computation**: Chunked processing for memory efficiency (grad_chunk_size=7, jacrev=50k, point=11k)

## Outputs
- **Fisher info tensors**: fisher_info_per_string_per_event_10000_{geom}_{version}.pt
- **Light yield tensors**: light_yield_per_string_10000_{geom}_{version}.pt
- **Pickle cache**: Signal events saved for batch reuse

## Related modules
- [EvanescentString geometry](../../nugget/geometries/EvanescentString.py): Detector geometry
- [Fisher information loss](../../nugget/losses/fisher_info.py): Computing resolution metrics
- [LLRnet](../../nugget/surrogates/LLRnet.py): For likelihood ratio computation
- [LightSabre](../../nugget/surrogates/LightSabre.py): Light yield simulation

## See also
- [res_test_make_geoms.py](example-res_test_make_geoms.md): Geometry optimization using precomputed Fisher info
- [test_evaluation.ipynb](example-test_evaluation.md): Evaluates optimized geometries

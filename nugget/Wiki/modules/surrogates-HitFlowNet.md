---
type: module
status: draft
sources:
  - ../../nugget/surrogates/HitFlowNet.py
updated: 2026-04-18
---

# HitFlowNet.py

Hybrid architecture: normalizing flow + conditioning MLP with
Fourier-feature inputs. Combines flow expressivity on hit-time
distributions with a learned amplitude net.

See [L1](../../nugget/surrogates/HitFlowNet.py#L1) onward. Uses
`FourierFeatures` and `LogTransform`.

## See also
- [surrogates](surrogates.md), [surrogates-HitFlow](surrogates-HitFlow.md), [surrogates-ChargeNet](surrogates-ChargeNet.md)

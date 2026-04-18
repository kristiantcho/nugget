---
type: module
status: draft
sources:
  - ../../nugget/surrogates/HitFlow.py
updated: 2026-04-18
---

# HitFlow.py

Normalizing flow learning the photon-arrival-time distribution (PATD).

## Key items
- Flow model class — [L34](../../nugget/surrogates/HitFlow.py#L34).
- Includes a `LogTransform` preconditioner for times.
- Trained on simulated PATD; evaluates log-density for hit times given event+point features.

## See also
- [surrogates](surrogates.md), [surrogates-HitFlowNet](surrogates-HitFlowNet.md), [surrogates-pandel](surrogates-pandel.md)

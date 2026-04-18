# Wiki Log

Append-only. Newest entries at the bottom. Format:
`YYYY-MM-DD HH:MM  OP  summary`.

---

2026-04-18 00:00  INIT    Wiki bootstrapped. Schema (CLAUDE.md), index.md,
                          folder skeleton (modules/, concepts/, entities/,
                          sources/, queries/) created.
2026-04-18 00:30  INGEST  Module team (7 parallel agents) ingested
                          geometries/, losses/, surrogates/, samplers/,
                          utils/, examples/, other/ + repo-root sources.
                          → 43 module pages, 19 entity pages, 5 source pages.
2026-04-18 01:00  INGEST  Concept team extracted 12 cross-cutting concepts:
                          llr, fisher-information, light-yield, pandel-timing,
                          effective-area, trigger, figure-of-merit,
                          detector-geometry, string-parameterization,
                          hungarian-matching, surrogate-modeling,
                          alm-optimization.
2026-04-18 01:15  MERGE   index.md rebuilt as full catalog of modules,
                          concepts, entities, sources.
2026-04-18 02:00  INGEST  Concept deep-dive (4 parallel general-purpose
                          agents with WebSearch/WebFetch). Rewrote all 12
                          concept pages in place with formal definitions,
                          math, physics/stats context, codebase usage
                          with source-line anchors, and external_refs:
                          - inference: llr, fisher-information, figure-of-merit
                          - optical: light-yield, pandel-timing, effective-area
                          - geometry: detector-geometry, string-parameterization,
                            hungarian-matching, trigger
                          - ML: surrogate-modeling, alm-optimization
                          Sources: IceCube 1612.05093, KM3NeT 1601.07459/2103.09885,
                          P-ONE 2008.04323/2108.04310, Baikal-GVD 2005.09493,
                          Pandel thesis, AMANDA astro-ph/0407044, MMC hep-ph/0407075,
                          Smith & Baker 1981, SkyLLH 2203.07316, IceCube-Gen2 2108.05292,
                          ConFIG 2408.11104, PCGrad 2001.06782, MGDA NeurIPS 2018,
                          Tancik 2006.10739, Eller 2308.13249, Nocedal & Wright ch.17,
                          plus Wikipedia references for each core concept.
2026-04-18 03:00  INGEST  Mermaid enhancement (4 parallel general-purpose
                          agents). Added diagrams in place:
                          - overview/pipeline: index.md (3-layer arch),
                            modules/examples.md (workflow),
                            modules/utils-basic_optimizer.md (training step),
                            modules/utils.md, modules/samplers.md.
                          - module hierarchies: modules/geometries.md,
                            modules/losses.md, modules/surrogates.md
                            (classDiagram + surrogate->loss flow).
                          - inference/ML concepts: llr, fisher-information,
                            figure-of-merit, surrogate-modeling,
                            alm-optimization (classical ALM + nugget step).
                          - physics/geometry concepts: light-yield,
                            pandel-timing, effective-area, trigger (6-step),
                            string-parameterization, detector-geometry,
                            hungarian-matching.
                          Total: 20 pages now carry Mermaid diagrams.
2026-04-18 03:30  INIT    Root CLAUDE.md created at repo root pointing every
                          future Claude session to Wiki/index.md + Wiki/CLAUDE.md.
                          Makes "always read the wiki" load-bearing via
                          Claude Code's auto-loaded CLAUDE.md mechanism.

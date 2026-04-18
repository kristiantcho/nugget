# nugget — Agent Instructions

This repository has a persistent **LLM Wiki** that serves as the project's
second brain. It is the primary context for any non-trivial task.

## Before doing anything

1. Read [nugget/Wiki/index.md](nugget/Wiki/index.md) — the full catalog of
   modules, concepts, entities, sources.
2. Read [nugget/Wiki/CLAUDE.md](nugget/Wiki/CLAUDE.md) — the wiki schema and
   operating rules. Every interaction in this repo follows that schema.
3. Consult the relevant wiki pages before opening raw source. Prefer wiki
   citations over re-deriving from code.

## Operating rules (summary — full rules in the wiki schema)

- **Ingest / Query / Lint** are the only sanctioned ops against the wiki.
- After any non-trivial investigation of raw sources, **write findings back**
  to the wiki so the work compounds. Never edit raw sources as part of a
  wiki operation.
- **Log every op** (ingest, query, lint) as one line in
  [nugget/Wiki/log.md](nugget/Wiki/log.md).
- **Cite everything.** Every factual claim links to a wiki page or a
  raw-source line range.
- The wiki ingests **only files in this repository.** No external web
  content as fact unless captured under `external_refs` in a concept page.

## File pointers

- Schema: [nugget/Wiki/CLAUDE.md](nugget/Wiki/CLAUDE.md)
- Catalog: [nugget/Wiki/index.md](nugget/Wiki/index.md)
- Log: [nugget/Wiki/log.md](nugget/Wiki/log.md)
- Folders: `modules/`, `concepts/`, `entities/`, `sources/`, `queries/` under
  [nugget/Wiki/](nugget/Wiki/).

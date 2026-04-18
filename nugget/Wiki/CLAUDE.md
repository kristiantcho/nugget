# Project Wiki — Schema & Operating Rules

This directory is an **LLM Wiki** (pattern by Andrej Karpathy, gist `442a6bf5...`).
It is a persistent, compounding second brain for the **nugget** project
(NeUtrino experiment Geometry optimization and General Evaluation Tool).

The wiki is owned by the LLM. The human curates sources (the code + files in
this repo) and asks questions. The LLM does the rest: ingest, synthesize,
cross-link, maintain.

---

## 1. Three-Layer Architecture

1. **Raw sources** — everything under `c:/Project/nugget/` *outside* `Wiki/`:
   Python modules, notebooks, CSVs, model checkpoints, README, setup.py.
   Sources are **immutable** from the wiki's point of view — never edited
   as part of wiki work.
2. **The wiki** — every markdown file under `Wiki/`. Fully LLM-owned.
   Rewrite freely when new information arrives.
3. **The schema** — this file. Defines structure, conventions, operations.
   Updated only deliberately, with a `log.md` entry.

---

## 2. Folder Conventions

```
Wiki/
  CLAUDE.md         ← this schema (never auto-rewritten)
  index.md          ← content-oriented catalog of all pages
  log.md            ← append-only chronological record
  modules/          ← one page per source-code module/file
  concepts/         ← cross-cutting ideas (e.g. Fisher information, LLR,
                      surrogate modelling, detector geometry)
  entities/         ← named concrete things: classes, notebooks,
                      datasets, model checkpoints
  sources/          ← pointers + summaries of non-code sources
                      (README, setup.py, CSV geometries, notebooks)
  queries/          ← synthesized answers to user questions that are
                      worth keeping. One file per question.
```

Rules:
- File names: `kebab-case.md`. No spaces. ASCII only.
- Every page begins with an H1 title matching the filename.
- Every page ends with a `## See also` section of wiki links
  (`[[relative/path]]` or `[text](relative/path.md)`).
- Pages cite raw sources with clickable relative links:
  `[file.py:42](../../nugget/losses/LLR.py#L42)`.
- No page may exceed ~400 lines. If it grows larger, split.

---

## 3. Page Front-Matter

Every wiki page (except `index.md`, `log.md`, `CLAUDE.md`) starts with:

```markdown
---
type: module | concept | entity | source | query
status: stub | draft | stable
sources:
  - ../../nugget/path/to/file.py
updated: YYYY-MM-DD
---
```

`status` ladder: `stub` → `draft` → `stable`. Move up only after the page
has been cross-checked against its sources in a later pass.

---

## 4. Operations

### 4.1 Ingest
Input: one or more raw-source paths.
Steps:
1. Read the source(s) fully.
2. For each distinct unit (module, class, concept, dataset), create or
   update a page in the appropriate folder.
3. Add/refresh cross-links to related pages.
4. Update `index.md` (add new entries, keep alphabetical within each
   section).
5. Append one line to `log.md`:
   `YYYY-MM-DD HH:MM  INGEST  <source paths>  →  <pages touched>`.

### 4.2 Query
Input: a user question.
Steps:
1. Search the wiki first (`Wiki/**/*.md`).
2. If sufficient, answer with citations to wiki pages **and** raw sources.
3. If insufficient, ingest the missing sources, then answer.
4. If the question + answer are reusable, save to
   `queries/<slug>.md` and link from `index.md`.
5. Append to `log.md`:
   `YYYY-MM-DD HH:MM  QUERY   "<question>"  →  <pages consulted/created>`.

### 4.3 Lint
Periodic health-check. Steps:
1. List orphan pages (not linked from `index.md` or any other page).
2. Find broken wiki links and broken source links.
3. Flag contradictions between pages.
4. Flag stale pages (`updated` older than last source mtime).
5. Write findings to `log.md` under a `LINT` entry; fix trivial issues
   immediately, open follow-up items for the rest.

---

## 5. Interaction Rules (for the LLM)

Every future interaction in this project follows this schema:

1. **Read first.** Always read `index.md` and relevant wiki pages before
   touching raw sources.
2. **Prefer wiki over re-derivation.** If a fact is already on a stable
   page, cite it instead of re-reading source.
3. **Write back.** After any non-trivial investigation of raw sources,
   update the wiki so the work compounds.
4. **Never edit raw sources** as part of a wiki operation. Wiki work is
   read-only with respect to the project.
5. **Log every operation.** Ingest, query, lint — all get one line in
   `log.md`.
6. **Cite everything.** Each factual claim links to either a wiki page
   or a raw-source line range.
7. **Small, focused pages.** One concept per page. Split when in doubt.
8. **Be honest about uncertainty.** Mark unverified claims with
   `> TODO: verify against <source>`.

---

## 6. Agent Teams

Ingestion and linting are parallelizable. Dispatch sub-agents as teams:

- **Module team** — one Explore agent per top-level source folder
  (`geometries/`, `losses/`, `surrogates/`, `samplers/`, `utils/`,
  `examples/`, `other/`). Each produces pages under `modules/`.
- **Concept team** — after modules exist, a second wave extracts
  cross-cutting concepts (Fisher info, LLR, light yield, Pandel, etc.)
  into `concepts/`.
- **Lint team** — a single agent runs the lint checklist above.

Team outputs must conform to §2 and §3. The main agent merges results
and updates `index.md` + `log.md`.

---

## 7. Scope

The wiki ingests **only** files inside this repository. No external web
content, no training-data recollection presented as fact. If external
context is needed, it lives in the user's question, not the wiki.

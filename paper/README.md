# ViPragSent NAACL paper production

This directory is the complete write boundary for the finished paper task.
The finalization pass included the authorized bounded manuscript compression,
the evidence-label correction, mechanical Markdown-to-LaTeX conversion, final
Tectonic compilation, and rendered-page inspection. The current files are a
completed artifact set rather than an unresolved author/production handoff.

## Current state

The final PDF is `ViPragSent_NAACL.pdf` (13 pages total; main research content
through page 9, References beginning on page 10, appendix ending on page 13).
Its SHA-256 is
`54445332F071EE4ECC25A0D5DDBFA3CA897B068A1123B8117C4B9059C0031ACA`, matching
`build/main.pdf`. The final rendered pages are in
`build/qa-render/page-01.png` through `build/qa-render/page-13.png`; all 13
pages were rendered and the Q1a/Q1b/Q2/Q3/Q4 pages were visually rechecked,
including the new expanded Q1a bar chart. The architecture source PDF remains
in `figures/`.

The manuscript SHA-256 is
`C7E492A5887465FCC8746D59AD9D1E2F7171C7BEA461D3CF9F0BB0B3BB597F91`, and the
conversion manifest records exact source parity. The evidence brief SHA-256 is
`36BEF23274DFC5B37CB21F1EF6C99066D51F6DB61401B52A916384E56BBF2D02`.

The evidence remains artifact-level verified: the paper does not claim an
independent training/inference rerun, statistical significance, or stronger
reproducibility than the supplied artifacts support. Existing metric values
were retained; added rows remain provenance-labelled.

## Required source contract

The production entry point is `main.tex`; `tools/convert_manuscript.py`
regenerates the required section files from `manuscript.md`. The required
sections are:

1. `sections/abstract.tex`
2. `sections/introduction.tex`
3. `sections/related_work.tex`
4. `sections/method.tex`
5. `sections/experiments.tex`
6. `sections/results.tex`
7. `sections/discussion.tex`
8. `sections/limitations.tex`
9. `sections/ethics.tex`
10. `sections/conclusion.tex`
11. optional `sections/appendix.tex`
12. `references.bib`

The converter preserves citations, table values, figure references, and
uncertainty labels. The final build has a non-empty 6,301-byte `main.bbl` and
zero undefined citation/reference diagnostics. The compiler may emit
non-blocking box diagnostics; visual review found no clipping, overlap, or
unreadable table content in the inspected PDF.

## Build commands

From PowerShell:

```powershell
.\build_paper.ps1 -Smoke
.\build_paper.ps1 -Final
```

The converter runs before compilation. `-Final` checks the required sections,
bibliography, citations, placeholder markers, title, and compiler diagnostics,
then emits `ViPragSent_NAACL.pdf`. The QA render directory is produced with the
local Poppler `pdftoppm` command for visual inspection. `build/current.txt`,
`build/main_layout.txt`, and `build/main.txt` are the regenerated text previews;
older named check extracts are historical diagnostics.

The Q1b presentation cleanup is complete. The Q1a Vistral CoT-only clean rerun
remains explicitly provisional at 2/3 completed test seeds. The exact
`q1a_cot_only_vistral_clean_rerun_003__seed_20260522` directory exists on the
inspected Hugging Face artifact repos, but its test-generation/evaluation stages
remain `NOT_STARTED` and no completed test prediction/metric artifacts are
present; its three-seed mean and sample SD are therefore not recomputed or
imputed.
Independent rerun and significance testing remain intentionally out of scope
and are reported as evidence limitations.

# ViPragSent NAACL production report

## Status: FINAL VERIFIED

Date: 2026-09-21

The expanded-table authoring and production pass is complete for the current
evidence boundary. Existing metric values were retained while Q1a/Q1b were
expanded with provenance-labelled baseline rows, Q1b Table 2 was expanded to
the six comparable three-seed rows plus the explicitly contextual three-seed
PhoBERT inventory row, Azure/identity-mismatched and unresolved GPT records
were omitted from the metric table, the dataset statistics table was added,
and Q3 cohort labels/missing cells were made explicit. The Q2 development
ECE audit was independently recomputed from all 18 saved runs; no metric was
fabricated. An expanded 12-row Q1a macro-pragmatic F1 bar chart was generated
from the canonical row inventory and inserted with explicit n/status labels.

### Final outputs and parity

- Final PDF: `D:\vipragsent\paper\ViPragSent_NAACL.pdf`
- PDF pages: 13 total; main research content reaches page 9, with References
  beginning on page 10 and the appendix ending on page 13.
- Final PDF SHA-256:
  `54445332F071EE4ECC25A0D5DDBFA3CA897B068A1123B8117C4B9059C0031ACA`
- `build\main.pdf` has the identical SHA-256 and output parity.
- `manuscript.md` SHA-256:
  `C7E492A5887465FCC8746D59AD9D1E2F7171C7BEA461D3CF9F0BB0B3BB597F91`
- `evidence\evidence_brief.md` SHA-256:
  `36BEF23274DFC5B37CB21F1EF6C99066D51F6DB61401B52A916384E56BBF2D02`
- The conversion manifest records the same manuscript SHA-256.
- `main.bbl` is present and non-empty at 6,301 bytes; undefined citation or
  reference diagnostics: 0.

### Final visual review

All 13 pages were rendered to the current QA render set under
`D:\vipragsent\paper\build\qa-render\page-01.png` through
`page-13.png` and the changed dataset/Q1a/Q1b/Q2/Q3/Q4 pages were inspected.
No clipping, missing rows, unreadable figures, or bibliography defects were
found. The expanded dataset, Q1a, Q1b, and Q3 tables are readable; the Q1a
table remains dense and the compiler retains non-blocking box diagnostics.
The new Q1a bar chart is legible with its coverage annotations. Main-content
pagination is nine pages.

Figure 1 is included in the compiled manuscript as the clean high-resolution
PNG; its source PDF remains preserved. Page 4 was specifically rechecked:
box outlines and the hidden-sequence/memory path are continuous, `Hidden
sequence H` fits inside its box, and the decoder-to-objective route stays below
the teacher-forced-token box. The training-only shifted-token path is explicit.
The PNG include avoids the PDF-import stroke artifacts.

Current architecture hashes:

- `figures\figure_architecture_schematic.pdf`:
  `8F05496D68BAFE9644847F41BDD5AA7C26C945976811D024FA5093166291C304`
- `figures\figure_architecture_schematic.png`:
  `F8112E355D6B78EFA0415EBF6BE15B75849DC61FB4A98E89EAA4BE017F3CEDAC`

Q3 and Q4 figures were checked in the final rendered pages and remain
readable. The regenerated Q3 figure leaves visible gaps for incomplete
PhoBERT standard-cohort cells; its PNG SHA-256 is
`19355F9B70B4DC66667F235515C22F7A9117FBCF696E5BD5C3CCBAACBA00B85B`.
The compiler log contains non-blocking box diagnostics; none has a
corresponding visible clipping or collision in the inspected pages.

### Evidence boundary

The paper retains artifact-analysis wording. The evidence package reports the
XLM-R-large comparison and saved prediction artifacts, but no independent
training/inference rerun or significance test was performed. Those remain
limitations of the evidence, not open production gates. No new research,
metric, ethics, or reproducibility claim was added.

### Remaining evidence boundary

The Vistral CoT-only clean rerun still has verified test evidence for seeds
20260521 and 20260523. The exact Hugging Face seed-20260522 run directory is
present in the current `overflow-006` and Vistral-checkpoint trees and contains
development-side predictions/metrics plus checkpoint/receipt artifacts. The
development metric is not a test result; no completed test prediction or
test-metric artifact is present. It remains
`PROVISIONAL_2_OF_3` in Q1a; no three-seed mean or sample SD is imputed.
Independent rerun and significance testing remain explicitly out of scope and
are not represented as completed.

The detailed exact-run audit is recorded in
[`q1a_cot_seed22_hf_audit.json`](revision_q1a_extra/q1a_cot_seed22_hf_audit.json)
and [`q1a_cot_seed22_hf_audit.md`](revision_q1a_extra/q1a_cot_seed22_hf_audit.md).
The exhaustive live-Hugging-Face audit covered all 29 artifact repositories and
their paginated `main` trees, including records, snapshots, packs, epochs, and
prediction paths. The seed-22 COT lineage has zero `test_predictions`, zero
`test_reasoning_metrics`, and zero `test_metrics` files. The dev-side artifact
in
[`overflow-006`](https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-006)
is PASS at 1,999/1,999 rows; [`overflow-021`](https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-021)
is in a pending-approval/preflight state with training marked `RUNNING`, has
checkpoint manifests but no test outputs, and still reports test stages
`NOT_STARTED`. The similarly
named `explanation_only` run is a different recipe and is not substituted.

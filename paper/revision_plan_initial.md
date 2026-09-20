# Initial paper/reference gap review and staged revision plan

Date: 2026-09-20  
Scope: first production pass only; evidence-dependent row expansion remains pending.

## Evidence read before editing

- Current author source: `manuscript.md`, with the current Q1a/Q1b/Q2/Q3/Q4
  tables and the artifact-analysis qualification language.
- Current production source: `main.tex`, `build_paper.ps1`, and
  `tools/convert_manuscript.py`.
- Current accepted PDF: `ViPragSent_NAACL.pdf`, 11 pages at the time of the
  snapshot; the previous production report described an 8-page main-content
  target, but the present request explicitly prioritizes complete comparison
  coverage over that earlier bound.
- Structural reference: `D:/vipragsent-pr/tmp/pdf_reference/main.pdf`, 14
  pages. It is useful for section/table/figure organization only. Its metrics,
  five-seed Vistral-backed study, annotation/ethics statements, and baseline
  claims do not match the current XLM-R-large artifact cohort and are not
  imported as evidence.

## Gaps found

1. The current Q1a source contains a six-system subset with recorded values,
   but coverage is still under investigation; it is not declared scientifically
   complete at this stage. Its orientation was opposite to the requested
   production schema: phenomena were rows and systems were columns. The source
   table has now been transposed without changing any value: the single primary
   `ViPragSent XLM-R-large` row is followed by the five currently recorded
   comparison rows; the six pragmatic metrics and macro-pragmatic F1 are now
   columns.
2. Q1b already has the requested orientation (models as rows; UIT-VSFC,
   UIT-VSMEC, AIVIVN, and ordinary-F1 aggregate as columns). Its current three
   rows are retained provisionally. Additional rows or restored baselines must
   come only from the evidence worker package.
3. A Q1a macro-F1 plot exists in `figures/`, but the current production set
   deliberately leaves it uninserted. It is a preserved point/error-bar plot,
   not the requested model-level bar plot, and its source package is the
   earlier five-baseline snapshot. It should not be silently treated as the
   final figure after row restoration.
4. No Q1b grouped external-retention plot is currently inserted. The paper has
   the table and prose but no visual profile of dataset-specific transfer and
   variability.
5. Q2 currently reports the no-polarity ECE cell as `N/A`. This value is
   explicitly frozen until `revision_q2` returns; this pass does not reinterpret
   or overwrite it.

## Planned revision after evidence arrives

### Q1a table and figure

- Reconcile the exhaustive worker ledger with the current five baseline rows;
  add only rows with verified comparable artifacts and record every excluded
  family with its reason.
- Keep exactly one primary ViPragSent XLM-R-large row. Do not turn contextual
  prior work or incomplete GPT/Vistral variants into measured rows without the
  required comparable evidence.
- Preserve the transposed schema: model rows; implicit sentiment, sarcasm,
  irony, idiom/figurative, code-switching, mocking, and macro-pragmatic F1 as
  columns. Keep mean +/- sample SD and explicit missingness.
- Generate an informative model-level bar plot of macro-pragmatic F1 with
  sample-SD error bars. Highlight the primary row, use a neutral baseline
  palette, sort only if this does not obscure the named primary, and annotate
  the plot as descriptive over saved runs. It must use the verified ledger,
  not hand-entered values.

### Q1b table and figure

- Preserve the current requested schema: model rows; each verified external
  dataset and the ordinary-F1 aggregate as columns.
- Restore only evidence-backed external model rows. Do not impute missing
  dataset cells or turn heterogeneous label semantics into a pooled metric.
- Add a grouped external-retention visualization after evidence integration.
  Use horizontal grouped bars when the verified model count is large, or
  faceted dataset panels when separate panels make missingness and labels more
  readable. Show dataset-wise mean +/- sample SD and treat ordinary-F1 as a
  clearly distinguished summary group, not as another dataset. The plot is
  required as an informative visual artifact; layout changes are allowed, but
  dropping it solely because the table has many rows is not.

### Production integration

- Add only plots that answer Q1a/Q1b questions and are not duplicates of the
  table or existing Q3/Q4 figures.
- Update the converter's placement/captions and keep manuscript Markdown as
  the author source. Rebuild generated section files from that source.
- Render every final PDF page and inspect tables/figures for clipping,
  overlap, legibility, and honest missingness. Verify final PDF/build PDF
  byte/hash parity and report source/figure hashes.

## Explicit hold points

- `paper/revision_evidence/` is owned by Leibniz and must be read as a worker
  handoff; this worker does not duplicate its searches or overwrite it.
- `paper/revision_q2/` is owned by Nash and must be read as a worker handoff;
  this worker does not duplicate its searches or overwrite it.
- Until both handoffs are present and internally consistent, do not expand Q1a
  or Q1b rows, and do not modify the Q2 `N/A` cell.

## Backup made before edits

The accepted source/PDF snapshot is preserved at
`paper/revision_backup/20260920_initial/` (`manuscript.md`, `main.tex`,
`build_paper.ps1`, `tools/convert_manuscript.py`, and
`ViPragSent_NAACL.pdf`).

## Structural reference inventory

The supplied 14-page `main.pdf` contains the following organization. These are
expected structural rows/figures to reconcile against evidence, not values to
copy into the current paper:

- Reference Table 1: dataset inventory, not a model comparison.
- Reference Table 2 (main pragmatic comparison): rows for PhoBERT
  single-task, PhoBERT fine-tune, XLM-R-large, Sailor-7B SFT, Vistral-7B SFT,
  GPT-4o-mini zero-shot, GPT-4o-mini 8-shot, ViPragSent without auxiliary
  loss, ViPragSent CoT-only, ViPragSent explanation-only, and ViPragSent
  (Vistral); columns are the six pragmatic metrics plus macro-pragmatic F1.
- Reference Table 3 (external retention): rows for PhoBERT single-task,
  PhoBERT fine-tune, XLM-R-large, Sailor-7B SFT, Vistral-7B SFT, GPT-4o-mini
  zero-shot, GPT-4o-mini 8-shot, and ViPragSent (ours); columns are UIT-VSFC,
  UIT-VSMEC, and AIVIVN. The current request additionally requires the
  ordinary-F1 aggregate column.
- Reference Table 4: multi-task ablations, which is structurally analogous to
  current Q2 but not evidence for the current Q2 cohort.
- Reference Table 5: training/evaluation cost, not a Q1 comparison table.
- Reference Figure 1: pipeline architecture; Figure 2: per-phenomenon
  comparison bars; Figure 3: per-phenomenon multi-task gain; Figure 4:
  sarcasm F1 versus labelled-sarcasm budget; Figure 5: row-normalized
  confusion matrix; Figure 6: test-set F1 across training epochs; Figure 7:
  reliability diagrams; Figure 8: qualitative prediction cases.

The current revision will use this inventory to check for missing coverage,
but will not import the reference's historical five-seed values, Vistral-backed
primary model, GPT rows, annotation claims, or qualitative examples without
matching current evidence. The planned Q1a model-level bar and Q1b
horizontal/faceted grouped visualization are deliberately more informative
than reproducing the reference's potentially table-duplicative Figure 2.

## Identity and visualization rules added after parent review

- Preserve the model identity recorded in each artifact. A reference or
  artifact label mentioning GPT-4o-mini must not be silently relabeled as
  GPT-4.1-mini or any other model family.
- Keep the sole primary system label as ViPragSent XLM-R-large. Any legitimate
  Vistral-backed ViPragSent ablation or comparison variant is a distinct
  comparison record: it must retain its Vistral-backed identity, cohort,
  protocol, and evidence status rather than being discarded or described as an
  XLM-R result.
- The final Q1a visualization must expose per-head behavior as well as the
  macro summary. Preferred form is a faceted six-head bar plot with mean +/-
  sample SD and a separate macro panel, or a verified per-head gain plot when
  that communicates the comparison more clearly. A macro-only bar is not
  sufficient if it hides materially different pragmatic-head behavior.
- These rules are planning constraints only until `revision_evidence/` and
  `revision_q2/` are present and reviewed; no numerical integration is
  performed in this interim state.

## Approved-evidence integration gates added after manuscript review

- Use one approved canonical Q1a/Q1b table source as the sole upstream for
  manuscript tables, plotted values, captions, prose claims, and row-level
  sample sizes. Do not hand-maintain parallel numeric copies.
- Preserve artifact identity exactly. Newly reported `GPT-4.1-mini` rows must
  be named GPT-4.1-mini; they must not be crosswalked to the reference's
  GPT-4o-mini rows. The standard XLM-R baseline remains a separate row from
  `ViPragSent (ours, XLM-R-large)`.
- Reconcile the expanded Q1a row inventory before revising the abstract,
  Section 5.1, Section 6.1, conclusion, or any repeated "highest among five
  baselines/all heads" wording. Retain a scoped comparable-subset statement
  when contextual cohorts or GPT rows do not share the same comparability
  contract.
- Q1b captions and prose must report row-specific `n` when GPT-4.1-mini is
  single-run or otherwise differs from three-seed rows; never retain the blanket
  claim that all entries are mean +/- SD over three runs.
- The main Q1b table may include the approved seven baseline recipes plus one
  primary ours row. Do not add a separate PhoBERT-backed ViPragSent contextual
  row to the main table when the approved scope is one primary ours model;
  preserve its evidence in provenance/crosswalk notes if needed.
- Replace any blanket Appendix wording that all Vistral-backed variants are
  excluded once legitimate, evidence-approved comparison variants are added.
  Distinguish inserted approved comparison rows from still-excluded or
  unresolved variants.
- Keep Q2 frozen until the complete 18-run report proves the split-specific
  field semantics. The seed-21 smoke observation is not sufficient to change
  the current `N/A`, and a `test_metrics` field must not be described as saved
  development ECE without per-seed verification.

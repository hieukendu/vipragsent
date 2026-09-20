# ViPragSent manuscript handoff notes

Updated: 2026-09-17

## Author status

`manuscript.md` is the canonical publication-style manuscript. `references.bib`
contains the closed, verified citation set, and `citation_verification.md`
records the source checks. The prose owner retains `manuscript.md`,
`references.bib`, and this file.

Production owns the mechanical conversion and maintenance of all eleven
`sections/*.tex` files. The section files should preserve the Markdown's
scientific meaning, citation keys, table values, uncertainty labels, figure
references, and appendix boundaries; they should not introduce new claims.

## Scientific narrative

The paper presents ViPragSent XLM-R-large as the sole primary model. The
pragmatic heads, polarity head, and emotion head are separate task groups. The
main contribution is the implemented shared encoder and its training-only
teacher-forced rationale objective, evaluated through Q1a--Q4.

The primary Q1a result is 93.67 +/- 0.19 versus 92.93 +/- 0.14 macro-pragmatic
F1 for standard XLM-R, an observed +0.73 percentage-point difference. The
target has the highest saved mean on all six heads among the five complete
baseline families. This is a descriptive saved-prediction comparison on the
same 2,000 IDs and six gold-label tuples, not a significance test, identical-
training claim, or independent rerun.

Q1b is mixed external retention. Q2 is a separate follow-up cohort with
linear scheduling, 0.10 warmup, and patience 2; its full result is not an
ablation of Q1a v37. Q3 is a separate budget cohort with contextual baseline
curves. Q4 extracts calibration from same-seed Q1a v37 checkpoints and does
not support a blanket calibration-superiority claim.

## Final evidence corrections incorporated

- Appendix B normalizes Q2 relative cost to the full XLM-R Q2 run using mean
  GPU-hours, not to PhoBERT.
- Q2 reports the selected artifact field `polarity_dev_ece` as saved
  three-way polarity ECE; the manuscript does not expand it into an
  unsupported top-label recomputation or confuse it with Q4's pragmatic ECE.
- The objective equation includes task multipliers, clamped learned log
  variances, and `+ beta L_rat` with beta 0.3.
- Appendix A preserves the local processed fingerprint and separately records
  remote target `B906C090...C24F9` and baseline `A13573...AF0D` fingerprints.
- Baseline recipes are described per family; the target v37 scheduler,
  patience, and multipliers are not assigned to comparison systems.
- The method now records the target run-level
  `approved_generated_rationales_train.jsonl` artifact with 7,998 rows and
  states that its provider is unspecified in the inspected evidence.
- Sailor-7B and Vistral-7B-Chat identity claims now have primary release/model
  citations in `references.bib` and `citation_verification.md`.
- Ordinary F1 is described as an unweighted aggregate, never as ordinal F1.
- ViSoBERT is contextual prior work and is not presented as an evaluated Q1a
  baseline.
- The ethics section uses only bounded package-source, split, risk, and
  missing-evidence statements.

## Production semantics

Use the all-head Q1a table as the main result. The Q3 budget figure is useful
context; the Q1a macro plot is optional if space is constrained. The canonical
Q3 prose does not assign an explicit figure number, because the architecture
schematic occupies Figure 1 in production. Preserve the
Q2 table's `Saved polarity ECE x 10^3` label and the Appendix B denominator
sentence. Render the full objective with the multiplier inside the
uncertainty-weighted classification sum and the rationale term outside that
sum. The conclusion has been shortened after the preview review; production
should re-sync the sections from the canonical Markdown without adding back
the removed schedule and rerun caveats. Keep the reproducibility appendix
factual and do not import claims from the structure-reference PDF.

## Final two-check review response (2026-09-17)

Q2 split/label check is resolved. In the selected Q2 `*_test_metrics.json`
records mapped by `source_manifest.jsonl`, the F1 source is explicitly
`prediction_file: test_predictions.jsonl`. The saved calibration field is
`polarity_dev_ece` (present for the 15 rows with a polarity head; the three
no-polarity rows are N/A). Thus Table 3 has mixed split semantics: F1 is test;
the saved polarity ECE is development-split. It must not be captioned as if
all columns were from the follow-up test split.

Production exact Table 3 caption replacement:

`\caption{Q2 follow-up ablations. Macro-pragmatic F1 is evaluated on the Q2 test split; the saved \texttt{polarity\_dev\_ece} field is a development-split three-way polarity ECE, shown multiplied by $10^3$; relative cost is normalized to the full Q2 XLM-R run. Entries are mean $\pm$ sample SD over $n=3$ runs.}`

Vistral identity check is resolved for the paper claim. The cached baseline
review summaries and the expected-run manifest identify the records as
`backbone: vistral_7b`, display name `Vistral-7B pragmatic SFT`, and
`model_repository: Viet-Mistral/Vistral-7B-Chat` (the corresponding Sailor
records are `sailor_7b` / `sail/Sailor-7B`). A literal `base_model` key is not
present in the cached Vistral baseline review summary, so the 7B statement
should be attributed to the recorded backbone/display name/repository rather
than to an absent field. `ViT5/Vistral` in the evidence brief is a stale raw
label, not the identity of the evaluated baseline.

Production exact identity correction: use `Vistral-7B SFT` (or
`Vistral-7B pragmatic SFT`) and the existing Vistral citation; remove
`ViT5/Vistral` from manuscript-facing prose. Do not broaden the claim to a
new ViT5 comparison.

Handoff status: the two requested checks are recorded above. No blanket
evidence-clearance statement is authorized until production synchronizes the
caption and baseline wording; no other manuscript rewrite is requested.

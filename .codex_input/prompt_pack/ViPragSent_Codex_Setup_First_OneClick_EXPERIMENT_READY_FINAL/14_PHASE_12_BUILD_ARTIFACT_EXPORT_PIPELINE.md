> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 12 — BUILD THE AUTOMATIC ARTIFACT EXPORT PIPELINE

Prepare an automatic pipeline that produces all experiment outputs after the full run.

Required tables:

- Q1 pragmatic results.
- Q1 external-retention results.
- Q2 ablations.
- Q3 low-resource results.
- Q4 calibration results.
- Statistical significance results.
- Cost and latency results.

Required figures:

- Per-phenomenon F1.
- Multi-task gain.
- Q3 low-resource learning curve.
- Dev-set learning curves.
- Reliability diagrams.

Required machine-readable artifacts:

- per-run metrics JSON;
- predictions and logits;
- thresholds;
- bootstrap distributions;
- Azure usage;
- hardware/runtime/VRAM reports;
- result provenance index;
- final manifest;
- checksums.

Rules: no hard-coded values, every figure has a backing CSV, every table cell is traceable, export is deterministic, and the paper is not modified.

Test the complete artifact pipeline using synthetic result fixtures.

# LOCKED PAPER-FACING ARTIFACTS

Implement the exact schemas in `27_OUTPUT_ARTIFACT_SCHEMA.md`.

Also export:

```text
experiment_artifacts/tables/backbone_sensitivity.csv
experiment_artifacts/manual/error_analysis_candidates.csv
experiment_artifacts/manual/error_analysis_annotation_template.csv
experiment_artifacts/manual/qualitative_candidates.jsonl
experiment_artifacts/manual/qualitative_approval_template.csv
```

Do not generate the old six-class pragmatic-polarity confusion matrix or any artifact named Figure 5.


# DATASET AND ANNOTATION SUMMARY ARTIFACTS

Export dataset-facing artifacts required to rewrite the dataset table and annotation section:

```text
experiment_artifacts/tables/table_1_dataset_summary.csv
experiment_artifacts/tables/vipragsent_label_distribution.csv
experiment_artifacts/tables/human_iaa_summary.csv
experiment_artifacts/backing_data/split_and_label_counts.csv
```

These artifacts must be generated from the frozen V8 files and recomputed IAA reports, not from the old manuscript.

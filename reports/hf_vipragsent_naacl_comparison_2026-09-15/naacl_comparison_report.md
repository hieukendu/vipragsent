# ViPragSent NAACL comparison artefact package

Generated: `2026-09-17T11:34:20.708857Z`  
Verification state: **VERIFIED_ARTIFACTS** (remote HF artefacts, hashes,
completion manifests, and derived tables verified; no independent training or
inference rerun was performed).

## Scope and evidence

This package follows the experiment schema in `main.pdf` (Q1a/Q1b/Q2/Q3/Q4,
baseline comparison, low-resource curve, retention, calibration and training
history). The PDF is used as a schema reference only; its reported numbers are
not copied into these tables.

- Source: authenticated Hugging Face API using the token from `.env`; no GitHub code was used.
- Fresh HF recheck: see [`hf_remote_recheck_summary.json`](hf_remote_recheck_summary.json).
- Live 2026-09-17 inventory retry: repository metadata remained unchanged for all 30 repositories, but tree pagination was rate-limited by HF HTTP 429; it is recorded in [`metadata_reconciliation.json`](metadata_reconciliation.json) and does not replace the complete 2026-09-15 snapshot.
- Account inventory: 30 repositories, 480,738 tree entries, and complete pagination for all 30 repositories.
- Selected structured sources fetched or cached in this package: 195.
- Primary target: **ViPragSent with XLM-R-large**, logical seed labels 21/22/23. The remote optimization manifests retain date-coded seed values 20260521/20260522/20260523; the reconciliation is recorded in [`artifact_verification_records.jsonl`](artifact_verification_records.jsonl) and [`q1a_target_artifact_manifest.json`](q1a_target_artifact_manifest.json).
- Artifact-level completion: all three primary target `optimization_manifest.json` files report `PASS`; their paths and hashes are recorded in [`artifact_verification_records.jsonl`](artifact_verification_records.jsonl).
- Complete target tree inventory: [`q1a_target_artifact_manifest.json`](q1a_target_artifact_manifest.json) records required config, metric, prediction, checkpoint, training, and resource files for all three seeds.
- Ordinary baselines remain in the comparison. ViPragSent variants using Vistral are recorded as discovered but excluded from the primary scope, following the original filtering instruction.
- The PDF describes five seeds; this HF snapshot provides three requested seeds (21/22/23) for the primary Q1a/Q2/Q3/Q4 groups. No five-seed claim is made here.

## Q1a baseline table

Primary ViPragSent XLM-R-large macro-pragmatic F1: **93.7 ± 0.2** (mean ± sample SD over available seeds).

Leaderboard assertion: **PASS** — the target is highest on all six pragmatic heads and macro-pragmatic F1 against the five complete standard baselines (see [`q1a_leaderboard_verification.json`](q1a_leaderboard_verification.json)).

| System | Backbone | n | Implicit | Sarcasm | Irony | Idiom/figurative | Code-switching | Mocking | Macro-prag |
|---|---|---|---|---|---|---|---|---|---|
| PhoBERT (single-task) | PhoBERT-base | 3 | 88.0 ± 1.3 | 70.9 ± 1.3 | 97.1 ± 1.1 | 91.2 ± 0.8 | 91.1 ± 1.5 | 79.2 ± 3.0 | 86.2 ± 0.5 |
| PhoBERT (fine-tune) | PhoBERT-base | 3 | 85.8 ± 0.7 | 72.6 ± 0.5 | 97.5 ± 0.2 | 86.6 ± 0.7 | 91.7 ± 0.5 | 78.5 ± 1.0 | 85.5 ± 0.4 |
| XLM-R-large (baseline) | XLM-R-large | 3 | 92.2 ± 0.1 | 88.1 ± 0.4 | 97.9 ± 0.1 | 92.9 ± 0.7 | 94.3 ± 0.0 | 92.3 ± 0.4 | 92.9 ± 0.1 |
| Sailor-7B SFT | Sailor-7B | 3 | 85.2 ± 0.1 | 72.2 ± 0.5 | 96.4 ± 0.6 | 91.2 ± 0.4 | 90.5 ± 0.9 | 77.8 ± 0.1 | 85.6 ± 0.1 |
| Vistral-7B SFT | Vistral-7B | 3 | 87.7 ± 0.4 | 75.4 ± 1.1 | 97.0 ± 0.0 | 91.5 ± 0.4 | 93.2 ± 0.5 | 81.2 ± 1.2 | 87.7 ± 0.1 |
| ViPragSent (XLM-R-large) | XLM-R-large | 3 | 92.7 ± 0.2 | 89.4 ± 0.9 | 98.4 ± 0.2 | 93.3 ± 0.7 | 94.8 ± 0.6 | 93.3 ± 0.7 | 93.7 ± 0.2 |

The machine-readable version is [`tables/table2_q1a_baselines.csv`](tables/table2_q1a_baselines.csv), and the grouped-bar and gain plots are [`figures/figure2_q1a_baseline_grouped_bars.png`](figures/figure2_q1a_baseline_grouped_bars.png) and [`figures/figure3_gain_over_phobert_single_task.png`](figures/figure3_gain_over_phobert_single_task.png).

### Q1a paper-schema coverage ledger

The following ledger keeps every Q1a row represented in the reference PDF. It
does not turn a missing or excluded HF run into a number. The rank-A primary
comparison above therefore contains only the standard baselines plus the
XLM-R-large ViPragSent target; the ledger documents the remaining rows and
their evidence state.

| System | Backbone | Scope | Status | n | Seeds | Macro-prag |
|---|---|---|---|---|---|---|
| PhoBERT (single-task) | PhoBERT-base | baseline | COMPLETE_SEEDS | 3 | 21,22,23 | 86.2 ± 0.5 |
| PhoBERT (fine-tune) | PhoBERT-base | baseline | COMPLETE_SEEDS | 3 | 21,22,23 | 85.5 ± 0.4 |
| XLM-R-large (baseline) | XLM-R-large | baseline | COMPLETE_SEEDS | 3 | 21,22,23 | 92.9 ± 0.1 |
| Sailor-7B SFT | Sailor-7B | baseline | COMPLETE_SEEDS | 3 | 21,22,23 | 85.6 ± 0.1 |
| Vistral-7B SFT | Vistral-7B | baseline | COMPLETE_SEEDS | 3 | 21,22,23 | 87.7 ± 0.1 |
| GPT-4.1-mini zero-shot | GPT-4.1-mini | missing_or_incomplete | MISSING | 0 | — | N/A |
| GPT-4.1-mini 8-shot | GPT-4.1-mini | missing_or_incomplete | MISSING | 0 | — | N/A |
| ViPragSent-no-aux (Vistral) | Vistral-7B | excluded_non_xlmr_vipragsent | COMPLETE_SEEDS | 3 | 21,22,23 | 85.5 ± 2.6 |
| ViPragSent-CoT-only (Vistral) | Vistral-7B | excluded_non_xlmr_vipragsent | PARTIAL | 2 | 21,23 | 42.5 ± 0.0 |
| ViPragSent-explanation-only (Vistral) | Vistral-7B | excluded_non_xlmr_vipragsent | COMPLETE_SEEDS | 3 | 21,22,23 | 46.9 ± 0.0 |
| ViPragSent full (Vistral) | Vistral-7B | excluded_non_xlmr_vipragsent | COMPLETE_SEEDS | 3 | 21,22,23 | 81.2 ± 12.2 |
| ViPragSent (XLM-R-large) | XLM-R-large | primary | COMPLETE_SEEDS | 3 | 21,22,23 | 93.7 ± 0.2 |

The full ledger is [`tables/table2_q1a_paper_schema_coverage.csv`](tables/table2_q1a_paper_schema_coverage.csv). GPT-4.1-mini zero-shot and 8-shot have only `NOT_STARTED` receipts in the current HF tree; the exact receipt records are preserved in [`baseline_status_records.jsonl`](baseline_status_records.jsonl). The CoT-only and explanation-only rows have direct reasoning metrics, but are excluded from the primary table because they are Vistral-backbone ViPragSent variants.

## Q1b ordinary-sentiment retention

| System | n | Seeds | UIT-VSFC | UIT-VSMEC | AIVIVN-2019 | Ord.F1 |
|---|---|---|---|---|---|---|
| Sailor-7B SFT | 3 | 21,22,23 | 34.4 ± 3.2 | 40.7 ± 0.7 | 48.7 ± 1.4 | 41.3 ± 1.2 |
| ViPragSent (XLM-R-large) | 3 | 21,22,23 | 37.9 ± 2.8 | 40.0 ± 0.7 | 47.9 ± 2.2 | 42.0 ± 1.8 |
| Vistral-7B SFT | 3 | 21,22,23 | 30.7 ± 6.4 | 39.4 ± 1.5 | 49.0 ± 0.8 | 39.7 ± 2.5 |

This table is intentionally limited to systems for which the HF snapshot
contains `external_retention_metrics.json`. Missing systems are listed in
[`coverage_matrix.csv`](coverage_matrix.csv), not replaced with paper values.

## Q2 XLM-R-large ablations

| Variant | n | Seeds | Prag.F1 | ECE ×10³ ↓ | Ord.F1 | Cost |
|---|---|---|---|---|---|---|
| ViPragSent XLM-R: Full | 3 | 21,22,23 | 92.8 ± 0.6 | 78.3 ± 5.9 | N/A | 1.00 |
| ViPragSent XLM-R: − emotion auxiliary | 3 | 21,22,23 | 91.7 ± 0.7 | 61.8 ± 27.0 | N/A | 0.65 |
| ViPragSent XLM-R: − explanation/CoT auxiliary | 3 | 21,22,23 | 92.3 ± 1.3 | 68.4 ± 21.2 | N/A | 0.20 |
| ViPragSent XLM-R: − multi-task bundle | 3 | 21,22,23 | 74.7 ± 2.3 | 115.3 ± 70.1 | N/A | 1.10 |
| ViPragSent XLM-R: − polarity auxiliary | 3 | 21,22,23 | 92.8 ± 0.2 | N/A | N/A | 0.81 |
| ViPragSent XLM-R: − task-uncertainty weighting | 3 | 21,22,23 | 92.9 ± 0.3 | 81.6 ± 10.8 | N/A | 0.82 |

Ordinary-sentiment F1 is **N/A** because the selected
XLM-R Q2 test metrics do not expose ordinary-sentiment F1. Normalized cost is
computed from the HF `training/resource_usage.json` GPU-hour means relative
to the full XLM-R run; it is not copied from the PDF. ECE is shown as ×10³ to
mirror the paper's table convention. Full data are in
[`tables/table4_q2_xlmr_ablation.csv`](tables/table4_q2_xlmr_ablation.csv).

## Q3 low-resource curve

| System | Budget | n | Seeds | Sarcasm F1 | Macro-prag F1 |
|---|---|---|---|---|---|
| PhoBERT (fine-tune) | 32 | 3 | 21,22,23 | 69.7 ± 1.5 | 84.2 ± 0.7 |
| PhoBERT (fine-tune) | 64 | 3 | 21,22,23 | 70.8 ± 2.8 | 77.7 ± 11.8 |
| PhoBERT (fine-tune) | 128 | 3 | 21,22,23 | 71.8 ± 2.2 | 84.0 ± 1.2 |
| PhoBERT (fine-tune) | 256 | 3 | 21,22,23 | 73.4 ± 1.2 | 84.5 ± 0.9 |
| Vistral-7B SFT | 32 | 3 | 21,22,23 | 70.5 ± 2.4 | 86.2 ± 0.2 |
| Vistral-7B SFT | 64 | 3 | 21,22,23 | 72.5 ± 0.9 | 86.7 ± 0.5 |
| Vistral-7B SFT | 128 | 3 | 21,22,23 | 71.3 ± 0.6 | 86.2 ± 0.4 |
| Vistral-7B SFT | 256 | 3 | 21,22,23 | 64.0 ± 5.1 | 70.3 ± 14.2 |
| Vistral-7B SFT | 512 | 3 | 21,22,23 | 73.9 ± 1.2 | 87.2 ± 0.3 |
| Vistral-7B SFT | full | 3 | 21,22,23 | 75.3 ± 0.8 | 87.4 ± 0.2 |
| ViPragSent (XLM-R-large) | 32 | 3 | 21,22,23 | 84.2 ± 1.6 | 91.1 ± 1.3 |
| ViPragSent (XLM-R-large) | 64 | 3 | 21,22,23 | 83.7 ± 1.3 | 91.2 ± 0.8 |
| ViPragSent (XLM-R-large) | 128 | 3 | 21,22,23 | 85.2 ± 0.7 | 91.6 ± 0.2 |
| ViPragSent (XLM-R-large) | 256 | 3 | 21,22,23 | 85.3 ± 2.6 | 91.6 ± 1.2 |
| ViPragSent (XLM-R-large) | 512 | 3 | 21,22,23 | 87.2 ± 1.2 | 92.3 ± 0.7 |
| ViPragSent (XLM-R-large) | full | 3 | 21,22,23 | 85.0 ± 2.3 | 91.1 ± 1.3 |

The line graph is [`figures/figure4_q3_sarcasm_budget_lines.png`](figures/figure4_q3_sarcasm_budget_lines.png). Missing seed/budget combinations remain visible as partial coverage rather than being imputed.

## Q4 calibration and training history

| System | n | Seeds | Macro pragmatic ECE ↓ |
|---|---|---|---|
| ViPragSent (XLM-R-large) | 3 | 21,22,23 | 0.0386 ± 0.0069 |
| Vistral-7B SFT | 3 | 21,22,23 | 0.0345 ± 0.0041 |

The reliability diagram is [`figures/figure7_q4_reliability_diagrams.png`](figures/figure7_q4_reliability_diagrams.png), and the available dev training histories are [`figures/figure6_q4_training_curves.png`](figures/figure6_q4_training_curves.png). A pooled six-head confusion-matrix figure based on the three primary Q1a prediction JSONL files is [`figures/figure5_pragmatic_confusion_matrices.png`](figures/figure5_pragmatic_confusion_matrices.png).

## Q1a cost and resource provenance

| System | n | Seeds | GPU-h mean | GPU-h SD | Azure $ mean | Azure cost status |
|---|---|---|---|---|---|---|
| PhoBERT (single-task) | 3 | 21,22,23 | 0.711 | 0.069 | N/A | NOT_RECORDED |
| PhoBERT (fine-tune) | 3 | 21,22,23 | 0.044 | 0.000 | N/A | NOT_RECORDED |
| XLM-R-large (baseline) | 3 | 21,22,23 | 0.229 | 0.001 | N/A | NOT_RECORDED |
| Sailor-7B SFT | 3 | 21,22,23 | 1.068 | 0.027 | N/A | NOT_RECORDED |
| Vistral-7B SFT | 3 | 21,22,23 | 1.136 | 0.006 | N/A | NOT_RECORDED |
| ViPragSent (XLM-R-large) | 3 | 21,22,23 | 1.594 | 0.012 | N/A | NOT_RECORDED |

This primary-scope table reports only GPU-hour/resource values actually present
in the HF run artefacts for the standard baselines and XLM-R-large target.
Azure/API cost is shown as a status when HF records
`NOT_APPLICABLE` or does not record a numeric value; the PDF's annotation and
API cost values are not copied into this package. Machine-readable files are
[`tables/table5_cost_inventory.csv`](tables/table5_cost_inventory.csv),
[`cost_inventory_q1a.csv`](cost_inventory_q1a.csv), and
[`resource_usage_records.csv`](resource_usage_records.csv).

## Coverage, exclusions and reproducibility boundary

- Complete-seed coverage rows in the matrix: 36/41.
- `coverage_matrix.csv` records incomplete GPT/CoT/explanation runs, excluded non-XLM-R ViPragSent variants, partial Q3 baselines, and unavailable fields.
- `candidate_baseline_inventory.csv` records the wider HF candidate tree scan.
- [`xlmr_weight_inventory.csv`](xlmr_weight_inventory.csv) is the weight metadata inventory from the audited XLM-R scope. Weight payloads were not downloaded or deserialized; this package is an analysis artefact, not a checkpoint-resume operation.
- [`selected_run_metrics.csv`](selected_run_metrics.csv), [`calibration_records.jsonl`](calibration_records.jsonl), [`external_retention_records.csv`](external_retention_records.csv), and [`source_manifest.jsonl`](source_manifest.jsonl) preserve the per-seed values and SHA-256 provenance used here.
- [`baseline_status_records.jsonl`](baseline_status_records.jsonl) preserves the GPT baseline receipt status rather than treating an unfinished run as a metric.
- [`resource_usage_records.csv`](resource_usage_records.csv) preserves the Q1a/Q2 resource records used for Table 4 cost normalization and Table 5.
- [`review_checks.json`](review_checks.json) and [`artifact_hashes.sha256`](artifact_hashes.sha256) are the final local integrity checks.
- [`metadata_reconciliation.json`](metadata_reconciliation.json) records stale local pre-experiment gates, HF status-attention rows, and the exact resolution used for this final package.

This is an artifact-level verification package, not an independent
reproducibility rerun. The three-seed scope, GPT missingness, and all excluded
backbone variants remain explicit; no missing score is imputed and no claim of
an independent training/inference rerun is made.

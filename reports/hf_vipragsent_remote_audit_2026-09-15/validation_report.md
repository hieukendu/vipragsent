## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-15T12:37:34.509209Z
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- **Source**: authenticated Hugging Face account inventory and fetched artefacts
- **Overall Confidence**: CAUTION

### Scope and Filter

- HF collection coverage: 30/30 repositories complete; 480738 tree entries inventoried; 1941/1941 selected text artefacts fetched; errors: 0.
- Parsed structured records: 2693
- XLM-R-large records: 2690
- Selected records Q1a/Q2/Q3/Q4: 2476
- Selected question counts: {'Q1a': 225, 'Q2': 1265, 'Q3': 710, 'Q4': 276}
- Canonical run-level records: 42 (duplicates removed: 0)
- Canonical backbone confirmation: 42 companion-confirmed; 0 unconfirmed path-only.
- Independent review checks: PASS.
- Seed 21/22/23 complete across selected records: True
- Weight inventory: 1368 files / 4322857019471 bytes across all repos; XLM-R scope 172 files / 1194719792814 bytes.
- Backbone policy: exclude explicit PhoBERT, Sailor, Vistral, and other non-XLM-R-large records.

### Interpretation Boundary

The attached PDF is treated as a reference schema for Q1-Q4 and metric names. Its prose and numbers are not used as remote evidence. This report does not execute GitHub code, rerun training, or declare publication readiness.

### Statistical Findings

| Question | Experiment | Metric | n | Seeds | Mean | Std | Min | Max | 21/22/23 complete |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| Q1a | q1a_xlmr_pragmatic_finetune | pragmatic_f1 | 3 | 21,22,23 | 0.929343 | 0.0014479 | 0.928502 | 0.931014 | True |
| Q1a | q1a_xlmr_pragmatic_finetune | sarcasm_f1 | 3 | 21,22,23 | 0.88123 | 0.00429794 | 0.878713 | 0.886193 | True |
| Q1a | q1a_xlmr_pragmatic_finetune | implicit_f1 | 3 | 21,22,23 | 0.921637 | 0.00130814 | 0.920147 | 0.922596 | True |
| Q2 | xlmr_followup_q2_full | pragmatic_f1 | 3 | 23,22,21 | 0.927589 | 0.00565987 | 0.921064 | 0.931171 | True |
| Q2 | xlmr_followup_q2_full | sarcasm_f1 | 3 | 23,22,21 | 0.87409 | 0.00485115 | 0.870288 | 0.879554 | True |
| Q2 | xlmr_followup_q2_full | implicit_f1 | 3 | 23,22,21 | 0.915023 | 0.0131114 | 0.899976 | 0.923999 | True |
| Q2 | xlmr_followup_q2_full | ece | 3 | 23,22,21 | 0.078263 | 0.00590189 | 0.0728097 | 0.0845292 | True |
| Q2 | xlmr_followup_q2_no_emotion_auxiliary | pragmatic_f1 | 3 | 23,22,21 | 0.91734 | 0.00660666 | 0.910851 | 0.924058 | True |
| Q2 | xlmr_followup_q2_no_emotion_auxiliary | sarcasm_f1 | 3 | 23,22,21 | 0.869664 | 0.0178986 | 0.85226 | 0.88802 | True |
| Q2 | xlmr_followup_q2_no_emotion_auxiliary | implicit_f1 | 3 | 23,22,21 | 0.903805 | 0.0163564 | 0.889476 | 0.921625 | True |
| Q2 | xlmr_followup_q2_no_emotion_auxiliary | ece | 3 | 23,22,21 | 0.0617847 | 0.0270392 | 0.0305673 | 0.0778698 | True |
| Q2 | xlmr_followup_q2_no_multitask | pragmatic_f1 | 3 | 23,21,22 | 0.747318 | 0.0233926 | 0.720523 | 0.763671 | True |
| Q2 | xlmr_followup_q2_no_multitask | sarcasm_f1 | 3 | 23,21,22 | 0.743817 | 0.0804151 | 0.652259 | 0.802989 | True |
| Q2 | xlmr_followup_q2_no_multitask | implicit_f1 | 3 | 23,21,22 | 0.735964 | 0.255222 | 0.441341 | 0.88927 | True |
| Q2 | xlmr_followup_q2_no_multitask | ece | 3 | 23,21,22 | 0.115298 | 0.0701389 | 0.0516315 | 0.190483 | True |
| Q2 | xlmr_followup_q2_no_polarity_auxiliary | pragmatic_f1 | 3 | 23,22,21 | 0.92806 | 0.00188936 | 0.926681 | 0.930213 | True |
| Q2 | xlmr_followup_q2_no_polarity_auxiliary | sarcasm_f1 | 3 | 23,22,21 | 0.877468 | 0.0101361 | 0.86642 | 0.886337 | True |
| Q2 | xlmr_followup_q2_no_polarity_auxiliary | implicit_f1 | 3 | 23,22,21 | 0.91764 | 0.00378402 | 0.914116 | 0.92164 | True |
| Q2 | xlmr_followup_q2_no_rationale | pragmatic_f1 | 3 | 22,21,23 | 0.923316 | 0.0128743 | 0.908509 | 0.931867 | True |
| Q2 | xlmr_followup_q2_no_rationale | sarcasm_f1 | 3 | 22,21,23 | 0.857212 | 0.0201399 | 0.833957 | 0.86887 | True |
| Q2 | xlmr_followup_q2_no_rationale | implicit_f1 | 3 | 22,21,23 | 0.909602 | 0.0174522 | 0.89056 | 0.924835 | True |
| Q2 | xlmr_followup_q2_no_rationale | ece | 3 | 22,21,23 | 0.068387 | 0.0212117 | 0.0442436 | 0.0840308 | True |
| Q2 | xlmr_followup_q2_no_uncertainty_weighting | pragmatic_f1 | 3 | 23,22,21 | 0.928969 | 0.0029312 | 0.925585 | 0.930747 | True |
| Q2 | xlmr_followup_q2_no_uncertainty_weighting | sarcasm_f1 | 3 | 23,22,21 | 0.875972 | 0.00585134 | 0.869263 | 0.880021 | True |
| Q2 | xlmr_followup_q2_no_uncertainty_weighting | implicit_f1 | 3 | 23,22,21 | 0.923955 | 0.0037293 | 0.92061 | 0.927976 | True |
| Q2 | xlmr_followup_q2_no_uncertainty_weighting | ece | 3 | 23,22,21 | 0.0816403 | 0.0107602 | 0.0694598 | 0.0898536 | True |
| Q3 | xlmr_followup_q3_full_128 | pragmatic_f1 | 3 | 22,21,23 | 0.915887 | 0.00168201 | 0.914055 | 0.917361 | True |
| Q3 | xlmr_followup_q3_full_128 | sarcasm_f1 | 3 | 22,21,23 | 0.851652 | 0.0073287 | 0.8464 | 0.860025 | True |
| Q3 | xlmr_followup_q3_full_128 | implicit_f1 | 3 | 22,21,23 | 0.906129 | 0.00101973 | 0.905065 | 0.907098 | True |
| Q3 | xlmr_followup_q3_full_256 | pragmatic_f1 | 3 | 23,22,21 | 0.916462 | 0.0121609 | 0.909279 | 0.930503 | True |
| Q3 | xlmr_followup_q3_full_256 | sarcasm_f1 | 3 | 23,22,21 | 0.852867 | 0.0256484 | 0.828244 | 0.87943 | True |
| Q3 | xlmr_followup_q3_full_256 | implicit_f1 | 3 | 23,22,21 | 0.909722 | 0.0108093 | 0.901878 | 0.922052 | True |
| Q3 | xlmr_followup_q3_full_32 | pragmatic_f1 | 3 | 23,21,22 | 0.911238 | 0.0130065 | 0.901087 | 0.925899 | True |
| Q3 | xlmr_followup_q3_full_32 | sarcasm_f1 | 3 | 23,21,22 | 0.841611 | 0.015978 | 0.829667 | 0.859761 | True |
| Q3 | xlmr_followup_q3_full_32 | implicit_f1 | 3 | 23,21,22 | 0.906134 | 0.0107689 | 0.896251 | 0.917611 | True |
| Q3 | xlmr_followup_q3_full_512 | pragmatic_f1 | 3 | 23,21,22 | 0.923438 | 0.00736766 | 0.915376 | 0.929822 | True |
| Q3 | xlmr_followup_q3_full_512 | sarcasm_f1 | 3 | 23,21,22 | 0.871946 | 0.0121763 | 0.860574 | 0.884793 | True |
| Q3 | xlmr_followup_q3_full_512 | implicit_f1 | 3 | 23,21,22 | 0.908125 | 0.00650018 | 0.900888 | 0.913469 | True |
| Q3 | xlmr_followup_q3_full_64 | pragmatic_f1 | 3 | 21,22,23 | 0.912099 | 0.00813764 | 0.905818 | 0.921292 | True |
| Q3 | xlmr_followup_q3_full_64 | sarcasm_f1 | 3 | 21,22,23 | 0.83693 | 0.0125951 | 0.82596 | 0.850684 | True |
| Q3 | xlmr_followup_q3_full_64 | implicit_f1 | 3 | 21,22,23 | 0.89997 | 0.00656738 | 0.89282 | 0.905734 | True |
| Q3 | xlmr_followup_q3_full_full | pragmatic_f1 | 3 | 21,22,23 | 0.911187 | 0.0132908 | 0.902216 | 0.926456 | True |
| Q3 | xlmr_followup_q3_full_full | sarcasm_f1 | 3 | 21,22,23 | 0.850398 | 0.0226367 | 0.836716 | 0.876527 | True |
| Q3 | xlmr_followup_q3_full_full | implicit_f1 | 3 | 21,22,23 | 0.895831 | 0.0220668 | 0.881207 | 0.921214 | True |
| Q4 | xlmr_followup_q4_full | ece | 3 | 23,21,22 | 0.0385906 | 0.00693768 | 0.0343699 | 0.0465976 | True |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Reproducibility | No training rerun was performed; status is ANALYZED, not VERIFIED. | All results |
| Backbone evidence | Canonical metric files are often path-inferred, but each is cross-checked against an explicit XLM-R-large companion manifest/review artifact when available. | XLM-R selection |
| Weight handling | Large weight payloads are inventoried using HF tree metadata; scalar aggregation uses downloaded structured artefacts. | Checkpoint files |

### Fallacy Scan

- **Coverage**: 11/11 checked at the audit level.

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's paradox | NOTE | No grouped causal comparison was inferred from the remote artefacts. |
| Ecological fallacy | NOTE | Unit of analysis is experiment/run; no individual-level causal inference made. |
| Berkson/collider bias | CAUTION | Selection is intentionally restricted to the user's XLM-R-large filter; do not generalize to all backbones. |
| Base-rate neglect | CAUTION | Rare-class metrics must be reported with support/prevalence when available. |
| Regression to mean | NOTE | No pre/post intervention claim made. |
| Survivorship bias | CAUTION | Missing or failed runs are not treated as successful; coverage files must be checked. |
| Look-elsewhere effect | CAUTION | Multiple Q1a/Q2/Q3/Q4 metrics may exist; no significance claim is made without an explicit correction plan. |
| Garden of forking paths | CAUTION | Remote manifests may encode post-hoc selection; provenance is retained for review. |
| Correlation vs causation | NOTE | This audit reports associations/differences only. |
| Reverse causality | NOTE | No causal direction is asserted. |
| Other structural checks | NOTE | No evidence sufficient to elevate to RED_FLAG in this metadata-only pass. |

### Reproducibility

- **Method**: not run; remote artefact audit only.
- **Verdict**: CANNOT_VERIFY.

### Output Files

- `discovery.json`, `inventory_status.json`, `repo_summaries.jsonl`, `tree_manifest.jsonl`: account/repo/tree provenance.
- `content_manifest.jsonl`, `parsed_record_summaries.jsonl`: fetched-content hashes and parsed records.
- `weight_inventory.csv`, `xlmr_weight_inventory.csv`, `weight_inventory_summary.json`: complete weight tree/LFS metadata; payloads were not downloaded or deserialized.
- `xlmr_large_records.jsonl`, `selected_q1a_q2_q3_q4_records.jsonl`, `canonical_q1a_q2_q3_q4_records.jsonl`, `selected_metrics_long.csv`, `selected_seed_metrics.csv`, `run_coverage.csv`, `q1a_q2_q3_q4_aggregates.csv`: paper-preparation tables.
- `analysis_status.json`, `validation_report.md`, `paper_summary.md`, `review_checks.json`, and `artifact_hashes.sha256`: summaries, deterministic review checks, and top-level provenance receipt.

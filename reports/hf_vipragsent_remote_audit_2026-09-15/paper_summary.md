# ViPragSent XLM-R-large: paper-preparation summary

This summary is computed from authenticated Hugging Face artefacts only. The attached PDF supplied the Q1a/Q2/Q3/Q4 schema; its reported numbers were not copied into these results.

## Audit status

- Verification status: **ANALYZED** (remote artefact analysis; no training rerun).
- Coverage: 30/30 repositories complete; 1941/1941 selected text artefacts fetched; fetch errors 0.
- Canonical runs: 42; explicit companion backbone confirmation: 42/42.
- Deterministic review checks: **PASS**.
- Requested seeds: 21, 22, 23; every one of the 14 experiment groups has all three seeds.

## Mean +/- sample SD over seeds 21/22/23

| Question | Experiment group | Pragmatic F1 | Sarcasm F1 | Implicit F1 | ECE |
|---|---|---:|---:|---:|---:|
| Q1a | q1a_xlmr_pragmatic_finetune | 0.929343 +/- 0.001448 | 0.881230 +/- 0.004298 | 0.921637 +/- 0.001308 | N/A |
| Q2 | xlmr_followup_q2_full | 0.927589 +/- 0.005660 | 0.874090 +/- 0.004851 | 0.915023 +/- 0.013111 | 0.078263 +/- 0.005902 |
| Q2 | xlmr_followup_q2_no_emotion_auxiliary | 0.917340 +/- 0.006607 | 0.869664 +/- 0.017899 | 0.903805 +/- 0.016356 | 0.061785 +/- 0.027039 |
| Q2 | xlmr_followup_q2_no_multitask | 0.747318 +/- 0.023393 | 0.743817 +/- 0.080415 | 0.735964 +/- 0.255222 | 0.115298 +/- 0.070139 |
| Q2 | xlmr_followup_q2_no_polarity_auxiliary | 0.928060 +/- 0.001889 | 0.877468 +/- 0.010136 | 0.917640 +/- 0.003784 | N/A |
| Q2 | xlmr_followup_q2_no_rationale | 0.923316 +/- 0.012874 | 0.857212 +/- 0.020140 | 0.909602 +/- 0.017452 | 0.068387 +/- 0.021212 |
| Q2 | xlmr_followup_q2_no_uncertainty_weighting | 0.928969 +/- 0.002931 | 0.875972 +/- 0.005851 | 0.923955 +/- 0.003729 | 0.081640 +/- 0.010760 |
| Q3 | xlmr_followup_q3_full_128 | 0.915887 +/- 0.001682 | 0.851652 +/- 0.007329 | 0.906129 +/- 0.001020 | N/A |
| Q3 | xlmr_followup_q3_full_256 | 0.916462 +/- 0.012161 | 0.852867 +/- 0.025648 | 0.909722 +/- 0.010809 | N/A |
| Q3 | xlmr_followup_q3_full_32 | 0.911238 +/- 0.013007 | 0.841611 +/- 0.015978 | 0.906134 +/- 0.010769 | N/A |
| Q3 | xlmr_followup_q3_full_512 | 0.923438 +/- 0.007368 | 0.871946 +/- 0.012176 | 0.908125 +/- 0.006500 | N/A |
| Q3 | xlmr_followup_q3_full_64 | 0.912099 +/- 0.008138 | 0.836930 +/- 0.012595 | 0.899970 +/- 0.006567 | N/A |
| Q3 | xlmr_followup_q3_full_full | 0.911187 +/- 0.013291 | 0.850398 +/- 0.022637 | 0.895831 +/- 0.022067 | N/A |
| Q4 | xlmr_followup_q4_full | N/A | N/A | N/A | 0.038591 +/- 0.006938 |

## Metric interpretation

- `pragmatic_f1`: test macro-pragmatic F1 when the canonical run metric file exposes it.
- `sarcasm_f1` and `implicit_f1`: test per-label F1 values selected from explicit F1 paths; support/count fields are excluded.
- `ece`: macro pragmatic expected calibration error; lower is better, and it is only populated where the artefact reports it.
- Standard deviation is the sample SD across the three requested seeds; no significance or causal claim is made.

## Provenance files

- `selected_seed_metrics.csv`: one canonical run row per experiment and seed.
- `q1a_q2_q3_q4_aggregates.csv`: machine-readable long-form mean/std/min/max table.
- `run_coverage.csv`: seed completeness by experiment group.
- `selected_q1a_q2_q3_q4_records.jsonl` and `content_manifest.jsonl`: selected structured artefacts and SHA-256 provenance.
- `xlmr_weight_inventory.csv`: XLM-R-scope weight paths, sizes, LFS/Xet IDs, and commit metadata; payloads were not downloaded or deserialized.
- `artifact_hashes.sha256`: SHA-256 hashes for top-level audit outputs.

The results remain **ANALYZED**, not VERIFIED, until an independent rerun or equivalent reproducibility check is performed.

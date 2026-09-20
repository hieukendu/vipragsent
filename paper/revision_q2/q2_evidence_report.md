## Material Passport

- ID: q2-no-polarity-evidence-2026-09-20
- Type: validation report / reproducibility audit
- Verification Status: ANALYZED (live artifact fetch and deterministic recomputation; no fresh training rerun)
- Scope: Q2 XLM-R follow-up, six variants, three seeds per variant

## Finding

The no-polarity diagnosis is asserted from the completed evidence rows, not hardcoded: proof={"all_15_head_present_dev_split_match": true, "all_15_head_present_test_split_match": true, "all_18_source_ok": true, "all_3_no_polarity_head_removed_confirmed": true, "ece_tolerance": 1e-12}. This is structural inapplicability, not an unresolved missing-data case: all three source-complete rows expose no polarity head/task and have zero polarity fields in both saved splits, so no polarity probability source exists to recompute. The canonical cell is therefore `not applicable (polarity head removed)`; no pragmatic ECE, zero, or surrogate value is inserted.

The canonical table follows the verified Q2 protocol: F1 is from `metrics/test_metrics.json`; ECE is the ten-bin top-label ECE on `metrics/dev_metrics.json` / dev predictions. Relative cost uses successful GPU hours normalized to the mean full Q2 run.

| Variant | Macro-pragmatic F1, test (%) | Polarity dev ECE x 10^3 | Relative cost to full Q2 XLM-R | n |
|---|---:|---:|---:|---:|
| Full follow-up | 92.8 +/- 0.6 | 81.1 +/- 11.7 | 1.00 | 3 |
| No emotion auxiliary | 91.7 +/- 0.7 | 56.2 +/- 22.5 | 0.65 | 3 |
| No polarity auxiliary | 92.8 +/- 0.2 | not applicable (polarity head removed) | 0.81 | 3 |
| No explanation auxiliary | 92.3 +/- 1.3 | 68.1 +/- 22.2 | 0.20 | 3 |
| No multitask bundle | 74.7 +/- 2.3 | 118.9 +/- 69.1 | 1.10 | 3 |
| No task-uncertainty weighting | 92.9 +/- 0.3 | 90.2 +/- 5.9 | 0.82 | 3 |

## Per-seed split-match evidence

The tolerance is absolute ECE <= 1.0e-12.

| Run | Saved dev field | Recomputed dev | Abs delta | Dev verdict | Saved test field | Recomputed test | Abs delta | Test verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|
| xlmr_followup_q2_full_20260521 | 0.08015336569635316 | 0.08015336569635316 | 0.0 | True | 0.08452923804521562 | 0.08452923804521562 | 0.0 | True |
| xlmr_followup_q2_full_20260522 | 0.0698599011943363 | 0.0698599011943363 | 0.0 | True | 0.07745010578632355 | 0.07745010578632355 | 0.0 | True |
| xlmr_followup_q2_full_20260523 | 0.09318917259208676 | 0.09318917259208676 | 0.0 | True | 0.07280974346399306 | 0.07280974346399306 | 0.0 | True |
| xlmr_followup_q2_no_emotion_auxiliary_20260521 | 0.0332176581718374 | 0.0332176581718374 | 0.0 | True | 0.030567316412925732 | 0.030567316412925732 | 0.0 | True |
| xlmr_followup_q2_no_emotion_auxiliary_20260522 | 0.057224404474924426 | 0.057224404474924426 | 0.0 | True | 0.07691690023243428 | 0.07691690023243428 | 0.0 | True |
| xlmr_followup_q2_no_emotion_auxiliary_20260523 | 0.07820133494400511 | 0.07820133494400511 | 0.0 | True | 0.07786980956792833 | 0.07786980956792833 | 0.0 | True |
| xlmr_followup_q2_no_polarity_auxiliary_20260521 | None | None | None | not_applicable_head_removed | None | None | None | not_applicable_head_removed |
| xlmr_followup_q2_no_polarity_auxiliary_20260522 | None | None | None | not_applicable_head_removed | None | None | None | not_applicable_head_removed |
| xlmr_followup_q2_no_polarity_auxiliary_20260523 | None | None | None | not_applicable_head_removed | None | None | None | not_applicable_head_removed |
| xlmr_followup_q2_no_rationale_20260521 | 0.04260601381351499 | 0.04260601381351499 | 0.0 | True | 0.04424364565312865 | 0.04424364565312865 | 0.0 | True |
| xlmr_followup_q2_no_rationale_20260522 | 0.08318062252614303 | 0.08318062252614303 | 0.0 | True | 0.08403081753849978 | 0.08403081753849978 | 0.0 | True |
| xlmr_followup_q2_no_rationale_20260523 | 0.07842915698610106 | 0.07842915698610106 | 0.0 | True | 0.07688652858138087 | 0.07688652858138087 | 0.0 | True |
| xlmr_followup_q2_no_multitask_20260521 | 0.05308565621139884 | 0.05308565621139884 | 0.0 | True | 0.051631503850221624 | 0.051631503850221624 | 0.0 | True |
| xlmr_followup_q2_no_multitask_20260522 | 0.19080188634873863 | 0.19080188634873863 | 0.0 | True | 0.1904833410680294 | 0.1904833410680294 | 0.0 | True |
| xlmr_followup_q2_no_multitask_20260523 | 0.112740290633078 | 0.112740290633078 | 0.0 | True | 0.10377938014268878 | 0.10377938014268878 | 0.0 | True |
| xlmr_followup_q2_no_uncertainty_weighting_20260521 | 0.09303992770504149 | 0.09303992770504149 | 0.0 | True | 0.08985356684029104 | 0.08985356684029104 | 0.0 | True |
| xlmr_followup_q2_no_uncertainty_weighting_20260522 | 0.08341203842418321 | 0.08341203842418321 | 0.0 | True | 0.06945975990593437 | 0.06945975990593437 | 0.0 | True |
| xlmr_followup_q2_no_uncertainty_weighting_20260523 | 0.09415716775182847 | 0.09415716775182847 | 0.0 | True | 0.0856076027303934 | 0.0856076027303934 | 0.0 | True |

## Split diagnostic for production

The Q2 protocol file `configs\experiments\q2\protocol.yaml` lines 8-11 explicitly declares `ece.split: vipragsent_dev`, head `intended_polarity_3way`, and ten-bin top-label ECE. Runtime proof={"all_15_head_present_dev_split_match": true, "all_15_head_present_test_split_match": true, "all_18_source_ok": true, "all_3_no_polarity_head_removed_confirmed": true, "ece_tolerance": 1e-12}. Although both split files use the legacy key `polarity_dev_ece`, the per-seed recomputation shows the dev-file value matches dev predictions and the test-file value matches test predictions; the key name does not change the split source. The old paper table used the test-file values; those legacy aggregate values are recorded here: `{"full": "78.3 +/- 5.9", "no_emotion_auxiliary": "61.8 +/- 27.0", "no_multitask": "115.3 +/- 70.1", "no_polarity_auxiliary": "not applicable (polarity head removed)", "no_rationale": "68.4 +/- 21.2", "no_uncertainty_weighting": "81.6 +/- 10.8"}`. Because the production caption says dev ECE and the protocol says dev, the canonical table above uses the dev aggregates instead. This is a split-label/value mismatch, not an imputation.

## Provenance

- Live HF tree API inventories are pinned to each repository commit SHA and paginated through the API `Link: rel=next` cursor; raw files use the same pinned revision.
- Machine-readable per-seed URLs, statuses, sizes, SHA-256 values, head evidence, probabilities, and split deltas: `source_inventory.json` and `run_evidence.jsonl`.
- Persisted raw payload mirror for audit/replay: `raw_sources/`; the earlier 12-row pre-normalization and four-row pre-component-bundle checkpoints remain in the `*_before_*` backup files.
- The source gate requires the Q2 evidence core (run manifest, review summary, dev/test metrics, and dev/test predictions); optional config/selection omissions are retained per run in `optional_missing_raw_files` and are not misreported as missing Q2 metrics.
- No-multitask manifests are accepted only under the recorded component-bundle protocol: `status=NOT_STARTED` is paired with `execution_kind=component_bundle`, `direct_classification_outputs_used=true`, `synthetic_results=false`, and `review_summary.RUN_STATUS=PASS`; this is recorded as `manifest_status_ok=true`, not relabeled as a generic PASS.
- Complete no-polarity artifacts: overflow-011 (20260521), overflow-010 (20260522), overflow-009 (20260523).
- Checkpoint-tree candidates are inventoried separately under `Thundergod2007/vipragsent-xlmr-checkpoints`; checkpoint payloads were not downloaded.
- Local alternate-manifest candidates are retained. The partial no-rationale 20260523 overflow-006 tree is not promoted over complete overflow-018 artifacts.
- Structural-only reference PDF inspection: `D:/vipragsent-pr/tmp/pdf_reference/main.pdf`; no numerical claims were extracted from it.

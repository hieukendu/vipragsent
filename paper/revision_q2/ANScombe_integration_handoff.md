# Q2 evidence handoff to Anscombe

- Integration worker: `01a0bd9e-d2b4-7b70-9f30-7648e156d8b3`
- Worker scope: Luna Q2 evidence worker; no manuscript edits made.
- Verification status: `ANALYZED` — live HF artifact fetches, persisted raw payloads, and deterministic recomputation; no fresh training rerun.

## Final status

The final checkpoint is `complete` with 18/18 evidence rows and 18/18 `source_ok=true`. All 18 artifact trees returned HTTP 200 from the HF tree API at pinned repository revisions. All 15 rows with a polarity head have dev/test ECE recomputation absolute deltas `0.0` under tolerance `1e-12`.

The seed-20260521 smoke values did not change after the full pass or label fix: full-run dev ECE `0.08015336569635316`, test ECE `0.08452923804521562`, both deltas `0.0`.

## No-polarity diagnosis

For all three no-polarity seeds (20260521/22/23), the selected artifacts are overflow-011/010/009. Each has 1,999 dev and 2,000 test prediction records; polarity counts are zero for gold, logits, predictions, and probabilities in both splits; saved dev/test polarity ECE fields are absent; and both review active-head lists and active uncertainty-task lists exclude polarity:

`[implicit_sentiment, sarcasm, irony, idiom_figurative, code_switching, mocking, emotion]`

Therefore the canonical value is exactly `not applicable (polarity head removed)`. This is structural inapplicability, not an unresolved missing saved field: no polarity head/task exists and no polarity probability source exists from which ECE could be recomputed. No pragmatic ECE, zero, or surrogate was inserted.

## Corrected canonical Q2 table

| Variant | Macro-pragmatic F1, test (%) | Polarity dev ECE x 10^3 | Relative cost |
|---|---:|---:|---:|
| Full follow-up | 92.8 +/- 0.6 | 81.1 +/- 11.7 | 1.00 |
| No emotion auxiliary | 91.7 +/- 0.7 | 56.2 +/- 22.5 | 0.65 |
| No polarity auxiliary | 92.8 +/- 0.2 | not applicable (polarity head removed) | 0.81 |
| No explanation auxiliary | 92.3 +/- 1.3 | 68.1 +/- 22.2 | 0.20 |
| No multitask bundle | 74.7 +/- 2.3 | 118.9 +/- 69.1 | 1.10 |
| No task-uncertainty weighting | 92.9 +/- 0.3 | 90.2 +/- 5.9 | 0.82 |

F1 is taken from test metrics. The Q2 protocol source `configs/experiments/q2/protocol.yaml:8-11` declares dev ECE, the intended polarity 3-way head, and ten equal-width bins. The legacy field name `polarity_dev_ece` occurs in both split files, but per-seed recomputation proves the dev file contains dev ECE and the test file contains test ECE. The old table used the test-file values; the canonical table uses protocol/caption dev values.

## Gate trail and data-format issue

The first full pass completed 12 rows, then stopped because later saved gold labels were strings such as `"neutral"` rather than integer indices. The recomputation now uses the repository-defined order `negative, neutral, positive`; smoke seed-20260521 values were unchanged.

The next pass identified four false `source_ok` rows. No-rationale/20260523 lacked only optional `config_snapshot.yaml` (HTTP status 0 after bounded retries). Each no-multitask row lacked the optional top-level files `training/resolved_training_config.json`, `selection/best_checkpoint.json`, and `selection/selection_metric.json` because its manifest declares `execution_kind=component_bundle`; required manifest/review, metrics, predictions, counts, F1, and GPU-hour evidence were present. The final gate accepts this status only when `NOT_STARTED + component_bundle + direct_classification_outputs_used=true + synthetic_results=false + review PASS`; it is recorded as `manifest_status_ok=true`, not relabeled as generic PASS.

## Validation command and log

Remote audit command (final pass):

`python paper/revision_q2/compute_q2_evidence.py` -> `EXIT:0`; emitted `runs: 18`, checkpoint `complete`.

Offline validation command (no network):

`python paper/revision_q2/validate_q2_evidence.py` -> `status=offline_validation_pass`, `evidence=18`, `headed_rows=15`, `no_polarity_rows=3`, `raw_files=198`, `EXIT:0`.

The offline validator checks checkpoint completion, 18/18 source rows, 18 artifact-tree HTTP 200 statuses, all no-polarity head/prediction proofs, all 15 headed-row split matches and deltas, smoke-versus-final equality for seed 20260521, the optional omission names, and component-bundle manifest fields.

## Remaining limitations

This is an artifact audit (`ANALYZED`), not a fresh training rerun. Checkpoint payload weights were not downloaded or inspected; therefore this handoff does not claim weight-level verification. The no-rationale/20260523 optional config snapshot remains unavailable, and no-multitask top-level training/selection files remain absent by component-bundle design; neither omission removes the required Q2 manifest/review/metric/prediction evidence. The structural no-polarity conclusion is based on saved head/task manifests and prediction schemas, not on checkpoint-weight inspection.

## Evidence files

- [q2_evidence_report.md](q2_evidence_report.md)
- [q2_canonical_table.csv](q2_canonical_table.csv)
- [q2_checkpoint.json](q2_checkpoint.json)
- [run_evidence.jsonl](run_evidence.jsonl)
- [source_inventory.json](source_inventory.json)
- [compute_q2_evidence.py](compute_q2_evidence.py)
- [validate_q2_evidence.py](validate_q2_evidence.py)
- Raw source payload mirror: `raw_sources/`
- Earlier checkpoints are preserved as `*_before_label_normalization.*`, `*_before_core_source_gate.*`, `*_before_retry_gate.*`, and `*_before_component_bundle_acceptance.*`.

# ViPragSent sequential experiment run: q1b_vipragsent_full_phobert_20260522

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `q1b_vipragsent_full_phobert_20260522`
- Research question: `Q1b`
- System ID: `vipragsent_full_phobert`
- Display name: Full ViPragSent PhoBERT
- Variant: `table3_checkpoint`
- Backbone: `phobert_base`
- Seed: `20260522`
- Budget: ``
- Task: `polarity;emotion`
- Split: `external_test`
- Dependencies: `approved_source_checkpoint`
- Required Phase 15 assets: `model_weights;tokenizer;runtime_profile`
- Execution kind: `evaluation_only`
- Expected artifacts: `external_predictions;metrics`
- Selection metric: `ord_external_f1`
- Evaluation protocol: `q1b_external_retention_v1`
- Reusable checkpoint key: `vipragsent_full_phobert:20260522`
- Protocol resolution: `RESOLVED`
- CLI kind: `experiment`
- Resolved execution stage plan: `q1b_evaluation_only`


## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id q1b_vipragsent_full_phobert_20260522 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id q1b_vipragsent_full_phobert_20260522 --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id q1b_vipragsent_full_phobert_20260522 --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: `preflight, resolve_approved_source, evaluate_external_tests, export_artifacts, validate_artifacts, generate_review_summary`.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id q1b_vipragsent_full_phobert_20260522` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

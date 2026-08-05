# ViPragSent sequential experiment run: q1a_sailor_pragmatic_sft_20260522

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `q1a_sailor_pragmatic_sft_20260522`
- Research question: `Q1a`
- System ID: `sailor_pragmatic_sft`
- Display name: Sailor-7B pragmatic SFT
- Variant: `pragmatic_sft`
- Backbone: `sailor_7b`
- Seed: `20260522`
- Budget: ``
- Task: `pragmatic`
- Split: `vipragsent_test`
- Dependencies: `preflight_validation`
- Required Phase 15 assets: `model_weights;tokenizer;runtime_profile`
- Execution kind: `trainable`
- Expected artifacts: `predictions;metrics;history`
- Selection metric: `macro_prag_f1_dev`
- Evaluation protocol: `q1a_frozen_dev_threshold_v1`
- Reusable checkpoint key: `sailor_pragmatic_sft:20260522`
- Protocol resolution: `RESOLVED`
- CLI kind: `experiment`
- Resolved execution stage plan: `trainable_classifier`


## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id q1a_sailor_pragmatic_sft_20260522 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id q1a_sailor_pragmatic_sft_20260522 --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id q1a_sailor_pragmatic_sft_20260522 --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: `preflight, train, evaluate_dev, freeze_selection, evaluate_test, export_artifacts, validate_artifacts, generate_review_summary`.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id q1a_sailor_pragmatic_sft_20260522` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

# ViPragSent sequential experiment run: q3_azure_gpt41_mini_8shot_64

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `q3_azure_gpt41_mini_8shot_64`
- Research question: `Q3`
- System ID: `azure_gpt41_mini_8shot`
- Display name: azure_gpt41_mini_8shot
- Variant: `q3_eight_shot`
- Backbone: `azure`
- Seed: `None`
- Budget: `64`
- Task: `sarcasm`
- Split: `test`
- Dependencies: `azure_prompted_baselines`
- Required Phase 15 assets: `azure_deployment;prompt_manifest`
- Expected artifacts: `q3_predictions;usage`
- Selection metric: `sarcasm_dev_f1`
- Evaluation protocol: `q3_low_resource_masked_v1`
- Reusable checkpoint key: `azure_gpt41_mini_8shot:64`
- Protocol resolution: `RESOLVED`

## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id q3_azure_gpt41_mini_8shot_64 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id q3_azure_gpt41_mini_8shot_64 --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id q3_azure_gpt41_mini_8shot_64 --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: preflight, train_or_run, evaluate_dev, freeze_selection, evaluate_test, export_artifacts, validate_artifacts, generate_review_summary.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id q3_azure_gpt41_mini_8shot_64` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

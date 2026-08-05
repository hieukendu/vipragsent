# ViPragSent sequential experiment run: q1b_azure_gpt41_mini

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `q1b_azure_gpt41_mini`
- Research question: `Q1b`
- System ID: `azure_gpt41_mini`
- Display name: azure_gpt41_mini
- Variant: `dedicated_prompts`
- Backbone: `azure`
- Seed: `None`
- Budget: ``
- Task: `polarity;emotion`
- Split: `external_test`
- Dependencies: `azure_prompted_baselines`
- Required Phase 15 assets: `azure_deployment;prompt_manifest`
- Execution kind: `azure`
- Expected artifacts: `external_predictions;metrics`
- Selection metric: `ord_external_f1`
- Evaluation protocol: `q1b_external_retention_v1`
- Reusable checkpoint key: `azure_gpt41_mini:dedicated_prompts`
- Protocol resolution: `RESOLVED`

## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id q1b_azure_gpt41_mini --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id q1b_azure_gpt41_mini --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id q1b_azure_gpt41_mini --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: preflight, train_or_reuse, evaluate_dev, freeze_selection, evaluate_test, export_artifacts, validate_artifacts, generate_review_summary.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id q1b_azure_gpt41_mini` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

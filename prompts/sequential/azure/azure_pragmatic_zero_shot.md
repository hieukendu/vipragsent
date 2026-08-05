# ViPragSent sequential Azure job: azure_pragmatic_zero_shot

This runbook names exactly one Azure job. Do not start another job, experiment, batch, or global matrix.

## Locked job

- Job ID: `azure_pragmatic_zero_shot`
- Job type: `pragmatic_zero_shot`
- Research question: `Q1a`
- Task: `pragmatic`
- Budget: ``
- Split: `vipragsent_test`
- Required assets: `azure_deployment;prompt_manifest`
- Model family: `gpt-4.1-mini`

## Required command sequence

1. Run `python scripts/run_single_azure_job.py --job-id azure_pragmatic_zero_shot --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste the report into the Codex chat and stop.
3. After preflight passes, execute exactly this stage order: `preflight` -> `execute_api_job` -> `validate_responses` -> `export_artifacts` -> `validate_artifacts` -> `generate_review_summary`.
4. Run that locked sequence with `python scripts/run_single_azure_job.py --job-id azure_pragmatic_zero_shot --stage all`.
5. On interruption, resume only this job with `python scripts/run_single_azure_job.py --job-id azure_pragmatic_zero_shot --resume`.
6. Use only the frozen prompt/schema manifest for this job. Do not log secrets, use the direct OpenAI endpoint, or silently change demonstrations, deployment, budget, or retry policy.

## Required review handoff

Complete the sequential stages applicable to this job, print the complete review summary with `python scripts/print_run_review_summary.py --run-id azure_pragmatic_zero_shot`, and paste it into the Codex chat. Include request/token usage, invalid-output accounting when applicable, artifact hashes, `RUN_STATUS`, `USER_REVIEW_STATUS`, and `NEXT_RUN_ALLOWED`.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next job automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

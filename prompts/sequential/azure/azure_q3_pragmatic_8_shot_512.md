# ViPragSent sequential Azure job: azure_q3_pragmatic_8_shot_512

This runbook names exactly one Azure job. Do not start another job, experiment, batch, or global matrix.

## Locked job

- Job ID: `azure_q3_pragmatic_8_shot_512`
- Job type: `q3_budget_specific_pragmatic_8_shot`
- Research question: `Q3`
- Task: `sarcasm`
- Budget: `512`
- Split: `vipragsent_test`
- Required assets: `azure_deployment;prompt_manifest`
- Model family: `gpt-4.1-mini`

## Required command sequence

1. Run `python scripts/run_single_azure_job.py --job-id azure_q3_pragmatic_8_shot_512 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste the report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_azure_job.py --job-id azure_q3_pragmatic_8_shot_512 --stage all`.
4. On interruption, resume only this job with `python scripts/run_single_azure_job.py --job-id azure_q3_pragmatic_8_shot_512 --resume`.
5. Use only the frozen prompt/schema manifest for this job. Do not log secrets, use the direct OpenAI endpoint, or silently change demonstrations, deployment, budget, or retry policy.

## Required review handoff

Complete the sequential stages applicable to this job, print the complete review summary with `python scripts/print_run_review_summary.py --run-id azure_q3_pragmatic_8_shot_512`, and paste it into the Codex chat. Include request/token usage, invalid-output accounting when applicable, artifact hashes, `RUN_STATUS`, `USER_REVIEW_STATUS`, and `NEXT_RUN_ALLOWED`.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next job automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

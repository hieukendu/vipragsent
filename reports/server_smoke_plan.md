# Server Smoke Plan

Status: `PLAN_ONLY`

This plan is intentionally not executed in the repair task. It must be run only
after the exact final repair SHA is checked out on the target server and the
user reviews the result.

1. Verify the checked-out commit equals the final repair SHA and the branch is
   `codex/phase-14-5-production-repair`.
2. Run the runtime preflight and verify the selected model-family snapshot,
   tokenizer revision, CUDA device, quantization settings, and physical batch
   probe without starting the global DAG.
3. Download and verify only the approved lightweight model family in Phase 15;
   record repository, revision, tokenizer revision, file checksums, and the
   complete smoke report.
4. Run exactly one approved model-family smoke entry, including checkpoint
   reload, first-batch device report, dev selection, frozen test gate, and
   artifact checksums.
5. Stop with `USER_REVIEW_STATUS=PENDING` and
   `NEXT_RUN_ALLOWED=NO`. Do not launch another run until explicit approval.

Required stop conditions:

- Any protocol conflict or hash mismatch: `BLOCKED`.
- Missing runtime assets, CUDA/Azure preflight failure, or checkpoint mismatch:
  `BLOCKED`.
- Never execute the global production DAG, an unapproved experiment, or a live
  Azure job as part of this smoke plan.

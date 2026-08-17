# Rollback and recovery

The optimization branch is isolated from `/root/vipragsent`. Historical checkpoints and artifacts are immutable evidence and are not rewritten by this PR.

## Software rollback

- Select the prior commit on the production branch and the legacy `sequential_review_gated` scheduler mode.
- Do not use `git reset --hard`, force-push, or delete the production worktree as part of rollback.
- Preserve the optimization branch and review reports for diagnosis. Re-run the CPU/mock test matrix after any controlled checkout.

## Scientific-state rollback

- Do not restart or control the paused legacy run merely to roll back code.
- Resume only from a validated persisted boundary after exact identity reconciliation; otherwise classify the path `BLOCKED`/`RESUME_REVIEW_REQUIRED`.
- Never replace an immutable checkpoint, per-epoch DEV artifact, reasoning chunk, judge result, or selection manifest in place.
- If a new runtime stage fails, recover from its last committed checkpoint/chunk/manifest and keep the old-engine artifacts namespace separate.

## Scheduler recovery

Use the durable journal, validate campaign authorization and bound hashes, inspect lease heartbeat/ownership, validate produced artifacts, and only then classify a stage as completed, resumable, retryable, or blocked. A PID alone is not proof of a live lease. A safe-stop request must persist the reached boundary and prevent new admission.

## External-state safety

This task performed no HF writes, Azure calls, network-side production actions, or paid API operations. Any future backup/cleanup must first reach remote verification and must never delete the last valid resume boundary.

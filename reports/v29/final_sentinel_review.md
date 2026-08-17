# ViPragSent V29 runtime convergence evidence

## Status

The V29 source implementation is converged at `9f540f3043c85cd60ea6c8706175d1bb44bcae0e`.
This evidence update is report-only; the resulting PR head must receive a fresh exact-head
CI run and Sentinel review before final completion.

## Finding closure

- **P0-1 — GenerationChunkStore:** committed rows, sample IDs, next index, and chunk metadata are retained in memory after one initialization validation. Commits inspect only the manifest signature, reconcile only appended manifest entries, validate only new rows, and perform one full chunk validation at completion. Contiguous chunk indexes and exact sample-record order are enforced. The scaling regression counts both historical JSONL reads and manifest stat calls and rejects quadratic growth.
- **P0-2 — Lease renewal:** immutable campaign/run/stage/host/PID/instance ownership is compared separately from mutable heartbeat state. Renewal advances heartbeat and expiry while preserving acquisition time.
- **P1-1 — Generation identity:** each trained epoch writes and verifies its canonical checkpoint before DEV generation. The DEV generation contract consumes that checkpoint SHA; the full live model-state hash is only a justified fallback when no persisted checkpoint exists.
- **P1-2 — Explanation source verification:** the resolver physically verifies the exact checkpoint once and returns an immutable validated-source boundary. Direct construction of that boundary is rejected, and request fingerprinting/runtime construction reuse the verified identity without rehashing the checkpoint.
- **P1-3 — Lazy stages:** generation and explanation judge/metrics stages classify and validate existing artifacts before device, snapshot, tokenizer, or large-model resolution. Train/generate paths retain model loading.
- **P1-4 — Reuse authenticity:** every reuse hash field must be a canonical 64-hex SHA-256 digest; malformed equal values block reuse, valid exact values verify, and valid disagreements invalidate.

## P2 checkpoint-copy decision

**Explicitly deferred as a post-convergence optimization.** The current checkpoint manager still
writes separate epoch, best, and latest checkpoint artifacts. Replacing those writes with immutable
canonical epoch files plus atomic `best_checkpoint.json`/`latest_checkpoint.json` pointer manifests
would touch best selection, latest resume, rollback, approval provenance, and artifact export in one
cross-cutting redesign. No safe copy primitive or real-model resume/rollback validation was authorized
in this code-only pass. Keeping the existing writes preserves exact checkpoint semantics and avoids
claiming an unmeasured performance gain.

A future P2 change must include exact path/epoch/SHA pointer validation, atomic pointer updates,
best/latest/resume/rollback regressions, and separately authorized real-model profiling. P2 is not
claimed as implemented in V29.

## Validation bound to the source implementation

- Exact source head: `9f540f3043c85cd60ea6c8706175d1bb44bcae0e`.
- Exact-head GitHub Actions `cpu-ci`: run `31993165762`, job `95280143214`, success.
- Local V29-focused regressions: **75 passed**.
- Local permitted CPU/mock suite: **404 passed** using the established exclusions (`not server`, `not gpu`, `not azure_live`, `not model_download`).
- `ruff check src tests`: pass.
- `python -m compileall -q src`: pass.
- `git diff --check`: pass.

No production training, GPU workload, live Azure request, model download, Hugging Face mutation,
TEST-data profiling, process-control action, or merge was performed.

## Frozen protocol checks

Q3 remains 36 retained local rows plus four seedless Azure rows; XLM-R and budgets 64/256 remain
excluded. Q2 remains six variants by three seeds. Q1b remains evaluation-only with exact source,
checkpoint, seed, graph, and approval provenance. Frozen TEST access remains unchanged.

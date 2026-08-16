# Final validation — code-only runtime optimization

## Scope and identity

- Master prompt: V26, SHA-256 `457da887b325625b395b2dc63576bed95c02e95c601be68c62f61f97f53a8ed0`.
- Manager branch: `codex/naacl-runtime-optimization`.
- Isolated source base: `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`.
- Production worktree `/root/vipragsent` was not edited. Its live loaded-code identity remains `LIVE_CODE_IDENTITY_UNCERTAIN`; the observed run is stale/paused after epoch 2 with no active PID, `PENDING_USER_APPROVAL`, and `next run = NO`.
- Current epoch-2 checkpoint evidence is read-only: local SHA-256 `d3fe7e99bb0758e527c4967fe2a8502c722fce8d86cd7664ea776440a3b41a77`, matching the current Vistral HF metadata revision/path. This does not promote reuse or resume because source identity and approval bindings are incomplete.

## Review-gated implementation

- Wave 0: accepted by independent Sentinel; path-specific identity/reuse blockers remain recorded.
- Wave 1: checkpoint/resume, read-only reuse, and NAACL profile chains accepted by Sentinel round 4.
- Wave 2: generation chain accepted by Sentinel round 2; decoder-safe left padding, causal variable-length equivalence, committed chunk persistence, and best-epoch artifact binding are integrated.
- Wave 3: Azure, explanation-only, scheduler, and estimator slices accepted by independent Sentinel. Scheduler/estimator rework `0cec7982ca412b4fe1b9efc50dc6e07e7f5ba2ec` was independently re-reviewed and passed.

## Tests and static checks

The integrated CPU/mock-only selected matrix passed **124 tests** after the scheduler rework. It included the Wave-3 packages, Wave-2 generation/resume/red-team coverage, reuse/profile/Q1b coverage, and pre-experiment closure tests. The broader CPU/mock-only repository run passed **292 tests** in 4:36 with CUDA hidden and an isolated temporary pytest cache.

Additional evidence:

- Wave-3 independent Sentinel: 29 focused tests passed; scheduler re-review: 12 passed.
- Python 3.11 compilation passed for all new runtime modules.
- `git diff --check` passed for reviewed commits and reports.
- Ruff 0.6.9 reports 13 pre-existing findings on `origin/main` and 19 additional style findings in the reviewed runtime modules; no auto-fix was applied after the implementation reviews. Mypy was not run.
- No real model/data benchmark, training, evaluation, Azure request, HF mutation, process control, or secret-bearing environment dump occurred.

## Gate result

`PROJECTED_GATE_CONDITIONAL`. The implementation removes safe execution waste and provides a resource-constrained estimator, but the final campaign makespan still depends on later authorized production-hardware measurements: Vistral generation throughput, Azure transport throughput, and any PhoBERT concurrency-2 profile. This code-only task does not claim `MEASURED_GATE_PASS`.

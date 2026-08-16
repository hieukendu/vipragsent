# Final validation — V27 code-only runtime optimization

- Governing overlay: V27, SHA-256 `d77f564135d9196ff88e1da24a0fb15fd7600c9a19041d0f7097cfd5850a4580`.
- Scientific authority: V26 remains authoritative, recorded digest `457da887b325625b395b2dc63576bed95c02e95c601be68c62f61f97f53a8ed0`.

## Scope and identity

- Manager branch: `codex/naacl-runtime-optimization`.
- Reviewed code head: `16181334b5d00a6e3f622dffa826575a1b18915d`.
- Base: `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`.
- The convergence artifacts are a report-only descendant of the reviewed code head; no source behavior changes occur in that descendant.
- Production worktree `/root/vipragsent` was not edited or controlled.
- No production, Azure, Hugging Face, model-download, benchmark, process-control, TEST-environment, or merge action occurred.

## Scientific invariants

- Q3 retains 36 local cells plus four seedless Azure rows; XLM-R and budgets `64/256` remain explicit exclusions.
- Q1b remains evaluation-only and binds canonical producer identity, checkpoint/seed, graph/source digests, strict training-prohibition fields, and complete approval records through source, retention, review, and aggregation.
- Q2 retains exactly six variants across the three locked seeds.
- Shared GPU occupancy is serialized by default; only the validated PhoBERT pair exception is allowed.
- Generation speedup divides duration by exact factors 1.0/1.5/2.0/2.5/3.0/4.0.
- `latest` preserves the last completed epoch independently of persisted `best` selection.
- Explicit source adapters require hash-bound approved runs; injected predictors are fixture-only.

## Checks

- Targeted approval/source suite: **55 passed**.
- Broad CPU/mock-only suite: **331 passed in 70.76 seconds**.
- Fixture DAG, execution registry, schemas, sequential prompts, NAACL profile validation, production implementation audit, and final production correctness audit: **PASS**.
- Compilation, `git diff --check`, Ruff 0.6.9, and Ruff 0.16.3: **PASS**.
- Final correctness audit records `CI_STATUS: NOT_RUN`; the previous remote Ruff failure (`31958375122`) is stale.

## Gate result

`PROJECTED_GATE_CONDITIONAL`. The code review is complete, but fresh remote CI on the pushed head and final PR synchronization remain required. This code-only task does not claim measured campaign readiness. PR #10 remains a draft and was **not merged**.

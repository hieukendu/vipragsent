# Final validation — V27 code-only runtime optimization

- Governing overlay: V27, SHA-256 `d77f564135d9196ff88e1da24a0fb15fd760c9a19041d0f7097cfd5850a4580` (see `debug_convergence/governing_prompt_provenance.json`).
- Scientific authority: V26 remains authoritative, recorded digest `457da887b325625b395b2dc63576bed95c02e95c601be68c62f61f97f53a8ed0`. The exact V26 bytes were not included in the current attachment, so no missing V26 rule is inferred.

## Scope and identity

- Manager branch: `codex/naacl-runtime-optimization`.
- Reviewed Manager head: `e0c502a3879fb3a65305841e6941a6bae24e5778`.
- Base: `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`.
- Production worktree `/root/vipragsent` was not edited. Its live loaded-code identity remains uncertain; the observed run is stale/paused with no active PID, `PENDING_USER_APPROVAL`, and `next run = NO`.
- No production, Azure, Hugging Face, model-download, benchmark, or process-control action occurred.

## Scientific invariants

- Q3 profile validation passes with 36 local cells plus four Azure rows, Azure `seed: null`, and fail-closed complete-inventory validation. XLM-R and budgets `64/256` remain explicit exclusions.
- Q1b remains evaluation-only: canonical producer kind/ID/run/checkpoint/seed and graph/source digests are emitted and bound through source, external manifest/metrics, and review summary; conflicting or type-coerced provenance blocks aggregation.
- Q2 retains exactly six variants across the three locked seeds.
- Shared GPU occupancy is serialized by default; only the validated PhoBERT pair exception is allowed.
- Generation speedup divides duration by the exact factors 1.0/1.5/2.0/2.5/3.0/4.0.
- `latest` preserves the last completed epoch independently of persisted `best` selection.

## Checks

- Focused correction suite: **47 passed**.
- Broad CPU/mock-only suite: **325 passed in 57.74 seconds**.
- Fixture DAG, execution registry, schemas, sequential prompt validation, NAACL profile validation, production implementation audit, and final production correctness audit: **PASS**.
- Compilation, `git diff --check`, Ruff 0.6.9, and Ruff 0.16.3: **PASS**.
- Final correctness audit reports `ci_status: NOT_RUN`; fresh remote CI is still required.

## Gate result

`PROJECTED_GATE_CONDITIONAL`. This code-only task does not claim measured campaign readiness. PR #10 remains a draft pending final read-only review and fresh remote CI. The PR was **not merged**.

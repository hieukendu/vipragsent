# NAACL runtime optimization — code-only review PR

## Summary

PR #10 contains bounded runtime engineering changes for ViPragSent: provenance-bound checkpoint/resume, decoder-safe resumable generation, explanation-only inference reuse, a shared-GPU scheduler mutex, a default-off estimator, strict Q1b provenance/approval gates, and fail-closed source adapters. The V27 overlay preserves V26 science: three local Q3 systems at budgets `32/128/512/full` across three locked seeds plus four protocol-defined seedless Azure rows; XLM-R and budgets `64/256` remain excluded.

## Safety and scope

- No production training/evaluation, model download, benchmark, real model/data run, or Azure request was performed.
- Hugging Face was not mutated.
- The paused CoT run and dirty `/root/vipragsent` worktree were not controlled or edited.
- The scheduler remains opt-in/default-off; legacy sequential review-gated behavior remains intact.
- TEST remains sealed and this task will not merge the PR.

## Validation at reviewed code head

- Reviewed code head: `23332af0bf5454958ea48b630ef43a4e45a61feb`.
- Broad CPU/mock-only suite: **331 passed in 70.76s**.
- Targeted approval/source suite: **55 passed**.
- Fixture DAG, execution registry, schemas, sequential prompts, NAACL profile validation, production implementation audit, and final correctness audit: **PASS**.
- Python compilation, `git diff --check`, Ruff 0.6.9, and Ruff 0.16.3: **PASS**.
- Dalton's final whole-repository read-only review: **PASS** with no actionable findings.
- Fresh GitHub Actions `cpu-ci` run `31968069469` passed on the exact reviewed code head; `31958375122` is a stale pre-current-head Ruff failure.

## Readiness

`PROJECTED_GATE_CONDITIONAL` remains the correct code-only status. Campaign authorization still requires later user-authorized DEV-only throughput/concurrency measurements and exact live-code/provenance reconciliation. The branch is synchronized and ready for the draft-to-ready transition.

## Delivery note

Target is `main`. Fresh CI is green; the PR is ready to be marked for review and remains **not merged**.

# NAACL runtime optimization — code-only review PR

## Summary

PR #10 contains bounded runtime engineering changes for ViPragSent: provenance-bound checkpoint/resume, decoder-safe resumable generation, explanation-only inference reuse, a shared-GPU scheduler mutex, and a default-off estimator. The V27 scientific overlay retains the three local Q3 systems at budgets `32/128/512/full` across the three locked seeds plus four protocol-defined seedless Azure rows; XLM-R and budgets `64/256` remain excluded.

## Safety and scope

- No production training/evaluation, model download, benchmark, or real model/data run was performed.
- No Azure request or spend occurred.
- Hugging Face was not mutated.
- The paused CoT run and dirty `/root/vipragsent` worktree were not controlled or edited.
- The new scheduler mode remains opt-in/default-off; legacy sequential review-gated behavior remains intact.
- TEST remains sealed and the PR will not be merged by this task.

## Validation at Manager head

- Head: `e0c502a3879fb3a65305841e6941a6bae24e5778`.
- Broad CPU/mock-only suite: **325 passed in 57.74s**.
- Focused correction suite: **47 passed**.
- Fixture DAG, execution registry, schemas, sequential prompts, NAACL profile validation, implementation audit, and final correctness audit: **PASS**.
- Python compilation and `git diff --check`: **PASS**.
- Ruff 0.6.9 and Ruff 0.16.3: **PASS**.
- Remote CI has not yet been rerun on this head; the previous Ruff failure (`31958375122`) is stale.

## Readiness

`PROJECTED_GATE_CONDITIONAL` remains the correct status. Campaign authorization still requires later user-authorized DEV-only throughput/concurrency measurements and exact live-code/provenance reconciliation. This branch is ready for final read-only review, then a fresh remote CI run; it remains a draft until all gates are green.

## Delivery note

Target is `main`. PR #10 will be pushed as a draft and marked ready only if final review and remote CI pass. The PR was **not merged**.

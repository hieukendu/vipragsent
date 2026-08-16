# NAACL runtime optimization — code-only review PR

## Summary

This PR refactors ViPragSent's future execution runtime without starting a production campaign. It adds provenance-bound checkpoint/resume contracts, decoder-safe resumable generation, bounded mock-only judging, explanation-only inference reuse, and a default-off resource-aware scheduler/estimator.

The only scientific scheduling change is the authorized NAACL-balanced Q3 profile: retain the three local systems, budgets `32/128/512/full`, and three seeds; exclude the Q3 XLM-R sweep and budgets `64/256`.

## Safety and scope

- No production training/evaluation or real model/data benchmark was run.
- No Azure request or spend occurred.
- Hugging Face was read-only; no upload, delete, rewrite, or retag occurred.
- The existing paused CoT run and dirty production worktree were not controlled or edited.
- New scheduler mode is opt-in/default-off; legacy sequential review-gated mode remains available.

## Validation

- Broad CPU/mock-only suite: `292 passed in 4:36`.
- Integrated selected suite: `124 passed`.
- Independent Wave-3 review: `29 passed`; scheduler re-review: `12 passed`.
- Python compilation and `git diff --check` passed.
- Ruff/mypy were unavailable.

## Known gates before production

The runtime gate remains `PROJECTED_GATE_CONDITIONAL`. A later user-authorized DEV-only Vistral profile, dedicated PhoBERT concurrency profile, bounded Azure transport profile, and exact live-code/provenance reconciliation are required before campaign authorization. TEST remains sealed.

## Delivery note

This local branch is ready for review. Push and draft-PR creation are pending because the environment does not contain the required authenticated `gh` CLI; no merge is requested or performed.

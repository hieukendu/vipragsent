# ViPragSent V28 runtime optimization and convergence

## Scope

This PR contains bounded runtime and provenance engineering for ViPragSent. It preserves the V26/V27 scientific protocol and adds the V28 convergence repairs for resumable generation identity, explanation-only reuse, bounded Azure execution, cache recovery, and NAACL profile parity.

No production training/evaluation, real Azure request, Hugging Face mutation, model download, benchmark, TEST-environment action, process-control action, or merge was performed.

## Scientific invariants preserved

- Q3 retains the authorized 36 local cells plus four seedless Azure rows; XLM-R and budgets `64/256` remain excluded.
- Q1b remains evaluation-only with exact producer, checkpoint, seed, graph/source, and approval provenance.
- Q2 retains its locked six-variant-by-three-seed matrix.
- Shared GPU occupancy remains serialized by default, with only the validated PhoBERT exception.
- Generation remains strict, causal, checkpoint-aware, and judge-bound to generated reasoning only.

## V28 repairs

- Generation chunk manifests now carry a versioned canonical contract covering source, code, model, tokenizer, checkpoint, config/protocol, dataset/data, split, record content/order, seed, system, and budget identity. Production resume fails closed on missing or changed identity.
- Explanation-only runtime uses the same generation persistence boundary and binds its approved source/checkpoint/config/dataset identity into that contract; CPU fixture mode is explicit.
- Async and synchronous Azure paths enforce finite logical-request, transport-attempt, input/output/total-token, concurrency, and verified-spend ceilings, with actual response usage checked before persistence.
- Retryable Azure transport failures are not reusable cache entries; bounded stage loops stop and mark remaining work after a safety ceiling.
- The checked-in NAACL profile enforces parity between retained rows, exclusions, and the YAML policy.

## Validation and exact-head binding

The Manager records the exact final PR head, current-head `cpu-ci` run, review state, and Sentinel result in `reports/v28/` after the final evidence push. The local source integration head for this repair wave is `acc6467864bcea299862f5b0e29c7247cef7afde`; the final PR head is the exact SHA reported by GitHub after the evidence commit.

Local validation on the integrated source tree:

- **346 CPU/mock-only tests passed** (`not server`, `not gpu`, `not azure_live`, `not model_download`).
- Generation, explanation, Azure, and profile regression suites passed, including negative identity, actual-usage, retry/cache-recovery, and exclusion-parity cases.
- `compileall`, Ruff, and `git diff --check` passed.

## Readiness

`PROJECTED_GATE_CONDITIONAL` remains the honest campaign-readiness status: no real-model throughput/concurrency measurement or live Azure spend evidence was authorized in this code-only loop, and the paused historical run remains blocked from automatic reuse until its identity is reconciled. Code correctness and PR delivery are kept separate from unmeasured production performance.

## Rollback and delivery

The changes are isolated to the PR branch and can be reverted by commit. PR #10 targets `main`, remains open and unmerged, and must not be merged as part of this task.

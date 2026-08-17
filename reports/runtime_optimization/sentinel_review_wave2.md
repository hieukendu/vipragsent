# LUNA_SENTINEL Wave-2 Generation Review — Round 2

Decision: **PASS**. Zero open CRITICAL/HIGH findings. One MEDIUM residual is accepted below.

## Exact final chain

| Item | Commit | Parent | Result |
|---|---|---|---|
| Wave-2 generation | `efe09a31769d753da9611201c4ee7b940557b2ce` | `522689971dd4dba39d111887aeb1483466fe40b8` | reviewed |
| Round-1 descendant | `f3cba05b7885979f184d0b88f8f57fe6245014e3` | `efe09a31769d753da9611201c4ee7b940557b2ce` | reviewed |
| Round-2 final | `2cf464b7d63a9dd65777b04a00bc885684e8336e` | `f3cba05b7885979f184d0b88f8f57fe6245014e3` | PASS |

The exact full final hash resolves as the clean head of `codex/naacl-opt/generation` in `/root/vipragsent-runtime-opt-generation`.

## Round-2 closure

- **GEN-H001 — CLOSED.** Variable-length inference now removes inactive tokens and left-pads active rows. Continuations are sliced after the common padded input length. A causal fixture derives each continuation from the row’s last active token and verifies batch 1, 2, and 4 equivalence.
- **GEN-M002 — CLOSED.** Batch sizes above one require an explicit `True` value for `profiled`, `measured`, or `approved`; omission is rejected. Missing production profile remains safe batch one.
- **GEN-M001 — SUBSTANTIALLY CLOSED.** DEV reuse now requires matching `best_epoch`, selected/best checkpoint hash, reasoning/prediction/judge hashes, metrics hash, and chunk-manifest hash. The marker test covers the binding and tamper rejection.

## Accepted residual

- **GEN-M001-R — MEDIUM, accepted.** Reuse does not independently compare `checkpoints/epoch_<best_epoch>/model.pt` to `checkpoints/best/model.pt`, nor validate `dev_artifacts.source_root` against the epoch field. The published selection/best hash and all DEV artifact hashes are bound and fail closed in normal publication; future hardening can add the epoch-checkpoint hash and source-root assertion.
- **GEN-L001 — LOW, accepted.** Fixture-only stage adapters retain direct `torch.save` calls; production/trainable generation uses the canonical checkpoint helper, so no duplicate production serializer was introduced.

## Requirement disposition

- Decoder-safe left padding and continuation slicing: **PASS**.
- Causal variable-length batch 1/2/4 equivalence: **PASS**.
- Explicit true evidence for batch >1 and safe default batch one: **PASS**.
- Atomic/idempotent committed-before-judge chunk resume: **PASS**.
- DEV best-epoch/checkpoint/metrics/chunk-manifest binding: **PASS**, with GEN-M001-R accepted hardening residual.
- Per-sample stopping and reversible context: **PASS**.
- TEST sealing and no benchmark/external effects: **PASS**.

## Evidence

Cache-free targeted tests and static checks:

- `tests/test_generation_wave2.py` + `tests/test_luna_max_01_generation.py`: **33 passed**.
- `tests/test_luna_max_08_red_team.py` + `tests/test_preexperiment_closure.py`: **26 passed**.
- Python compilation and `git diff --check`: **passed**.
- No real model/data, Azure, Hugging Face, network, or benchmark execution.

No branch or production source was modified. The manager worktree’s unrelated pre-existing `agent_task_ledger.md` modification was left untouched.

# LUNA_SENTINEL Wave-1 Review — Round 4

Decision: **PASS**. No open CRITICAL or HIGH findings. Two MEDIUM residuals are accepted below with rationale.

## Reviewed heads

| Builder | Commit | Parent | Decision |
|---|---|---|---|
| COT / checkpoint-resume | `beb823d0f0e2e65cfee88b26ce5da03916adc821` | `326ad50cab09d87a53512309eb7b154ec1470ed6` | PASS |
| NAACL policy profile | `371b8717160be7e8a9be04eda0ca95415096ea76` | `008f16231b2b174b022838057e92b9f4c51e7017` | PASS |
| read-only reuse | `4b721704448b3781b6ff6a379f4c5bb208f575b7` | `2672c295cb82ae99758a42d0b106bc87e4112ad0` | PASS |

All three requested heads resolved exactly on their named branches; all three worktrees were clean. The reviewed commits are scoped to the requested Builder changes. No production or Builder branch was modified.

## Round-4 disposition

- **COT-H002 — CLOSED.** `stage_registry` rejects missing and placeholder `context.metadata.data_hash` values before injected-executor use or production model resolution. The negative test verifies `_execution_spec` is not reached and no checkpoint is written.
- **COT fixture/legacy safety — CLOSED.** Placeholder provenance is accepted only with explicit `fixture_mode=True`, and fixture mode is CPU-only. Legacy loading also requires explicit fixture mode. Production construction and checkpoint save/load require real provenance.
- **COT identity/resume invariants — PASS.** Dataset identity/hash, model identity/artifact, and tokenizer identity/artifact are validated and compared. The canonical checkpoint path retains model/optimizer/scheduler/RNG/run-state/data-order/provenance fields; no duplicate generation serializer was added.
- **COT-M002/M003 — CLOSED.** Targeted resume/state and legacy-gating coverage passed.
- **Profile source-drift closure — CLOSED.** Q1b/Q2/Q3 bindings, source digests, retained/excluded systems, budgets, seeds, metrics, and expected cells are validator/test checked. Default execution remains off and Q1b dependency wiring is covered.
- **Reuse prior H001/H002/H003 — CLOSED.** The read-only path has no network/HF/filesystem mutation or subprocess use and retains identity/hash verification and evidence-bearing promotion checks.

## Accepted residuals

- **COT-M001 — MEDIUM, accepted.** Checkpoint payload and provenance sidecar are published with separate atomic replacements. A crash between replacements can leave a missing sidecar; load fails closed rather than silently accepting incomplete state. A future single-container publication would remove this operational window.
- **PROFILE-M002 — MEDIUM, accepted.** The exact Azure Q3 exclusion is protected by the checked-in profile/report tests and the profile is default-off, but the standalone validator does not independently require that named exclusion. A future hardening change should encode that exclusion directly in validator policy.

## Evidence

Cache-free targeted tests completed without model/data/network access:

- COT checkpoint/generation/device-contract tests: **35 passed**.
- Profile balanced-profile and Q1b dependency tests: **19 passed**.
- Reuse read-only safety/identity tests: **13 passed**.
- Final Python compilation checks and `git diff --check`: passed.

No real model, dataset, benchmark, Azure, Hugging Face, or external call was run.


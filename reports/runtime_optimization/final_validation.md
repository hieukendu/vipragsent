# V28 final validation dossier

This dossier is maintained with exact-head semantics. The integrated source repair head is `acc6467864bcea299862f5b0e29c7247cef7afde`; the final PR head, current-head CI run, and final Sentinel verdict are recorded after the evidence push in `reports/v28/`.

## Scope and safety

- Governing specification: V28, SHA-256 `47866393782c0e761c2556849413e6f73c4c0f4ee77e8660e31102a89007ce96`.
- Base: `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`.
- PR: #10, branch `codex/naacl-runtime-optimization`, target `main`.
- The user worktree `/root/vipragsent` and its paused historical run were preserved.
- No production, Azure, Hugging Face, model-download, benchmark, TEST, process-control, or merge action occurred.

## Local evidence

- Broad CPU/mock-only suite: **346 passed** with server, GPU, live-Azure, and model-download tests excluded.
- Generation, explanation-runtime, Azure safety/cache, and NAACL profile regression suites: **PASS**.
- `compileall`: **PASS**.
- Ruff: **PASS**.
- `git diff --check`: **PASS**.

## V28 completion-gate status

| Gate | Status | Evidence boundary |
|---|---|---|
| R1 exact baseline/inventory | PASS for policy/profile; live campaign baseline remains read-only | Frozen protocol and profile validators |
| R2 runtime correctness | PASS for CPU/mock and negative-path regressions | No real-model launch claimed |
| R3 protocol/inventory fidelity | PASS | 36 local + 4 seedless Azure rows; exclusion parity enforced |
| R4 measured optimization | NOT MEASURED | No speedup/concurrency credit claimed |
| R5 training correctness | PASS for tested contracts | Production launch remains conditional |
| R6 evaluation correctness | PASS for tested contracts | Live Azure/model evaluation not run |
| R7 external cost bound | BOUNDED CODE PATH | Actual live pricing/usage remains unobserved |
| R8 artifact reproducibility | PASS for new contract; paused-run reuse BLOCKED | Historical code identity requires reconciliation |

Overall campaign readiness is `PROJECTED_GATE_CONDITIONAL`, not a measured performance claim. The PR remains open and unmerged.

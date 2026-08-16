# V27 Debug Convergence Bug Register

Round 0 is frozen at remote head `c2e78c62ee51c7566629930b3b0a115920735f30`; Manager has integrated the first fix wave at `183126e7a53fc7027b1ac3774c77dd3c5016426a`.
PR #10 is open and draft. The remote `cpu-ci` run `31957116632` failed at Ruff before
fixture generation or tests.

| Bug | Severity | Status | Owner | Evidence-backed symptom |
|---|---|---|---|---|
| BUG-KNOWN-001 | HIGH | FIXED_PENDING_REVIEW | LUNA_WORKER_SCIENCE_PROFILE | Reduced Q3 profile drops protocol-defined Azure comparison rows and conflates local-cell count with total rows. |
| BUG-KNOWN-002 | HIGH | FIXED_PENDING_REVIEW | LUNA_WORKER_SCHEDULER_RESOURCES | Resource-aware scheduler permits independent GPU-family lanes to overlap on one allocation. |
| BUG-KNOWN-003 | HIGH | FIXED_PENDING_REVIEW | LUNA_WORKER_ESTIMATOR_CI | Generation speedup is applied as a duration multiplier and the test oracle accepts reversed semantics. |
| BUG-KNOWN-004 | HIGH | OPEN | LUNA_WORKER_ESTIMATOR_CI + hygiene pass | Exact GitHub Ruff gate is red; the first scoped hygiene pass left residual findings outside its writable scope. |

All four findings are seeded from the V27 external review and reproduced/located against
the current Manager head in `bug_register.json`. The first three have integrated fixes;
BUG-KNOWN-004 remains open until the exact remote Ruff gate is green. No production
campaign, model/data benchmark, real Azure request, Hugging Face mutation, or process
control was performed.

# V27 Debug Convergence Bug Register

Round 0 is frozen at Manager and remote head `16ff2b0f403eb694ed0c53d6808a5b441d903735`.
PR #10 is open and draft. The remote `cpu-ci` run `31957116632` failed at Ruff before
fixture generation or tests.

| Bug | Severity | Status | Owner | Evidence-backed symptom |
|---|---|---|---|---|
| BUG-KNOWN-001 | HIGH | OPEN | pending | Reduced Q3 profile drops protocol-defined Azure comparison rows and conflates local-cell count with total rows. |
| BUG-KNOWN-002 | HIGH | OPEN | pending | Resource-aware scheduler permits independent GPU-family lanes to overlap on one allocation. |
| BUG-KNOWN-003 | HIGH | OPEN | pending | Generation speedup is applied as a duration multiplier and the test oracle accepts reversed semantics. |
| BUG-KNOWN-004 | HIGH | OPEN | pending | Exact GitHub Ruff gate is red; branch and legacy findings are not yet corrected. |

All four findings are seeded from the V27 external review and reproduced/located against
the current Manager head in `bug_register.json`. No production campaign, model/data
benchmark, real Azure request, Hugging Face mutation, or process control was performed.

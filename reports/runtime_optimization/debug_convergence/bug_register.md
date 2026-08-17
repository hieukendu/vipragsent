# V28 Debug Convergence Bug Register

The V28 repair wave is integrated at source head `acc6467864bcea299862f5b0e29c7247cef7afde`. The final evidence push, current-head CI, and independent Sentinel re-review are still required; no earlier report-only review is treated as final evidence.

| Finding | Severity | State | Evidence |
|---|---:|---|---|
| SENTINEL-001 | HIGH | FIXED_PENDING_REVIEW | PR dossier and convergence reports refreshed; exact final PR head still pending. |
| SENTINEL-002 | HIGH | FIXED_PENDING_REVIEW | Canonical generation/explanation contract plus changed-identity and record-order regressions pass. |
| SENTINEL-003 | HIGH | FIXED_PENDING_REVIEW | Async/sync Azure global request, attempt, token, concurrency, spend, and actual-usage checks pass. |
| SENTINEL-004 | MEDIUM | FIXED_PENDING_REVIEW | Retryable failures are excluded from cache reuse; cross-run recovery regression passes. |
| SENTINEL-005 | MEDIUM | FIXED_PENDING_REVIEW | NAACL retained/exclusion parity is enforced and contradictory-report regression passes. |

Local evidence: **346 CPU/mock-only tests passed**, impacted generation/explanation/Azure/profile suites passed, compilation passed, Ruff passed, and `git diff --check` passed. The PR remains open and unmerged. No production, Azure, Hugging Face, model-download, benchmark, TEST, process-control, or merge action occurred.

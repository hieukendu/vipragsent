# V28 Debug Convergence Bug Register

The V28 repair wave is integrated at source head `168254eb5df094924a49f0363d2403af4c87b35c`. Current-head `cpu-ci` run `31988858252` (job `95268598734`) is green. The independent Sentinel re-review is still required; no earlier report-only review is treated as final evidence.

| Finding | Severity | State | Evidence |
|---|---:|---|---|
| SENTINEL-001 | HIGH | FIXED_PENDING_REVIEW | PR dossier and convergence reports refreshed for source head `168254e`; live final PR-head verification remains pending. |
| SENTINEL-002 | HIGH | FIXED_PENDING_REVIEW | Canonical generation/explanation contract plus changed-identity and record-order regressions pass. |
| SENTINEL-003 | HIGH | FIXED_PENDING_REVIEW | Async/sync Azure global request, attempt, token, concurrency, spend, and actual-usage checks pass. |
| SENTINEL-004 | MEDIUM | FIXED_PENDING_REVIEW | Retryable failures are excluded from cache reuse; cross-run recovery regression passes. |
| SENTINEL-005 | MEDIUM | FIXED_PENDING_REVIEW | NAACL retained/exclusion parity is enforced and contradictory-report regression passes. |
| F-002 | HIGH | FIXED_PENDING_REVIEW | Sync/async Azure ceilings reject non-finite spend and non-integral integer values;  regressions pass. |
| F-003 | MEDIUM | FIXED_PENDING_REVIEW | Sync Azure cache verifies embedded key and request identity; poisoning regressions pass. |

Local evidence: **389 CPU/mock-only tests passed** and **65 Azure/cache/ceiling regressions passed**; impacted generation/explanation/Azure/profile suites passed, compilation passed, Ruff passed, and `git diff --check` passed. The PR remains open and unmerged. No production, Azure, Hugging Face, model-download, benchmark, TEST, process-control, or merge action occurred.

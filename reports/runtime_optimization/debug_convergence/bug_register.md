# V27 Debug Convergence Bug Register

The final read-only review is `PASS` for reviewed code head `23332af0bf5454958ea48b630ef43a4e45a61feb`, against base `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`. The report files are a report-only descendant of that code head; this distinction prevents report commits from being mistaken for source changes.

PR #10 is open and synchronized. Fresh `cpu-ci` run `31968069469` passed on the reviewed code head; the previous remote run (`31958375122`) failed at Ruff before the current head and is stale. The PR is ready for review and remains unmerged.

| Bug | Severity | Status | Closure |
|---|---:|---|---|
| BUG-KNOWN-001 | HIGH | CLOSED | Exact 36 local + 4 Azure Q3 profile and full authorized inventory validation. |
| BUG-KNOWN-002 | HIGH | CLOSED | Shared GPU boundary with only the validated PhoBERT exception. |
| BUG-KNOWN-003 | HIGH | CLOSED | Exact 49/S estimator factors and monotonic makespan. |
| BUG-KNOWN-004 | HIGH | CLOSED | Both local Ruff versions and fresh `cpu-ci` run `31968069469` pass. |
| BUG-REV-001 | CRITICAL | CLOSED | Legacy Q3 shape rejected; retained profile enforced. |
| BUG-REV-002 | HIGH | CLOSED | Exact Q1b producer/consumer and Q2 matrix gates. |
| BUG-REV-003 | HIGH | CLOSED | Explanation reuse requires an approved exact source. |
| BUG-REV-004 | HIGH | CLOSED | Production checkpoint identities require SHA-256. |
| BUG-REV-005 | MEDIUM | CLOSED | Persisted best-selection resume contract is fail-closed. |
| BUG-REV-006 | HIGH | CLOSED | Current code head and report-only descendant semantics are cross-file consistent. |
| BUG-REV-007 | HIGH | CLOSED | Q1b canonical provenance propagates through real fixture-backed outputs. |
| BUG-REV-008 | HIGH | CLOSED | Q1b graph/source digest binding and conflict rejection. |
| BUG-REV-009 | HIGH | CLOSED | Explicit strict Q1b training-prohibition fields. |
| BUG-REV-010 | HIGH | CLOSED | Azure Q3 seed is literal JSON null. |
| BUG-REV-011 | HIGH | CLOSED | Complete Q3 inventory validation before filtering. |
| BUG-REV-012 | MEDIUM | CLOSED | latest and best checkpoint semantics are independent. |
| BUG-REV-013 | MEDIUM | CLOSED | Predictor → retention → review and resume-history integration coverage. |
| BUG-REV-014 | HIGH | CLOSED | Missing Q1b prohibition fields block aggregation. |
| BUG-REV-015 | HIGH | CLOSED | Q1b seed types are not coerced. |
| BUG-REV-016 | HIGH | CLOSED | Null/empty/conflicting provenance is rejected. |
| BUG-REV-017 | HIGH | CLOSED | Explanation approval is complete and hash-bound. |
| BUG-REV-018 | MEDIUM | CLOSED | Non-advancing resume is rejected; metrics stay defined. |
| BUG-REV-019 | MEDIUM | CLOSED | Existing resume history is appended, not replaced. |
| BUG-REV-020 | MEDIUM | CLOSED | Direct Q1b source resolution uses the full approval validator. |
| BUG-REV-021 | HIGH | CLOSED | Final aggregation validates complete approval and state consistency. |
| BUG-REV-022 | HIGH | CLOSED | All source consumers require approved state and full approval; legacy export is rejected. |
| BUG-REV-023 | HIGH | CLOSED | Injected Q1b predictors require explicit fixture mode. |

Local evidence: **331 CPU/mock-only tests passed in 70.76s**, 55 targeted approval/source tests passed, fixture DAG and all required validators/audits passed, both Ruff versions passed, and scientific hashes are unchanged. Fresh remote `cpu-ci` run `31968069469` passed, including final runtime integration and readiness audits. No production, Azure, Hugging Face, model-download, benchmark, process-control, TEST-environment, or merge action occurred.

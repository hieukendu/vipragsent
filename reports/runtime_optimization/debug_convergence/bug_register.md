# V27 Debug Convergence Bug Register

Round 2 is recorded against Manager head `e0c502a3879fb3a65305841e6941a6bae24e5778`, base `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`, and stale remote head `c2e78c62ee51c7566629930b3b0a115920735f30`. PR #10 remains open and draft. The last remote run (`31958375122`) failed at Ruff before the new head was pushed; local Ruff 0.6.9 and 0.16.3 both pass.

| Bug | Severity | Status | Closure evidence |
|---|---:|---|---|
| BUG-KNOWN-001 | HIGH | FIXED_PENDING_FINAL_REVIEW | Q3 validates 36 local + 4 Azure rows and the complete authorized 78-row inventory. |
| BUG-KNOWN-002 | HIGH | FIXED_PENDING_FINAL_REVIEW | Shared GPU occupancy and bounded PhoBERT exception pass scheduler tests. |
| BUG-KNOWN-003 | HIGH | FIXED_PENDING_FINAL_REVIEW | Exact 49/S speedup projections and monotonic makespan pass. |
| BUG-KNOWN-004 | HIGH | FIXED_PENDING_FINAL_REVIEW | Both local Ruff versions pass; fresh remote CI is pending. |
| BUG-REV-001 | CRITICAL | FIXED_PENDING_FINAL_REVIEW | Legacy Q3 shape is rejected; exact profile slice is enforced. |
| BUG-REV-002 | HIGH | FIXED_PENDING_FINAL_REVIEW | Q1b producer/consumer and Q2 six-by-three matrix gates are enforced. |
| BUG-REV-003 | HIGH | FIXED_PENDING_FINAL_REVIEW | Direct unapproved explanation sources are rejected. |
| BUG-REV-004 | HIGH | FIXED_PENDING_FINAL_REVIEW | Production checkpoint hashes require canonical SHA-256. |
| BUG-REV-005 | MEDIUM | FIXED_PENDING_FINAL_REVIEW | Persisted best selection is validated on resume. |
| BUG-REV-006 | HIGH | FIXED_PENDING_FINAL_REVIEW | Required reports and PR text are being regenerated against the current head. |
| BUG-REV-007 | HIGH | FIXED_PENDING_FINAL_REVIEW | Q1b producer kind, IDs, key, seed, and digests flow through real artifacts. |
| BUG-REV-008 | HIGH | FIXED_PENDING_FINAL_REVIEW | Required graph/source digests and conflict rejection are implemented. |
| BUG-REV-009 | HIGH | FIXED_PENDING_FINAL_REVIEW | Boolean/integer training fields are strict and fail closed. |
| BUG-REV-010 | HIGH | FIXED_PENDING_FINAL_REVIEW | Azure Q3 seed remains JSON `null`; sentinels are rejected. |
| BUG-REV-011 | HIGH | FIXED_PENDING_FINAL_REVIEW | Rogue or duplicate full-inventory rows block aggregation. |
| BUG-REV-012 | MEDIUM | FIXED_PENDING_FINAL_REVIEW | `latest` preserves the last epoch independently of `best`. |
| BUG-REV-013 | MEDIUM | FIXED_PENDING_FINAL_REVIEW | New tests exercise canonical aggregation and latest-checkpoint paths. |

The two independent read-only review waves found BUG-REV-001 through BUG-REV-013. All code findings have fixes in the Manager branch; the final Dalton review is pending. Local evidence is green: 325 CPU/mock-only tests in 57.74 seconds, fixture DAG, schemas, registry, sequential prompts, profile validation, implementation audit, final correctness audit, compilation, diff check, and both Ruff versions. No production campaign, real Azure request, HF mutation, model download, benchmark, or process control was performed. The PR was not merged.

# ViPragSent V30 remaining-fixes evidence draft

## Binding

This is a report-only draft bound to exact source head
f438a61a078e713cfa94c5624b6b0e19b719651e. It is intentionally not a final
Sentinel result: exact-head CI and the independent affected-scope Sentinel are
pending.

The existing PR is PR #10
(https://github.com/hieukendu/vipragsent/pull/10). It remains open, unmerged,
and is the single PR for this branch.

## Frozen profile

The policy-only balanced profile retains 80 rows:

- Q3: 36 local rows and 4 seedless Azure comparison rows;
- Q2: 18 rows (six variants across three seeds);
- Q1b: 22 evaluation-only consumers.

The 54 training-applicable rows/units are the 36 local Q3 rows plus the 18
Q2 rows. Q1b has zero optimizer steps. XLM-R Q3 and budgets 64 and 256
remain excluded. The profile is default-off, execution-disabled, and
real execution is prohibited.

No V30 time credit is claimed:

- exact REUSE: 0;
- exact RESUME: 0;
- persisted V30 BLOCKED: 0;
- saved-time credit: 0 hours.

The absence of status credit is deliberate. A policy row is not a completed
run, and historical blocked rows do not become savings.

## Implementation packets

R1 adds transactional, append-delta generation-manifest reconciliation.
R2 binds explanation artifacts to a one-time physically verified source
receipt and removes repeated artifact-only checkpoint hashing. R3 uses
canonical epoch checkpoints with tiny latest/best selection pointers while
retaining legacy readers. R4 requires an exact provenance binding for reuse
and distinguishes paused resume from uncertain or conflicting work.

These implementation states are recorded as
IMPLEMENTED_PENDING_VALIDATION. This draft does not claim targeted tests,
the full CPU/mock suite, Ruff, compilation, diff checks, CI, or Sentinel
success for this exact head.

## Scientific and safety boundary

The accepted Q3/Q2/Q1b protocol, seeds, budgets, DEV selection, TEST
isolation, approval requirements, and fail-closed behavior are unchanged.
No production training, GPU workload, live Azure request, model download,
Hugging Face mutation, process-control action, or merge was performed.

## Closure state

The V28 historical failure is preserved and the V29 historical evidence is
preserved. Draft descendant-closure entries now point at the V30 source head,
with CI and Sentinel explicitly marked pending. Final-pass claims must be
written only after those checks run against the final exact head.

# ViPragSent V30 remaining-fixes evidence

## Binding

This final evidence is bound to exact source/implementation head
`79b0a925479ee5255aab6e2fa799b1867542cd10`, with implementation code at `66d4e6bd5a29e7986027afa8da045151b369235b`.
The exact-head GitHub CI run and the independent affected-scope Sentinel both
passed. The eventual closure commit is report-only and does not change the
implementation head.

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

- **R1 — PASS:** transactional, append-delta generation-manifest reconciliation
  validates candidate state before publication and leaves the in-memory store
  unchanged on corrupt external append.
- **R2 — PASS:** explanation artifacts use a one-time physically verified
  source receipt; artifact-only stages validate the receipt without rehashing
  or loading the checkpoint payload.
- **R3 — PASS:** canonical epoch checkpoints use tiny atomic latest/best
  selection pointers with exact path, epoch, SHA, sidecar, provenance, variant,
  and metric validation; legacy readers remain available.
- **R4 — PASS:** reuse requires the complete exact provenance binding, while
  paused work can resume and uncertain or conflicting work blocks.

## Validation

- Independent focused validation: **119 passed**.
- Manager focused validation: **111 passed**.
- Permitted CPU/mock suite: **453 passed**.
- `ruff check src tests`: **PASS**.
- `python -m compileall -q src`: **PASS**.
- `git diff --check`: **PASS**.
- Exact-head GitHub Actions `cpu-ci`: run
  `31999651740`, job `95297452139`, **SUCCESS** on `79b0a925479ee5255aab6e2fa799b1867542cd10`
  ([run link](https://github.com/hieukendu/vipragsent/actions/runs/31999651740)).
- Independent Sentinel: **PASS** on `79b0a925479ee5255aab6e2fa799b1867542cd10`; no closure blockers.

## Scientific and safety boundary

The accepted Q3/Q2/Q1b protocol, seeds, budgets, DEV selection, TEST
isolation, approval requirements, and fail-closed behavior are unchanged.
No production training, GPU workload, live Azure request, model download,
Hugging Face mutation, process-control action, TEST-data profiling, or merge
was performed.

## Closure state

V28's historical failure remains preserved. V29's historical P2 deferral
remains preserved, while V30 R3 implements and validates the descendant
pointer repair. V28 and V29 descendant closures are **PASS**. Overall V30
completion is **100% for the code/report validation scope**; it does not claim
that the real-model experiment has been executed.

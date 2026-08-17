# ViPragSent V29 runtime convergence evidence

## Historical V29 evidence

The V29 source implementation and its original validation remain historical:
the exact V29 source head was `9f540f3043c85cd60ea6c8706175d1bb44bcae0e`,
and its prior green run was `31993165762`. The earlier report-only retry
failure is preserved in the JSON evidence; it is not treated as a green
result.

The V29 findings P0-1 through P1-4 remain **PASS**. The historical P2
checkpoint-copy decision remains explicitly **DEFERRED** in V29.

## V30 descendant closure

V30 closes the descendant state at exact source/implementation head
`79b0a925479ee5255aab6e2fa799b1867542cd10` (implementation commit `66d4e6bd5a29e7986027afa8da045151b369235b`):

- R1 transactional manifest reconciliation: **PASS**.
- R2 one-time validated explanation source receipt: **PASS**.
- R3 canonical checkpoint files with atomic latest/best pointers: **PASS**.
- R4 exact reuse/resume/blocked classification: **PASS**.
- Scientific protocol invariants: **PASS**; no accepted cells, seeds,
  budgets, DEV selection, TEST access, or Q1b evaluation-only semantics
  changed.

Validation is exact and independent:

- focused V30 validation: **119 passed**;
- permitted CPU/mock suite: **453 passed**;
- Ruff, compileall, and diff check: **PASS**;
- exact-head GitHub Actions `cpu-ci`: run `31999651740`, job
  `95297452139`, **SUCCESS** on `79b0a925479ee5255aab6e2fa799b1867542cd10`
  ([run link](https://github.com/hieukendu/vipragsent/actions/runs/31999651740));
- independent affected-scope Sentinel: **PASS** on `79b0a925479ee5255aab6e2fa799b1867542cd10`.

The historical V29 P2 deferral is not rewritten. Its descendant resolution is
that V30 R3 implements the pointer design, with regression coverage for
selection, resume, rollback, approval, export, and freeze readers. V29
descendant closure is **PASS**.

No production training, GPU workload, live Azure request, model download,
Hugging Face mutation, TEST-data profiling, process-control action, or merge
was performed.

## Frozen protocol checks

Q3 remains 36 retained local rows plus four seedless Azure rows; XLM-R and
budgets 64/256 remain excluded. Q2 remains six variants by three seeds. Q1b
remains evaluation-only with exact source, checkpoint, seed, graph, and
approval provenance. Frozen TEST access remains unchanged.

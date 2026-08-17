# V28 Sentinel review dossier

## Historical review preserved

The historical V28 Sentinel result remains preserved exactly as a failure:
the prior review recorded `SENTINEL-001` through `SENTINEL-005`, followed by
F-002/F-003 Azure findings. The repair wave and its historical successful
source validation remain recorded in the JSON dossier.

## V30 descendant closure

The V30 descendant closes the historical failure at exact
source/implementation head `79b0a925479ee5255aab6e2fa799b1867542cd10` (implementation commit
`66d4e6bd5a29e7986027afa8da045151b369235b`):

- descendant R1–R4 implementation: **PASS**;
- focused V30 validation: **119 passed**;
- permitted CPU/mock suite: **453 passed**;
- Ruff, compileall, and diff check: **PASS**;
- exact-head GitHub Actions `cpu-ci`: run `31999651740`, job
  `95297452139`, **SUCCESS** on `79b0a925479ee5255aab6e2fa799b1867542cd10`
  ([run link](https://github.com/hieukendu/vipragsent/actions/runs/31999651740));
- independent affected-scope Sentinel: **PASS** on `79b0a925479ee5255aab6e2fa799b1867542cd10`.

The historical failure is not erased: `historical_review.status=FAIL` remains
true in the JSON. The descendant closure is **PASS** because the repaired
implementation and its exact-head evidence now satisfy the affected scope.
No production training, GPU workload, live Azure request, model download,
Hugging Face mutation, TEST-data profiling, process-control action, or merge
was performed.

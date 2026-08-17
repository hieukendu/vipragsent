# V28 ownership matrix

The final source implementation head is `64945bc0fb4f154f59062647b1c91cb9c4dfa003`; exact source-head CI
run `32001664211`/job `95303031902` is green. Reviewed report-head CI
run `32002103939`/job `95304262130` is also green. The final independent
Sentinel is PASS and PR #10 remains open and unmerged.
The implementation Workers have completed their disjoint repair scopes. Existing builder
worktrees remain preserved and are not concurrently edited.

| Task | Worker | Write scope | Read scope | Shared interface | Dependency | Runtime resource | Integration order | State |
|---|---|---|---|---|---|---|---|---|
| Production/state reconciliation | Manager / prior Worker A | baseline and live-state reports | worktrees, processes, checkpoints, environment | source identity | discovery | read-only | 1 | complete with paused-run identity blocked |
| Provenance/artifact identity | prior Worker B | artifact-reuse and provenance modules | manifests, hashes, approvals | exact identity contract | baseline | CPU/mock | 2 | source review passed |
| Generation runtime | prior Worker C | generation/checkpoint modules | tokenizer/model contracts, persistence | chunk/resume contract | provenance | CPU/mock | 3 | source review passed |
| Training runtime | prior Worker D | training/runtime contracts | optimizer/checkpoint state | resume state contract | provenance | CPU/mock | 3 | no new wave active |
| Scheduler/resource safety | prior Worker E | scheduler/estimator modules | DAG, leases, storage policy | resource/lease contract | inventory | CPU/mock | 3 | source review passed |
| Azure pipeline | prior Worker F | bounded async judge modules | schemas, retry/budget policy | request/cache contract | provenance | mocked only | 3 | no live calls |
| Protocol/inventory | prior Worker G | profile/inventory reports and validators | frozen protocol files | exact row identity | discovery | read-only | 2 | 36 local + 4 Azure validated |
| Recovery/storage | prior Worker H | persistence/recovery contracts | checkpoint and journal schemas | atomic boundary | provenance | CPU/mock | 3 | source review passed |
| Test/CI | prior Worker I / Manager | tests and audit scripts | all relevant modules | validation commands | integration | CPU/mock | 4 | exact current head green |
| Final Sentinel | Independent exact-head V30 Sentinel | `reports/v28/final_sentinel_review.*` only | whole repository and PR #10 | none | exact-head validation | read-only | 5 | PASS; historical V28 FAIL preserved |
| Generation provenance repair | GENE-V28 | generation persistence/executor modules and generation tests | generation manifests and checkpoint contracts | canonical generation identity | SENTINEL-002 | CPU/mock | 6A | integrated; regressions pass |
| Azure safety/cache repair | AZURE-V28 | Azure client/async judge/stage registry and Azure tests | request/retry/budget contracts | global Azure ceiling contract | SENTINEL-003/004 | CPU/mock | 6B | integrated; regressions pass |
| Profile parity repair | PROFILE-V28 | NAACL profile validator/report/profile tests | Q2/Q3/YAML/JSON policy | profile exclusion contract | SENTINEL-005 | CPU/mock | 6C | integrated; regressions pass |
| Exact-head evidence repair | Manager | V28 reports, convergence register, PR body | Sentinel findings, CI, PR metadata | exact-head evidence binding | SENTINEL-001 | read-only/CPU | 7 | final source/report evidence bound to V30 auxiliary correction at 64945bc0fb4f154f59062647b1c91cb9c4dfa003 |

Workers may not write the same physical file concurrently. Any new finding
will receive one explicit owner and a disjoint write scope before activation.

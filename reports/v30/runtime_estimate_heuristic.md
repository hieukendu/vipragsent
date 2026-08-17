# V30 heuristic runtime estimate draft

## Scope and evidence boundary

This is an analytical engineering estimate, not a measured runtime proof. It
is bound to source head
f438a61a078e713cfa94c5624b6b0e19b719651e; exact-head CI and the independent
Sentinel review are still pending.

The frozen balanced profile contains 80 retained rows: 36 local Q3 rows, 4
seedless Azure Q3 comparison rows, 18 Q2 rows, and 22 Q1b evaluation-only
consumers. The estimate treats 54 rows/units as training-applicable and 40
Q3 rows as generation rows. Q1b has zero optimizer steps.

No exact V30 REUSE, RESUME, or persisted BLOCKED status is present. All three
counts are zero and receive zero saved-time credit.

## Historical anchor

The only timing anchor is the repository's historical approximately 49-hour
run for a 1,999-example Vistral DEV split. It is used as a rough proxy for
the 24 retained local 7B Q3 cells only. It is not a measurement of this
source head, and no new benchmark or profiling workload was run.

## Model

The pre-optimization proxy is:

- 24 local 7B Q3 cells × 49 hours = 1,176 hours;
- 30 smaller training-applicable units × 2 hours = 60 hours;
- 4 Azure rows plus 22 Q1b evaluation consumers × 2 hours = 52 hours;
- serial gates and QA = 48 hours.

That yields 1,336 hours, or 55.7 days, before modeled optimization effects.

The central case models 366 hours of conditional savings:

- generation batching and removal of per-record launch overhead: 300 hours;
- removal of historical persistence rereads: 40 hours;
- repeated checkpoint hashing: 1 hour;
- duplicate large latest/best checkpoint copies: 1 hour;
- exact artifact reuse/resume: 0 hours;
- safe independent CPU/I/O/smaller-work overlap: 24 hours.

The resulting central estimate is 970 hours, or 40.4 days. The batching
factor is deliberately below theoretical linear scaling; padding, decoding,
memory pressure, and scheduler overhead remain.

The conservative estimate is 1,523 hours, or 63.5 days. It gives no exact
reuse/resume credit, does not assume concurrent 7B jobs on one constrained
allocation, and retains a campaign-level contingency for queueing, serial
gates, and unresolved throughput variability.

| Estimate | Hours | Days |
| --- | ---: | ---: |
| Before optimization proxy | 1,336 | 55.7 |
| Central post-optimization heuristic | 970 | 40.4 |
| Conservative post-optimization heuristic | 1,523 | 63.5 |

The central estimate does not defensibly fit the requested 30-day target.
The report therefore records the minimum credible modeled result and the
remaining bottleneck rather than inventing a <=30-day claim. Further reduction
would require changing the accepted scientific cells, seeds, provenance,
selection, or TEST semantics, or obtaining authorized throughput evidence.

## Critical path and limitations

The dominant path is the 24 retained local 7B Q3 training/generation cells,
followed by DEV judging/persistence, DEV selection and freeze, and later TEST
evaluation/export. Q1b remains evaluation-only and cannot be treated as
training savings.

The 4-5 GB checkpoint and 100-200 MB/s storage assumptions are used only to
bound I/O/hash/copy savings. They do not imply measured device throughput.
No live Azure request, production run, model download, or real-model benchmark
was performed.

## Validation state

This is a draft bound to the exact source head above. Exact-head CI and the
independent affected-scope Sentinel are PENDING; the final report must be
rewritten if either the source head or evidence head changes.

# Runtime estimate after optimization

As of `2026-08-16T19:02:21Z`, against reviewed code head `16181334b5d00a6e3f622dffa826575a1b18915d`, the strongest defensible result is `PROJECTED_GATE_CONDITIONAL`, not a measured campaign duration. The live run has no active PID, the production worktree is dirty, loaded-code identity is uncertain, telemetry is partial, and the code-only phase forbids real production-model/data benchmarks and real Azure calls.

The only timing evidence is the last-known historical reference of approximately **49 hours per 1,999-example Vistral DEV split**. The sensitivity table below is therefore a per-split planning reference, not a measured speedup or a total-experiment estimate.

| Vistral generation speedup | Reference time per 1,999-example DEV split | Evidence |
|---:|---:|---|
| 1.0× | 49.0000 h | last-known historical reference |
| 1.5× | 32.6667 h | projected only |
| 2.0× | 24.5000 h | projected only |
| 2.5× | 19.6000 h | projected only |
| 3.0× | 16.3333 h | projected only |
| 4.0× | 12.2500 h | projected only |

The total remaining wall-clock is intentionally **not given a fabricated number**. Exact stage counts, current completion state beyond the observed epoch-2 boundary, Azure throughput, storage contention, and source identity are not jointly authoritative. The new estimator can calculate lower-bound and scheduler-policy makespan once those values are injected; its fixture tests do not become production throughput evidence.

No numeric savings are credited yet. The reconciliation is therefore:

```text
unknown baseline remaining
- 0 proven reuse/elimination credit
- 0 measured speedup credit
- 0 unmeasured overlap credit
+ known future profile/identity/storage overhead
= conditional remaining estimate
```

The current CPU/mock-only validation is 331 passed in 70.76 seconds; it is correctness evidence, not campaign throughput evidence. Central and conservative planning remain conditional on an authorized DEV-only Vistral profile, bounded Azure transport accounting, exact remaining inventory, and identity reconciliation. PhoBERT concurrency remains 1 by default; concurrency 2 is only a sensitivity case after a validated aggregate-throughput gain of at least 25%. This task does not claim `MEASURED_GATE_PASS`.

The exact JSON fields, source hashes, Task-A conservative classifications, environment fingerprint, live-state credit, and optimization IDs are in [`runtime_estimate_before.json`](runtime_estimate_before.json) and [`runtime_estimate_after.json`](runtime_estimate_after.json).

# V30 runtime estimate — corrected topology draft

## Required classification

**HEURISTIC ENGINEERING ESTIMATE — NOT A MEASURED RUNTIME PROOF**

This estimate is bound to source implementation head `64945bc0fb4f154f59062647b1c91cb9c4dfa003`. It uses
the frozen execution topology, registry, dataset counts, locked training
configuration, model family, executor kind, optimizer steps, and explicit
unit-cost assumptions. No runtime proof, GPU benchmark, TEST access, live
Azure call, model download, or Hugging Face mutation was performed.

## Methodology

```
estimated wall clock
= sum(topology-specific workload-unit costs)
  + unavoidable serial overhead
  - exact reuse/resume credit
  - eliminated redundant I/O
  - safe dependency overlap
```

Exact REUSE, RESUME, and persisted BLOCKED counts are zero, so no time credit
is claimed. The prior approximately 49-hour anchor is not transferable to
retained Q3 classification/multitask training and is excluded from the
arithmetic.

## Workload breakdown

| Scope | Topology | Units | Step topology | Central h | Conservative h |
| --- | --- | ---: | --- | ---: | ---: |
| Q3 PhoBERT | single-model classification | 12 cells | 2,500 optimizer steps/cell | 12.0 | 18.0 |
| Q3 Vistral pragmatic SFT | 7B classification SFT | 12 cells | 1,500 steps/cell | 96.0 | 144.0 |
| Q3 full ViPragSent Vistral | 7B multitask + rationale training; inference off | 12 cells | 1,500 steps/cell | 132.0 | 192.0 |
| Q3 Azure | external 8-shot comparison | 4 rows | no optimizer | 3.0 external | 8.0 external |
| Q2 single joint | five variants × three seeds | 15 cells | 2,500 steps/cell | 15.0 | 22.5 |
| Q2 no_multitask | eight independent components × three seeds | 24 units | 2,500 steps/component | 18.0 | 30.0 |
| Q1b seeded consumers | evaluation-only | 21 consumers | optimizer_steps = 0 | 3.15 | 6.3 |
| Q1b Azure output | external approved output | 1 consumer | no optimizer | 0.5 external | 1.0 external |
| serial orchestration/QA | preflight, checkpoints, DEV freeze, TEST/export | one campaign | dependency-bound | 24.0 | 48.0 |

Raw central total is **303.65 hours**. Raw conservative total is
**469.8 hours**. The corrected R5 pointer/hash work removes an estimated
1 hour of redundant hashing and 1 hour of redundant checkpoint-copy work in
the campaign model; safe dependency overlap credits 18 central hours. A
separate 180-hour conservative contingency covers queueing, throughput
uncertainty, and incomplete telemetry.

## Result

| Estimate | Hours | Days |
| --- | ---: | ---: |
| Corrected central | 283.65 | **11.82** |
| Corrected conservative | 647.8 | **26.99** |

Local GPU server contribution is 276.15 central hours / 412.8 conservative
hours (**11.51 / 17.20 days**). Azure/external contribution is 3.5 / 9.0
hours (**0.15 / 0.38 days**) and is not placed on the local GPU critical path.

The critical path is the 24 retained local Vistral 7B classification/multitask
training cells, followed by DEV selection/freeze and downstream TEST. Q1b is
evaluation-only. No retained V30 Q3 row is autoregressive reasoning
generation, so the historical reasoning-generation anchor is excluded.

## Assumptions and limitations

- Q3/Q2 counts use 7,998 training records, locked epochs, effective batches,
  and gradient accumulation from `configs/runtime/training.yaml`.
- PhoBERT and Vistral unit costs are planning assumptions, not measured
  throughput.
- `no_multitask` is counted as eight independent component-training units per
  seed, not one cell.
- Azure rows are external latency only; they are not local 7B GPU training.
- Exact reuse/resume/blocked credit is zero.
- Central and conservative values must be recalculated when authorized
  throughput or queue telemetry exists.

## Validation state

R5 focused tests are **11 passed**. Source-head CI for `64945bc0fb4f154f59062647b1c91cb9c4dfa003` and the
fresh independent Sentinel are **PENDING**. Parent-head CI is historical.

# ViPragSent V30 final convergence evidence

## Binding

This final dossier is bound to source implementation head
`64945bc0fb4f154f59062647b1c91cb9c4dfa003`. The previous parent report head
was `3334d98575f373eaa12f64cbf336748f531ccc66`; its CI is historical for this
auxiliary source change. V30 R1-R5 are PASS.

The reviewed report evidence head is
`3de2d5c1c9a34dbe7030f86a65cf22978751048`, with green source-head CI
`32001664211`/job `95303031902` and report-head CI
`32002103939`/job `95304262130`. The final PR head is a report-only descendant;
the source implementation SHA remains the exact code-validation binding.

PR #10 remains the single open, unmerged PR:
https://github.com/hieukendu/vipragsent/pull/10.

## R5 checkpoint-hash correction

The save-sidecar verification, canonical pointer validation, and pointer-based
load path now carry an observed checkpoint SHA-256 downstream instead of
rehashing the same multi-GB payload. The physical hash still occurs when the
boundary has no observed digest, and pointer, sidecar, provenance, epoch,
variant, metric, path, and corrupt-content checks remain fail-closed.

Focused regressions: **11 passed**, including exact one-hash counts for save,
pointer publication, and pointer-based load. The final focused regression set
passed **172 tests**; Ruff, compileall, diff checks, and the permitted CPU/mock
suite also pass.

## Corrected ETA scope

The prior draft's broad 7B reasoning-generation anchor is not used as a
multiplier for retained V30 Q3 cells. The retained Q3 topology is:

- 12 `phobert_pragmatic_finetune` classification cells;
- 12 `vistral_pragmatic_sft` classification SFT cells;
- 12 `vipragsent_full_vistral` multitask+rationale-training cells with
  `rationale_inference=false`;
- 4 Azure comparison rows, external latency only.

Q2 is split into 15 single-joint PhoBERT cells and 24 independent
`no_multitask` component units. Q1b remains evaluation-only with
`optimizer_steps=0); no Q1b training cost is included.

The corrected heuristic report is
`reports/v30/runtime_estimate_heuristic.json/.md`. It is explicitly a
**HEURISTIC ENGINEERING ESTIMATE — NOT A MEASURED RUNTIME PROOF**:

- central: **11.82 days**;
- conservative: **26.99 days**;
- local GPU server contribution: 11.51 central / 17.20 conservative days;
- Azure/external contribution: 0.15 central / 0.38 conservative days;
- under-30-day: **true under the stated planning assumptions**.

The historical approximately 49-hour Vistral autoregressive reasoning DEV
anchor is recorded but excluded: no retained V30 Q3 row uses
`rationale_inference=true` or the `generation_trainable` executor.

## Invariants and safety

Q3/Q2/Q1b rows, seeds, budgets, DEV selection, TEST isolation, approval
requirements, and Q1b evaluation-only semantics are unchanged. No production
training, GPU benchmark/profiling, TEST access, live Azure request, model
download, Hugging Face mutation, process-control action, or merge occurred.

## Final validation

An independent exact-head Sentinel found no actionable blocker across R1-R5,
the corrected ETA, protocol invariants, tests/static/CI evidence, V28/V29
closure, or PR state. PR #10 remains the single open, unmerged PR and has not
been merged.

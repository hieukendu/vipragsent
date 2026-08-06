
# Specification Completeness Audit

The setup is complete only if every item below is explicitly represented in code, configuration, tests,
the master DAG, and output schemas.

## Dataset and splits

- [ ] ViPragSent frozen V8 split validated.
- [ ] External datasets downloaded, normalized, and checksum-locked.
- [ ] No deduplication or near-overlap filtering.
- [ ] Human IAA recomputed before adjudication.

## Paper roles

- [ ] Table 2 uses full Vistral ViPragSent.
- [ ] Table 3 uses full PhoBERT ViPragSent.
- [ ] Table 4 uses the PhoBERT controlled-ablation family.
- [ ] Backbone-sensitivity comparison is explicit.

## Table 3

- [ ] `phobert_pol_single` exists.
- [ ] `phobert_emo_single` exists.
- [ ] Every system uses the correct polarity/emotion output.
- [ ] No external fine-tuning.
- [ ] Ord.F1 is the unweighted mean of three external Macro-F1 scores.

## Model variants

- [ ] Main full model uses classification heads at inference and drops the decoder.
- [ ] CoT-only trains generation only and reports parsed generated labels.
- [ ] No separate explanation-at-inference system exists; full and explanation-only variants drop the decoder and report classification-head outputs.
- [ ] Invalid generations are reported and scored without semantic guessing.

## Q2

- [ ] Six independent pragmatic PhoBERT checkpoints implement `no_multitask`.
- [ ] Ordinary single-task checkpoints provide ordinary metrics for that bundle.
- [ ] Pragmatic F1, ordinary F1, ECE, and cost definitions are locked.
- [ ] Relative cost denominator is full ViPragSent PhoBERT.

## Q3

- [ ] Frozen nested masks are used.
- [ ] All train negatives are fixed.
- [ ] Out-of-budget positives mask sarcasm and rationale only.
- [ ] Other tasks remain active.
- [ ] `pos_weight` is recomputed per budget.
- [ ] Dev/test stay fixed.
- [ ] Budget 512 is removed only if full positives are below 512.

## Architecture and optimization

- [ ] Pooling is locked by backbone family.
- [ ] Decoder cross-attends full token-level hidden states.
- [ ] Decoder dimensions and target lengths are locked.
- [ ] Optimizers, schedulers, warmup, weight decay, precision, and checkpoint tie-breaks are locked.

## Azure

- [ ] Exactly eight demonstrations with the required composition.
- [ ] Q3 demonstration eligibility is validated per budget.
- [ ] GPT-4.1-mini Azure deployment and version are recorded.
- [ ] No direct OpenAI endpoint.

## Q4 and statistics

- [ ] Q4 uses PhoBERT, Vistral SFT, and full ViPragSent Vistral.
- [ ] ECE uses intended-polarity, not a removed pragmatic-polarity head.
- [ ] Required paired comparisons are computed.
- [ ] Holm correction is applied within each comparison family.

## Cost, analysis, and artifacts

- [ ] Cost and latency measurement protocol is implemented.
- [ ] Exact table and backing-data schemas validate.
- [ ] Backbone-sensitivity artifact exists.
- [ ] Error-analysis candidates and templates exist.
- [ ] Qualitative candidates and templates exist.
- [ ] No old Figure 5 artifact exists.
- [ ] Every final figure has backing data.
- [ ] Every final table cell is traceable to source results.

The full-run preflight and final reproducibility audit must programmatically evaluate this checklist.

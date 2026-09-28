# Q1a matched XLM-R-large baseline inspection

Date: 2026-09-28

## Scope and decision

The existing Q1a XLM-R-large baseline is sufficiently identified to run a new
controlled matched experiment. The new experiment keeps the existing baseline
model semantics and changes only the locked optimization recipe requested for
the matched comparison. It uses the dedicated experiment name
`xlmr_large_ft_q1a_matched` and does not overwrite the existing
`q1a_xlmr_pragmatic_finetune_*` artifacts.

The inspection sources are the current implementation under `src/`, the
execution registry, the existing per-seed remote artifact audit under
`reports/hf_vipragsent_remote_audit_2026-09-15/`, and the current Q1a summary
under `reports/q1a_best_f1_table.*`.

## Existing baseline implementation

- **System and variant:** `xlmr_pragmatic_finetune`, resolved from
  `configs/experiments/system_execution_registry.yaml`.
- **Backbone:** `FacebookAI/xlm-roberta-large` at revision
  `c23d21b0620b635a76227c604d44e43a9f0ee389`; production loading uses the
  pinned local snapshot with `AutoModel.from_pretrained(...,
  add_pooling_layer=False, local_files_only=True)`.
- **Architecture:** XLM-R encoder followed by the shared `ClassificationHeads`
  module. The encoder output is pooled from the first attended token by the
  locked `encoder` pooling path. A dropout layer with probability `0.1` is
  applied in the classification heads before each linear classifier.
- **Heads:** exactly six independent binary pragmatic heads:
  `implicit_sentiment`, `sarcasm`, `irony`, `idiom_figurative`,
  `code_switching`, and `mocking`. Each head is `Linear(hidden_size, 1)`.
  The baseline has no polarity head, emotion head, rationale decoder,
  uncertainty parameters, auxiliary loss, or proposed-only component.
- **Loss:** binary cross entropy with logits for the six active pragmatic
  heads, combined by equal-weight sum/mean through the generic training
  engine. Positive weights are the train-only negative-to-positive ratio for
  each pragmatic label (`N_negative / N_positive`). Polarity and emotion
  weights may be serialized by the shared data utility, but they are not
  consumed by this six-head variant.
- **Data and preprocessing:** frozen ViPragSent train/dev/test splits, NFC
  normalization for XLM-R, tokenizer/model max length `128`, and the pinned
  tokenizer revision. The loader preserves the sample IDs and does not alter
  split membership.
- **Thresholds:** after each development evaluation, each pragmatic threshold
  is searched on `0.05, 0.06, ..., 0.95`; ties are ordered by distance to
  `0.5`, then by threshold value. The final thresholds are frozen from the
  selected development checkpoint before test evaluation.
- **Selection and test gate:** primary selection is highest
  `dev_macro_pragmatic_f1`. The training engine saves a best checkpoint on
  that development metric and exposes a gate that prohibits test evaluation
  before the checkpoint and thresholds are frozen. Prediction JSONL rows carry
  `sample_id`, six `gold` labels, six `probabilities`, six `predictions`, and
  six `logits`.

## Exact old run that produced the current row

The current Table-3-style summary row is the `XLM-R-large fine-tune` row in
`reports/q1a_best_f1_table.md` and `reports/q1a_best_f1_table.csv`. Its source
IDs are:

```text
q1a_xlmr_pragmatic_finetune_20260521
q1a_xlmr_pragmatic_finetune_20260522
q1a_xlmr_pragmatic_finetune_20260523
```

The row reports the following mean test binary macro-F1 percentages in the
canonical pragmatic-label order, followed by macro pragmatic F1:

```text
implicit_sentiment 92.163676
sarcasm            88.123035
irony              97.929522
idiom_figurative   92.850118
code_switching     94.265129
mocking            92.274024
macro_pragmatic    92.934251
```

The remote-audit artifact for seed `20260521` records best epoch `10`, best
development macro pragmatic F1 `0.928512759521461`, frozen thresholds
`{code_switching: 0.95, idiom_figurative: 0.93, implicit_sentiment: 0.90,
irony: 0.94, mocking: 0.89, sarcasm: 0.95}`, and 2,000 test predictions.
The old resolved artifact configuration records AdamW at `2e-5`, weight decay
`0.01`, physical batch `8`, accumulation `4`, effective batch `32`, bf16,
gradient clipping `1.0`, ten maximum epochs, linear scheduling, warmup ratio
`0.10`, and patience `2`. Those old scheduler and stopping values are
documented here as provenance; the matched experiment resolves the requested
proposed recipe below.

## Matched recipe to be used

The new runner keeps all baseline architecture, loss, pooling, threshold, and
selection semantics above and resolves these values before each seed starts:

| Field | Resolved value |
|---|---|
| model | `FacebookAI/xlm-roberta-large` |
| max sequence length | `128` |
| optimizer | AdamW |
| learning rate | `2e-5` |
| weight decay | `0.01` |
| maximum epochs | `10` |
| precision | bf16 |
| physical batch | `8` |
| gradient accumulation | `4` |
| effective batch | `32` |
| gradient clipping | `1.0` |
| scheduler | cosine |
| warmup ratio | `0.20` |
| patience | `10` |
| checkpoint selection | highest dev macro pragmatic F1 |
| threshold tuning | dev only, `0.05`–`0.95`, step `0.01`, nearest `0.5` tie rule |
| positive class weights | train-only `N_negative / N_positive` per pragmatic head |
| seeds | `20260521`, `20260522`, `20260523` |

No proposed model retraining, other Q1a model reruns, Q2–Q4 reruns, split
changes, label cooccurrence recomputation, or paper-file edits are part of
this baseline.

## Artifact and HF destination

The existing project uploader maps the XLM-R checkpoint family to
`Thundergod2007/vipragsent-xlmr-checkpoints`. The existing artifact uploader
uses the project’s established `Thundergod2007/vipragsent-experiment-artifacts`
family and its overflow allocation for run artifacts. Each new seed run will
contain only its final selected checkpoint under `checkpoints/best/model.pt`
plus reproducibility artifacts; intermediate engine checkpoints remain under
the uploader-excluded `_engine_checkpoints` directory. The seed is uploaded
and remotely verified before the next seed starts.

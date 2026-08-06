
# Paper Experiment Role Registry

This file is the source of truth for mapping experimental checkpoints to paper-facing results.

## Table 2 — headline pragmatic results

```yaml
model: vipragsent_full
backbone: vistral_7b
checkpoint_family: vipragsent_full_vistral
seeds: [20260521, 20260522, 20260523]
```

## Table 3 — cross-domain ordinary retention

```yaml
vipragsent_row:
  model: vipragsent_full
  backbone: phobert_base
  checkpoint: vipragsent_full_phobert

ordinary_single_task_row:
  polarity_checkpoint: phobert_pol_single
  emotion_checkpoint: phobert_emo_single
```

Every Table 3 system is trained on ViPragSent only and evaluated directly on the frozen external test sets.

## Table 4 — controlled ablation

```yaml
anchor:
  model: vipragsent_full
  backbone: phobert_base
  checkpoint: vipragsent_full_phobert
```

Every ablation must use PhoBERT and change only the named component.

## Q4 calibration

Compare:

- PhoBERT fine-tune;
- Vistral-7B SFT;
- full ViPragSent Vistral.

## Backbone sensitivity

Compare full ViPragSent PhoBERT with full ViPragSent Vistral using the same split, tasks, rationales,
seeds, metric definitions, and reporting code.

## Removed component

The old six-class pragmatic-polarity head and old Figure 5 are removed completely.
Do not train the head, compute the confusion matrix, export the figure, or preserve claims that depend on it.


# TABLE 2 VARIANT ROLES

```yaml
variants:
  no_auxiliary:
    checkpoint: vistral_pragmatic_sft
    inference: classification_heads

  cot_only:
    checkpoint: cot_only_vistral
    inference: parsed_generated_labels

  explanation_only:
    checkpoint: vistral_explanation_only
    training_heads: six_pragmatic_plus_rationale
    inference: classification_heads
    rationale_decoder_at_inference: false

  full:
    checkpoint: vipragsent_full_vistral
    training_heads: eight_classification_plus_rationale
    inference: classification_heads
    rationale_decoder_at_inference: false
```

There is no separate explanation-at-inference system.

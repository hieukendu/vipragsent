> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 04 — DEFINE ALL SCHEMAS AND CONFIGURATIONS

Freeze the complete experiment contract before implementing the core pipeline.

Create and validate dataset schemas, label maps, prediction schema, run-metadata schema, result schema, Azure request/response schemas, rationale schema, artifact schema, Q1–Q4 configs, seed/loss/threshold/calibration/statistics configs, an A100 20 GB runtime profile, and `configs/master_run.yaml`.

Use the following locked repositories, then resolve and pin their immutable Hugging Face commit revisions
without downloading weights:

```yaml
phobert_base:
  repo_id: vinai/phobert-base

xlmr_large:
  repo_id: FacebookAI/xlm-roberta-large

sailor_7b:
  repo_id: sail/Sailor-7B

vistral_7b:
  repo_id: Viet-Mistral/Vistral-7B-Chat
```

Record the resolved commit SHA, tokenizer revision, license, architecture, vocabulary size, and gated-access status.
If any repository becomes unavailable or incompatible, stop with a blocker rather than substituting another model.

Acceptance criteria: every schema and config validates, no protocol field is unresolved, and each model has an exact ID/revision or an explicit blocker.

# LOCKED EXPERIMENT CONFIGURATION ADDENDUM

This section overrides generic defaults.

## Paper roles

Create and validate `configs/paper_roles.yaml` using the exact registry in
`28_PAPER_EXPERIMENT_ROLE_REGISTRY.md`.

## Table 3 checkpoint matrix

Create `configs/experiments/q1b/checkpoint_matrix.yaml` with these roles:

```yaml
systems:
  phobert_ordinary_single_task:
    polarity_checkpoint: phobert_pol_single
    emotion_checkpoint: phobert_emo_single

  phobert_multitask:
    checkpoint: phobert_multitask_8head
    polarity_output: polarity_head
    emotion_output: emotion_head

  xlmr_multitask:
    checkpoint: xlmr_multitask_8head
    polarity_output: polarity_head
    emotion_output: emotion_head

  sailor_multitask:
    checkpoint: sailor_multitask_8head
    polarity_output: polarity_head
    emotion_output: emotion_head

  vistral_multitask:
    checkpoint: vistral_multitask_8head
    polarity_output: polarity_head
    emotion_output: emotion_head

  vipragsent:
    checkpoint: vipragsent_full_phobert
    polarity_output: polarity_head
    emotion_output: emotion_head

  azure_gpt41_mini:
    polarity_output: dedicated_polarity_prompt
    emotion_output: dedicated_emotion_prompt
```

`phobert_pol_single` optimizes only the intended-polarity loss.
`phobert_emo_single` optimizes only the emotion loss.
Every local Table 3 checkpoint is trained on ViPragSent train only.
No external benchmark examples may enter model training, checkpoint selection, prompt selection, or threshold tuning.

## Q2/Table 4 reporting contract

```yaml
table_4_metrics:
  pragmatic_f1:
    split: vipragsent_dev
    definition: arithmetic_mean_of_six_binary_macro_f1

  ordinary_f1:
    split:
      - uit_vsfc_test
      - uit_vsmec_test
      - aivivn_human_derived_3way_test
    definition: unweighted_mean_of_three_external_macro_f1_scores

  ece:
    split: vipragsent_dev
    head: intended_polarity_3way
    definition: top_label_ece_10_equal_width_bins

  relative_cost:
    denominator: vipragsent_full_phobert
    definition: measured_total_gpu_hours_variant_divided_by_measured_total_gpu_hours_full_phobert
```

For the `no_multitask` row:

- Pragmatic F1 comes from six independent PhoBERT pragmatic checkpoints and is aggregated across the six labels.
- Ordinary F1 and ECE come from the separate `phobert_pol_single` and `phobert_emo_single` checkpoints.
- Relative cost is the summed measured GPU-hours of the complete independent-checkpoint bundle required
  to produce that row, divided by the measured GPU-hours of `vipragsent_full_phobert`.

## Q3 locked protocol

```yaml
q3:
  source_of_truth: bundled_budget_mask_files
  budgets: [32, 64, 128, 256, 512, full]
  positive_subsets: nested
  negative_pool: all_train_sarcasm_negatives_fixed_across_budgets
  out_of_budget_positive:
    sarcasm_target_mask: 0
    rationale_loss_mask: 0
    other_task_targets: active
  pos_weight:
    recompute_per_budget: true
    formula: active_negative_count / selected_positive_count
  dev_test: fixed
  primary_metric: sarcasm_binary_macro_f1
  early_stopping_metric: dev_sarcasm_binary_macro_f1
  threshold: tune_on_dev_per_seed_and_budget
  if_full_positive_count_below_512: remove_512_budget
```

Do not regenerate the frozen budget masks unless validation proves they are corrupt.

## Exact artifact schemas

Validate all filenames and columns in `27_OUTPUT_ARTIFACT_SCHEMA.md`.

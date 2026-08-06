# Output Artifact Schema

## Per-run artifacts

- configuration snapshot;
- seed;
- model ID and revision;
- data fingerprint;
- Azure deployment and prompt metadata where applicable;
- checkpoint;
- predictions;
- logits and probabilities;
- thresholds;
- training histories;
- metrics;
- runtime, VRAM, and API cost;
- status and error information.

## Final tables

- Q1 pragmatic results.
- External retention.
- Q2 ablations.
- Q3 low-resource results.
- Q4 calibration.
- Significance.
- Cost and latency.

## Final figures

- Per-phenomenon F1.
- Multi-task gain.
- Q3 low-resource curve.
- Dev-set learning curves.
- Reliability diagrams.

## Final provenance

`results/result_provenance_index.csv` must map every final artifact to source files, scripts, hashes, and model or Azure deployment metadata.

Do not create paper or manuscript outputs.

# LOCKED FILENAMES AND COLUMNS

## Table 2 / Q1a

File:

```text
experiment_artifacts/tables/table_2_pragmatic.csv
```

Required columns:

```text
system,backbone,seed_count,
implicit_f1,implicit_ci_low,implicit_ci_high,
sarcasm_f1,sarcasm_ci_low,sarcasm_ci_high,
irony_f1,irony_ci_low,irony_ci_high,
idiom_f1,idiom_ci_low,idiom_ci_high,
code_switching_f1,code_switching_ci_low,code_switching_ci_high,
mocking_f1,mocking_ci_low,mocking_ci_high,
macro_prag_f1,macro_prag_ci_low,macro_prag_ci_high,
invalid_output_rate
```

## Table 3 / Q1b

File:

```text
experiment_artifacts/tables/table_3_external_retention.csv
```

Required columns:

```text
system,polarity_checkpoint,emotion_checkpoint,
vsfc_macro_f1,vsmec_macro_f1,aivivn_macro_f1,ord_f1,
seed_count,training_data,external_finetuning
```

`external_finetuning` must be `false` for every row.

## Table 4 / Q2

File:

```text
experiment_artifacts/tables/table_4_ablation.csv
```

Required columns:

```text
configuration,backbone,
prag_dev_f1,ord_external_f1,polarity_dev_ece,
gpu_hours,relative_cost_to_full_phobert,
seed_count,changed_components
```

## Q3 backing data

File:

```text
experiment_artifacts/backing_data/q3_low_resource.csv
```

Required columns:

```text
system,budget,selected_positive_count,fixed_negative_count,
seed,sarcasm_dev_f1,sarcasm_test_f1,
dev_threshold,pos_weight,data_hash,mask_hash
```

## Q4 calibration

File:

```text
experiment_artifacts/tables/q4_calibration.csv
```

Required columns:

```text
system,backbone,polarity_test_ece,
bin_count,binning,confidence_definition,
temperature_scaling,seed_count
```

## Significance

File:

```text
experiment_artifacts/tables/significance.csv
```

Required columns:

```text
comparison,metric,observed_delta,ci_low,ci_high,
raw_p_value,holm_adjusted_p_value,resamples,
bootstrap_seed,prediction_files
```

## Cost and latency

Files:

```text
experiment_artifacts/tables/cost_latency.csv
experiment_artifacts/backing_data/latency_measurements.csv
```

Required cost/latency methodology:

- GPU training cost: measured successful-run GPU-hours; failed/retried runs reported separately.
- Table 4 relative cost denominator: `vipragsent_full_phobert`.
- Batch-1 online latency and batch-32 throughput.
- Exclude model-load time.
- Use 50 warm-up iterations.
- Measure at least 500 examples, three repetitions.
- Call `torch.cuda.synchronize()` around GPU timing.
- Report mean, median, p95, peak VRAM, GPU model, and MIG profile.
- Azure reports actual request count, input tokens, output tokens, response latency, and actual Azure cost
  under the deployed pricing configuration.
- Rationale-generation API cost and prompted-baseline API cost must be separate.

## Backbone sensitivity

File:

```text
experiment_artifacts/tables/backbone_sensitivity.csv
```

Required columns:

```text
system,backbone,macro_prag_f1,ord_f1,polarity_ece,
gpu_hours,relative_cost,peak_vram_gb,
batch1_latency_ms,batch32_examples_per_second,seed_count
```

## Qualitative and manual analysis

Files:

```text
experiment_artifacts/manual/error_analysis_candidates.csv
experiment_artifacts/manual/error_analysis_final.csv
experiment_artifacts/manual/qualitative_candidates.jsonl
experiment_artifacts/manual/qualitative_final.jsonl
```

No old Figure 5 artifact is allowed.


# ADDITIONAL REQUIRED PROVENANCE FIELDS

Every local-model run manifest must include:

```text
preprocessing_name
preprocessing_version
tokenizer_revision
model_revision
physical_batch_size
gradient_accumulation_steps
effective_batch_size
inference_output_source
rationale_decoder_enabled_at_inference
```

Valid `inference_output_source` values:

- `classification_heads`
- `parsed_generated_labels`

Full and explanation-only models must use `classification_heads` and set
`rationale_decoder_enabled_at_inference=false`.
CoT-only must use `parsed_generated_labels`.


# DATASET SUMMARY SCHEMAS

## Dataset summary

File:

```text
experiment_artifacts/tables/table_1_dataset_summary.csv
```

Required columns:

```text
dataset,role,train_count,dev_count,test_count,total_count,
task,label_space,source_manifest,checksum,redistribution_status
```

## ViPragSent label distribution

File:

```text
experiment_artifacts/tables/vipragsent_label_distribution.csv
```

Required columns:

```text
split,label_group,label,count,total,rate
```

## Human IAA summary

File:

```text
experiment_artifacts/tables/human_iaa_summary.csv
```

Required columns:

```text
field,n,raw_agreement,cohen_kappa,krippendorff_alpha_nominal,disagreement_count
```

> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 07 — IMPLEMENT THE TRAINING ENGINE

Implement a configuration-driven, resume-safe, test-safe training and evaluation engine.

Required capabilities: optimizer/scheduler, mixed precision, gradient accumulation, gradient clipping, checkpointing, early stopping on dev only, threshold tuning on dev, frozen test evaluation, prediction/logit export, run hashes, safe resume, failure recovery, epoch-level histories, and runtime/VRAM/cost hooks.

Fixed hyperparameters:

```text
PhoBERT/XLM-R: learning rate 2e-5, effective batch 32, max 10 epochs.
Sailor/Vistral QLoRA: learning rate 1e-4, effective batch 16, 3 epochs.
Maximum length: 128. Gradient clipping: 1.0.
```

Use dummy/tiny fixtures to test interrupted-run resume, test-set access restrictions before checkpoint freeze, prediction-schema validation, and safe handling of completed runs.

Do not download real weights or run full experiments.

# LOCKED OPTIMIZATION AND CHECKPOINT-SELECTION DETAILS

This section overrides generic optimization wording.

## PhoBERT and XLM-R

```yaml
optimizer: adamw
learning_rate: 2.0e-5
weight_decay: 0.01
scheduler: linear
warmup_ratio: 0.10
effective_batch_size: 32
max_epochs: 10
precision: bf16
max_grad_norm: 1.0
early_stopping:
  patience: 2
  min_delta: 0.0001
```

## Sailor and Vistral QLoRA

```yaml
optimizer: paged_adamw_8bit
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine
warmup_ratio: 0.05
effective_batch_size: 16
epochs: 3
precision: bf16
max_grad_norm: 1.0
gradient_checkpointing: true
micro_batch_probe_order: [2, 1]
```

Probe micro-batch size once during Phase 15, freeze the successful value in the hardware profile, and
do not vary it across experimental conditions.

## Selection metrics

- Q1a, Q2, and full multi-task runs: dev macro-pragmatic F1.
- Q3: dev sarcasm binary macro-F1.
- `phobert_pol_single`: dev polarity macro-F1.
- `phobert_emo_single`: dev emotion macro-F1.
- CoT-only: dev macro-pragmatic F1 computed from parsed generated labels.
- Explanation-only and full models: dev macro-pragmatic F1 computed from classification-head outputs.

Checkpoint tie-break order:

1. higher primary dev metric;
2. lower total dev loss;
3. earlier checkpoint.

Use deterministic seed control and `torch.use_deterministic_algorithms(True, warn_only=True)`.


# MEMORY-SAFE BATCHING AND GENERATION VALIDATION

Effective batch sizes are fixed, but physical micro-batch sizes are selected once during Phase 15 and then frozen:

```yaml
phobert_probe_order: [32, 16, 8]
xlmr_probe_order: [8, 4, 2, 1]
qlora_7b_probe_order: [2, 1]
```

Use gradient accumulation to reach the required effective batch size.
Record the selected physical batch and accumulation steps in the frozen hardware profile.

For CoT-only checkpoint selection, parse generated labels on the full dev split using greedy decoding and compute
dev macro-pragmatic F1. Cache dev generations per checkpoint so checkpoint selection is reproducible.

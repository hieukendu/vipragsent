> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 06 — IMPLEMENT MODEL CODE WITHOUT REAL WEIGHTS

Implement every required architecture and variant using interfaces and tiny dummy backbones.

Implement six binary pragmatic heads, one three-way intended-polarity head, one seven-way emotion head, a two-layer rationale decoder, pooling adapters, uncertainty-weighted multi-task loss, and rationale loss with `beta=0.3`.

Support PhoBERT single-task, pragmatic-only fine-tunes, eight-head multi-task checkpoints, Sailor/Vistral QLoRA, no-auxiliary, full ViPragSent, CoT-only, explanation-only, and all Q2 ablation flags.

QLoRA: rank 16, alpha 32, dropout 0.05, NF4, q/k/v/o projections, and optional gradient checkpointing.

Required tests with tiny fixtures: forward shapes, finite losses, gradient flow, ablation toggles, rationale generation/parser behavior, and trainable-parameter reports.

Do not download real model weights and do not run full training.

# LOCKED ARCHITECTURE DETAILS

This section overrides generic architecture wording.

## Classification pooling

- PhoBERT and XLM-R: use the final hidden state of the first non-padding `<s>` token.
- Sailor and Vistral causal backbones: use attention-mask-aware mean pooling over all non-padding final hidden states.
- Classification dropout: 0.1 before every classification head.

Do not use last-token pooling for the causal backbones.

## Rationale decoder

Use:

```yaml
rationale_decoder:
  layers: 2
  hidden_size: 128
  attention_heads: 4
  feed_forward_size: 512
  dropout: 0.1
  teacher_forcing: true
  loss: token_cross_entropy
  label_smoothing: 0.0
  rationale_only_max_target_tokens: 96
  rationale_plus_labels_max_target_tokens: 160
  decoding: greedy
  tie_decoder_input_output_embeddings: true
  rationale_loss_beta: 0.3
```

The decoder must cross-attend to the complete token-level backbone hidden-state sequence, not only to a
single pooled vector. Project backbone hidden states to decoder hidden size 128 before cross-attention.
Use the corresponding backbone tokenizer vocabulary and its BOS/EOS conventions.
The decoder must have its own learnable `vocab_size × 128` token-embedding matrix. Tie this decoder input
embedding to the decoder output projection only. Do not tie it directly to the backbone token embeddings,
because backbone and decoder hidden dimensions differ.

## Variant-specific targets

- `vipragsent_full_*`: rationale-only decoder target.
- `cot_only_*`: rationale-plus-labels target; classification losses disabled.
- `explanation_only_*`: six pragmatic classification losses plus rationale-only auxiliary loss; polarity and emotion losses disabled; decoder disabled at inference.
- No separate `explanation_at_inference_*` implementation is permitted.


# BACKBONE LOADING CLASSES

Use `AutoModel`-style base transformer classes rather than causal language-model heads, because all generation is
performed by the custom rationale decoder.

- PhoBERT/XLM-R: encoder base model.
- Sailor/Vistral: decoder-only base transformer returning token-level hidden states.
- Do not allocate or train an unused pretrained LM head.
- Keep `output_hidden_states=False` unless a specific diagnostic requires it; use `last_hidden_state`.
- Set `trust_remote_code=False` by default. Any exception requires a pinned, reviewed code revision and an ADR.

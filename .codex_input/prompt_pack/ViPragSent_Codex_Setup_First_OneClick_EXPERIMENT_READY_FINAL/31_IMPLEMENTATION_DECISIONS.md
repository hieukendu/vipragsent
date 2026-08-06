
# Final Implementation Decisions

This file resolves remaining implementation ambiguities and overrides less-specific wording elsewhere.

## 1. Model repositories

```yaml
phobert_base: vinai/phobert-base
xlmr_large: FacebookAI/xlm-roberta-large
sailor_7b: sail/Sailor-7B
vistral_7b: Viet-Mistral/Vistral-7B-Chat
```

Resolve and freeze immutable commit SHAs before downloading weights.

## 2. Text processing

- Preserve immutable raw Unicode text.
- PhoBERT: deterministic VnCoreNLP word segmentation for source text and rationale targets.
- XLM-R, Sailor, Vistral: raw Unicode-NFC text.
- No global lowercasing.
- No lexical normalization.
- Never delete emojis, repeated punctuation, slang, or code-switched tokens.

## 3. Model family semantics

- Pragmatic fine-tune: six pragmatic heads only.
- Multi-task 8-head: six pragmatic heads plus polarity and emotion; no rationale.
- Full ViPragSent: eight classification heads plus rationale-only training auxiliary; decoder disabled at inference.
- Explanation-only: six pragmatic heads plus rationale-only training auxiliary; decoder disabled at inference.
- CoT-only: generation path only; reported labels parsed from generated strict JSON.
- No separate explanation-at-inference system.

## 4. Canonical label keys

```text
implicit_sentiment
sarcasm
irony
idiom_figurative
code_switching
mocking
polarity
emotion
```

## 5. Rationale decoder

- Two Transformer decoder layers.
- Hidden size 128.
- Four attention heads.
- Feed-forward size 512.
- Dropout 0.1.
- Cross-attend to full projected token-level backbone states.
- Separate decoder embedding matrix.
- Tie decoder input embedding and output projection only.
- Greedy decoding.
- Maximum 96 target tokens for rationale-only.
- Maximum 160 target tokens for rationale-plus-labels.

## 6. Full-model loss

Eight independent classification losses receive eight independent uncertainty parameters.
Rationale loss remains fixed at coefficient 0.3.

## 7. Hardware behavior

Effective batch sizes remain fixed. Physical micro-batch and accumulation settings are selected once during
model-load smoke tests and frozen for all comparable runs.

## 8. Paper-facing systems

- Table 2 headline: full Vistral ViPragSent.
- Table 3 retention: full PhoBERT ViPragSent.
- Table 4 ablations: PhoBERT family only.
- Q4: PhoBERT pragmatic fine-tune, Vistral pragmatic SFT, full Vistral ViPragSent.
- Old Figure 5 and the six-class pragmatic-polarity head are prohibited.


# 9. Azure few-shot task separation

- Pragmatic 8-shot: six phenomenon examples plus ordinary positive and negative controls.
- Polarity 8-shot: 3 negative, 2 neutral, 3 positive.
- Emotion 8-shot: one per seven classes plus one additional `other`.
- All demonstrations come from ViPragSent train and are frozen by sample ID.

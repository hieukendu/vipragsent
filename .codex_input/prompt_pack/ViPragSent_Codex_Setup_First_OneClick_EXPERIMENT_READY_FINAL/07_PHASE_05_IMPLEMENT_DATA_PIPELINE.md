> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 05 — IMPLEMENT THE DATA PIPELINE

Implement all loaders, collators, masks, and tokenization interfaces without downloading real model weights.

Required work:

- Typed dataset examples using dataclasses or Pydantic.
- Frozen split loading without re-splitting.
- Preserve Vietnamese text, emoji, slang, and punctuation.
- Maximum sequence length 128.
- Label encoders and decoders.
- Loss weights from the train split only.
- Q3 budget-mask loaders.
- External evaluation loaders.
- Rationale loaders.
- Deterministic samplers.
- Dataset fingerprinting.
- Dummy tokenizer adapter for tests.

Required tests: label round-trip, split immutability, no test sample in train, no external data in optimizer loaders, nested Q3 subsets, correct masks, deterministic batches, and train-only loss weights.

Do not download tokenizers or model weights and do not train real models.


# BACKBONE-SPECIFIC TEXT PREPROCESSING

Preserve an immutable raw-text column for every sample.

## PhoBERT

PhoBERT requires Vietnamese word-segmented input. Use VnCoreNLP RDRSegmenter in deterministic mode and cache:

```text
data/processed/tokenized_text/phobert/<split>.jsonl
```

Apply the same segmentation to rationale targets used with PhoBERT. Convert underscores back to spaces only for
human-facing decoded rationale display. Record the VnCoreNLP version and segmentation-resource checksum.

## XLM-R, Sailor, and Vistral

Use raw Unicode-NFC text without word segmentation.

## All backbones

- Do not lowercase globally.
- Do not remove emojis, repeated punctuation, hashtags, slang, or code-switched tokens.
- Do not apply lexical normalization by default because it may erase pragmatic cues.
- Tokenization caches must be keyed by sample ID, model revision, tokenizer revision, preprocessing version,
  and maximum sequence length.


# TOKENIZER IMPLEMENTATION DETAILS

- PhoBERT: load `AutoTokenizer` with `use_fast=False`; tokenize the cached VnCoreNLP-segmented text.
- XLM-R: use its SentencePiece tokenizer on raw Unicode-NFC text.
- Sailor and Vistral: if the tokenizer has no padding token, set `pad_token` to the existing EOS token without
  adding a new vocabulary item; use right padding for training and evaluation.
- Store `attention_mask` for every backbone and never include padding positions in causal-backbone mean pooling.
- Truncation must be deterministic and logged. Export truncation rates by backbone and split.

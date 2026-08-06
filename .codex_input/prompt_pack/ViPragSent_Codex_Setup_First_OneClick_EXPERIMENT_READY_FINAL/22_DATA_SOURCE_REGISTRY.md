# Data Source Registry

Codex must verify current access terms, schemas, official splits, and checksums at execution time.

## ViPragSent

```text
ViPragSent_Experiment_Dataset_FINAL_V8.zip
```

The split is frozen and must not be modified.

## UIT-VSFC

Prefer the official UIT source or official UIT Hugging Face organization. Verify the official test split and sentiment mapping.

## UIT-VSMEC

Prefer the official UIT source. Do not invent an official split from an unsplit mirror.

## AIVIVN original

Kaggle slug:

```text
mcocoz/aivivn-2019
```

The original binary dataset is used only for provenance. Q1b uses the bundled human-derived three-way split.

## Azure OpenAI

- Provider: Microsoft Azure OpenAI / Microsoft Foundry.
- Model family: GPT-4.1-mini.
- Expected version: `2025-04-14`.
- API: Azure OpenAI Responses API v1.
- Request `model` value: Azure deployment name.
- Strict Structured Outputs are required.


# LOCKED MODEL REPOSITORIES

```text
PhoBERT: vinai/phobert-base
XLM-R-large: FacebookAI/xlm-roberta-large
Sailor-7B: sail/Sailor-7B
Vistral-7B: Viet-Mistral/Vistral-7B-Chat
```

Resolve and freeze immutable commit revisions during Phase 04. Do not track moving `main` branches during the
experiment run.

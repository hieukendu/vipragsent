
# Runtime Preflight Checklist

The full one-click run must not start unless every applicable item passes.

## Data

- [ ] Dataset ZIP and internal checksums pass.
- [ ] ViPragSent split manifest and counts pass.
- [ ] Q3 budget masks pass nested-set validation.
- [ ] External official test files are present and validated.
- [ ] Restricted-data access and redistribution notes are recorded.
- [ ] Active rationale input manifest contains no legacy generator placeholder.

## Schemas and roles

- [ ] Canonical label keys validate.
- [ ] Table 2/3/4 paper roles validate.
- [ ] Exact Table 3 checkpoint matrix validates.
- [ ] No active config contains `explanation_at_inference`.
- [ ] No six-class pragmatic-polarity head exists.

## Preprocessing

- [ ] PhoBERT segmentation resources and checksums are available.
- [ ] Segmentation cache is complete for train/dev/test and rationale targets.
- [ ] Raw-text hashes are unchanged.
- [ ] Other backbones use raw Unicode-NFC text.

## Models and environment

- [ ] Exact Hugging Face commit SHAs are pinned.
- [ ] All model/tokenizer files are available offline.
- [ ] PyTorch/Transformers/PEFT/Accelerate/bitsandbytes compatibility smoke passes.
- [ ] A100/MIG profile is recorded.
- [ ] BF16 is supported.
- [ ] Frozen micro-batch settings pass forward/backward without OOM.
- [ ] Disk capacity is sufficient for models, checkpoints, predictions, and caches.

## Azure

- [ ] Azure endpoint and deployment are verified.
- [ ] Model family/version verification passes.
- [ ] Structured Output smoke passes.
- [ ] Rate-limit/retry behavior passes.
- [ ] API quota estimate covers rationale and baseline requests.
- [ ] No direct OpenAI endpoint appears in active configuration.
- [ ] No secret appears in Git or logs.

## Execution and artifacts

- [ ] Fixture one-click DAG passes.
- [ ] Resume and cache validation pass.
- [ ] Expected run count matches the master matrix.
- [ ] GPU and Azure pricing snapshots are present or monetary cost is explicitly unavailable.
- [ ] Exact table and figure schemas validate.
- [ ] Old Figure 5 output path does not exist.

Only after this checklist passes may the orchestrator write `FULL_RUN_PREFLIGHT_PASS=true`.


# ADDITIONAL PREFLIGHT ITEMS

- [ ] Java 17 and VnCoreNLP deterministic segmentation smoke pass.
- [ ] Task-specific Azure demonstration manifests pass coverage validation.
- [ ] Dataset-summary artifact schemas validate.
- [ ] Decoder-only backbones load without unused pretrained LM heads.

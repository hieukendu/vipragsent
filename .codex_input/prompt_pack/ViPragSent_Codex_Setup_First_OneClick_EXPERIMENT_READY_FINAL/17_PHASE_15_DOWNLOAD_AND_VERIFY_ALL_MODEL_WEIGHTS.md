> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 15 — DOWNLOAD AND VERIFY ALL MODEL WEIGHTS

This is the first phase in which downloading real model weights is allowed.

Download the locked revisions of PhoBERT, XLM-R-large, Sailor-7B, Vistral-7B, and their tokenizers/configurations.

Create:

```text
scripts/download_all_models.py
configs/models/download_manifest.yaml
data/model_cache_manifest.json
```

Requirements: exact revisions, resumable downloads, checksum verification where available, disk-space precheck, no duplicate downloads, configurable cache, size/license metadata, and offline verification.

Required load smoke tests:

- load and tokenize with PhoBERT;
- load and tokenize with XLM-R;
- load each 7B model in 4-bit QLoRA mode on the A100 20 GB;
- verify the intended micro-batch fits;
- run one tiny forward/backward pass;
- record peak VRAM and any OOM behavior.

Acceptance criteria: all required models are available offline, revisions match the registry, A100 20 GB smoke tests pass, and no full training has been performed.


# DOWNLOAD AND COMPATIBILITY REQUIREMENTS

Download only the exact pinned repository commits from `configs/models/model_registry.yaml`.
Prefer `safetensors` and reject untrusted remote code unless the locked model genuinely requires it and the code
revision has been reviewed and pinned.

Before accepting the environment:

- verify the installed PyTorch, Transformers, PEFT, Accelerate, and bitsandbytes versions are mutually compatible;
- record CUDA, driver, GPU/MIG profile, and library versions;
- verify VnCoreNLP segmentation resources are available and checksum-locked;
- verify all tokenizers can encode the canonical schema tokens;
- verify 4-bit QLoRA forward/backward on both Sailor and Vistral.


# VNCORENLP RUNTIME SMOKE

Verify Java 17 and the pinned VnCoreNLP RDRSegmenter resources.
Segment at least 100 representative train examples, including emoji, slang, punctuation repetition, and
code-switching. Confirm that raw text is preserved separately and that cache regeneration is deterministic.

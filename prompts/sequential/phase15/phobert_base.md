# ViPragSent Phase 15 model-family preparation: phobert_base

This future runbook is for exactly one model family: `phobert_base`. Do not download or verify any other model family in this run.

The setup task that generated this file must not execute Phase 15. Execute this runbook only after the setup is frozen and the user explicitly approves Phase 15.

## Locked model

- Model family: `phobert_base`
- Repository: `vinai/phobert-base`
- Revision: `01daacda68afe13d83023d16ec647239e344a1e6`
- Tokenizer revision: `01daacda68afe13d83023d16ec647239e344a1e6`
- Quantization: `none`

## Required sequence

1. Confirm the runtime preflight and server prerequisites from `32_RUNTIME_PREFLIGHT_CHECKLIST.md`.
2. Download only this family with `python scripts/download_all_models.py --manifest configs/models/download_manifest.yaml --model-family phobert_base`.
3. Run the offline revision/tokenizer/model verification for this family with `python scripts/verify_model_smoke.py --manifest data/model_cache_manifest.json --model-family phobert_base`.
4. Run the locked forward/backward smoke and physical-batch probe for this family when the runtime checklist permits it. Use exactly `python scripts/probe_model_batch.py --model-family phobert_base` for the physical-batch probe.
5. Record the exact local revision, tokenizer revision, quantization, physical batch, and verification hashes.
6. Print the complete Phase 15 report and paste it into the Codex chat.

Do not start Phase 16, an experiment, an Azure job, or another model family. Stop with `PENDING_USER_APPROVAL` and wait for explicit approval before any next Phase 15 prompt.

PHASE15_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO

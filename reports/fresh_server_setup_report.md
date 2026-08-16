# ViPragSent fresh-server setup report

## Final state

`SETUP_ONLY_STATUS: READY_WAITING_FOR_MASTER_PROMPT`

No scientific experiment, resume, approval, aggregation, next-job selection, production generation, or Azure live model request was started.

## Server and runtime

- Ubuntu 22.04.4, x86_64; 128 logical CPUs; 2.0 TiB RAM.
- One approved H100 MIG device: `NVIDIA H100 80GB HBM3 MIG 2g.20gb`, 19.625 GiB visible, compute capability 9.0, driver 550.54.15.
- PyTorch `2.4.0+cu124`, CUDA available, BF16 supported; free root disk approximately 5.9T.
- Python: `/root/vipragsent/.venv/bin/python`, version 3.11.0rc1, within the project range `>=3.11,<3.14`.
- Java/JDK 17.0.19. VnCoreNLP adapter `py-vncorenlp 0.1.4`, bridge `pyjnius 1.7.0`; resource checksum and JAR SHA-256 match the tracked dependency inventory. Deterministic Vietnamese segmentation passed.
- `pip check`, imports, compilation, environment check, schema validation, execution-registry validation, sequential-prompt validation, and CPU-safe tests passed.

The repository-root `.env` was not displayed or committed. `HF_TOKEN` and the required Azure settings were checked by presence only. `VNCORENLP_HOME` was configured as the non-secret relative path `data/model_cache/vncorenlp`.

## Hugging Face recovery

Authenticated Hub inspection passed for all five repositories. The detailed, secret-free inventory is in [hf_authenticated_repository_inventory.json](hf_authenticated_repository_inventory.json) and [hf_authenticated_repository_inventory.md](hf_authenticated_repository_inventory.md). The local recovery archive is `/root/vipragsent_hf_backup`, outside the Git checkout.

The authoritative 35-entry checkpoint inventory passed:

- Expected: 35
- Verified: 35
- Missing: 0
- Size mismatches: 0
- SHA-256 mismatches: 0
- Verified bytes: 107,567,669,982 (about 100.18 GiB)

The paused Vistral run was restored at `results/runs/q1a_cot_only_vistral_20260521/`. Its epoch-1 checkpoint and `best/model.pt` both match SHA-256 `27f81249de067555827099334acb999f009d0d5970df333e855be231f36fc7c0`. No epoch-2 checkpoint exists. The checkpoint was inspected only after hashing and is readable as schema 2 with optimizer, scheduler, and RNG state.

## Base models

All four exact pinned upstream revisions and tokenizer revisions were authenticated and downloaded into local caches. PhoBERT and XLM-R passed the repository’s actual local CPU smoke and physical GPU probe. Sailor and Vistral passed direct production-device NF4/QLoRA tokenizer, forward, backward, finite-loss, finite-gradient, and official physical-batch probes on `cuda:0` at physical batch 2 (effective batch 16, gradient accumulation 8).

The stock `verify_model_smoke.py` helper has a setup-only validation limitation for the 7B NF4 families: its CPU path cannot fit the model, and its CUDA path creates CPU synthetic inputs after loading the model on CUDA. This does not indicate a production loader failure—the direct production-device smoke and official batch probes pass—and no scientific code or protocol was changed.

## Data, Azure, and hard stop

The frozen ViPragSent V8 package passed semantic, split, ID, prompt-budget, and hash validation: 11,997 rows across train/dev/test (7,998/1,999/2,000). Existing line-ending normalization was verified without changing frozen data.

Azure client libraries, configuration, and credentials are present by safe presence checks. `AZURE_LIVE_REQUEST_TEST` is `NOT_RUN_SETUP_ONLY`.

The checkout remains dirty only from setup-generated cache/status/recovery-report artifacts; no reset, clean, push, upload, delete, visibility change, or other remote write was performed. Active ViPragSent GPU jobs: `0`. Active ViPragSent Azure jobs: `0`.

Setup is complete. No scientific experiment has been started. Waiting for the user's Master Prompt.

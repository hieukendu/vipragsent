# Reproducibility report

EXPERIMENT_REPOSITORY_READY=false
Status: BLOCKED

## Blockers
- Every model family must pass cache, actual offline smoke, and frozen physical-batch verification
- A100 20 GB or an A100 MIG profile is not available
- Java 17 LTS is required for VnCoreNLP
- PEFT is not installed for QLoRA
- bitsandbytes is not installed for NF4 QLoRA
- Phase 15 actual per-family model/tokenizer smoke reports are missing or incomplete
- complete Phase 16 production manifest is not present

## Scientific protocol conflicts
- None

The report distinguishes implementation checks from deferred runtime and protocol readiness.

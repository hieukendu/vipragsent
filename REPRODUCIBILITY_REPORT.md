# Reproducibility report

EXPERIMENT_REPOSITORY_READY=false
Status: BLOCKED

## Blockers
- Model weights have not passed Phase 15 offline verification
- A100 20 GB or an A100 MIG profile is not available
- Java 17 LTS is required for VnCoreNLP
- PEFT is not installed for QLoRA
- bitsandbytes is not installed for NF4 QLoRA
- Phase 15 model/tokenizer smoke report is missing
- complete Phase 16 production manifest is not present

## Scientific protocol conflicts
- None

The report distinguishes implementation checks from deferred runtime and protocol readiness.

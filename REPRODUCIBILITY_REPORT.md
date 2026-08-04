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
- unresolved scientific protocol conflict

## Scientific protocol conflicts
- SCIENTIFIC_PROTOCOL_CONFLICT_Q1A_VISTRAL_NO_AUXILIARY
- SCIENTIFIC_PROTOCOL_CONFLICT_Q4
- SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE

The report distinguishes implementation checks from deferred runtime and protocol readiness.

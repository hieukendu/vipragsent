# Setup readiness

SETUP_READY=false

The complete setup is intentionally not marked ready until all runtime preflight prerequisites pass.

## Blockers
- Azure deployment live verification failed or is missing
- Azure Responses API plain/strict Structured Output smoke has not passed
- Model weights have not passed Phase 15 offline verification
- A100 20 GB or an A100 MIG profile is not available
- Java 17 LTS is required for VnCoreNLP
- PEFT is not installed for QLoRA
- bitsandbytes is not installed for NF4 QLoRA
- Phase 15 model/tokenizer smoke report is missing

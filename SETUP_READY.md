# Setup readiness

SETUP_IMPLEMENTATION_READY=true
SETUP_FROZEN=false
RUNTIME_DEPENDENCIES_PENDING=true

Phase 15 model download and runtime smoke are intentionally deferred.

## Scientific protocol conflicts
- `SCIENTIFIC_PROTOCOL_CONFLICT_Q1A_VISTRAL_NO_AUXILIARY`
- `SCIENTIFIC_PROTOCOL_CONFLICT_Q4`
- `SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE`

## Implementation blockers
- None

## Deferred runtime requirements
- A100 or A100 MIG runtime
- Java 17 and VnCoreNLP resources
- PEFT
- bitsandbytes
- model downloads
- real Phase 15 model/tokenizer/QLoRA smoke

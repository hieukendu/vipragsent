# Phase 14.5 production implementation

The implementation repair completed the non-server code paths and exercised them with CPU and temporary synthetic/fake-runtime tests. Scientific protocol conflicts remain explicit and are not resolved by this repair.

## Inventory

- Expected runs: **156**
- Counts: `{"Q1a": 29, "Q1b": 22, "Q2": 18, "Q3": 78, "Q4": 3, "backbone_sensitivity": 6}`

## Deferred runtime

- Java 17 and VnCoreNLP resources
- PEFT
- bitsandbytes
- A100 or A100 MIG runtime
- model downloads
- real Phase 15 model/tokenizer/QLoRA smoke

## Conflicts

- `SCIENTIFIC_PROTOCOL_CONFLICT_Q1A_VISTRAL_NO_AUXILIARY`
- `SCIENTIFIC_PROTOCOL_CONFLICT_Q4`
- `SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE`

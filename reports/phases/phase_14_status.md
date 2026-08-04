# Phase 14 status

- Status: `BLOCKED`
- Tests passed: `True`
- Next phase ready: `False`

## Inputs read
- `30_SPEC_COMPLETENESS_AUDIT.md`
- `31_IMPLEMENTATION_DECISIONS.md`
- `32_RUNTIME_PREFLIGHT_CHECKLIST.md`

## Files created
- `SETUP_FREEZE_MANIFEST.json`
- `SETUP_CHECKSUMS.sha256`
- `SETUP_READY.md`
- `reports/semantic_config_audit.json`

## Tests run
- `configuration validation`
- `semantic configuration audit`
- `fixture DAG state check`
- `full runtime preflight`

## Blockers
- External dataset provenance is incomplete; official/manual-drop checks must pass
- Azure credentials/deployment are not configured
- Model weights have not passed Phase 15 offline verification
- A100 20 GB or an A100 MIG profile is not available
- Java 17 LTS is required for VnCoreNLP
- Pinned VnCoreNLP RDRSegmenter resources are missing
- PEFT is not installed for QLoRA
- bitsandbytes is not installed for NF4 QLoRA
- Phase 15 model/tokenizer smoke report is missing

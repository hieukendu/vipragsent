# ViPragSent — Setup-First / One-Click Experiment Prompt Pack

## Workflow objective

This prompt pack forces Codex to follow this order:

```text
complete all code, data, configuration, tests, and orchestration setup
→ freeze the complete setup
→ download model weights in the penultimate setup phase
→ execute the entire experiment DAG with one command
→ automatically export metrics, statistics, tables, figures, and artifacts
→ perform a final reproducibility audit
```

The initial project contains only:

```text
ViPragSent_Experiment_Dataset_FINAL_V8.zip
```

## Prompt order

1. `01_GLOBAL_PROJECT_CONTRACT.md`
2. `02_PHASE_00_BOOTSTRAP_REPOSITORY.md`
3. `03_PHASE_01_INGEST_VALIDATE_VIPRAGSENT.md`
4. `04_PHASE_02_DOWNLOAD_NORMALIZE_EXTERNAL_DATASETS.md`
5. `05_PHASE_03_CONFIGURE_AZURE_OPENAI.md`
6. `06_PHASE_04_DEFINE_SCHEMAS_AND_CONFIGS.md`
7. `07_PHASE_05_IMPLEMENT_DATA_PIPELINE.md`
8. `08_PHASE_06_IMPLEMENT_MODEL_CODE_NO_WEIGHTS.md`
9. `09_PHASE_07_IMPLEMENT_TRAINING_ENGINE.md`
10. `10_PHASE_08_IMPLEMENT_EVALUATION_STATISTICS.md`
11. `11_PHASE_09_IMPLEMENT_AZURE_PROMPTS_AND_CLIENT.md`
12. `12_PHASE_10_IMPLEMENT_RATIONALE_PIPELINE_NO_FULL_RUN.md`
13. `13_PHASE_11_BUILD_EXPERIMENT_MATRIX_AND_ORCHESTRATOR.md`
14. `14_PHASE_12_BUILD_ARTIFACT_EXPORT_PIPELINE.md`
15. `15_PHASE_13_TEST_EVERYTHING_WITH_DUMMIES_AND_TINY_FIXTURES.md`
16. `16_PHASE_14_FREEZE_COMPLETE_SETUP.md`
17. `17_PHASE_15_DOWNLOAD_AND_VERIFY_ALL_MODEL_WEIGHTS.md`
18. `18_PHASE_16_RUN_ALL_EXPERIMENTS_ONE_CLICK.md`
19. `19_PHASE_17_FINAL_REPRODUCIBILITY_AUDIT.md`

Files `20–27` are supporting reference files.

## Required phase handoff

Every phase must create:

```text
reports/phases/phase_XX_status.md
reports/phases/phase_XX_handoff.json
```

Minimum handoff schema:

```json
{
  "phase": "XX",
  "status": "PASS | BLOCKED | FAIL",
  "inputs_read": [],
  "files_created": [],
  "tests_run": [],
  "tests_passed": true,
  "blockers": [],
  "next_phase_ready": false
}
```

Proceed to the next phase only when `status=PASS` and `next_phase_ready=true`.

## Specification-completeness additions

Before running the full suite, Codex must also read:

- `28_PAPER_EXPERIMENT_ROLE_REGISTRY.md`
- `29_MANUAL_ERROR_AND_QUALITATIVE_ANALYSIS.md`
- `30_SPEC_COMPLETENESS_AUDIT.md`

These files lock the table-specific backbones, exact checkpoint roles, controlled-ablation definitions,
paper-facing artifact schemas, backbone-sensitivity analysis, and the human-reviewed analysis that
cannot be honestly automated.


## Final implementation-readiness files

Codex must read these additional files before implementing model or execution code:

- `31_IMPLEMENTATION_DECISIONS.md`
- `32_RUNTIME_PREFLIGHT_CHECKLIST.md`

`31_IMPLEMENTATION_DECISIONS.md` resolves the remaining technical ambiguities, including backbone-specific
preprocessing, exact baseline families, canonical label keys, full-model inference, the removed duplicate
explanation-at-inference system, uncertainty-loss aggregation, model repositories, and quantization defaults.

`32_RUNTIME_PREFLIGHT_CHECKLIST.md` defines the mandatory checks that must pass before the one-click full run.

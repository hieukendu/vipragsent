> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 17 — FINAL REPRODUCIBILITY AUDIT

Audit the full one-click experiment run from a clean state.

Required audit:

- install the environment in a clean checkout;
- verify dataset, model, configuration, and result checksums;
- verify Azure deployment and request provenance;
- verify no direct OpenAI endpoint is used;
- verify no test-set peeking;
- verify no external fine-tuning for Q1b;
- verify Q3 masks and demonstration eligibility;
- verify all three training seeds and all budgets;
- verify artifact backing data;
- regenerate at least one table and one figure;
- verify resume behavior;
- scan for prohibited components;
- scan for secrets;
- verify access and license notes.

Create:

```text
REPRODUCIBILITY_REPORT.md
RELEASE_MANIFEST.json
EXPERIMENT_MODEL_REGISTRY.md
DATASET_CARD.md
KNOWN_LIMITATIONS.md
FINAL_CHECKSUMS.sha256
```

Write `EXPERIMENT_REPOSITORY_READY=true` only when the audit fully passes.

# SPECIFICATION-COMPLETENESS AUDIT

The final audit must verify all items in `30_SPEC_COMPLETENESS_AUDIT.md`.

Explicitly fail the audit if:

- a paper role uses the wrong backbone;
- Table 3 lacks either ordinary single-task checkpoint;
- CoT-only uses classification-head losses;
- any separate explanation-at-inference system exists;
- Q3 fails to recompute `pos_weight` per budget;
- Q2 `no_multitask` is implemented as one shared model;
- Q4 uses a six-class pragmatic-polarity head;
- the old Figure 5 is generated;
- cost ratios use a denominator other than full ViPragSent PhoBERT for Table 4;
- an artifact schema differs from `27_OUTPUT_ARTIFACT_SCHEMA.md`.


# ADDITIONAL READINESS CHECKS

Verify:

- task-specific Azure 8-shot manifests have the required class/phenomenon coverage;
- Table 1 dataset artifacts match the frozen manifests;
- no unused pretrained causal LM head is loaded for Sailor or Vistral;
- persisted prediction schemas use only canonical keys;
- all full/explanation-only inference manifests report classification-head outputs and decoder disabled;
- CoT-only is the only local paper-facing system reporting parsed generated labels.

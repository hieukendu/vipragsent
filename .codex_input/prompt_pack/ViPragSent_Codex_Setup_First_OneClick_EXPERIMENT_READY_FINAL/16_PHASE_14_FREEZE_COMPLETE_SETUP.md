> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 14 — FREEZE THE COMPLETE SETUP

Freeze the entire experiment setup before downloading models or spending substantial compute/API budget.

Verify:

- all datasets are downloaded and normalized;
- dataset schemas and checksums pass;
- Azure deployment is verified;
- all configs validate;
- the master matrix and run count are complete;
- exact model IDs and revisions are resolved;
- the one-click fixture run passes;
- output schemas are frozen;
- disk, time, and Azure request estimates exist;
- no secrets are present;
- Git working state is clean.

Create:

```text
SETUP_FREEZE_MANIFEST.json
SETUP_CHECKSUMS.sha256
SETUP_READY.md
```

Write PASS in `SETUP_READY.md` only when every setup requirement has passed.

Do not download model weights and do not run full experiments.


# SEMANTIC CONFIGURATION AUDIT

Before writing `SETUP_READY.md`, run a machine-readable audit that verifies:

- no active config contains `explanation_at_inference`;
- full models drop/disable the rationale decoder at inference;
- CoT-only is the only local system that obtains reported labels from generated text;
- all persisted label keys are canonical;
- PhoBERT preprocessing includes frozen word segmentation;
- model repository IDs and commit SHAs are pinned;
- all Table 2/3/4 role assignments match `configs/paper_roles.yaml`;
- the old Figure 5 and six-class pragmatic-polarity head are absent.

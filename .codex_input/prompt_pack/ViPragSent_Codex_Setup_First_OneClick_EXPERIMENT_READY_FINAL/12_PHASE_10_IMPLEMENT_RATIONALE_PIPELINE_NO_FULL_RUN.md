> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 10 — IMPLEMENT THE RATIONALE PIPELINE WITHOUT A FULL RUN

Prepare the complete rationale-generation workflow so Phase 16 can execute it automatically.

Required work:

- Build rationale inputs from the ViPragSent train split.
- Define the Azure Structured Output rationale schema.
- Implement rationale validation.
- Render valid rationale text deterministically inside `<RATIONALE>...</RATIONALE>`.
- Build rationale-plus-gold-label targets deterministically for CoT variants.
- Implement cache, resume, retry, and failure manifests.
- Support synchronous execution and optional verified Azure Global Batch.
- Implement usage and cost reports.
- Implement repair and retry rules.

Testing scope: mocked responses and at most 5–20 Azure smoke samples if needed. Do not generate rationales for all 7,998 train samples.

Acceptance criteria: the full-generation command exists, dry-run reports exact request counts and expected outputs, and resume behavior passes.

# LOCKED TARGET CONTRACTS

## Full ViPragSent rationale target

```xml
<RATIONALE>
Vietnamese explanation only.
</RATIONALE>
```

Do not append gold labels.

## CoT-only generation target

```xml
<RATIONALE>
Vietnamese explanation.
</RATIONALE>
<LABELS>
{"implicit_sentiment":0,"sarcasm":0,"irony":0,"idiom_figurative":0,
 "code_switching":0,"mocking":0,
 "polarity":"neutral","emotion":"other"}
</LABELS>
```

The local parser must extract the exact `<LABELS>` block and validate it against a strict schema.
It may repair JSON punctuation only. It must not infer labels from rationale text.

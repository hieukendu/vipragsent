> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 09 — IMPLEMENT AZURE PROMPTS AND CLIENT

Complete the Azure prompt registry and execution layer without running the full dataset.

Create versioned and hashed prompts for rationale generation, pragmatic zero-shot/8-shot, polarity zero-shot/8-shot, emotion zero-shot/8-shot, and Q3 budget-specific 8-shot.

Prompt rules:

- Use strict JSON Schema.
- Require all label fields.
- Use exactly eight demonstrations for 8-shot prompts.
- Select demonstrations from ViPragSent train only.
- Q3 demonstrations must come from the eligible pool for the current budget.
- Do not use synonym-based regex parsing.

Implement Azure Responses API v1, optional verified Global Batch, caching, resume, bounded concurrency, `Retry-After`, exponential backoff, request/response IDs, token usage, content-filter logs, failure manifests, and secret-safe logging.

Use mocks and only the minimal Azure smoke capability established in Phase 03. Do not run full baseline inference.

# LOCKED 8-SHOT DEMONSTRATION COMPOSITION

Each 8-shot prompt must contain exactly eight unique ViPragSent train examples:

1. one implicit-sentiment example;
2. one sarcasm example;
3. one irony example;
4. one idiom/figurative example;
5. one code-switching example;
6. one mocking example;
7. one ordinary positive control;
8. one ordinary negative control.

Freeze exact sample IDs in a versioned demonstration manifest.
Use the same eight IDs for every test sample of the corresponding task.

For Q3, every demonstration must be eligible under the current budget mask:

- all non-sarcasm/control examples must be active train examples;
- every sarcasm-positive demonstration must be among the selected positives for that budget.

If exact coverage cannot be achieved for a budget, mark that budget BLOCKED rather than silently changing
the composition.


# CANONICAL STRUCTURED-OUTPUT SCHEMA

Persist only these exact keys:

```json
{
  "implicit_sentiment": 0,
  "sarcasm": 0,
  "irony": 0,
  "idiom_figurative": 0,
  "code_switching": 0,
  "mocking": 0,
  "polarity": "neutral",
  "emotion": "other"
}
```

The six designated phenomenon demonstrations must be eight distinct sample IDs and each designated example must
have its assigned pragmatic label equal to 1. The positive and negative control examples must have all six
pragmatic labels equal to 0 and the corresponding intended polarity.

Freeze one versioned demonstration manifest for general 8-shot evaluation and one manifest per Q3 budget.


# TASK-SPECIFIC 8-SHOT MANIFESTS

The user-selected six-phenomena-plus-two-controls composition applies to the pragmatic 8-shot prompt.

## Pragmatic prompt

- one designated positive example for each of the six pragmatic labels;
- one ordinary positive control with all pragmatic labels equal to 0;
- one ordinary negative control with all pragmatic labels equal to 0.

Output schema contains exactly the six canonical pragmatic keys.

## Polarity prompt

Use eight ViPragSent train examples with deterministic class coverage:

```text
negative: 3
neutral: 2
positive: 3
```

Output schema contains only `polarity`.

## Emotion prompt

Use one ViPragSent train example for each of the seven emotion classes plus one additional `other` example selected
deterministically from a different annotation batch. Output schema contains only `emotion`.

External benchmark text must never be used to choose demonstrations. Freeze exact sample IDs and prompt hashes.

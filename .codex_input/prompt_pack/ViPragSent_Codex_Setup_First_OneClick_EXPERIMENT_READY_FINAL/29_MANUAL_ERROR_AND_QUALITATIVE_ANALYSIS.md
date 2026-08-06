
# Manual Error and Qualitative Analysis Protocol

Some paper analyses require human judgment and must not be fabricated by an automated run.

## A. Error analysis

### Candidate generation

After all final predictions are frozen, automatically sample 400 unique test sample-label error pairs.

Use:

```text
sampling_seed = 20260525
```

Stratify as evenly as possible across:

- six pragmatic labels;
- PhoBERT fine-tune errors;
- Azure GPT-4.1-mini 8-shot errors;
- full ViPragSent Vistral errors.

Prioritize disagreement cases while preserving deterministic sampling.
Export the original text, gold label, three system predictions, probabilities/confidences, and sample ID.

### Human review

Two independent human reviewers assign one primary failure category:

1. missing broader discourse/context;
2. sarcasm or irony cue failure;
3. idiom/figurative interpretation failure;
4. code-switching or borrowed-token failure;
5. mocking target or stance failure;
6. ordinary sentiment/emotion confusion;
7. ambiguous or insufficient context;
8. probable annotation issue;
9. other.

A third adjudicator resolves disagreements.

Report raw agreement, Cohen's kappa, and nominal Krippendorff's alpha.
Do not report final error-category percentages before adjudication is complete.

## B. Qualitative examples

Automatically generate candidates satisfying:

- full ViPragSent Vistral is correct;
- PhoBERT or Azure GPT-4.1-mini 8-shot is incorrect;
- one approved sarcasm example;
- one approved code-switching example.

Rank candidates by the full model's confidence margin, but require human approval.
Do not expose private annotation notes.

## Completion states

The automated run may finish with:

```text
CORE_EXPERIMENTS_READY=true
MANUAL_PAPER_ANALYSIS_PENDING=true
```

After reviewed files are supplied, rerun the artifact exporter with `--resume` to produce:

```text
error_analysis_final.csv
qualitative_final.jsonl
```

Only then may the related paper analysis be treated as complete.

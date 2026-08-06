# ViPragSent Experiment Dataset FINAL V8

## Status

This package implements the dataset decisions agreed for the ViPragSent experiments.

### ViPragSent

- Total samples: **11,997**
- Source: **SEACrowd/ViSoBERT**
- Human annotator 1, human annotator 2, and human adjudicator
- Train/dev/test: **7,998 / 1,999 / 2,000**
- Split seed: **20260520**
- Split method: deterministic multilabel stratification across six pragmatic labels, polarity, and emotion

### Duplicate policy

No duplicate or near-duplicate filtering is performed.

- Comments differing by emoji, punctuation, or similar textual variants remain separate samples.
- Near-overlap with UIT-VSFC, UIT-VSMEC, or AIVIVN is retained by project decision.
- The split manifest is frozen without removing these rows.

### Annotation files

The original XLSX workbooks are preserved in `00_private_source_archive/` for audit. Their legacy note
strings contain incorrect `AI_*` wording inserted during packaging. Those strings are not used.

The active cleaned human annotation data are:

- `01_clean_human_annotations/01_annotator_1_clean.csv`
- `01_clean_human_annotations/02_annotator_2_clean.csv`
- `01_clean_human_annotations/03_gold_adjudicated_clean.csv`

### AIVIVN

The bundled AIVIVN data are treated as **AIVIVN-human-derived-3way** with project-defined train/dev/test
splits supplied by the project owner.

### Rationale

Rationales have not been generated yet. This package includes:

- a train-set JSONL input
- a rationale-only prompt contract for full ViPragSent
- a rationale-plus-labels contract for CoT-only / explanation-at-inference variants

### Q3 low-resource sarcasm

Train sarcasm positives: **545**

Budgets: **32, 64, 128, 256, 512, full**

All negative examples are retained. Positive samples outside a budget have the sarcasm target and
rationale loss masked.

### Table 3

The protocol is cross-domain retention/generalisation:

- train on ViPragSent
- do not fine-tune on the external benchmarks
- polarity head: UIT-VSFC and AIVIVN-human-derived-3way
- emotion head: UIT-VSMEC

UIT-VSFC and UIT-VSMEC are not bundled in this ZIP.

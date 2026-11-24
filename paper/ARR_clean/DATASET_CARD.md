# ViPragSent dataset card

This card documents the files currently delivered in this release. It was
prepared from the released CSVs, schema, manifests, annotation files and the
supplied standalone rubric; it is not a recovered historical dataset card.

## Task and annotation unit

Each instance is one Vietnamese social-media comment. Models receive only its
standalone text: no parent post, thread history, profile or inferred speaker
identity. Gold labels are adjudicated readings of text, not verified author intent.

The six independent binary pragmatic labels are `implicit_sentiment`, `sarcasm`,
`irony`, `idiom_figurative`, `code_switching`, and `mocking`. A value of 1 denotes
presence, not positive sentiment; labels may co-occur. Polarity is exactly one
of negative/neutral/positive. Emotion is exactly one of anger, disgust, enjoyment,
fear, other, sadness and surprise. Follow `ANNOTATION_RUBRIC.md` for the operational
definitions and their limits. Existing gold must not be retrospectively relabeled.

## Composition and splits

There are 11,997 unique sample IDs and 11,997 distinct exact text strings.
The fixed split is 7,998 train / 1,999 dev / 2,000 test, with no sample-ID overlap.
`split_manifest.csv` records membership and the split seed. Do not resplit for
comparison to the paper, and do not use test labels to select thresholds.

| Pragmatic label | Positive examples in all splits |
| --- | ---: |
| implicit_sentiment | 2,517 |
| sarcasm | 817 |
| irony | 831 |
| idiom_figurative | 844 |
| code_switching | 1,809 |
| mocking | 1,398 |

Training has 545 sarcasm positives and 7,453 negatives. Paper-reported Q3 masks select
nested positive budgets 32/64/128/256/512/full (545), retaining the fixed negative
pool. Out-of-budget positives have sarcasm and rationale losses masked; their
other task targets remain active. Negatives retain sarcasm and rationale loss.
Canonical masks in `q3_masks/` match the reported paper cohort. See `REPRODUCE.md`.

## Source and time semantics

Released rows record the public ViSoBERT corpus distributed through SEACrowd
as their source dataset, along with source-row IDs. `SOURCE_RECEIPT.json` locks
the current delivered bytes and summarizes retained row metadata. Its status is
`RECONSTRUCTED_CURRENT`: the original extraction receipt was not available, and
the upstream revision, extraction execution and filtering history have not been
independently revalidated. Do not interpret the observed source-ID range or ordering
as proof of random sampling or platform representativeness.

The timestamp `2026-08-21T07:00:00+07:00` is an approximate, user-reported
collection time. It is not the original comment publication time. Original
platform and publication timestamps are unavailable and stay empty, with explicit
`not_provided_by_upstream` statuses. No author account, personal link, email,
affiliation or workstation identity is included in the added materials.

## Annotation and agreement

Two first-pass annotation CSVs each cover all 11,997 IDs. The standalone rubric
describes independent first-pass annotation and final adjudication. The available
files permit recomputing pre-adjudication agreement, not rerunning the human
annotation process. Mean pragmatic nominal alpha is approximately 0.817; the
figurative-language alpha is approximately 0.5746, a material annotation boundary.
The legacy rubric-version string is not permission to introduce external context.

## Rationales and evaluation inputs

The two training JSONL files cover exactly the training IDs. Generator inputs
contain text and gold for bookkeeping; their presence does not mean gold labels
were transmitted to the rationale API. The experiment code constructs the
generation request separately. Supplied generated rationales are training-only
targets, not deployed explanations or evidence of explanation faithfulness.

`replay/` retains aggregate experimental numbers only. Added per-example
predictions and external gold-only CSVs were removed by request; the original
dataset above is unchanged. Confusion counts permit F1 validation and calibration
bins permit ECE validation. Saved paired-bootstrap CI/p-values are retained,
but cannot be resampled independently without paired records.
Numerical bindings cover all ten Table 1 rows, six Table 2 external rows, eight
primary Table 3 variants and eighteen Table 4 model/budget groups. Q3 uses
rationale supervision on all 545 positives and zero negatives at every budget. All six Q1b systems have confusion
counts for independent F1 recalculation. Vistral Q3 budget 32, seed 20260521 has
saved freeze/epoch-selection results and a search grid that independently
reproduces the original dev thresholds. Downloaded records are not retained.
No new training/API inference run is claimed.
See `REPRODUCE.md` for commands and precise verification boundaries.

## Intended use and limitations

Use for research on the fixed operational Vietnamese pragmatic label set and
the supplied splits. Annotation ambiguity, sparse positives, social-media domain
selection and unknown platform coverage limit generalization. Do not infer
author identity, demographics, original intent, population prevalence or causal
component effects from these data. Inspect sensitive or offensive user-generated
text before reuse; automated downstream deployment requires separate evaluation.

The release declares CC-BY-NC-4.0 in `LICENSE`; this card does not create additional
upstream or external-benchmark redistribution rights. External benchmarks are
not redistributed as text in this release. External held-out gold-only CSVs
have also been removed; only aggregate confusion counts and metrics remain.

Duplicate legacy packages, private workbooks and recovery/source plans have been
removed from this folder. Generated anonymous ZIPs under `submission/` are not
recursively included in the manifest-defined release.

This release is bound to PDF version 7 and contains no historical posthoc
annotation notes or original case-level adjudication records. Its Appendix B
describes 17 notes and section 2.3 mentions existing notes. The user confirmed
both descriptions and reported losing the supporting information/material.
The historical description remains user-reported and cannot be independently
verified from the available release artifacts. Zero available notes does not
establish that notes never existed. Recovery remains waived. See
`AVAILABILITY.json`; the operational rubric does not replace those records.
Frozen social-media text still contains
user-mention candidates. See `PRIVACY_REVIEW.json`; no author identity is inferred
from those mentions and no claim of complete user-text de-identification is made.

# ADR-Q4-PRAGMATIC-CALIBRATION

## Status

RESOLVED by the sequential experiment protocol approval.

## Decision

Q4 evaluates exactly `phobert_pragmatic_finetune`, `vistral_pragmatic_sft`, and
`vipragsent_full_vistral`. Each system exposes the same six pragmatic probabilities.
Calibration uses raw positive-class sigmoid probabilities, ten equal-width bins, no
temperature scaling, and independent computation for each training seed. Learning
dynamics use the frozen ViPragSent dev split and dev macro-pragmatic F1 by epoch.

## Consequence

The Q4 protocol no longer supports an intended-polarity calibration claim. Table 3
polarity performance remains separate. Real Q4 results require the sequential run,
artifact validation, and explicit user approval; setup and fixture runs produce only
synthetic validation artifacts.

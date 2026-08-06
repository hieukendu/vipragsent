# Master Experiment Matrix

## Q1a — ViPragSent pragmatic detection

- PhoBERT: six independent single-task models × 3 seeds.
- PhoBERT multi-label × 3 seeds.
- XLM-R-large × 3 seeds.
- Sailor-7B × 3 seeds.
- Vistral-7B × 3 seeds.
- Azure GPT-4.1-mini zero-shot.
- Azure GPT-4.1-mini 8-shot.
- ViPragSent no-auxiliary × 3 seeds.
- CoT-only × 3 seeds.
- Vistral explanation-only × 3 seeds.
- Full ViPragSent × 3 seeds.

## Q1b — External retention

Reuse trained checkpoints and evaluate:

- UIT-VSFC polarity;
- UIT-VSMEC emotion;
- AIVIVN-human-derived-3way polarity.

No external fine-tuning is allowed.

## Q2 — Controlled PhoBERT ablations

- Full.
- No emotion auxiliary task.
- No polarity auxiliary task.
- No rationale.
- No multi-task learning.
- No uncertainty weighting.


## Q3 — Low-resource sarcasm

Budgets:

```text
32 / 64 / 128 / 256 / 512 / full
```

Systems:

- PhoBERT.
- XLM-R.
- Vistral.
- Azure GPT-4.1-mini 8-shot.
- Full ViPragSent.

## Q4 — Calibration and learning dynamics

Reuse final predictions and training histories:

- intended-polarity ECE;
- reliability diagrams;
- dev-set learning curves.

The master orchestrator must deduplicate reusable checkpoints and runs.

# LOCKED TABLE-SPECIFIC ROLES

```yaml
table_2_headline: vipragsent_full_vistral
table_3_retention: vipragsent_full_phobert
table_4_ablation_anchor: vipragsent_full_phobert
q4_vipragsent_system: vipragsent_full_vistral
```

## Exact Table 3 systems

- `phobert_pol_single`
- `phobert_emo_single`
- `phobert_multitask_8head`
- `xlmr_multitask_8head`
- `sailor_multitask_8head`
- `vistral_multitask_8head`
- `vipragsent_full_phobert`
- Azure GPT-4.1-mini dedicated polarity prompt
- Azure GPT-4.1-mini dedicated emotion prompt

## Backbone sensitivity

- `vipragsent_full_phobert` × 3 seeds
- `vipragsent_full_vistral` × 3 seeds

## Q4 reliability systems

- PhoBERT fine-tune
- Vistral-7B SFT
- full ViPragSent Vistral

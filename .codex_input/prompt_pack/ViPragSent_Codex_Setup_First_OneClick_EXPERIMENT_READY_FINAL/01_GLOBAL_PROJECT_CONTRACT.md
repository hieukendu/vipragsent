> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# GLOBAL PROJECT CONTRACT

You are Codex. Your responsibility is to build the complete ViPragSent experimental repository from scratch using a **setup-first, one-click execution architecture**.

## 1. Locked dataset contract

Input ZIP:

```text
ViPragSent_Experiment_Dataset_FINAL_V8.zip
```

Verify these facts from the files instead of assuming them:

- ViPragSent contains 11,997 samples.
- Frozen train/dev/test sizes are 7,998 / 1,999 / 2,000.
- Six binary pragmatic labels: `implicit_sentiment`, `sarcasm`, `irony`, `idiom_figurative`, `code_switching`, and `mocking`.
- Intended polarity has three classes.
- Emotion has seven classes.
- Annotation sources are human annotator 1, human annotator 2, and a human adjudicator.
- Legacy `AI_*` strings in raw workbooks must not be used.
- Do not deduplicate.
- Treat comments that differ in emoji or punctuation as separate samples.
- Do not remove near-overlaps.
- The bundled AIVIVN dataset is named `AIVIVN-human-derived-3way`.

## 2. Azure OpenAI contract

- Provider: Microsoft Azure OpenAI / Microsoft Foundry.
- Model family: GPT-4.1-mini.
- Expected model version: `2025-04-14`.
- API requests must use the Azure deployment name through `AZURE_OPENAI_DEPLOYMENT`.
- Use the Azure OpenAI Responses API v1.
- Use strict Structured Outputs.
- Support API-key or Microsoft Entra ID authentication.
- Never fall back to the direct OpenAI endpoint.
- Full Azure jobs may run only in Phase 16.

## 3. Fixed random seeds

```text
split_seed      = 20260520
training_seeds  = [20260521, 20260522, 20260523]
subset_seed     = 20260524
bootstrap_seed  = 20260525
bootstrap_resamples = 1000
```

## 4. Fixed losses

- Pragmatic heads: weighted binary cross-entropy.
- Compute each pragmatic `pos_weight` from the ViPragSent train split only.
- Polarity and emotion: class-weighted cross-entropy.
- Compute class weights from the train split only.
- Full multi-task model: homoscedastic uncertainty weighting.
- Rationale loss coefficient: `beta = 0.3`.

## 5. Threshold selection and calibration

Threshold selection:

- One threshold per pragmatic label.
- Tune on the dev split for each training seed.
- Search grid: 0.05 through 0.95 with step 0.01.
- Objective: binary macro-F1 for the corresponding label.
- First tie-break: threshold closest to 0.5.
- Second tie-break: smaller numerical threshold.
- Freeze thresholds before testing.

Calibration:

- Three-way intended-polarity head.
- Top-label expected calibration error.
- Ten equal-width bins.
- Confidence is maximum softmax probability.
- No temperature scaling.

## 6. Statistics

- Trainable models: report the mean over three training seeds.
- Compute 95% confidence intervals using paired hierarchical bootstrap over seed runs and test examples.
- Azure fixed-prompt outputs: bootstrap over test examples only.
- Use 1,000 bootstrap resamples.

## 7. Research questions

- Q1a: pragmatic detection on ViPragSent.
- Q1b: cross-domain ordinary sentiment and emotion retention on UIT-VSFC, UIT-VSMEC, and AIVIVN-human-derived-3way.
- Q2: controlled multi-task ablations.
- Q3: low-resource sarcasm.
- Q4: calibration and dev-set learning dynamics.

## 8. Setup-first rule

Before Phase 15:

- Do not download full Hugging Face model weights.
- Do not train real models.
- Do not run Q1–Q4.
- Do not run full rationale generation.
- Do not run full Azure prompted baselines.

Allowed during setup:

- Resolve exact model IDs and revisions from metadata.
- Implement model code and configuration.
- Use dummy or tiny random fixtures.
- Make minimal Azure smoke requests.
- Run unit and integration tests.
- Dry-run the orchestrator.
- Download and normalize external datasets.
- Build all prompts, schemas, and configurations.

## 9. One-click execution rule

After Phase 15, the full experiment suite must run through exactly one entry point:

```bash
python scripts/run_all_experiments.py   --config configs/master_run.yaml   --mode full
```

The command must automatically run preflight checks, generate rationales, run Azure baselines, train and evaluate all local models, execute Q1–Q4, compute statistics, measure cost and latency, export artifacts, create final manifests, and resume safely after interruption.

A single A100 20 GB GPU may run only one GPU job at a time. “Run everything at once” means one command orchestrates the entire DAG; it does not mean launching multiple 7B models concurrently.

## 10. Prohibited actions

- Downloading model weights before Phase 15.
- Running full experiments before Phase 16.
- Test-set peeking.
- Fine-tuning on external datasets for Q1b/Table 3.
- Hard-label Azure GPT distillation.
- Surface-polarity annotation.
- A six-class pragmatic-polarity head.
- The old Figure 5.
- Test-set learning curves.
- Editing the paper or manuscript.
- Hard-coding result values.

## 11. Phase completion rule

```text
implement → test → inspect → fix → rerun → write handoff → stop
```

# LOCKED PAPER-EXPERIMENT ROLE REGISTRY

This section overrides any earlier generic model-role wording.

Create `configs/paper_roles.yaml` with exactly:

```yaml
paper_roles:
  table_2_headline:
    model: vipragsent_full
    backbone: vistral_7b

  table_3_retention:
    model: vipragsent_full
    backbone: phobert_base

  table_4_ablation:
    model: vipragsent_full
    backbone: phobert_base

  q4_calibration:
    systems:
      - phobert_finetune
      - vistral_7b_sft
      - vipragsent_full_vistral

  backbone_sensitivity:
    systems:
      - vipragsent_full_phobert
      - vipragsent_full_vistral
```

The full Vistral model is the headline ViPragSent system for Q1a/Table 2.
The full PhoBERT model is the ViPragSent system for Q1b/Table 3 and the controlled anchor for Q2/Table 4.
Do not mix backbones within a controlled ablation table.

# LOCKED INFERENCE BEHAVIOR

## Main full ViPragSent model

Train the eight classification heads and a rationale-only auxiliary decoder.
At ordinary inference, disable the rationale decoder and use only the classification heads.

## CoT-only

Train only the generative path with:

```xml
<RATIONALE>...</RATIONALE>
<LABELS>{strict label JSON}</LABELS>
```

Do not optimize classification-head losses for this variant.
At inference, generate text, parse the strict label JSON, and score the parsed labels.

## Training-only explanation behavior

The full ViPragSent model is trained with all eight classification heads plus a rationale-only auxiliary decoder.
At inference, remove or disable the rationale decoder and score only the classification-head outputs.
This is the required explanation-augmented training behavior.

Do not create a separate `explanation_at_inference` system because it would duplicate the full model under
this locked definition.

## Explanation-only controlled variant

For the distinct Table 2 explanation-only variant, train the six pragmatic classification heads plus the
rationale-only auxiliary decoder, remove the polarity and emotion auxiliary heads, and disable the rationale
decoder at inference. Score the six classification heads.

## CoT-only invalid generations

For CoT-only, unresolved invalid local generations must be counted as invalid predictions, reported in an
invalid-output rate, and scored as incorrect for the required labels. Structural repair may fix JSON punctuation
only; it must never infer or replace semantic labels.

# ViPragSent evidence and protocol brief

Status: evidence/protocol handoff for Lagrange (id01a0af75-64da-7d42-b4f3-ef72f496e815) and production Epicurus (id01a0af75-6698-7e22-bc6c-526117cd4d5c). This is an evidence brief for manuscript authoring, not a manuscript or template edit. Claims below are either recorded artifact facts or deterministic recomputations from preserved artifacts.

## Critical findings

1. The completed Q1a fairness gate covers 3 ViPragSent target seeds against 15 baseline model-seed artifacts: 45 pairwise checks, not 15. Every pair has 2,000 unique sample IDs and the same six pragmatic gold-label tuple by ID. All 18 run-level macro-F1 recomputations agree with the recorded metrics within 1e-12.
2. The gate establishes a same sample-ID and gold-label cohort, not proof of identical raw text, tokenizer inputs, or preprocessing. Prediction JSONL rows contain IDs, labels, probabilities, and logits, but no text or token IDs. The local and newer-worktree processed test CSV copies are byte-identical, which is useful corroboration but is not an embedded input attestation.
3. Table summaries use sample SD with n-1, computed with statistics.stdev. For the three target runs, mean macro-F1 is 0.9366583803341149 (93.665838 percentage points) and sample SD is 0.001871210373416054 (0.187121 percentage points). For the three XLM baseline runs, mean is 0.929342505569368 (92.934251 percentage points) and sample SD is 0.0014479038590645334 (0.144790 percentage points). The mean gap is 0.7315874764746821 percentage points. Population SD values remain in the machine-readable diagnostics but must not be used as the table SD.
4. The six heads are independent binary pragmatic-phenomenon heads: implicit sentiment, sarcasm, irony, idiom/figurative language, code-switching, and mocking. They are not six emotion classes. Polarity is a separate three-way head and emotion is a separate seven-way head.

## Actual ViPragSent implementation

The implementation evidence is in src/vipragsent/constants.py, data/labels.py, models/backbones.py, models/heads.py, models/variants.py, models/rationale_decoder.py, models/losses.py, and training/engine.py.

### Encoder, pooling, and heads

- The target registry records FacebookAI/xlm-roberta-large at revision c23d21b0620b635a76227c604d44e43a9f0ee389, with trust_remote_code disabled and no quantization.
- For encoder-style backbones, pooling selects the hidden state at the first non-padding position using the attention mask. This is not mean pooling for XLM-R.
- The six pragmatic outputs are six independent nn.Linear(hidden_size, 1) heads. Their logits are independently thresholded for binary labels.
- Separate heads produce polarity logits of size 3 and emotion logits of size 7. Label validation keeps pragmatic labels binary and does not encode them as emotion categories.

### Rationale decoder and teacher forcing

- The rationale component is a causal two-layer TransformerDecoder with decoder hidden size 128, four attention heads, feed-forward size 512, token embeddings, a Linear(backbone_hidden_size, decoder_hidden_size) memory projection, and a tied Linear(decoder_hidden_size, vocab_size, bias=false) output projection.
- During training, decoder input is the target rationale shifted right: target_ids[:, :-1] predicts target_ids[:, 1:]. Padding labels are ignored with -100 and a causal attention mask prevents access to future target tokens.
- Rationale token loss is token cross-entropy with ignore index -100 and label smoothing 0.
- The full model only runs this decoder when training and rationale target IDs are supplied. Inference output comes from the classification heads; rationale generation is disabled for the reported test predictions. Thus the decoder is an auxiliary teacher-forced training signal, not the source of pragmatic test labels.

### Objective, uncertainty weighting, and rationale beta

- Each pragmatic task uses binary cross-entropy with logits; polarity and emotion use masked categorical cross-entropy.
- The target uses uncertainty-weighted multitask aggregation for eight classification tasks: six pragmatic labels, polarity, and emotion. For task loss L and learned log variance v, the implementation contributes 0.5 times exp(-v) times L plus 0.5 times v, with the stored variance parameters clamped to [-5, 5].
- The target rationale coefficient is beta = 0.3. The engine applies task loss multipliers before the uncertainty/equal aggregation and adds the rationale contribution with this beta. Recorded target multipliers are 1.04 for code-switching, 1.01 for implicit sentiment, 1.05 for irony, 1.01 for sarcasm, and 1.00 for the remaining listed tasks.
- The target manifests record an approved_generated_rationales_train.jsonl artifact with 7,998 rows and SHA-256 B583AC45F4BC2FDFA41DA44376AB99585075CA3EBB1FC47BC3936D84B89B21D1. The V8 dataset README and package summary separately state rationales_generated:false at the package snapshot. These are different evidence layers; the run-manifest rationale artifact must not be rewritten as a claim about the original dataset annotation history.

## Q1a target training protocol

The Q1a target is the V37 optimized full ViPragSent XLM-R-large cohort, with actual training seeds 20260521, 20260522, and 20260523. The three recorded run IDs are:

- q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_20260521
- q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_20260522
- q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_20260523

The target manifests record full dataset fingerprint B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F, the 7,998/1,999/2,000 train/dev/test split, actual seeds 20260521/20260522/20260523 (logical labels 21/22/23), AdamW at learning rate 2e-5 with weight decay 0.01, physical batch size 8, gradient accumulation 4, effective batch size 32, maximum 10 epochs, bf16, gradient clipping 1.0, cosine scheduling, 0.2 warmup, dev_macro_pragmatic_f1 selection, fixed label order, and test export only after checkpoint freezing. Rationale training is enabled with beta 0.3, while rationale inference is false. The recorded training status is PASS. These are recorded run-manifest facts; no independent training rerun was performed.

## Q1a fairness and numeric evidence

The target fingerprint is B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F and the standard-baseline artifact fingerprint is A13573E38550ABD55D7F63E983C602BEA13C2765A1FFECA285F718478532AF0D. Fingerprints differ, so fingerprint equality is not the fairness criterion. The decisive deterministic check is the prediction-row cohort:

- 3 target seeds x 15 baseline model-seed artifacts = 45 pairs.
- For every pair: identical 2,000-ID set, identical ID order, and identical pragmatic gold tuple by ID.
- The prediction files do not carry text, token IDs, or tokenizer hashes. Report this as same verified sample-ID/gold-label cohort, not identical input text or tokenization.
- The test-cohort check does not establish identical training data preprocessing, augmentation, optimizer configuration, or checkpoint protocol across target and baselines. That is a separate comparability question.

The baseline families are PhoBERT single-task, PhoBERT fine-tune, XLM baseline, Sailor, and Vistral-7B SFT as named in the raw artifacts. The headline comparison uses the three XLM baseline seeds, not a pooled mixture of baseline families. The preserved table-input JSON contains every run value, sample SD, and explicit population-SD diagnostic.

## Q1a versus Q2, Q3, and Q4

These are different evidence lanes and must not be pooled as if they were one repeated-seed experiment.

- Q1a is the V37 optimized full target cohort above. Its recorded configuration uses cosine scheduling, 0.2 warmup, and patience 10.
- Q2 contains 18 distinct follow-up runs: six full/ablation families x the actual seeds 20260521, 20260522, and 20260523. The Q2 full run manifest records the same eight-task, rationale-enabled, beta-0.3, uncertainty-weighted family, but its recorded protocol uses linear scheduling, 0.1 warmup, and patience 2. The six family labels include full, no emotion auxiliary, no multitask, no polarity auxiliary, no rationale, and no uncertainty weighting.
- Q3 contains 18 low-resource runs: budgets 32, 64, 128, 256, 512, and full x the same three actual seeds. Budget masks and the low-resource regime change the training protocol; these are not additional Q1a seeds. The Q3 full lane records dev_sarcasm_binary_macro_f1 as its selection metric.
- Q4 contains three calibration/extraction artifacts linked to the same-seed Q1a V37 checkpoints. The provenance records additional_training:false, direct classification probabilities, classification-head inference, and rationale inference false. Its calibration records pass, but the outer extraction status is NOT_STARTED; Q4 is not evidence of a newly trained model.

The machine-readable protocol lineage records source repositories, run IDs, config snapshots, Q4 source checkpoint links, and the Q1a/Q2/Q3/Q4 boundary: protocol_lineage.json.

## Dataset source and ethics boundary

The V8 package README and PACKAGE_SUMMARY identify SEACrowd/ViSoBERT as the declared source and record 11,997 examples with 7,998 train, 1,999 development, and 2,000 test examples. The package records split seed 20260520 and deterministic multilabel stratification over the six pragmatic labels, polarity, and emotion. The package contains annotator1, annotator2, and adjudicated-gold files plus split and adjudication manifests. The processed test CSVs inspected here contain 2,000 rows and identify SEACrowd/ViSoBERT as the source dataset.

These are package provenance facts only. The inspected evidence does not establish recruitment details, informed consent, annotator payment, IRB or ethics approval, annotator compensation, or release permissions. It also does not support latency guarantees or a release claim. Do not copy those assertions from the structure-reference PDF. The external-benchmark README mentions UIT-VSFC, UIT-VSMEC, and AIVIVN as separate research-only sources, but their official files are not bundled in the inspected package; no external-benchmark result should be implied here.

## Evidence status and handoff files

- Recorded artifact evidence: run manifests, source manifests, raw prediction JSONL, raw metric JSON, dataset package files, and calibration records.
- Deterministic recomputation: Q1a F1 and cohort checks pass; the recomputation is not an independent model rerun.
- Independent rerun: NOT_PERFORMED. No training or paid inference was performed.
- GPT zero-shot and eight-shot records were not needed for this handoff; NOT_STARTED records are not imputed as results.

Machine-readable handoff files:

- [q1a_fairness_comparison.json](D:/vipragsent/paper/evidence/q1a_fairness_comparison.json)
- [q1a_fairness_run_table.csv](D:/vipragsent/paper/evidence/q1a_fairness_run_table.csv)
- [q1a_verified_table_inputs.json](D:/vipragsent/paper/evidence/q1a_verified_table_inputs.json)
- [q1a_numeric_claim_ledger.json](D:/vipragsent/paper/evidence/q1a_numeric_claim_ledger.json)
- [method_evidence.json](D:/vipragsent/paper/evidence/method_evidence.json)
- [protocol_lineage.json](D:/vipragsent/paper/evidence/protocol_lineage.json)
- [evidence_claim_ledger.json](D:/vipragsent/paper/evidence/evidence_claim_ledger.json)

Recommended manuscript wording for Q1a: “All target and baseline prediction artifacts were evaluated on the same verified 2,000-example sample-ID/gold-label cohort. Because the prediction rows do not include text or tokenizer attestations, this establishes label-cohort equivalence rather than byte-level input equivalence. The reported mean and table dispersion use sample SD across the three recorded seeds; the result is an artifact-backed deterministic recomputation, not an independent rerun.”

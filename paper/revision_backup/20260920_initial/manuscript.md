# ViPragSent: XLM-R-large for Vietnamese Pragmatic Sentiment Analysis

## Abstract

Pragmatic sentiment in Vietnamese text is not exhausted by polarity. Sarcasm,
irony, idiomaticity, implicitness, code-switching, and mocking can change how a
reader should interpret an utterance even when its surface sentiment is
positive or negative. We introduce ViPragSent, an XLM-R-large encoder with six
distinct binary pragmatic heads, separate three-way polarity and seven-way
emotion auxiliaries, and a training-only teacher-forced rationale objective.
The model uses a shared first-nonpadding-token representation and homoscedastic
uncertainty weighting to couple these tasks while keeping their semantic roles
separate. On a matched 2,000-example ID/gold test cohort, ViPragSent reaches
93.67 +/- 0.19 macro-pragmatic F1 versus 92.93 +/- 0.14 for the standard
XLM-R baseline, an observed +0.73 percentage-point difference. Its mean is
also highest on all six pragmatic heads and on the macro average among five
complete three-seed baseline families. Follow-up analyses show that the full
objective is an operating point rather than a uniform winner: Q2 reports
accuracy, calibration, and cost trade-offs; Q3 reports a non-monotonic
positive-label budget curve with contextual baselines; and Q4 reports target
ECE 0.0386 +/- 0.0069 versus Vistral 0.0345 +/- 0.0041. These results define
what the implemented model contributes while keeping rationale supervision
separate from inference.

## 1 Introduction

Sentiment analysis is often framed as a polarity problem, but pragmatic meaning
can reverse or qualify the interpretation suggested by a surface-positive or
surface-negative expression. A Vietnamese utterance may be sarcastic, ironic,
idiomatic, implicit, code-switched, or mocking without belonging to an emotion
category. These phenomena are therefore useful prediction targets in their own
right. Treating them as six emotion classes would collapse distinct
communicative functions and obscure the role of emotion as an auxiliary signal.
ViPragSent adopts the narrower formulation: six pragmatic phenomena are six
binary tasks, while polarity and emotion are separate auxiliary tasks.

Vietnamese offers a demanding setting. PhoBERT shows the value of
language-specific pretraining \citep{nguyen-tuan-nguyen-2020-phobert}, whereas
XLM-R provides a multilingual transfer regime across one hundred languages
\citep{conneau-etal-2020-unsupervised}. ViSoBERT applies the XLM-R architecture
to Vietnamese social-media tasks \citep{nguyen-etal-2023-visobert}. These
contrasting settings motivate testing a large multilingual encoder on a bundle
of pragmatic distinctions rather than one sentiment label.

The challenge is also objective design. Pragmatic labels may correlate with
polarity and emotion without being identical. Multi-task learning can use
related signals \citep{caruana1997multitask}, while uncertainty weighting adapts
relative loss scales \citep{kendall2018multitask}. These are empirical choices,
so we report the training losses, causal rationale decoder, and
classification-head inference interface explicitly.

This paper studies ViPragSent XLM-R-large through four linked questions:

* **Q1a--Q1b:** How does the primary model behave in-domain, and how much of
  that behavior is retained on external Vietnamese sentiment and emotion
  datasets?
* **Q2:** What changes when the polarity, emotion, explanation, multitask, or
  task-uncertainty components are removed?
* **Q3:** How does pragmatic performance change as the number of positive
  examples is constrained while the evaluation procedure is held fixed?
* **Q4:** How well calibrated are the resulting confidence estimates relative
  to comparison models under a common expected-calibration-error protocol?

The contribution is a focused analysis of one primary XLM-R-large model rather
than a leaderboard claim. We contribute (i) an implemented shared encoder with
separated pragmatic, polarity, emotion, and teacher-forced rationale
objectives; (ii) a six-head comparison on a matched ID/gold prediction cohort;
and (iii) follow-up analyses that expose accuracy, calibration, cost, and
budget trade-offs. All reported means summarize three saved runs, and
comparison claims concern the evaluated prediction artifacts rather than a
fresh training or inference rerun.

## 2 Related Work

### 2.1 Multilingual and Vietnamese encoders

XLM-R established a strong multilingual masked-language-model baseline by
scaling pretraining across languages and evaluating cross-lingual transfer
\citep{conneau-etal-2020-unsupervised}. PhoBERT demonstrates the complementary
case for a Vietnamese-specific encoder trained on word-segmented Vietnamese
data \citep{nguyen-tuan-nguyen-2020-phobert}. ViSoBERT moves toward Vietnamese
social-media text while retaining the XLM-R architecture
\citep{nguyen-etal-2023-visobert}. ViPragSent uses XLM-R-large as its primary
encoder so that the study can focus on the pragmatic multi-task formulation
and its auxiliary objectives.

The Q1a comparison contains standard XLM-R, PhoBERT, Sailor, and Vistral
prediction artifacts. ViSoBERT is discussed here as contextual prior work, not
as an evaluated Q1a baseline: no ViSoBERT prediction family is included in the
verified comparison table. This distinction prevents a related model family
from being mistaken for a measured result.

### 2.2 Pragmatic and affective supervision

The target formulation combines signals that are related but not
interchangeable. UIT-VSFC provides Vietnamese student feedback with
sentiment-oriented labels \citep{vannguyen2018uitvsfc}, whereas UIT-VSMEC is a
Vietnamese social-media emotion corpus \citep{ho2019emotion}. ViPragSent uses
these label families as auxiliary signals while retaining separate pragmatic
outputs.

Sarcasm detection studies show why context can matter beyond the current
utterance: conversation-aware models can outperform models that read only the
current turn \citep{ghosh-etal-2018-sarcasm}. A complementary line of work
distills teacher-generated rationales as additional supervision in a
multi-task objective \citep{hsieh-etal-2023-distilling}; earlier rationale
models likewise separate prediction from explanation
\citep{lei-etal-2016-rationalizing}. ViPragSent brings this
rationale-supervision idea to a shared XLM-R-large encoder and tests it
alongside Vietnamese pragmatic heads, while keeping rationale generation
outside the deployed prediction interface.

### 2.3 Multi-task objectives, rationale supervision, and calibration

Multi-task learning shares a representation across related objectives and can
transfer useful domain information between them \citep{caruana1997multitask}.
The uncertainty-weighted objective follows the motivation that task losses
need not contribute equally \citep{kendall2018multitask}. The implemented
rationale decoder supplies a teacher-forced auxiliary loss during training and
is discarded at inference.

Confidence is evaluated separately from discrimination: expected calibration
error measures the gap between confidence and empirical correctness across
bins, which can remain large when accuracy is strong
\citep{guo2017calibration}. Q4 is therefore a protocol-specific calibration
comparison, not a universal uncertainty claim.

## 3 Method

### 3.1 Task definition

Given a Vietnamese utterance $x$, ViPragSent predicts six binary pragmatic
labels in the manifest order: implicit sentiment, sarcasm, irony,
idiom/figurative language, code-switching, and mocking. These labels represent
distinct pragmatic phenomena. The model may also predict a three-way polarity
label and a seven-way emotion label. The auxiliary labels are not additional
pragmatic classes.

For each pragmatic head, the metric is binary macro-F1: the arithmetic mean of
the F1 values for class 0 and class 1. The macro-pragmatic F1 is the arithmetic
mean of the six binary macro-F1 values. This definition is important for the
head-level table because it differs from a positive-class-only F1.

### 3.2 XLM-R-large encoder and prediction heads

The primary model uses `FacebookAI/xlm-roberta-large` at revision
`c23d21b0620b635a76227c604d44e43a9f0ee389`. The encoder returns a hidden state
for each input token. For the XLM-R family, the implementation pools the first
nonpadding representation: the hidden state at the first position with an
active attention mask, rather than a mean over the sequence. A 0.1 dropout
layer feeds task-specific linear heads: six one-logit binary heads, one
three-logit polarity head, and one seven-logit emotion head. At inference,
predictions come from these classification heads.

### 3.3 Implemented auxiliary objective

The six pragmatic heads use weighted binary cross-entropy with logits, with
positive-class weights computed from the training split. Polarity and emotion
use class-weighted cross-entropy. For the full model, the eight classification
losses have independent learned log-variance parameters. Let $\mathcal{T}$ be
the eight classification tasks, let $m_t$ be the recorded task multiplier,
and let $s_t$ be the learned log variance. The implementation clamps the log
variance before applying uncertainty weighting:

$$
  s_t^{\mathrm{c}} = \operatorname{clip}(s_t,-5,5), \qquad
  \mathcal{L}_{\mathrm{cls+rat}} =
  \sum_{t\in\mathcal{T}}
  \left[\frac{1}{2}\exp(-s_t^{\mathrm{c}})
  \left(m_t\ell_t\right) + \frac{1}{2}s_t^{\mathrm{c}}\right]
  + \beta\mathcal{L}_{\mathrm{rat}}, \qquad \beta=0.3.
$$

Here $\ell_t$ is the task loss before the multiplier, so the multiplier is
applied inside the uncertainty-weighted classification term. The rationale
term is present only for the full rationale-enabled variant.

The rationale component is a two-layer causal `TransformerDecoder`. It uses a
memory projection from the encoder hidden size to a decoder hidden size of 128,
four attention heads, feed-forward size 512, dropout 0.1, causal masking, and
tied token input/output embeddings. During training it consumes rationale
tokens with teacher forcing and contributes token cross-entropy as
$\beta\mathcal{L}_{\mathrm{rat}}$ with $\beta=0.3$. For the target run, the
manifest identifies `approved_generated_rationales_train.jsonl` with 7,998
training rows. The inspected evidence does not identify a provider for these
generated targets, so we treat this as a run-level training artifact rather
than evidence about the original dataset's annotation history. The decoder is
discarded from the inference path and produces no reported prediction.

### 3.4 Training and selection

The frozen ViPragSent package is identified in the data handoff as the
SEACrowd/ViSoBERT source package and contains 11,997 rows: 7,998 training,
1,999 development, and 2,000 test examples. The split seed is 20260520.
Training uses date-coded seeds 20260521, 20260522, and 20260523, reported in
the paper as logical seeds 21, 22, and 23. The primary Q1a v37 configuration
uses AdamW with learning rate $2\times10^{-5}$, weight decay 0.01, a maximum of
10 epochs, bf16 precision, physical batch size 8 with four accumulation steps
(effective batch size 32), gradient clipping 1.0, cosine scheduling, and a
0.20 warmup ratio. Early stopping patience is 10. The primary loss
multipliers are 1.01 for implicit sentiment, 1.01 for sarcasm, 1.05 for
irony, and 1.04 for code-switching; the other task multipliers are 1.0. The
selection metric is development macro-pragmatic F1.

For each binary pragmatic head, the development threshold is selected from
0.05 through 0.95 in steps of 0.01 by binary macro-F1, with ties resolved
toward 0.5. Q3 applies this rule independently for each seed and budget. Q4
uses raw positive-class sigmoid probabilities and therefore does not threshold
the probabilities before computing ECE.

### 3.5 Provenance and comparison scope

The Q1a evidence package contains 18 saved prediction JSONL files: three
ViPragSent target runs and three runs for each of five complete baseline
families. Every one of the 45 target--baseline pairs has 2,000 rows with the
same sample-ID set and the same six pragmatic gold-label tuples when aligned by
sample ID. The Q1a scores were deterministically recomputed from those saved
rows using the metric above. This supports a direct score comparison for the
evaluated ID/gold cohort. It does not claim identical upstream input text,
identical training, a significance test, or an independent rerun.

## 4 Experimental Setup

### 4.1 Separate experimental cohorts

Q1a, Q2, Q3, and Q4 are related but distinct artifact cohorts. The distinction
is material when interpreting the full-model values.

| Cohort | Run family and purpose | Key protocol |
|---|---|---|
| Q1a | Optimized v37 ViPragSent and five complete baseline families | Cosine scheduler, warmup 0.20, patience 10, development macro-pragmatic-F1 selection; target task multipliers as above |
| Q2 | XLM-R follow-up full model and five ablations | Linear scheduler, warmup 0.10, patience 2; named auxiliary and uncertainty variants |
| Q3 | XLM-R follow-up nested positive-label budgets | Linear scheduler, warmup 0.10, patience 2; development sarcasm binary macro-F1 selection |
| Q4 | Calibration extraction | Same-seed extraction from each Q1a v37 full checkpoint; no new training |

The Q2 full value is not a rerun of Q1a v37: the cohorts differ in scheduler,
warmup ratio, and early-stopping patience. Q2 is therefore interpreted within
its own follow-up cohort, not as an explanation of the difference between
Q1a's 93.67 and Q2's 92.8.

### 4.2 Data and external evaluation

Q1a evaluates the 2,000-example pragmatic test cohort. Q1b reports retention
on UIT-VSFC, UIT-VSMEC, and AIVIVN. The external aggregate is called ordinary
F1 in this paper: it is the unweighted mean of the three external macro-F1
values. It is not ordinal F1 and does not encode an ordinal relation among
labels. External datasets have different domains and label semantics, so their
scores are not pooled with the primary macro-pragmatic F1.

### 4.3 Baselines and reporting conventions

ViPragSent XLM-R-large is the only primary model. Comparison records include
standard XLM-R, PhoBERT single-task and fine-tuned variants, Sailor, and
Vistral where complete three-seed summaries are available. Vistral
ViPragSent variants and incomplete GPT records are outside the primary Q1a
comparison. ViSoBERT is contextual prior work, not an evaluated baseline in
this study.

The available baseline manifests describe family-specific recipes rather than
one recipe shared with the target. The standard XLM-R baseline uses AdamW at
$2\times10^{-5}$, bf16, physical batch size 8 with four accumulation steps,
linear scheduling with 0.10 warmup, and a ten-epoch maximum; its selection
metric is development macro-pragmatic F1. The PhoBERT fine-tune uses AdamW at
$2\times10^{-5}$, bf16, physical batch size 32, linear scheduling with 0.10
warmup, and the same ten-epoch maximum. The PhoBERT single-task record is a
bundle of six separately trained component artifacts and exposes no single
aggregate optimizer recipe. Sailor-7B and Vistral-7B-Chat are causal-7B
pragmatic-SFT records \citep{dou-etal-2024-sailor,nguyen-etal-2023-vistral}
using paged AdamW 8-bit, learning rate $10^{-4}$, bf16, physical batch size 2
with eight accumulation steps, cosine scheduling, 0.05 warmup, and a
three-epoch maximum; their reported inference source is the classification
heads. These records are described for provenance and interpretation only:
the ViPragSent v37 cosine schedule, warmup, patience, and task multipliers are
not claimed for the baselines.

Unless otherwise stated, a value is the arithmetic mean over three saved runs
and the uncertainty is the sample standard deviation with $n=3$. Q1a and Q3
report F1 in percentage points. Q2 reports macro-pragmatic F1, ECE multiplied
by $10^3$, and relative cost. Q4 reports ECE on its native [0,1] scale. Lower
ECE is better. No significance test is applied to the three-seed summaries.

## 5 Results

### 5.1 Q1a: in-domain pragmatic performance

Table 1 gives the main result at the level of all six pragmatic phenomena and
their macro average. Values are mean +/- sample SD in percentage points over
seeds 21, 22, and 23. The target is highest in mean on every head and on the
macro average among the five complete-seed baseline families. The strongest
baseline is standard XLM-R for every row.

| Phenomenon | ViPragSent XLM-R-large | XLM-R baseline | PhoBERT single | PhoBERT fine-tune | Sailor | Vistral |
|---|---:|---:|---:|---:|---:|---:|
| Implicit sentiment | 92.67 +/- 0.24 | 92.16 +/- 0.13 | 88.00 +/- 1.29 | 85.76 +/- 0.72 | 85.23 +/- 0.11 | 87.68 +/- 0.42 |
| Sarcasm | 89.36 +/- 0.90 | 88.12 +/- 0.43 | 70.87 +/- 1.34 | 72.64 +/- 0.50 | 72.24 +/- 0.50 | 75.40 +/- 1.06 |
| Irony | 98.43 +/- 0.20 | 97.93 +/- 0.10 | 97.13 +/- 1.06 | 97.45 +/- 0.19 | 96.40 +/- 0.55 | 97.02 +/- 0.01 |
| Idiom/figurative | 93.35 +/- 0.66 | 92.85 +/- 0.68 | 91.21 +/- 0.78 | 86.63 +/- 0.66 | 91.17 +/- 0.41 | 91.52 +/- 0.35 |
| Code-switching | 94.84 +/- 0.56 | 94.27 +/- 0.04 | 91.08 +/- 1.51 | 91.73 +/- 0.53 | 90.48 +/- 0.87 | 93.22 +/- 0.45 |
| Mocking | 93.35 +/- 0.73 | 92.27 +/- 0.41 | 79.16 +/- 2.99 | 78.53 +/- 1.01 | 77.82 +/- 0.08 | 81.16 +/- 1.16 |
| **Macro-pragmatic F1** | **93.67 +/- 0.19** | **92.93 +/- 0.14** | 86.24 +/- 0.49 | 85.46 +/- 0.36 | 85.56 +/- 0.10 | 87.67 +/- 0.06 |

Against the strongest baseline, the observed mean gaps are +0.51 pp for
implicit sentiment, +1.24 pp for sarcasm, +0.50 pp for irony, +0.50 pp for
idiom/figurative language, +0.57 pp for code-switching, and +1.07 pp for
mocking. The two largest observed head gaps are therefore sarcasm and mocking,
but these are descriptive differences across saved predictions, not
significance claims. The macro difference is 93.665838 - 92.934251 = +0.731587
percentage points, reported as +0.73 pp.

### 5.2 Q1b: external retention

Table 2 shows mixed transfer rather than uniformly preserved performance. The
ViPragSent XLM-R record is higher than Sailor on UIT-VSFC and ordinary F1,
lower on UIT-VSMEC and AIVIVN, and differs in variability. Vistral is included
as comparison context.

| Model record | UIT-VSFC | UIT-VSMEC | AIVIVN | Ordinary F1 |
|---|---:|---:|---:|---:|
| Sailor | 34.4 +/- 3.2 | 40.7 +/- 0.7 | 48.7 +/- 1.4 | 41.3 +/- 1.2 |
| ViPragSent XLM-R-large | 37.9 +/- 2.8 | 40.0 +/- 0.7 | 47.9 +/- 2.2 | 42.0 +/- 1.8 |
| Vistral | 30.7 +/- 6.4 | 39.4 +/- 1.5 | 49.0 +/- 0.8 | 39.7 +/- 2.5 |

All entries are mean +/- sample SD over three saved runs. The external datasets
differ in domain and target semantics, so the table is a retention profile
rather than a single generalization score. The larger spread on some target
rows also cautions against interpreting a single mean as stable transfer.

### 5.3 Q2: auxiliary-task and uncertainty ablations

Table 3 reports the separate XLM-R follow-up cohort. Within this cohort, the
full configuration is not the highest reported macro-pragmatic F1: removing
task-uncertainty weighting gives 92.9 versus 92.8 for the full configuration.
The full configuration has lower reported ECE than the no-uncertainty variant,
while removing the explanation auxiliary reduces relative cost substantially
and changes F1 and ECE. Removing the multitask bundle causes a much larger F1
reduction. These are trade-offs within Q2, not evidence that one component is
universally beneficial.

| XLM-R variant | Macro-pragmatic F1 | Saved polarity ECE x $10^3$ | Relative cost |
|---|---:|---:|---:|
| Full follow-up | 92.8 +/- 0.6 | 78.3 +/- 5.9 | 1.00 |
| No emotion auxiliary | 91.7 +/- 0.7 | 61.8 +/- 27.0 | 0.65 |
| No explanation auxiliary | 92.3 +/- 1.3 | 68.4 +/- 21.2 | 0.20 |
| No multitask bundle | 74.7 +/- 2.3 | 115.3 +/- 70.1 | 1.10 |
| No polarity auxiliary | 92.8 +/- 0.2 | N/A | 0.81 |
| No task-uncertainty weighting | 92.9 +/- 0.3 | 81.6 +/- 10.8 | 0.82 |

The metric artifacts expose `polarity_dev_ece`; Table 3 combines test-split
macro-pragmatic F1 with this saved development-split three-way ECE on the
$10^3$ scale. It is not Q4's six-head calibration metric or a new top-label
recomputation. Because Q2 is a separate cohort, its scientific comparison is
the pattern across named variants, not an explanation of Q1a's absolute score.

### 5.4 Q3: pragmatic label-budget behavior

Table 4 reports the XLM-R follow-up budget profile together with contextual
PhoBERT and Vistral baseline curves. Values are mean +/- sample SD over three
runs. Thresholds are tuned on the fixed development split; the displayed
metrics are evaluated on the fixed test split.

| Positive examples | ViPragSent sarcasm / macro | PhoBERT sarcasm / macro | Vistral sarcasm / macro |
|---:|---:|---:|---:|
| 32 | 84.2 +/- 1.6 / 91.1 +/- 1.3 | 69.7 +/- 1.5 / 84.2 +/- 0.7 | 70.5 +/- 2.4 / 86.2 +/- 0.2 |
| 64 | 83.7 +/- 1.3 / 91.2 +/- 0.8 | 70.8 +/- 2.8 / 77.7 +/- 11.8 | 72.5 +/- 0.9 / 86.7 +/- 0.5 |
| 128 | 85.2 +/- 0.7 / 91.6 +/- 0.2 | 71.8 +/- 2.2 / 84.0 +/- 1.2 | 71.3 +/- 0.6 / 86.2 +/- 0.4 |
| 256 | 85.3 +/- 2.6 / 91.6 +/- 1.2 | 73.4 +/- 1.2 / 84.5 +/- 0.9 | 64.0 +/- 5.1 / 70.3 +/- 14.2 |
| 512 | 87.2 +/- 1.2 / 92.3 +/- 0.7 | -- | 73.9 +/- 1.2 / 87.2 +/- 0.3 |
| Full | 85.0 +/- 2.3 / 91.1 +/- 1.3 | -- | 75.3 +/- 0.8 / 87.4 +/- 0.2 |

The target exceeds the available contextual baseline rows at displayed
budgets, but the curves are non-monotonic: macro-pragmatic F1 rises from 91.1
at 32 positives to 92.3 at 512 and returns to 91.1 at full budget. Missing
512/full PhoBERT cells limit the baseline curve. This is a sample-efficiency
diagnostic, not evidence that less data is generally preferable; the supplied
plot visualizes the same cohort.

### 5.5 Q4: calibration

Under the Q4 extraction protocol, the ViPragSent XLM-R target has ECE
0.0386 +/- 0.0069, while the Vistral comparison record has ECE
0.0345 +/- 0.0041. Lower ECE is better, so the comparison does not support
target calibration dominance. Q4 extracts raw positive sigmoid probabilities
from same-seed Q1a v37 checkpoints without retraining; ECE depends on its
split, ten bins, and probability source.

## 6 Analysis and Discussion

### 6.1 What the primary result establishes

Q1a establishes a direct score comparison for the saved matched cohort: all 45
pairs share 2,000 sample IDs and six pragmatic gold tuples, and macro-F1 was
deterministically recomputed. ViPragSent records 93.67 versus 92.93 for
standard XLM-R (+0.73 pp), leads on all six heads, and has the largest observed
gaps on sarcasm (+1.24 pp) and mocking (+1.07 pp). It has the highest recorded
mean among the five complete baseline families, but the result remains
descriptive because recipes differ.

### 6.2 Auxiliary supervision is an operating-point choice

Q2 shows that auxiliary supervision is an operating-point choice. The full
variant has 92.8 macro-pragmatic F1, while removing uncertainty weighting is
slightly higher; the other variants trade F1, ECE, and cost in different
directions, with the no-multitask bundle showing a large F1 reduction. These
three-run comparisons do not isolate a causal mechanism. Q2 is a separate
cohort, not an explanation of Q1a's 93.67.

The rationale component is auxiliary teacher-forced token supervision during
training; classification heads remain the deployed interface. The results
therefore support a representation-learning interpretation, not faithful
natural-language explanations at inference.

### 6.3 Transfer and calibration expose different weaknesses

Q1b and Q4 keep in-domain performance from carrying the argument: external
retention is mixed, and Vistral has lower ECE under the supplied Q4 protocol.
Discrimination, transfer, and calibration answer different questions, so
deployment selection should match the operating point rather than a single
macro-F1 value.

### 6.4 Label budgets and variance

Q3 shows why positive-label budgets need run variation. The best displayed
target mean occurs at 512 positives, while the full budget does not dominate;
missing PhoBERT cells limit the contextual comparison. Threshold selection,
class balance, and optimization variance may contribute, but the artifacts do
not isolate them. Budget studies therefore need a fixed negative pool,
explicit thresholds, contextual baselines, and uncertainty summaries.

## 7 Limitations

The paper analyzes saved artifacts rather than an independent training or
inference rerun; deterministic recomputation verifies the Q1a arithmetic but
not fresh-run reproducibility. Q1a is scoped to the matched ID/gold prediction
cohort, while Q2 and Q3 are separate follow-up run families and Q4 extracts
calibration from Q1a v37 checkpoints. External retention is finite and
heterogeneous across UIT-VSFC, UIT-VSMEC, and AIVIVN, and ECE depends on its
specified split, binning, and probability source. The operational pragmatic
labels do not exhaust discourse or speaker intent, and the evidence does not
establish group-conditional fairness, annotation history, consent, licensing,
or institutional-review details.

## 8 Ethics and Responsible Use

The data handoff identifies a frozen ViPragSent package sourced through
SEACrowd/ViSoBERT and records raw-text preservation, split sizes, and label
fields. The current project record does not provide sufficient evidence for
this paper to assert consent procedures, institutional review or exemption,
compensation, licensing terms, or de-identification beyond those manifest
fields. We make no such claims.

Vietnamese text with pragmatic and affective signals may contain offensive,
personal, or socially sensitive language. Classification errors may
mischaracterize speakers or amplify stereotypes. The model is therefore an
analysis aid requiring human review, not an autonomous adjudicator of a
person's intent, emotion, credibility, or identity. Aggregate macro-F1 does
not establish equal performance across social groups, domains, or writing
styles. The manuscript does not reproduce individual example text, and the
system is not intended for high-impact decisions or inference of protected
attributes.

## 9 Conclusion

ViPragSent is an XLM-R-large multi-task model with separate pragmatic,
polarity, and emotion heads and a training-only rationale objective. On the
matched Q1a ID/gold cohort it records 93.67 +/- 0.19 macro-pragmatic F1 versus
92.93 +/- 0.14 for standard XLM-R (+0.73 pp), with the highest saved mean on
all six heads among five complete baseline families. Q1b is mixed, Q2 exposes
accuracy/calibration/cost trade-offs, Q3 is non-monotonic with contextual
baselines, and Q4 shows no blanket calibration advantage. The primary result
is descriptive and cohort-specific, but it supports the implemented model as
a focused account of Vietnamese pragmatic prediction.

## Appendix A. Reproducibility record

The following record is part of the manuscript rather than a future action list.
All paths are relative to the paper workspace unless a source path is explicitly
identified.

| Item | Recorded value or artifact |
|---|---|
| Primary backbone | `FacebookAI/xlm-roberta-large`, revision `c23d21b0620b635a76227c604d44e43a9f0ee389` |
| Model interface | First nonpadding-token pooling; six binary pragmatic heads, three-way polarity head, seven-way emotion head; classification heads are the inference source |
| Rationale implementation | Two-layer causal TransformerDecoder with memory projection; hidden 128, four heads, feed-forward 512, dropout 0.1; tied token embeddings; teacher forcing; beta 0.3; target maximum length 160; decoder discarded at inference |
| Rationale target provenance | Target manifest artifact `approved_generated_rationales_train.jsonl`; 7,998 training rows; provider unspecified in the inspected evidence; not treated as original annotation-history evidence |
| Dataset package | SEACrowd/ViSoBERT source package; 11,997 rows; train/dev/test = 7,998/1,999/2,000; split seed 20260520; local processed fingerprint `7C39BEEBC462D1F9076F5DF1565E924BD884263D5503F23398BD83A6BF4205EB` |
| Q1a target artifact provenance | Remote target dataset fingerprint `B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F9` |
| Q1a baseline artifact provenance | Remote baseline dataset fingerprint `A13573E38550ABD55D7F63E983C602BEA13C2765A1FFECA285F718478532AF0D` |
| Training seeds | Date-coded seeds 20260521, 20260522, 20260523; logical labels 21, 22, 23 |
| Primary Q1a optimizer | AdamW, learning rate 2e-5, weight decay 0.01, bf16, physical batch 8, accumulation 4, effective batch 32, maximum 10 epochs, patience 10, gradient clipping 1.0 |
| Primary Q1a schedule | Cosine, warmup ratio 0.20, development macro-pragmatic-F1 selection; task multipliers implicit 1.01, sarcasm 1.01, irony 1.05, code-switching 1.04, remaining tasks 1.0 |
| Follow-up Q2/Q3 schedule | AdamW, learning rate 2e-5, bf16, physical batch 8, accumulation 4, effective batch 32, maximum 10 epochs, linear schedule, warmup ratio 0.10, patience 2 |
| Threshold rule | Candidate thresholds 0.05--0.95, step 0.01; development binary macro-F1; ties toward 0.5 |
| Q3 mask rule | Nested positive subsets of 32, 64, 128, 256, 512, and full; fixed negative pool; positive weights recomputed per budget; development threshold per seed/budget |
| Q4 rule | Six pragmatic heads; raw positive sigmoid probabilities; ten equal-width bins; no temperature scaling; test split; mean and sample SD across seeds; no new training |
| Q1a evidence | `evidence/q1a_fairness_comparison.json`, `evidence/q1a_fairness_run_table.csv`, `evidence/q1a_verified_table_inputs.json`, `evidence/q1a_numeric_claim_ledger.json` |
| Method evidence | `evidence/method_evidence.json`, `evidence/evidence_claim_ledger.json` |
| Protocol lineage | `evidence/protocol_lineage.json`; records distinct Q1a v37, Q2, Q3, and Q4 run families and Q4 source-checkpoint lineage |
| Prediction rows | `raw_fairness/*.jsonl`; 18 files, 2,000 rows per file; deterministic metric recomputation matches recorded values |

The Q1a comparison covers 45 target--baseline pairs across the five complete
baseline families. Each pair has the same sample-ID set and six-label gold
tuples after alignment by ID. The comparison is descriptive over saved
predictions. No independent training or inference rerun is part of this record.
The three fingerprints above are retained as separate local and remote
provenance identifiers; the direct score comparison is scoped to the verified
same-ID, same-gold cohort rather than treated as a fingerprint-equality test.

## Appendix B. Table-specific protocol and missingness

Q1a uses the optimized v37 target and the complete three-seed baseline families
listed in Table 1. The target and standard XLM-R macro means are 93.665838 and
92.934251, with sample SDs 0.187121 and 0.144790 percentage points. Their
difference is 0.731587 percentage points. The per-head means and SDs in Table 1
were recomputed from the same saved JSONL prediction rows.

Q1b uses the external retention records and reports ordinary F1 as the
unweighted mean of the three external macro-F1 values. Q2 uses the six
follow-up variants: full, no emotion auxiliary, no polarity auxiliary, no
rationale, no multitask, and no uncertainty weighting. The selected Q2 metric
artifacts expose the field `polarity_dev_ece`; Table 3 reports
macro-pragmatic F1 on the Q2 test split and the saved three-way polarity ECE
from the development split on the $10^3$ scale. The paper does not expand
that field into an unverified top-label recomputation. Relative cost is
normalized to the full XLM-R Q2 run using the recorded mean GPU-hour values.
Q3 reports the six nested target budgets and the available contextual baseline
rows; missing baseline budget cells are not imputed. Q4 extracts raw
probabilities from the same-seed Q1a v37 full checkpoints.

Incomplete GPT records, ViSoBERT prediction records, and excluded ViPragSent
Vistral variants are not inserted into any table. Missing ECE for the Q2
no-polarity row is reported as N/A. These exclusions define the comparison
scope and do not support claims about systems without verified rows.

The structure-reference PDF supplied organization only. Its historical
metrics, training protocol, annotation or ethics assertions, hardware claims,
and generative-decoder description are not evidence for this manuscript.

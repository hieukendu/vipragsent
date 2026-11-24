# ViPragSent Standalone Annotation Rubric

**Purpose:** Standalone operational rubric for annotating one Vietnamese comment at a time with the ViPragSent label space.

**Source fidelity:** This rubric is aligned to the paper's task definition, annotation-scheme description, and representative gold examples. It intentionally does **not** invent decision rules that are not specified there.

---

## 1. Annotation unit and context

Annotate **one standalone Vietnamese comment at a time**.

Use only the text of that comment.

Do **not** use:

- the parent post;
- earlier or later replies;
- thread history;
- speaker/user profile information;
- outside assumptions about the speaker's intent;
- external situational context that is not recoverable from the comment itself.

The gold labels are **adjudicated readings of the standalone text**, not verified author intent.

---

## 2. Output label space

For every comment, assign:

### 2.1 Six pragmatic labels — multi-label binary decisions

Each pragmatic label is judged **independently**:

```text
implicit_sentiment = 0 or 1
sarcasm            = 0 or 1
irony              = 0 or 1
idiom_figurative   = 0 or 1
code_switching     = 0 or 1
mocking            = 0 or 1
```

`1` means the pragmatic phenomenon is **present**.

`0` means the pragmatic phenomenon is **absent**.

A positive pragmatic label means **presence of that phenomenon**, not positive sentiment.

Several pragmatic labels may be positive for the same comment.

### 2.2 Polarity — exactly one class

```text
polarity = negative | neutral | positive
```

Polarity represents the **intended evaluative polarity** of the standalone comment.

### 2.3 Emotion — exactly one dominant class

```text
emotion =
    enjoyment |
    sadness |
    anger |
    surprise |
    fear |
    disgust |
    other
```

Assign one **dominant emotion**.

The paper does not specify a more detailed tie-breaking rule when multiple emotions appear plausible; do not invent one beyond the project annotation convention.

---

# 3. Pragmatic-label decision rubric

## 3.1 Implicit sentiment

### Definition

**Evaluative stance without an overt sentiment expression.**

Assign:

```text
implicit_sentiment = 1
```

when the standalone comment conveys an evaluation or sentiment **implicitly**, rather than stating the sentiment overtly.

Assign:

```text
implicit_sentiment = 0
```

when this criterion is not met.

### Representative gold example

> **“Đào nhà t nở đc có 4-5 bông, cháu t vặt cả nụ lẫn hoa :))”**  
> *“The peach tree at my house only managed to bloom four or five flowers. My niece or nephew picked the buds and the flowers :)).”*

### Important

Do not infer `implicit_sentiment` merely because another pragmatic label is positive. Each label is independent.

---

## 3.2 Sarcasm

### Definition

**Target-directed, derisive or taunting incongruity.**

The paper describes sarcasm as requiring **incongruity that derides or taunts a target**.

Assign:

```text
sarcasm = 1
```

when both are supported by the standalone text:

1. there is an incongruity/mismatch in the expression; and
2. that incongruity functions to deride or taunt a target.

Assign:

```text
sarcasm = 0
```

when this combination is not supported.

### Representative gold example

> **“Ơ lần đầu à? Lần đầu thì xác định đi nhé, nó không thơm ngon như mọi người kể đâu =)))”**  
> *“Oh, your first time? Brace yourself. It is not as pleasant as everyone says =))).”*

### Distinction from irony and mocking

Sarcasm is **not automatically identical** to either irony or mocking.

- Irony requires a literal/intended mismatch.
- Mocking requires ridicule.
- Sarcasm specifically combines incongruity with target-directed derision or taunting.

Annotate all three independently.

---

## 3.3 Irony

### Definition

**Mismatch between literal wording and intended evaluation.**

Assign:

```text
irony = 1
```

when the standalone text supports a mismatch between what is literally expressed and the intended evaluation.

Assign:

```text
irony = 0
```

when such a mismatch is not supported.

### Representative gold example

> **“kênh hay mà ra video chậm nhỉ quá là buồn”**  
> *“A good channel, but it releases videos slowly—what a pity.”*

### Boundary caution

The paper explicitly notes that some adjudicated gold readings go beyond what the written criteria alone make unambiguous.

Therefore:

- do not force an irony label solely from a superficial cue;
- do not assume that praise-plus-complaint automatically implies irony;
- preserve adjudicated gold labels when evaluating existing gold data rather than retroactively relabeling them from this rubric alone.

---

## 3.4 Idiom / figurative language

### Definition

Positive cases include:

- a **conventional idiom**;
- a **metaphor**;
- a **nonliteral comparison**;
- **clearly nonliteral hyperbole**.

Assign:

```text
idiom_figurative = 1
```

when the expression is clearly nonliteral under one of these categories.

Assign:

```text
idiom_figurative = 0
```

when the text does not meet this criterion.

### Dictionary guidance

A Vietnamese dictionary may be used as a **lexical reference**.

However:

- dictionary membership is not a whitelist;
- dictionary absence does not automatically make an expression figurative;
- clear novel nonliteral expressions may still qualify.

### Representative gold example

> **“CT này chán như con zán”**  
> *“This program is as boring as a cockroach.”*

### Boundary caution

The paper reports this label as the most difficult annotation boundary among the six pragmatic labels. Apply the nonliteral criterion conservatively and do not equate unusual wording with figurative language automatically.

---

## 3.5 Code-switching

### Definition

**Alternation between Vietnamese and material from another language.**

Assign:

```text
code_switching = 1
```

when the standalone Vietnamese comment contains meaningful material from another language as part of the utterance.

Assign:

```text
code_switching = 0
```

when this criterion is not met.

### Representative gold example

> **“Haha trend mới mà”**  
> *“Haha, it is a new trend.”*

### Important

Code-switching does not automatically imply sentiment, irony, sarcasm, mocking, or figurative language.

Judge those labels separately.

---

## 3.6 Mocking

### Definition

**Derisive expression directed at a target.**

The paper distinguishes mocking from irony by noting that mocking requires **ridicule without necessarily requiring a literal-to-intended mismatch**.

Assign:

```text
mocking = 1
```

when the comment directs ridicule or derision at a target.

Assign:

```text
mocking = 0
```

when target-directed ridicule is not supported.

### Representative gold example

> **“Flash thua theo chiến thuật làm anh tưởng đánh yếu lo vãi”**  
> *“Flash lost strategically and made me think he was weak. That was frightening.”*

### Distinction from sarcasm and irony

- `mocking` requires ridicule.
- `irony` requires literal/intended mismatch.
- `sarcasm` requires target-directed derisive/taunting incongruity.

They may co-occur, but none should be mechanically inferred from another.

---

# 4. Cross-label rules

## Rule 1 — Labels are independent

For every comment, evaluate all six pragmatic labels separately.

Do not use a shortcut such as:

```text
sarcasm = 1  => irony = 1
mocking = 1  => sarcasm = 1
code_switching = 1 => implicit_sentiment = 1
```

No such implication rule is licensed.

---

## Rule 2 — Multi-label cases are valid

A comment can have multiple pragmatic phenomena simultaneously.

The paper gives the gold example:

> **“Haha trend mới mà”**

with three positive pragmatic labels:

```text
implicit_sentiment = 1
irony              = 1
code_switching     = 1
```

The example demonstrates **co-occurrence**, not a rule that any one of these labels implies the others.

---

## Rule 3 — Sarcasm / irony / mocking must be separated conceptually

Use these core tests:

| Label | Core question |
|---|---|
| Sarcasm | Is there target-directed derisive/taunting **incongruity**? |
| Irony | Is there a **mismatch between literal wording and intended evaluation**? |
| Mocking | Is there **target-directed ridicule**, even without literal/intended mismatch? |

Evaluate each independently.

---

## Rule 4 — Figurative language is not inferred from lexical novelty

A novel, slang, misspelled, or uncommon expression is not automatically figurative.

The expression must support a nonliteral interpretation under the project definition.

---

## Rule 5 — Code-switching is a linguistic-form label

Foreign-language material can co-occur with any sentiment or pragmatic function.

Do not infer the other labels simply because code-switching is present.

---

# 5. Polarity rubric

Assign exactly one of:

```text
negative
neutral
positive
```

Use the comment's **intended evaluation**, rather than automatically following isolated surface words.

The paper's motivating example illustrates why this matters:

> “Tuyệt thật, lại phải chờ hai tiếng.”  
> *“Great, another two-hour wait.”*

Surface praise can coexist with an intended negative judgment.

### Decision rule

Choose the class that best represents the intended overall evaluation supported by the standalone comment:

- `positive` — intended overall evaluation is positive;
- `negative` — intended overall evaluation is negative;
- `neutral` — no sufficiently positive or negative intended evaluation is supported.

Do not derive polarity mechanically from the pragmatic labels.

---

# 6. Emotion rubric

Assign exactly one dominant emotion:

```text
enjoyment
sadness
anger
surprise
fear
disgust
other
```

Use only the standalone comment.

### Important limitation

The paper specifies the seven-category space and that one **dominant emotion** is assigned, but it does not provide detailed per-category inclusion/exclusion criteria or a formal tie-breaking procedure.

Therefore this standalone rubric should not invent additional emotion-specific rules.

When applying the original ViPragSent gold standard, preserve the established gold/adjudicated emotion label.

---

# 7. Annotation workflow

## 7.1 First pass

Two native-Vietnamese annotators independently assign:

- all six pragmatic labels;
- one polarity label;
- one dominant emotion label.

The first-pass annotators work from standalone comments only.

They do not use:

- parent posts;
- thread history;
- speaker profiles.

---

## 7.2 Finalization / adjudication

Final labels are established through **discussion among three annotators**, including review of difficult cases and cases arising from first-pass disagreement or uncertainty.

The final adjudicated decisions form the gold label set.

### Important

The paper notes that adjudicated gold can preserve discussion-based readings beyond what the written criteria alone make fully explicit.

Therefore the rubric is an operational guide; it is not a license to overwrite established gold labels purely because a later reader would make a different standalone judgment.

---

# 8. Agreement reporting

Agreement is measured **before adjudication**.

The paper uses:

- nominal **Krippendorff's alpha** for each pragmatic label;
- **Cohen's kappa** for polarity;
- **Cohen's kappa** for emotion.

This section is for annotation-quality reporting, not for deciding an individual label.

---

# 9. Per-comment annotation checklist

For each comment:

```text
[ ] I used only the standalone comment.
[ ] I did not use thread history, parent posts, profiles, or outside context.
[ ] I judged each pragmatic label independently.
[ ] I did not interpret "positive pragmatic label" as positive sentiment.
[ ] I distinguished sarcasm, irony, and mocking using their separate criteria.
[ ] I used a dictionary only as a lexical reference for figurative-language questions.
[ ] I allowed clear novel nonliteral expressions to count as figurative when justified.
[ ] I did not infer figurativeness from unusual wording alone.
[ ] I assigned exactly one polarity class.
[ ] I assigned exactly one dominant emotion class.
```

Then record:

```yaml
implicit_sentiment: 0
sarcasm: 0
irony: 0
idiom_figurative: 0
code_switching: 0
mocking: 0

polarity: neutral

emotion: other
```

---

# 10. Recommended annotation output format

For manual annotation:

```json
{
  "comment": "<original Vietnamese comment>",
  "pragmatic": {
    "implicit_sentiment": 0,
    "sarcasm": 0,
    "irony": 0,
    "idiom_figurative": 0,
    "code_switching": 0,
    "mocking": 0
  },
  "polarity": "neutral",
  "emotion": "other"
}
```

Optional annotator note:

```json
{
  "note": "Short explanation only when the decision is genuinely ambiguous."
}
```

Do not require a rationale for every routine item unless the annotation protocol explicitly asks for one.

---

# 11. Quick-reference table

| Label | Positive criterion |
|---|---|
| `implicit_sentiment` | Evaluative stance without an overt sentiment expression |
| `sarcasm` | Target-directed, derisive or taunting incongruity |
| `irony` | Mismatch between literal wording and intended evaluation |
| `idiom_figurative` | Conventional idiom, metaphor, nonliteral comparison, or clearly nonliteral hyperbole |
| `code_switching` | Alternation between Vietnamese and material from another language |
| `mocking` | Derisive expression directed at a target |
| `polarity` | Intended evaluation: negative / neutral / positive |
| `emotion` | One dominant category among enjoyment / sadness / anger / surprise / fear / disgust / other |

---

# 12. Representative gold examples

| Label | Vietnamese example | English gloss |
|---|---|---|
| Implicit sentiment | “Đào nhà t nở đc có 4-5 bông, cháu t vặt cả nụ lẫn hoa :))” | “The peach tree at my house only managed to bloom four or five flowers. My niece or nephew picked the buds and the flowers :)).” |
| Sarcasm | “Ơ lần đầu à? Lần đầu thì xác định đi nhé, nó không thơm ngon như mọi người kể đâu =)))” | “Oh, your first time? Brace yourself. It is not as pleasant as everyone says =))).” |
| Irony | “kênh hay mà ra video chậm nhỉ quá là buồn” | “A good channel, but it releases videos slowly—what a pity.” |
| Idiom/figurative | “CT này chán như con zán” | “This program is as boring as a cockroach.” |
| Code-switching | “Haha trend mới mà” | “Haha, it is a new trend.” |
| Mocking | “Flash thua theo chiến thuật làm anh tưởng đánh yếu lo vãi” | “Flash lost strategically and made me think he was weak. That was frightening.” |

---

# 13. What this rubric does **not** specify

To remain faithful to the paper, this rubric does **not** invent:

- external conversational context;
- speaker-intent verification;
- a hidden hierarchy among the six pragmatic labels;
- automatic implication rules between labels;
- detailed emotion-specific lexical rules;
- a formal emotion tie-breaker beyond selecting one dominant emotion;
- a requirement that every comment receive a written rationale;
- retroactive relabeling of existing adjudicated gold based only on this document.

When the written rubric and an established adjudicated ViPragSent gold label differ, treat that case as an annotation-boundary example rather than silently changing the gold label.

# ViPragSent Standalone Annotation Manual

## Purpose

This manual formalizes the operational annotation scheme used for ViPragSent. It covers six binary pragmatic labels, three-way polarity, seven-way emotion, cross-label relations, idiom/figurative-language decisions, dictionary consultation, and independent annotation with adjudication.

The manual documents the labeling rules. It does not redefine the dataset or retroactively change the original agreement statistics.

---

## 1. Annotation unit and context

The annotation unit is **one standalone Vietnamese comment**.

Annotators must judge only the text of the current comment. Do **not** use the parent post, previous/following comments, thread history, user profiles, usernames, URLs, identity information, or other unseen metadata/context.

If a label cannot be supported from the standalone comment, do not infer it from imagined context.

---

## 2. Output label schema

### 2.1 Six binary pragmatic labels

Annotate each independently as `0` or `1`:

1. `implicit_sentiment`
2. `sarcasm`
3. `irony`
4. `idiom_figurative`
5. `code_switching`
6. `mocking`

These labels are **multi-label** and **not mutually exclusive**.

### 2.2 Polarity

Choose exactly one:

- `negative`
- `neutral`
- `positive`

### 2.3 Emotion

Choose exactly one dominant emotion:

- `enjoyment`
- `sadness`
- `anger`
- `surprise`
- `fear`
- `disgust`
- `other`

---

## 3. Pragmatic-label definitions

### 3.1 Implicit sentiment

Label `1` when the comment conveys an evaluative stance or sentiment without relying on a direct, overt sentiment expression as the main signal. Typical cues include implication, indirect approval/disapproval, or evaluative meaning that must be inferred from the described event/situation.

Label `0` when no implicit evaluative stance is supported by the standalone comment.

Representative example:

> “Đào nhà t nở đc có 4-5 bông, cháu t vặt cả nụ lẫn hoa :))”

### 3.2 Sarcasm

Label `1` when the comment contains **target-directed, derisive or taunting evaluative incongruity**.

Sarcasm generally involves a target plus ridicule/derision/taunting and an evaluative contrast between surface wording and intended meaning. Do not label sarcasm merely because a comment is negative, humorous, or insulting.

Representative example:

> “Ơ lần đầu à? Lần đầu thì xác định đi nhé, nó không thơm ngon như mọi người kể đâu =)))”

### 3.3 Irony

Label `1` when there is a broader **mismatch between literal wording and intended evaluation**.

Irony does not require the target-directed derision characteristic of sarcasm.

Key distinction:

- **irony**: literal–intended evaluative incongruity;
- **sarcasm**: target-directed derisive or taunting incongruity.

A comment may be both ironic and sarcastic if both definitions are independently satisfied.

Representative example:

> “kênh hay mà ra video chậm nhỉ quá là buồn”

### 3.4 Idiom / figurative language

Label `1` when the comment contains a clearly nonliteral expression belonging to one or more of the following categories:

- conventional idiom;
- metaphor;
- nonliteral or figurative comparison/simile;
- clearly nonliteral hyperbole.

Representative example:

> “CT này chán như con zán”

#### Include

- conventional idioms;
- metaphors;
- clearly figurative comparisons;
- productive figurative expressions;
- clearly nonliteral exaggeration/hyperbole.

#### Exclude

Do **not** label `idiom_figurative = 1` solely because the comment contains:

- a literal comparison;
- ordinary slang used literally;
- profanity alone;
- an exclamation alone;
- a meme phrase/catchphrase alone;
- an unusual phrase whose nonliteral function is unsupported by the comment.

A meme/catchphrase may still receive the label when it is actually functioning nonliterally in the comment.

#### Dictionary guidance

Use **Từ điển tiếng Việt** (Viện Ngôn ngữ học, 1994; Hoàng Phê, chủ biên) as a lexical/conventional-meaning reference.

The dictionary is **not a whitelist**. An expression may still be annotated as figurative when absent from the dictionary if its productive nonliteral use is clear from the standalone comment.

### 3.5 Code-switching

Label `1` when the comment alternates between Vietnamese and lexical/material elements from another language.

Code-switching is independent of the other pragmatic labels and may co-occur with any of them.

Representative example:

> “Haha trend mới mà”

Do not infer pragmatic intent merely from the presence of foreign-language material.

### 3.6 Mocking

Label `1` when the comment contains **target-directed ridicule or derision**.

Unlike sarcasm, mocking does **not** require literal–intended incongruity. A mocking comment may be literal, and it may also be sarcastic when both definitions are satisfied.

Representative example:

> “Flash thua theo chiến thuật làm anh tưởng đánh yếu lo vãi”

---

## 4. Cross-label relations

The six pragmatic labels must be decided **independently**. Do not impose a hierarchy or mutual-exclusion rule.

Allowed relations include:

- `sarcasm` may co-occur with `irony`;
- `mocking` may co-occur with `sarcasm`;
- `mocking` may co-occur with `irony`;
- `implicit_sentiment` may co-occur with `irony` or `mocking`;
- `code_switching` may co-occur with any pragmatic label;
- `idiom_figurative` may co-occur with any other pragmatic label when its own definition is independently satisfied.

Do not infer one label solely because another label is present. In particular:

- sarcasm does not automatically imply irony;
- irony does not automatically imply sarcasm;
- mocking does not automatically imply sarcasm;
- profanity does not automatically imply mocking;
- code-switching does not automatically imply sentiment or another pragmatic label.

---

## 5. Polarity annotation

Assign one overall **intended polarity**.

### Positive
Use `positive` when the intended overall evaluation is favorable, approving, appreciative, or otherwise positive.

### Negative
Use `negative` when the intended overall evaluation is unfavorable, critical, disapproving, hostile, or otherwise negative.

### Neutral
Use `neutral` when the comment does not express a sufficiently supported positive or negative overall evaluation.

### Sarcasm and irony
For sarcastic or ironic comments, annotate **intended polarity**, not merely literal surface wording. Do not infer polarity from unavailable conversational context.

---

## 6. Emotion annotation

Assign exactly one **dominant emotion**:

- `enjoyment`
- `sadness`
- `anger`
- `surprise`
- `fear`
- `disgust`
- `other`

Choose the emotion most strongly supported by the standalone comment.

Use `other` when no listed emotion clearly dominates, the affective state is outside the six named emotions, or the evidence is insufficient to assign one of them reliably.

For sarcasm and irony, use the intended affective meaning rather than blindly following literal wording.

Polarity and emotion are separate decisions: a polarity label does not mechanically determine the emotion label.

---

## 7. Recommended annotation order

For each standalone comment:

1. Read the full comment once without assigning labels.
2. Annotate each of the six pragmatic labels independently.
3. Assign overall intended polarity.
4. Assign one dominant emotion.
5. Re-check possible co-occurrence, especially sarcasm/irony/mocking, figurative language with other pragmatic phenomena, and code-switching with any other label.
6. For uncertain lexical/conventional expressions, consult **Từ điển tiếng Việt** as a reference.
7. Do not change one label merely to make the label set look internally “clean”; each label must satisfy its own operational definition.

---

## 8. Independent annotation and adjudication

The annotation workflow uses:

- **Annotator 1**: labels independently;
- **Annotator 2**: labels independently;
- **Adjudicator**: resolves disagreements.

The two primary annotators should not coordinate their decisions before initial independent annotation is complete.

During adjudication:

1. identify the disputed label(s);
2. apply the operational definition in this manual;
3. use only the standalone comment;
4. consult the dictionary when the disagreement concerns lexical/conventional figurative meaning;
5. assign the final adjudicated label.

Adjudication should not introduce unavailable external context.

---

## 9. Quick decision checklist

- [ ] I used only the standalone comment.
- [ ] I evaluated all six pragmatic labels independently.
- [ ] I did not force sarcasm, irony, and mocking to be mutually exclusive.
- [ ] I did not label figurative language merely because of slang, profanity, or a meme phrase.
- [ ] If I consulted the dictionary, I used it as a reference rather than a whitelist.
- [ ] Polarity reflects intended overall evaluation.
- [ ] Emotion is one dominant class from the seven-class inventory.
- [ ] Code-switching did not automatically determine any pragmatic or sentiment label.
- [ ] Any co-occurring labels independently satisfy their own definitions.

---

## 10. Annotation schema

```text
comment
implicit_sentiment: 0/1
sarcasm: 0/1
irony: 0/1
idiom_figurative: 0/1
code_switching: 0/1
mocking: 0/1
polarity: negative | neutral | positive
emotion: enjoyment | sadness | anger | surprise | fear | disgust | other
```

---

## 11. Reference

Viện Ngôn ngữ học. 1994. *Từ điển tiếng Việt*. Hoàng Phê (chủ biên). Trung tâm Từ điển học / Nhà xuất bản Giáo dục, Hà Nội.

The dictionary is used as a lexical/conventional-meaning reference for annotation decisions; it does not define the complete pragmatic label inventory and is not treated as an exhaustive whitelist.

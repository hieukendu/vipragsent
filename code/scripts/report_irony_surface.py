"""Export interpretation and an explicitly exploratory lexical diagnostic."""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from analyze_irony_surface import OUT, ROOT, SEEDS, cues, load_split, metrics, write_csv
from analyze_priority3_positive_class import _full_run_root, _read_jsonl

# Close readings of the 21 cases selected before inspecting their wording.
# These are assistant interpretations, not independent human reannotations.
NOTES = {
    "05851": ("evaluative_contrast", "partial", "Praise of another person's hand contrasts with self-criticism. Both clauses can be literal; reversal of intended stance is not explicit."),
    "06427": ("counterfactual_regret", "unclear", "Regret and a conditional intention to book are visible. Emoji alone does not establish a literal-versus-intended reversal."),
    "06911": ("affect_behavior_incongruity", "partial", "Feeling guilt for the poster contrasts with reading and laughing. This is an explicit behavioral contrast, without an overt reversal of literal wording."),
    "08771": ("literal_gratitude", "unclear", "Thanks to a mother for enduring childbirth can be read literally. A reversed intended evaluation cannot be established from this standalone comment."),
    "09059": ("standalone_context_ambiguity", "unclear", "A narrated return home after parental pressure has no clearly expressed opposite stance; interpreting irony would require additional assumptions."),
    "09527": ("mock_praise", "clear", "The closing praise of honesty occurs within a critical account of contradictory behavior and laughing/upside-down emoji. Mock praise is a plausible standalone reading."),
    "09542": ("evaluation_before_use", "partial", "The contrast is satisfaction before trying on the item. It could reflect an incongruous evaluation or a literal first impression; the text does not settle intent."),
    "09758": ("general_vs_self_contrast", "partial", "Belief that happiness exists contrasts with doubt about obtaining it. This is explicit evaluative contrast, but may be sincere rather than verbally ironic."),
    "10882": ("literal_consolation", "unclear", "Advice and consolation can be interpreted literally. The final question does not establish irony without a conflicting intended stance."),
    "11607": ("literal_product_praise", "unclear", "A product review contrasts skin sensitivity with satisfaction. This can be a coherent literal recommendation rather than a reversed evaluation."),
    "00490": ("fragment_missing_context", "unclear", "A short fragment about a mask supplies no recoverable reversal in isolation. Gold irony is negative."),
    "04631": ("literal_affective_assertion", "unclear", "The expression of enjoyment with elongated màaa can be literal. It resembles an emphatic contrast particle but is not the exact token mà."),
    "06940": ("direct_mocking_with_laughter", "partial", "Direct football-related ridicule with :))) is present; gold sarcasm and mocking are positive, irony negative. This illustrates the operational boundary between derision and irony."),
    "11452": ("complaint_with_laughter", "partial", "A complaint about overlooked paternal effort ends in :))). Gold sarcasm and mocking are positive, irony negative; laughter does not uniquely mark irony."),
    "02285": ("context_dependent_praise", "unclear", "A short comment about placement beside two attractive women has =)), but intended reversal cannot be verified without the unseen referent."),
    "03693": ("direct_taunt_with_laughter", "unclear", "A derogatory title and kkk convey ridicule. No explicit praise-to-criticism reversal is visible."),
    "05327": ("metalinguistic_irony_mention", "unclear", "The comment mentions mỉa mai. Mentioning an ironic act is distinct from demonstrating an ironic reversal within this text."),
    "05582": ("metalinguistic_irony_mention", "unclear", "The expression bé mỉa mai refers to someone being sarcastic/ironic, but the standalone utterance does not clearly perform a reversed evaluation."),
    "08089": ("hyperbolic_praise", "partial", "Repeated thanks to every participant in a purchase can support hyperbolic mock praise. A sincere enthusiastic reading remains possible without further context."),
    "08223": ("explicit_polarity_reversal", "clear", "Congratulations on opening immediately contrasts with impending closure. The positive formula and adverse outcome supply a clear standalone reversal."),
    "11779": ("contrast_without_reversal", "partial", "Avoiding society contrasts with enjoyment of solitary phone use. This can be a literal account of social discomfort rather than reversed intent."),
}


def main():
    output = OUT
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    frequency = list(csv.DictReader((output / "cue_frequencies.csv").open(encoding="utf-8-sig")))
    cases = list(csv.DictReader((output / "qualitative_cases.csv").open(encoding="utf-8-sig")))
    data = {s: load_split(s) for s in ("train", "dev", "test")}
    # These lexical rules are exploratory: motivated by the training-fitted
    # coefficient inspection, after the primary test analysis was available.
    rules = {"mà": r"(?<!\w)mà(?!\w)", "nhưng": r"(?<!\w)nhưng(?!\w)",
             "mà_or_nhưng": r"(?<!\w)(?:mà|nhưng)(?!\w)"}
    lexical = []
    for split, rows in data.items():
        y = np.array([int(r["irony"]) for r in rows])
        for name, pattern in rules.items():
            p = np.array([bool(re.search(pattern, unicodedata.normalize("NFC", r["text"]).casefold())) for r in rows])
            tp, fp = int(np.sum(p & (y == 1))), int(np.sum(p & (y == 0)))
            lexical.append({"split": split, "cue": name, "irony_count": tp, "irony_total": int(y.sum()),
                            "irony_percent": 100 * tp / y.sum(), "non_irony_count": fp,
                            "non_irony_total": int((y == 0).sum()), "non_irony_percent": 100 * fp / (y == 0).sum(),
                            "positive_precision": float(precision_score(y, p, zero_division=0)),
                            "positive_recall": float(recall_score(y, p, zero_division=0)),
                            "positive_f1": float(f1_score(y, p, zero_division=0)),
                            "binary_macro_f1": float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)),
                            "tp": tp, "fp": fp, "fn": int(y.sum()) - tp, "tn": int((y == 0).sum()) - fp,
                            "scope": "exploratory_lexical_diagnostic"})
    write_csv(output / "exploratory_lexical_cues.csv", lexical)
    annotated = []
    for row in cases:
        category, strength, note = NOTES[row["sample_id"].rsplit("_", 1)[1]]
        annotated.append({**row, "text_pattern": category, "standalone_ironic_reading": strength,
                          "interpretive_note": note, "review_type": "assistant_close_reading_not_human_reannotation"})
    write_csv(output / "qualitative_review.csv", annotated)
    def lexical_present(text):
        return bool(re.search(rules["mà_or_nhưng"], unicodedata.normalize("NFC", text).casefold()))
    by_id = {r["sample_id"]: r for r in data["test"]}
    strata = []
    for seed in SEEDS:
        root = _full_run_root(seed)
        rows = _read_jsonl(root / "predictions/test_predictions.jsonl")
        thresholds = json.loads((root / "selection/thresholds.json").read_text(encoding="utf-8"))
        marker = np.array([lexical_present(by_id[r["sample_id"]]["text"]) for r in rows])
        y = np.array([r["gold"]["irony"] for r in rows])
        p = np.array([r["probabilities"]["irony"] for r in rows])
        for name, mask in (("marker_present", marker), ("marker_absent", ~marker)):
            score = metrics(y[mask], p[mask], thresholds["irony"])
            # The present stratum contains only positives; macro-F1 is not a
            # useful between-class diagnostic in that stratum.
            if len(set(y[mask])) == 1:
                score["binary_macro_f1"] = None
            strata.append({"seed": seed, "stratum": name, "n": int(mask.sum()),
                           "gold_irony": int(y[mask].sum()), **score})
    write_csv(output / "xlmr_lexical_strata.csv", strata)
    norm = lambda text: " ".join(unicodedata.normalize("NFC", text).casefold().split())
    train_text = {norm(r["text"]) for r in data["train"]}
    exact_overlap = sum(norm(r["text"]) in train_text for r in data["test"])
    result["exploratory_lexical_cues"] = lexical
    result["xlmr_lexical_strata"] = strata
    result["normalized_train_test_exact_overlap_rows"] = exact_overlap
    result["qualitative_review_scope"] = "Assistant close reading of 10 stable TP, all four unique FP and all seven unique FN; not model attribution, human agreement, or gold-label correction."
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    def number(row, k, percent=True):
        value = row[k + "_mean"] * (100 if percent else 1)
        sd = row[k + "_sd"]
        return f"{value:.2f}" + (f" ± {100 * sd:.2f}" if sd is not None else "")
    lines = ["# Irony: surface cues, shallow baseline, and qualitative review", "",
             "Train/dev/test have 7,998/1,999/2,000 comments, including 554/138/139 gold irony-positive comments. Test has 1,861 non-irony comments. No main model is retrained. The normalized exact-text train/test overlap count is " + str(exact_overlap) + ".", "",
             "## A. Cue frequencies fixed before the primary analysis", "",
             "Presence is counted once per comment. Families may overlap. Definitions and regexes are exported in parameters.json; quoted fragments, punctuation, emoji and capitalization are descriptive surface features, not proof of author intent. The manual has no fixed discourse-particle whitelist, so no such lexicon is added to this primary table.", "",
             "| Cue | Irony n/139 | P(cue | irony), % | Non-irony n/1861 | P(cue | non-irony), % |", "|---|---:|---:|---:|---:|"]
    for row in frequency:
        if row["split"] == "test":
            lines.append(f"| {row['cue']} | {row['irony_count']} | {float(row['irony_percent']):.2f} | {row['non_irony_count']} | {float(row['non_irony_percent']):.2f} |")
    lines += ["", "The tested orthographic cues do not individually cover most irony positives. Quotation marks do not occur in either test class, so this split cannot establish their diagnostic value.", "",
              "## B. Character TF-IDF + Logistic Regression", "",
              "TF-IDF uses character n-grams 2–5, lowercase=True, min_df=2, sublinear_tf=True, L2 normalization and no accent stripping or feature cap. Vocabulary and IDF are fitted only on train. Six candidate models combine C ∈ {0.1, 1, 10} and class_weight ∈ {None, balanced}; selection maximizes dev binary macro-F1 using a dev-only threshold grid 0.05–0.95 in 0.01 steps, ties toward 0.5 then the smaller threshold. Equal model scores prefer lower C, then no class weighting. Solver=liblinear, max_iter=2000, tol=1e-4, random_state=20260520. All candidates converged. The selected model uses C=10, class_weight=balanced, threshold=0.72 and 71,772 features. It is not refitted on train+dev.", "",
              "The shallow result is one fit, with no invented seed SD. XLM-R entries are mean ± sample SD over the three existing seeds. Precision, recall and positive F1 refer to irony=1; macro-F1 averages the two classes. Values below are percentages.", "",
              "| Model | Threshold strategy | Binary macro-F1 | Positive precision | Positive recall | Positive F1 | AP |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["comparison"]:
        lines.append(f"| {row['model']} | {row['strategy']} | " + " | ".join(number(row, k) for k in ("binary_macro_f1", "positive_precision", "positive_recall", "positive_f1", "average_precision")) + " |")
    lines += ["", "At the selected threshold, the shallow model has TP=121, FP=2, FN=18, TN=1,859. Its binary macro-F1 is 2.52 percentage points below the ViPragSent mean; its positive F1 is 4.71 points below. At threshold 0.5 the two models' binary macro-F1 values are nearly equal. These are descriptive comparisons without a significance claim.", "",
              "## Exploratory lexical shortcut diagnostic", "",
              "Inspecting the training-fitted character coefficients highlights mà and nhưng. The following whole-token diagnostic was added after the primary test results were available, so it is exploratory; it is not represented as a prespecified independent validation. No lexical-rule parameter is optimized on test.", "",
              "| Split | Cue | Irony count | Irony % | Non-irony count | Non-irony % |", "|---|---|---:|---:|---:|---:|"]
    for row in lexical:
        lines.append(f"| {row['split']} | {row['cue']} | {row['irony_count']} | {row['irony_percent']:.2f} | {row['non_irony_count']} | {row['non_irony_percent']:.3f} |")
    lines += ["", "On test, the exact-token rule ‘mà or nhưng present ⇒ irony’ yields TP=129, FP=0, FN=10, TN=1,861: precision 100.00%, recall 92.81%, positive F1 96.27%, binary macro-F1 98.00%. The same lexical association is already strong on train: 504/554 positives versus 2/7,444 negatives contain either marker.", "",
              "All three ViPragSent seeds correctly predict the 129 marker-present positives. Among the ten marker-absent positives they correctly predict 4, 3 and 5 respectively: positive recall 40.00 ± 10.00%. This small stratum is descriptive; it does not establish causally that the main model uses these exact words. Nevertheless, a near-ceiling lexical rule prevents interpreting the overall irony score as evidence that shallow lexical cues have been ruled out.", "",
              "## C. Qualitative review", "",
              "Ten TP were sampled deterministically from 132 comments classified positive by all three seeds. All four unique FP and all seven unique FN across seeds were reviewed; errors are not padded to ten or repeated as independent comments. FN includes comments missed by at least one seed. Gold labels and co-occurring labels are preserved. Categories and notes are assistant close readings, not independent human annotation, author-intent verification or main-model feature attribution.", "",
              "The TP sample includes explicit mock praise, behavioral/evaluative contrast, and several comments with a plausible literal reading. FP includes emphatic enjoyment and direct derision with laughter; two have gold sarcasm/mocking but not irony. FN includes metalinguistic mentions of mỉa mai, ambiguous taunts, hyperbolic praise, and one particularly clear polarity reversal: ‘chúc mừng quán, vừa mở đã chuẩn bị đóng lại :D’. That example is verified gold irony-positive and missed by all three seeds. It has gold implicit_sentiment=1 and sarcasm/idiom_figurative/code_switching/mocking=0.", "",
              "Some gold-positive examples are difficult to reconcile with a strict standalone literal-versus-intended reversal. These observations motivate human review of the irony boundary; they do not independently establish that a gold label is wrong and are not a prevalence estimate of annotation ambiguity.", "",
              "| Group | Comment | Visible text pattern | Standalone ironic reading | Note |", "|---|---|---|---|---|"]
    for row in annotated:
        clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join(clean(row[k]) for k in ("group", "text", "text_pattern", "standalone_ironic_reading", "interpretive_note")) + " |")
    lines += ["", "## Scientific interpretation", "",
              "Orthographic markers alone do not separate the classes, but the selected shallow model performs strongly and an exploratory rule based only on mà/nhưng nearly matches the main model. The present split therefore contains a strong lexical shortcut. The analyses do not justify a broad claim that high irony performance requires deep pragmatic understanding; their generalization beyond this cue–label association remains unresolved.", "",
              "Implementation reference: [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "qualitative_review_count": len(annotated),
                      "lexical_test": [r for r in lexical if r["split"] == "test"],
                      "train_test_exact_overlap_rows": exact_overlap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

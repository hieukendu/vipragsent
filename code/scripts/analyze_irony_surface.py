"""Surface-cue frequencies, a cheap irony baseline, and sampled error cases."""
from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
import warnings
from pathlib import Path

import emoji
import numpy as np
import regex
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from _bootstrap import ROOT
from analyze_priority3_positive_class import SEEDS, _validate_sources
from vipragsent.constants import EXPECTED_SPLIT_COUNTS
from vipragsent.evaluation.thresholds import tune_binary_threshold

CUES = {
    "emoji": "At least one Unicode emoji recognized by emoji.emoji_list.",
    "emoticon": "ASCII smile/frown faces (:, ;, =, 8 prefix), xD, ^^, T_T, >_<; raw case preserved.",
    "emoji_or_emoticon": "Union of Unicode emoji and ASCII emoticon presence.",
    "exclamation_or_question": "At least one ! or ?.",
    "repeated_punctuation": "At least two consecutive !/? characters, at least three periods, or Unicode ellipsis.",
    "laughter_marker": "Standalone repeated ha/he/hi (including spaces), kk+, or laughing :))/=))/xD faces.",
    "quotation_mark": "Double or typographic quote; ASCII single quotes count only when paired outside words.",
    "repeated_letters": "At least three consecutive copies of a Unicode letter, ignoring case.",
    "uppercase_token": "A token consisting of at least two uppercase Unicode letters; abbreviations can count.",
}
PATTERNS = {
    "emoticon": r"(?<!\p{L})(?:[:;=8][\-^']?(?:\)+|\(+|[dDpP/\\|])|[xX][dD]+|\^{2,}|[Tt][_.-][Tt]|[><][_.-][><])",
    "exclamation_or_question": r"[!?]",
    "repeated_punctuation": r"[!?]{2,}|\.{3,}|…",
    "laughter_marker": r"(?<!\p{L})(?:(?:ha|he|hi)(?:\s*(?:ha|he|hi))+|k{2,})(?!\p{L})|[:=]\){2,}|[xX][dD]+",
    "quotation_mark": r'''["“”„«»‘’]|(?<!\p{L})'[^'\n]+'(?!\p{L})''',
    "repeated_letters": r"(\p{L})\1{2,}",
    "uppercase_token": r"(?<!\p{L})\p{Lu}{2,}(?!\p{L})",
}
FLAGS = {"laughter_marker": regex.IGNORECASE, "repeated_letters": regex.IGNORECASE}
COMPILED = {k: regex.compile(p, FLAGS.get(k, 0)) for k, p in PATTERNS.items()}
OUT = ROOT / "reports/irony_surface_analysis_20261006"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cues(text):
    text = unicodedata.normalize("NFC", text)
    result = {k: bool(pattern.search(text)) for k, pattern in COMPILED.items()}
    result["emoji"] = bool(emoji.emoji_list(text))
    result["emoji_or_emoticon"] = result["emoji"] or result["emoticon"]
    return {key: int(result[key]) for key in CUES}


def load_split(split):
    with (ROOT / "data/processed/vipragsent" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == EXPECTED_SPLIT_COUNTS[split], "Unexpected split size")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate IDs")
    require(all(r["split"] == split for r in rows), "Wrong split designation")
    require(all(r["irony"] in ("0", "1") for r in rows), "Nonbinary irony label")
    return rows


def metrics(y, p, threshold):
    prediction = (p >= threshold).astype(int)
    return {"binary_macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "positive_precision": float(precision_score(y, prediction, zero_division=0)),
            "positive_recall": float(recall_score(y, prediction, zero_division=0)),
            "positive_f1": float(f1_score(y, prediction, zero_division=0)),
            "average_precision": float(average_precision_score(y, p)),
            "tp": int(np.sum((y == 1) & (prediction == 1))),
            "fp": int(np.sum((y == 0) & (prediction == 1))),
            "fn": int(np.sum((y == 1) & (prediction == 0))),
            "tn": int(np.sum((y == 0) & (prediction == 0)))}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    # Fix features and candidate choices before calculating test statistics.
    protocol = {"cue_definitions": CUES, "patterns": PATTERNS,
                "case_insensitive_patterns": list(FLAGS), "normalization_for_cues": "Unicode NFC only",
                "discourse_lexicon": "Not included: the manual provides no fixed lexical/particle whitelist.",
                "tfidf": {"analyzer": "char", "ngram_range": [2, 5], "lowercase": True,
                          "min_df": 2, "sublinear_tf": True, "norm": "l2", "max_features": None,
                          "strip_accents": None},
                "logistic_regression": {"solver": "liblinear", "C_candidates": [0.1, 1.0, 10.0],
                                        "class_weight_candidates": [None, "balanced"],
                                        "max_iter": 2000, "tol": 0.0001, "random_state": 20260520},
                "selection": "Train-only vocabulary/IDF/model fit. Select C and class weight by dev binary macro-F1 with dev-selected threshold. Ties: smaller C, then unweighted model.",
                "threshold_grid": {"start": 0.05, "stop": 0.95, "step": 0.01,
                                   "tie_break": "nearest 0.5, then smaller threshold"},
                "error_selection": "All unique FP across three seeds; up to ten stable TP and ten unique FN sampled without replacement with seed 20260526. FP/FN lists may overlap TP in another seed; counts are not pooled sample sizes.",
                "versions": {"sklearn": sklearn.__version__, "emoji": emoji.__version__, "regex": regex.__version__}}
    (output / "parameters.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    data = {split: load_split(split) for split in ("train", "dev", "test")}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        require({r["sample_id"] for r in data[left]}.isdisjoint(r["sample_id"] for r in data[right]), "Split IDs overlap")
    features = {split: [cues(r["text"]) for r in rows] for split, rows in data.items()}
    counts = {split: {"n": len(rows), "irony": sum(int(r["irony"]) for r in rows)} for split, rows in data.items()}
    frequency = []
    for split, rows in data.items():
        y = np.array([int(r["irony"]) for r in rows])
        for cue in CUES:
            present = np.array([r[cue] for r in features[split]])
            positive = int(np.sum((y == 1) & (present == 1)))
            negative = int(np.sum((y == 0) & (present == 1)))
            npos, nneg = int(y.sum()), int((y == 0).sum())
            frequency.append({"split": split, "cue": cue, "irony_count": positive, "irony_total": npos,
                              "irony_percent": 100 * positive / npos, "non_irony_count": negative,
                              "non_irony_total": nneg, "non_irony_percent": 100 * negative / nneg,
                              "difference_pp": 100 * (positive / npos - negative / nneg),
                              "irony_given_cue_percent": 100 * positive / (positive + negative) if positive + negative else None})
    write_csv(output / "cue_frequencies.csv", frequency)
    train_x = [r["text"] for r in data["train"]]
    dev_x = [r["text"] for r in data["dev"]]
    ytrain = np.array([int(r["irony"]) for r in data["train"]])
    ydev = np.array([int(r["irony"]) for r in data["dev"]])
    vectorizer = TfidfVectorizer(**{**protocol["tfidf"], "ngram_range": (2, 5)})
    xtrain = vectorizer.fit_transform(train_x)
    xdev = vectorizer.transform(dev_x)
    print(f"Train {counts['train']}; dev {counts['dev']}; {xtrain.shape[1]} character features", flush=True)
    models, candidate_rows = [], []
    for c in (0.1, 1.0, 10.0):
        for weight in (None, "balanced"):
            started = time.monotonic()
            model = LogisticRegression(C=c, class_weight=weight, solver="liblinear", max_iter=2000,
                                       tol=1e-4, random_state=20260520)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(xtrain, ytrain)
            require(not any(issubclass(w.category, ConvergenceWarning) for w in caught), "Baseline did not converge")
            prob = model.predict_proba(xdev)[:, 1]
            threshold = tune_binary_threshold(ydev, prob)
            result = metrics(ydev, prob, threshold)
            candidate_rows.append({"C": c, "class_weight": str(weight), "threshold": threshold,
                                   "iterations": int(model.n_iter_[0]), "seconds": time.monotonic() - started, **result})
            models.append((model, threshold, c, weight, result))
            print(f"C={c}, weight={weight}, dev macro-F1={result['binary_macro_f1']:.6f}, t={threshold:.2f}", flush=True)
    write_csv(output / "baseline_dev_selection.csv", candidate_rows)
    selected = sorted(models, key=lambda r: (-r[4]["binary_macro_f1"], r[2], r[3] is not None))[0]
    model, threshold, c, weight, dev_metrics = selected
    test = data["test"]
    ytest = np.array([int(r["irony"]) for r in test])
    xtest = vectorizer.transform([r["text"] for r in test])
    ptest = model.predict_proba(xtest)[:, 1]
    baseline = {"model": "Character TF-IDF + Logistic Regression", "C": c, "class_weight": weight,
                "selected_threshold": threshold, "feature_count": xtrain.shape[1], "fit_count": len(ytrain),
                "dev": dev_metrics, "test_dev_threshold": metrics(ytest, ptest, threshold),
                "test_threshold_0.5": metrics(ytest, ptest, 0.5)}
    coefficients = model.coef_[0]
    names = vectorizer.get_feature_names_out()
    top = np.r_[np.argsort(coefficients)[-50:][::-1], np.argsort(coefficients)[:50]]
    write_csv(output / "baseline_top_coefficients.csv", [{"ngram": names[i], "coefficient": float(coefficients[i])} for i in top])
    write_csv(output / "baseline_test_predictions.csv", [{"sample_id": r["sample_id"], "gold_irony": int(y),
              "probability": float(p), "prediction": int(p >= threshold)} for r, y, p in zip(test, ytest, ptest)])
    loaded, _ = _validate_sources()
    full_metrics = []
    baseline_by_id = {r["sample_id"]: float(p) for r, p in zip(test, ptest)}
    test_by_id = {r["sample_id"]: r for r in test}
    per_id = {}
    for seed in SEEDS:
        for key in ("proposed_multitask", "baseline_pragmatic_only"):
            entry = loaded[seed][key]
            rows = entry["rows"]
            y = np.array([r["gold"]["irony"] for r in rows])
            p = np.array([r["probabilities"]["irony"] for r in rows])
            t = entry["thresholds"]["irony"]
            for strategy, cut in (("dev_binary_macro_f1", t), ("fixed_0.5", 0.5)):
                full_metrics.append({"model": key, "seed": seed, "strategy": strategy, "threshold": cut, **metrics(y, p, cut)})
            for r in rows:
                require(str(r["gold"]["irony"]) == test_by_id[r["sample_id"]]["irony"], "Gold mismatch")
                if key == "proposed_multitask":
                    item = per_id.setdefault(r["sample_id"], {"sample_id": r["sample_id"], "text": test_by_id[r["sample_id"]]["text"],
                        "gold_irony": int(r["gold"]["irony"]), "full_predictions": {}, "full_probabilities": {},
                        "co_labels": {k: r["gold"][k] for k in ("implicit_sentiment", "sarcasm", "idiom_figurative", "code_switching", "mocking")}})
                    item["full_predictions"][seed] = int(r["predictions"]["irony"])
                    item["full_probabilities"][seed] = float(r["probabilities"]["irony"])
    write_csv(output / "xlmr_irony_per_seed.csv", full_metrics)
    rng = np.random.default_rng(20260526)
    groups = {"TP": [r for r in per_id.values() if r["gold_irony"] == 1 and all(r["full_predictions"].values())],
              "FP": [r for r in per_id.values() if r["gold_irony"] == 0 and any(r["full_predictions"].values())],
              "FN": [r for r in per_id.values() if r["gold_irony"] == 1 and not all(r["full_predictions"].values())]}
    cases = []
    for group, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["sample_id"])
        chosen = rows if group == "FP" or len(rows) <= 10 else [rows[i] for i in sorted(rng.choice(len(rows), 10, replace=False))]
        for r in chosen:
            cases.append({"group": group, "sample_id": r["sample_id"], "text": r["text"], "gold_irony": r["gold_irony"],
                          "full_positive_seeds": sum(r["full_predictions"].values()),
                          **{f"p_{seed}": r["full_probabilities"][seed] for seed in SEEDS},
                          "surface_baseline_probability": baseline_by_id[r["sample_id"]],
                          "surface_baseline_prediction": int(baseline_by_id[r["sample_id"]] >= threshold),
                          "surface_cues": ";".join(k for k, v in cues(r["text"]).items() if v),
                          "co_labels": json.dumps(r["co_labels"], ensure_ascii=False)})
    write_csv(output / "qualitative_cases.csv", cases)
    comparison = []
    fields = ("binary_macro_f1", "positive_precision", "positive_recall", "positive_f1", "average_precision")
    for model_key in ("proposed_multitask", "baseline_pragmatic_only"):
        for strategy in ("dev_binary_macro_f1", "fixed_0.5"):
            rows = [r for r in full_metrics if r["model"] == model_key and r["strategy"] == strategy]
            comparison.append({"model": model_key, "strategy": strategy, "runs": 3,
                               **{f"{k}_{s}": float(np.mean([r[k] for r in rows]) if s == "mean" else np.std([r[k] for r in rows], ddof=1)) for k in fields for s in ("mean", "sd")}})
    for strategy, row in (("dev_binary_macro_f1", baseline["test_dev_threshold"]), ("fixed_0.5", baseline["test_threshold_0.5"])):
        comparison.append({"model": "char_tfidf_logistic_regression", "strategy": strategy, "runs": 1,
                           **{f"{k}_{s}": row[k] if s == "mean" else None for k in fields for s in ("mean", "sd")}})
    write_csv(output / "baseline_comparison.csv", comparison)
    result = {"counts": counts, "baseline": baseline, "comparison": comparison,
              "qualitative_population": {k: len(v) for k, v in groups.items()},
              "qualitative_selected": {k: sum(r["group"] == k for r in cases) for k in groups}}
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output_dir)

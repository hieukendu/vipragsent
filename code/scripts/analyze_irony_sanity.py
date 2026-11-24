"""Five-head evaluation and close reading of all ten cue-negative irony cases."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from _bootstrap import ROOT
from analyze_priority3_positive_class import SEEDS, MODEL_KEYS, _validate_sources
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.statistics.bootstrap import paired_bootstrap_comparison

LABELS = tuple(PRAGMATIC_LABELS)
KEPT = tuple(k for k in LABELS if k != "irony")
NAMES = {"proposed_multitask": "ViPragSent", "baseline_pragmatic_only": "XLM-R pragmatic-only"}
OUT = ROOT / "reports/irony_sanity_20261006"
NOTES = {
    "00005": ("plausible_but_ambiguous", "Apparent low social status versus unexpectedly high family wealth; the explicit ai ngờ contrast.",
              "Expectation reversal is visible, but the comment can literally describe a surprising situation. Situational incongruity does not by itself establish verbal irony.",
              "Possible boundary tension between situational surprise and a literal-versus-intended evaluative reversal."),
    "02285": ("insufficient_standalone_evidence", "Praise of placement beside two attractive women and the laughing emoticon =)).",
              "Mock praise is possible, but the intended opposite evaluation is not recoverable without the unseen referent.",
              "Potential tension with the standalone-context rule; sarcasm=1 is not independent proof of irony."),
    "03693": ("insufficient_standalone_evidence", "A derogatory title and the laughter marker kkk.",
              "Ridicule is explicit; a reversal of the literal evaluation is not. Direct taunting is excluded by the manual unless a separate ironic incongruity is present.",
              "Potential direct-derision versus verbal-irony boundary issue."),
    "05327": ("insufficient_standalone_evidence", "The explicit lexical mention mỉa mai.",
              "The text describes or comments on an ironic act; mentioning irony is not sufficient to show the utterance itself performs an ironic reversal.",
              "Potential metalinguistic-mention versus performed-irony boundary issue."),
    "05582": ("insufficient_standalone_evidence", "The metalinguistic characterization bé mỉa mai.",
              "A person is described as sarcastic/ironic, but no opposite intended evaluation is recoverable from this fragment alone.",
              "Potential metalinguistic-mention versus performed-irony boundary issue."),
    "06081": ("insufficient_standalone_evidence", "Literal praise zễ thương and the friendly emoticon ^.^.",
              "The visible wording supports sincere praise. An ironic reading requires additional contextual assumptions.",
              "Potential direct-praise versus irony boundary issue, despite all seeds matching the gold label."),
    "08089": ("plausible_but_ambiguous", "Intense product praise and repeated thanks to every participant in the purchase.",
              "Hyperbolic mock praise is plausible; sincere enthusiasm is also possible. Exaggeration alone is not sufficient under the manual.",
              "Possible hyperbole versus ironic mock-praise ambiguity; the gold figurative/sarcasm labels do not settle author intent."),
    "08223": ("clear_standalone_irony", "The positive formula chúc mừng immediately juxtaposed with impending closure, plus :D.",
              "Congratulations on opening conflicts with preparing to close. The evaluative reversal is recoverable within this standalone comment.",
              "No evident conflict with the irony definition in this close reading; every seed nevertheless misses it."),
    "09651": ("insufficient_standalone_evidence", "A literal appearance comparison to a named person.",
              "The comparison alone supplies no literal-versus-intended reversal. A humorous or derisive interpretation would require unstated background.",
              "Potential literal-comparison versus irony boundary issue, despite all seeds matching the gold label."),
    "11779": ("plausible_but_ambiguous", "Social withdrawal contrasts with enjoyment of being alone; sad emoji.",
              "The contrast can be a sincere, coherent statement of preference or distress. It is not sufficient to establish an opposite intended stance.",
              "Possible ordinary-contrast versus verbal-irony boundary issue."),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_f1_from_codes(codes):
    # codes: model x seed x examples x head; 0=TN, 1=FP, 2=FN, 3=TP.
    tn, fp, fn, tp = [(codes == c).sum(axis=2) for c in (0, 1, 2, 3)]
    pos_denom, neg_denom = 2 * tp + fp + fn, 2 * tn + fp + fn
    pos = np.divide(2 * tp, pos_denom, out=np.zeros_like(tp, dtype=float), where=pos_denom != 0)
    neg = np.divide(2 * tn, neg_denom, out=np.zeros_like(tn, dtype=float), where=neg_denom != 0)
    return (pos + neg) / 2


def bootstrap(codes, *, resamples, seed):
    rng = np.random.default_rng(seed)
    excluded = [LABELS.index(k) for k in KEPT]
    distribution = np.empty((resamples, 2))
    for i in range(resamples):
        chosen_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
        chosen_rows = rng.integers(0, codes.shape[2], size=codes.shape[2])
        # Same aligned row sample across models and seed draws, matching Q1a.
        scores = binary_f1_from_codes(codes[:, chosen_seeds][:, :, chosen_rows, :])
        distribution[i, 0] = scores[0].mean() - scores[1].mean()
        distribution[i, 1] = scores[0, :, excluded].mean() - scores[1, :, excluded].mean()
    return distribution


def run(output, resamples=10000, bootstrap_seed=20260525):
    started = time.monotonic()
    loaded, _ = _validate_sources()
    rows_by_seed = {seed: {model: loaded[seed][model]["rows"] for model in MODEL_KEYS} for seed in SEEDS}
    gold = np.array([[[r["gold"][k] for k in LABELS] for r in rows_by_seed[s][MODEL_KEYS[0]]] for s in SEEDS])
    predictions = np.array([[[[r["predictions"][k] for k in LABELS] for r in rows_by_seed[s][model]] for s in SEEDS] for model in MODEL_KEYS])
    codes = 2 * gold[None, ...] + predictions
    per_head = binary_f1_from_codes(codes)
    require(codes.shape == (2, 3, 2000, 6), "Unexpected paired data shape")
    excluded = [LABELS.index(k) for k in KEPT]
    seed_rows, head_rows = [], []
    for m, model in enumerate(MODEL_KEYS):
        for j, seed in enumerate(SEEDS):
            for h, label in enumerate(LABELS):
                independent = f1_score(gold[j, :, h], predictions[m, j, :, h], labels=[0, 1], average="macro", zero_division=0)
                require(np.isclose(independent, per_head[m, j, h], atol=1e-14, rtol=0), "F1 differs from sklearn")
                head_rows.append({"model": NAMES[model], "seed": seed, "label": label, "binary_macro_f1_percent": 100 * independent})
            seed_rows.append({"model": NAMES[model], "seed": seed,
                              "macro_all_six_percent": 100 * float(per_head[m, j].mean()),
                              "macro_excluding_irony_percent": 100 * float(per_head[m, j, excluded].mean())})
    # Verify a small complete distribution against the existing Q1a helper.
    left = [(gold[j], predictions[0, j]) for j in range(3)]
    right = [(gold[j], predictions[1, j]) for j in range(3)]
    def reference_metric(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        return float(np.mean([f1_score(y[:, h], prediction[:, h], labels=[0, 1], average="macro", zero_division=0) for h in excluded]))
    reference = paired_bootstrap_comparison(left, right, reference_metric, resamples=12, seed=bootstrap_seed)
    test_distribution = bootstrap(codes, resamples=12, seed=bootstrap_seed)
    require(np.allclose(reference.distribution, test_distribution[:, 1], atol=1e-14, rtol=0), "Vectorized bootstrap differs from existing protocol")
    print("PASS: paired saved decisions, six-head F1, and bootstrap implementation validated", flush=True)
    distribution = bootstrap(codes, resamples=resamples, seed=bootstrap_seed)
    summary = []
    for model in MODEL_KEYS:
        rows = [r for r in seed_rows if r["model"] == NAMES[model]]
        summary.append({"model": NAMES[model], **{f"{metric}_{stat}": float(np.mean([r[metric] for r in rows]) if stat == "mean" else np.std([r[metric] for r in rows], ddof=1))
                       for metric in ("macro_all_six_percent", "macro_excluding_irony_percent") for stat in ("mean", "sd")}})
    intervals = []
    for i, metric in enumerate(("macro_all_six", "macro_excluding_irony")):
        observed = (per_head[0].mean() - per_head[1].mean() if i == 0 else
                    per_head[0, :, excluded].mean() - per_head[1, :, excluded].mean())
        lo, hi = np.percentile(distribution[:, i], [2.5, 97.5], method="linear")
        intervals.append({"metric": metric, "difference_pp": 100 * float(observed),
                          "ci_low_pp": 100 * float(lo), "ci_high_pp": 100 * float(hi),
                          "resamples": resamples, "bootstrap_seed": bootstrap_seed})
    with (ROOT / "data/processed/vipragsent/test.csv").open(encoding="utf-8-sig", newline="") as stream:
        canonical = list(csv.DictReader(stream))
    has_marker = lambda text: bool(re.search(r"(?<!\w)(?:mà|nhưng)(?!\w)", unicodedata.normalize("NFC", text).casefold()))
    subset = sorted([r for r in canonical if r["irony"] == "1" and not has_marker(r["text"])], key=lambda r: r["sample_id"])
    require(len(subset) == 10, "Cue-negative positive subset does not contain exactly ten comments")
    lookup = {s: {m: {r["sample_id"]: r for r in rows_by_seed[s][m]} for m in MODEL_KEYS} for s in SEEDS}
    cases = []
    for r in subset:
        suffix = r["sample_id"].rsplit("_", 1)[1]
        assessment, candidate_cues, reasoning, boundary = NOTES[suffix]
        result = {"sample_id": r["sample_id"], "text": r["text"], "gold_irony": 1, "assessment": assessment,
                  "visible_candidate_cues": candidate_cues, "reasoning": reasoning, "potential_boundary_issue": boundary,
                  "actual_annotator_reason": "Not available in the evaluated label table; candidate cues are interpretive.",
                  "review_type": "assistant_close_reading_not_independent_human_reannotation"}
        for model in MODEL_KEYS:
            key = "full" if model == "proposed_multitask" else "pragmatic_only"
            for seed in SEEDS:
                prediction = lookup[seed][model][r["sample_id"]]
                require(prediction["gold"]["irony"] == 1, "Subset gold mismatch")
                result[f"{key}_prediction_{seed}"] = prediction["predictions"]["irony"]
                result[f"{key}_probability_{seed}"] = prediction["probabilities"]["irony"]
                result[f"{key}_threshold_{seed}"] = loaded[seed][model]["thresholds"]["irony"]
            result[f"{key}_correct_seeds"] = sum(result[f"{key}_prediction_{s}"] for s in SEEDS)
        for label in LABELS:
            result[f"gold_{label}"] = int(r[label])
        cases.append(result)
    assessed_counts = {k: sum(r["assessment"] == k for r in cases) for k in ("clear_standalone_irony", "plausible_but_ambiguous", "insufficient_standalone_evidence")}
    subset_counts = []
    for model in MODEL_KEYS:
        prefix = "full" if model == "proposed_multitask" else "pragmatic_only"
        for seed in SEEDS:
            tp = sum(r[f"{prefix}_prediction_{seed}"] for r in cases)
            subset_counts.append({"model": NAMES[model], "seed": seed, "n": 10, "gold_positive": 10, "tp": tp,
                                  "fn": 10 - tp, "positive_recall_percent": 10 * tp})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in {"macro_per_seed": seed_rows, "macro_summary": summary, "per_head_f1": head_rows,
                       "paired_intervals": intervals, "cue_negative_all_ten": cases, "cue_negative_per_seed": subset_counts}.items():
        write_csv(output / f"{name}.csv", rows)
    parameters = {"included_heads": list(KEPT), "excluded_head": "irony", "test_count": 2000,
                  "decision_rule": "Existing per-model/per-seed dev-selected thresholds unchanged.",
                  "bootstrap": "Paired hierarchical seed-then-aligned-test-row bootstrap; shared row draw across selected seeds, matching the Q1a implementation.",
                  "bootstrap_replicates": resamples, "bootstrap_seed": bootstrap_seed,
                  "cue_absence": "Neither mà nor nhưng as a whole Unicode word after NFC/casefold; no change to comment text.",
                  "review_scope": "All ten gold-positive cue-negative comments, no sampling, no gold relabeling; interpretation under manual section 6."}
    result = {"macro_summary": summary, "paired_intervals": intervals, "review_counts": assessed_counts,
              "cue_negative_per_seed": subset_counts, "full_correct_all_seeds": sum(r["full_correct_seeds"] == 3 for r in cases),
              "full_missed_all_seeds": sum(r["full_correct_seeds"] == 0 for r in cases)}
    (output / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Irony sanity checks", "",
             "No model is retrained or reselected. The existing development-selected thresholds are retained. Excluding irony changes only the evaluation average: the five retained heads are implicit sentiment, sarcasm, idiom/figurative, code-switching and mocking. Irony was still an objective during the original training.", "",
             "## Macro-pragmatic F1 excluding irony", "",
             "Entries are percent, mean ± sample SD over the same three seeds on the same 2,000 test comments.", "",
             "| Model | All six heads | Excluding irony |", "|---|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['model']} | {r['macro_all_six_percent_mean']:.4f} ± {r['macro_all_six_percent_sd']:.4f} | {r['macro_excluding_irony_percent_mean']:.4f} ± {r['macro_excluding_irony_percent_sd']:.4f} |")
    for r in intervals:
        lines += ["", f"ViPragSent minus pragmatic-only, {r['metric']}: {r['difference_pp']:+.4f} pp; paired percentile 95% CI [{r['ci_low_pp']:+.4f}, {r['ci_high_pp']:+.4f}] pp ({resamples:,} resamples)."]
    lines += ["", "The five-head mean favors ViPragSent in each of the three seeds. This addresses whether the reported mean advantage depends entirely on including the easy irony head; it does not estimate the training contribution of that head, nor remove its supervision from the original models. This is a post-hoc sensitivity analysis, not a new primary endpoint.", "",
              "## All ten cue-negative gold irony positives", "",
              "The whole-word rule is applied to the original test texts. These ten comments are the complete positive subset without mà/nhưng. Gold labels and model decisions are preserved. The evaluated label table contains no individual annotator explanations, so the visible cues below are candidate interpretations, not claims about annotator reasoning. These assessments are assistant close readings under the existing manual; they are not independent human reannotation or corrected gold labels.", "",
              "| Comment | Gold | Correct seeds / 3 | Assessment | Candidate textual cue and reasoning |", "|---|---:|---:|---|---|"]
    clean = lambda text: str(text).replace("|", "\\|").replace("\n", " ")
    for r in cases:
        lines.append(f"| {clean(r['text'])} | 1 | {r['full_correct_seeds']} | {r['assessment']} | {clean(r['visible_candidate_cues'] + ' ' + r['reasoning'])} |")
    lines += ["", "Close-reading categories: one clear standalone polarity reversal, three plausible but ambiguous cases, and six with insufficient standalone evidence for a literal-versus-intended reversal. This categorization is interpretive and does not establish that six labels are wrong. The latter cases include direct praise/comparison, direct taunting, metalinguistic mentions of irony, and a context-dependent fragment, which create potential tensions with the manual's exclusions.", "",
              "ViPragSent predicts 4/10, 3/10 and 5/10 positives across the three seeds (positive recall 40.00 ± 10.00%). Three comments are correctly predicted by every seed, five are missed by every seed, and two vary across seeds. The clearly ironic congratulations-on-closure comment is missed by all three. The pragmatic-only comparator predicts 5/10, 4/10 and 4/10 (43.33 ± 5.77%). On this ten-positive subset there are no negative examples, so binary macro-F1 would not be an informative two-class assessment and is not reported.", "",
              "These results establish low gold-positive recall in this small cue-negative subset. The semantic ambiguities prevent treating that recall alone as a clean estimate of generalization to independently validated irony without the lexical markers. Human adjudication under the standalone-comment definition would be needed to determine actual annotation inconsistencies.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    archive = output.parent / "irony_sanity_results_20261006.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("summary.md", "macro_per_seed.csv", "macro_summary.csv", "per_head_f1.csv", "paired_intervals.csv", "cue_negative_all_ten.csv", "cue_negative_per_seed.csv", "parameters.json", "results.json"):
            z.write(output / name, arcname=name)
    print(json.dumps({"status": "PASS", "elapsed_seconds": round(time.monotonic() - started, 2), **result}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260525)
    args = parser.parse_args()
    run(args.output_dir, args.resamples, args.bootstrap_seed)

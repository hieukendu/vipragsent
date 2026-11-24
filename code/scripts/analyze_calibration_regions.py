"""Descriptive score-region, language-script, and within-corpus checks."""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from analyze_dataset_diagnostics import ROOT, OUT, LABELS, SEEDS, _validate_sources, normalized, scalar_ece, summarize


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded, _ = _validate_sources()
    train_dev = pd.concat([pd.read_csv(ROOT / f"data/processed/vipragsent/{s}.csv") for s in ("train", "dev")])
    test = pd.read_csv(ROOT / "data/processed/vipragsent/test.csv").set_index("sample_id")
    reference = {normalized(t, True) for t in train_dev.text}
    excluded = {i for i,t in test.text.items() if normalized(t, True) in reference}
    rows, sensitivity = [], []
    for seed in SEEDS:
        for model in ("proposed_multitask", "baseline_pragmatic_only"):
            predictions = loaded[seed][model]["rows"]
            keep = np.array([r["sample_id"] not in excluded for r in predictions])
            scores = []
            for label in LABELS:
                y = np.array([r["gold"][label] for r in predictions])
                p = np.array([r["probabilities"][label] for r in predictions])
                pred = np.array([r["predictions"][label] for r in predictions])
                scores.append(f1_score(y[keep], pred[keep], labels=[0,1], average="macro", zero_division=0))
                for name, mask in (("p_below_0.5", p < .5), ("p_at_least_0.5", p >= .5)):
                    rows.append({"model": model, "seed": seed, "label": label, "region": name, "n": int(mask.sum()), "positive_n": int(y[mask].sum()), "ece": scalar_ece(y[mask],p[mask]), "brier": float(np.mean((p[mask]-y[mask])**2))})
            sensitivity.append({"model": model, "seed": seed, "n": int(keep.sum()), "excluded": len(excluded), "macro_pragmatic_f1": float(np.mean(scores))})
    pd.DataFrame(rows).to_csv(OUT / "calibration_score_regions.csv",index=False)
    summary = summarize(rows,["model","label","region"],["n","ece","brier"])
    pd.DataFrame(summary).to_csv(OUT / "calibration_score_regions_summary.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "in_domain_sensitivity.csv",index=False)
    d = pd.read_csv(ROOT / "data/processed/vipragsent/vipragsent_full.csv")
    positive = d[d.code_switching.eq(1)]
    pats = {"Han":r"[\u3400-\u4dbf\u4e00-\u9fff]", "Hangul":r"[\uac00-\ud7af\u1100-\u11ff]", "Kana":r"[\u3040-\u30ff]", "Cyrillic":r"[\u0400-\u04ff]", "Arabic":r"[\u0600-\u06ff]", "Thai":r"[\u0e00-\u0e7f]"}
    scripts = {key: sum(bool(re.search(pat,str(t))) for t in positive.text) for key,pat in pats.items()}
    notes = {"code_switch_gold_positive":len(positive),"script_counts":scripts,"foreign_language_annotations_present":False,"script_presence_does_not_determine_language":True,"in_domain_relaxed_overlap_test_rows":len(excluded)}
    (OUT / "additional_notes.json").write_text(json.dumps(notes,indent=2),encoding="utf-8")
    print(json.dumps(notes,indent=2))
    print(pd.DataFrame(sensitivity).groupby('model').macro_pragmatic_f1.agg(['mean','std']))
    print(pd.DataFrame(summary).query("model == 'proposed_multitask' and region == 'p_at_least_0.5'").to_string(index=False))


if __name__ == "__main__":
    main()

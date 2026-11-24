"""Re-score saved Q1b predictions after excluding overlap with train/development."""
from __future__ import annotations
import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from analyze_dataset_diagnostics import ROOT, OUT, normalized

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_dev = pd.concat([pd.read_csv(ROOT / f"data/processed/vipragsent/{split}.csv") for split in ("train", "dev")])
    references = {mode: {normalized(t, relaxed=(mode == "relaxed")) for t in train_dev.text} for mode in ("strict", "relaxed")}
    pairs = pd.read_csv(OUT / "overlap_pairs.csv")
    near = pairs[pairs.left.isin(["ViPragSent_train", "ViPragSent_dev"]) & pairs.kind.eq("near")]
    result = []
    for tag, folder, gold_column, labels in (
        ("uit_vsfc", "uit_vsfc", "polarity", ["negative", "neutral", "positive"]),
        ("uit_vsmec", "uit_vsmec", "emotion", ["anger", "disgust", "enjoyment", "fear", "other", "sadness", "surprise"]),
        ("aivivn", "aivivn_human_derived_3way", "label", ["negative", "neutral", "positive"]),
    ):
        frame = pd.read_csv(ROOT / f"data/processed/external/{folder}/test.csv")
        frame = frame.rename(columns={"id": "sample_id", "comment": "text"}).set_index("sample_id")
        masks = {"original": np.ones(len(frame), dtype=bool)}
        masks["exclude_strict_train_dev_overlap"] = np.array([normalized(t) not in references["strict"] for t in frame.text])
        masks["exclude_relaxed_train_dev_overlap"] = np.array([normalized(t, True) not in references["relaxed"] for t in frame.text])
        near_ids = set(near[near.right.eq(folder + "_test")].right_id)
        masks["exclude_relaxed_and_near_train_dev_overlap"] = masks["exclude_relaxed_train_dev_overlap"] & ~frame.index.isin(near_ids)
        for seed in (20260521, 20260522, 20260523):
            for model in ("ViPragSent", "XLM-R_8H"):
                if model == "ViPragSent":
                    path = ROOT / f"paper/revision_evidence/q1b_http_cache/primary_{seed}_{tag}_test_predictions.bin"
                    text = path.read_text(encoding="utf-8")
                else:
                    path = ROOT / f"paper/revision_evidence/q1b_http_cache/baseline_{seed}_tar.bin"
                    with tarfile.open(path) as archive:
                        text = archive.extractfile(f"predictions/{tag}_test_predictions.jsonl").read().decode("utf-8")
                rows = [json.loads(line) for line in text.splitlines() if line]
                assert len(rows) == len(frame)
                by_id = {r["sample_id"]: r for r in rows}
                assert len(by_id) == len(frame) and set(by_id) == set(frame.index)
                y = np.array([by_id[i]["gold"] for i in frame.index])
                p = np.array([by_id[i]["prediction"] for i in frame.index])
                assert all(y == frame[gold_column].to_numpy())
                for mode, mask in masks.items():
                    result.append({"model": model, "dataset": tag, "seed": seed, "filter": mode, "n": int(mask.sum()), "excluded": int((~mask).sum()), "macro_f1": f1_score(y[mask], p[mask], labels=labels, average="macro", zero_division=0)})
    pd.DataFrame(result).to_csv(OUT / "external_sensitivity_per_seed.csv", index=False)
    summaries = []
    for (model, dataset, mode), rows in pd.DataFrame(result).groupby(["model", "dataset", "filter"], sort=False):
        summaries.append({"model": model, "dataset": dataset, "filter": mode, "n": int(rows.n.iloc[0]), "excluded": int(rows.excluded.iloc[0]), "mean": rows.macro_f1.mean(), "sd": rows.macro_f1.std(ddof=1)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "external_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

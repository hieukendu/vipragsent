"""Recompute Q1b means and sample SDs from q1b_raw_extraction.csv.

Artifact-only: no training and no remote writes. The routed PhoBERT
single-task row is composed per seed from polarity (AIVIVN/VSFC) and emotion
(VSMEC) component packs.
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "q1b_raw_extraction.csv"


def mean_sd(values):
    values = [float(x) for x in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else None
    return mean, sd


with RAW.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

required = {"actual_recipe", "run_id", "seed", "aivivn_macro_f1", "vsfc_macro_f1", "vsmec_macro_f1", "ordinary_f1"}
if not rows or not required.issubset(rows[0]):
    raise SystemExit(f"missing columns: {required - set(rows[0]) if rows else required}")


def numeric(row, key):
    value = row.get(key, "")
    return None if value in ("", "NOT_APPLICABLE") else float(value)


groups = {
    name: [row for row in rows if row["actual_recipe"] == name]
    for name in ("phobert_multitask_8head", "xlmr_multitask_8head", "xlmr_followup_full", "vipragsent_full_phobert")
}

expected_stochastic = {"phobert_multitask_8head", "xlmr_multitask_8head", "xlmr_followup_full", "vipragsent_full_phobert"}
for recipe in expected_stochastic:
    seeds = [row["seed"] for row in groups[recipe]]
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(seeds) != {"20260521", "20260522", "20260523"}:
        raise SystemExit(f"seed coverage failure for {recipe}: {seeds}")
pol = {row["seed"]: row for row in rows if row["actual_recipe"] == "phobert_pol_single"}
emo = {row["seed"]: row for row in rows if row["actual_recipe"] == "phobert_emo_single"}
composite = []
for seed in sorted(pol):
    if seed not in emo:
        raise SystemExit(f"missing same-seed emotion component for {seed}")
    values = [numeric(pol[seed], "aivivn_macro_f1"), numeric(pol[seed], "vsfc_macro_f1"), numeric(emo[seed], "vsmec_macro_f1")]
    composite.append({"aivivn_macro_f1": values[0], "vsfc_macro_f1": values[1], "vsmec_macro_f1": values[2], "ordinary_f1": sum(values) / 3})
groups["phobert_single_routed"] = composite

for name, items in groups.items():
    print(name)
    for key in ("aivivn_macro_f1", "vsfc_macro_f1", "vsmec_macro_f1", "ordinary_f1"):
        values = [item[key] for item in items if item.get(key) is not None]
        if values:
            mean, sd = mean_sd(values)
            print(f"  {key}: mean={mean:.10f} sd_sample={sd:.10f} n={len(values)}")

canonical = {
    "phobert_multitask_8head": {"aivivn_macro_f1": (0.4468401714, 0.0127433079), "vsfc_macro_f1": (0.3359740942, 0.0301657685), "vsmec_macro_f1": (0.3636745529, 0.0071397335), "ordinary_f1": (0.3821629395, 0.0145570073)},
    "xlmr_multitask_8head": {"aivivn_macro_f1": (0.4832125097, 0.0123299607), "vsfc_macro_f1": (0.3911861088, 0.0109309963), "vsmec_macro_f1": (0.4016255677, 0.0020133352), "ordinary_f1": (0.4253413954, 0.0052422067)},
    "xlmr_followup_full": {"aivivn_macro_f1": (0.4793643700, 0.0215713924), "vsfc_macro_f1": (0.3788982264, 0.0278404735), "vsmec_macro_f1": (0.4004046714, 0.0070773594), "ordinary_f1": (0.4195557559, 0.0178125657)},
}
tol = 1e-9
for name, expected in canonical.items():
    for key, (expected_mean, expected_sd) in expected.items():
        values = [numeric(row, key) for row in groups[name] if numeric(row, key) is not None]
        actual_mean, actual_sd = mean_sd(values)
        if abs(actual_mean - expected_mean) > tol or abs(actual_sd - expected_sd) > tol:
            raise SystemExit(f"canonical mismatch {name} {key}: {(actual_mean, actual_sd)} != {(expected_mean, expected_sd)}")
print("canonical_tolerance_ok")

# Validate every CSV schema in the evidence directory, including quoted fields.
for path in ROOT.glob("*.csv"):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        width = len(header)
        bad = [(index, len(row)) for index, row in enumerate(reader, start=2) if len(row) != width]
    if bad:
        raise SystemExit(f"CSV schema failure in {path.name}: {bad[:5]}")
    print(f"csv_ok {path.name} columns={width}")

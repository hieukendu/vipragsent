"""Repair/validate Q1b CSV evidence with csv.DictReader/csv.writer.

The script is intentionally bounded to revision_evidence CSVs. It performs no
training, fetching, or writes outside this directory.
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def repair_canonical(path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            extras = row.pop(None, None)
            if extras:
                row[fieldnames[-1]] = row[fieldnames[-1]] + "," + ",".join(extras)
            rows.append(row)
    write_rows(path, fieldnames, rows)


canonical_path = ROOT / "q1b_canonical_table.csv"
repair_canonical(canonical_path)

# Preserve metric rows while removing two malformed source fields rather than
# carrying a non-verifiable pseudo-hash into the canonical provenance ledger.
raw_path = ROOT / "q1b_raw_extraction.csv"
raw_fields = [
    "actual_recipe", "run_id", "seed", "aivivn_macro_f1", "vsfc_macro_f1",
    "vsmec_macro_f1", "ordinary_f1", "remote_repo", "remote_ref",
    "artifact_sha256", "metric_file_sha256", "status",
]
raw_rows = read_rows(raw_path)
for row in raw_rows:
    if row["artifact_sha256"] and len(row["artifact_sha256"]) != 64:
        row["artifact_sha256"] = "NOT_RECORDED"
        row["status"] = "ANALYZED_ARTIFACT_HASH_UNAVAILABLE"
write_rows(raw_path, raw_fields, raw_rows)

provenance_path = ROOT / "q1b_provenance.csv"
provenance_fields = [
    "canonical_row", "run_id", "seed", "remote_repo", "remote_ref",
    "model_repository", "model_revision", "code_commit", "data_fingerprint",
    "external_manifest_hash", "aivivn_macro_f1", "vsfc_macro_f1",
    "vsmec_macro_f1", "ordinary_f1", "aivivn_gold_hash", "vsfc_gold_hash",
    "vsmec_gold_hash", "artifact_sha256", "metric_file_sha256", "status",
]
provenance_rows = read_rows(provenance_path)
for row in provenance_rows:
    if row["code_commit"] and len(row["code_commit"]) not in {40, 64}:
        row["code_commit"] = "NOT_RECORDED"
        row["status"] = row["status"].replace("ANALYZED", "ANALYZED_CODE_COMMIT_UNAVAILABLE", 1)
    if row["artifact_sha256"] and len(row["artifact_sha256"]) != 64:
        row["artifact_sha256"] = "NOT_RECORDED"
        row["status"] = row["status"].replace("ANALYZED", "ANALYZED_ARTIFACT_HASH_UNAVAILABLE", 1)
write_rows(provenance_path, provenance_fields, provenance_rows)

csv_paths = sorted(ROOT.glob("*.csv"))
for path in csv_paths:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        bad = [(n, len(row)) for n, row in enumerate(reader, 2) if len(row) != len(header)]
    if bad:
        raise SystemExit(f"CSV width failure: {path.name}: {bad[:3]}")

provenance = read_rows(ROOT / "q1b_provenance.csv")
hash_fields = {
    "code_commit", "data_fingerprint", "external_manifest_hash",
    "aivivn_gold_hash", "vsfc_gold_hash", "vsmec_gold_hash",
    "artifact_sha256", "metric_file_sha256",
}
for row in provenance:
    for field in hash_fields:
        value = row[field]
        allowed_lengths = {"code_commit": {40, 64}}.get(field, {64})
        if value and value not in {"NOT_RECORDED", "NOT_APPLICABLE"} and len(value) not in allowed_lengths:
            raise SystemExit(f"hash width failure: {row['run_id']} {field} {len(value)}")
primary = [r for r in provenance if r["canonical_row"] == "ViPragSent XLM-R-large primary"]
if {r["seed"] for r in primary} != {"20260521", "20260522", "20260523"}:
    raise SystemExit("primary seed coverage failure")

expected = {
    "aivivn_macro_f1_mean": 0.4793643700,
    "aivivn_macro_f1_sd": 0.0215713924,
    "vsfc_macro_f1_mean": 0.3788982264,
    "vsfc_macro_f1_sd": 0.0278404735,
    "vsmec_macro_f1_mean": 0.4004046714,
    "vsmec_macro_f1_sd": 0.0070773594,
    "ordinary_f1_mean": 0.4195557559,
    "ordinary_f1_sd": 0.0178125657,
}
row = next(r for r in read_rows(canonical_path) if r["canonical_row"] == "ViPragSent XLM-R-large primary")
for key, value in expected.items():
    if not math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-10):
        raise SystemExit(f"canonical numeric failure: {key}={row[key]} expected={value}")

working_path = ROOT.parent / "figures" / "source_data" / "q1b_expanded_working_draft.csv"
working = read_rows(working_path)
working_fields = [
    "model", "actual_recipe", "backbone", "scope", "record_status", "source",
    "seed_coverage", "seeds", "n", "identity_or_scope_note",
    "cohort_comparability_note", "UIT_VSFC_mean", "UIT_VSFC_sd",
    "UIT_VSMEC_mean", "UIT_VSMEC_sd", "AIVIVN_mean", "AIVIVN_sd",
    "ordinary_f1_mean", "ordinary_f1_sd",
]
with working_path.open(newline="", encoding="utf-8-sig") as fh:
    reader = csv.reader(fh)
    header = next(reader)
    bad = [(n, len(row)) for n, row in enumerate(reader, 2) if len(row) != len(header)]
if header != working_fields or bad:
    raise SystemExit(f"working CSV schema failure: header={header} bad={bad[:3]}")
expected_models = {
    "PhoBERT single-task", "PhoBERT fine-tune (actual multitask recipe)",
    "XLM-R-large standard classification baseline", "Sailor-7B SFT",
    "Vistral-7B SFT", "Azure deployment gpt-4.1-mini (version unrecorded)",
    "GPT-4o-mini 8-shot (reference identity)", "ViPragSent XLM-R-large",
    "ViPragSent full PhoBERT (contextual inventory)",
}
if {r["model"] for r in working} != expected_models:
    raise SystemExit("working CSV expected-row coverage failure")
numeric_fields = ["UIT_VSFC_mean", "UIT_VSFC_sd", "UIT_VSMEC_mean", "UIT_VSMEC_sd", "AIVIVN_mean", "AIVIVN_sd", "ordinary_f1_mean", "ordinary_f1_sd"]
for r in working:
    if r["record_status"] == "UNRESOLVED":
        if any(r[f] not in {"", "—", "â€”"} for f in numeric_fields):
            raise SystemExit(f"unresolved row has numeric value: {r['model']}")
    else:
        for f in numeric_fields:
            if r["n"] == "1" and f.endswith("_sd") and r[f] in {"", "—", "â€”"}:
                continue
            if r[f] in {"", "—", "â€”"}:
                raise SystemExit(f"numeric row missing value: {r['model']} {f}")
            float(r[f])
print(f"csv_width_ok files={len(csv_paths)}")
print(f"provenance_hash_width_ok rows={len(provenance)} primary_seeds=3")
print("primary_canonical_numeric_ok")
print(f"working_csv_ok rows={len(working)} numeric_rows={sum(r['record_status'] != 'UNRESOLVED' for r in working)} unresolved_rows={sum(r['record_status'] == 'UNRESOLVED' for r in working)}")

"""Build the bounded expanded Q1b working table from two paper-local inputs.

This is a layout-review artifact, not a scientific finalizer. The canonical
CSV supplies the newly extracted rows; the existing local manuscript subset
supplies the already-present ours/Sailor/Vistral rows. The contextual
PhoBERT-backed ViPragSent row is deliberately omitted from the main draft.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "revision_evidence" / "q1b_canonical_table.csv"
LOCAL = ROOT / "figures" / "source_data" / "q1b_external_provisional_subset.csv"
OUT_DIR = ROOT / "q1b_working_draft"
CONSOLIDATED = ROOT / "figures" / "source_data" / "q1b_expanded_working_draft.csv"
DATASETS = [
    ("AIVIVN", "AIVIVN"),
    ("UIT_VSFC", "UIT-VSFC"),
    ("UIT_VSMEC", "UIT-VSMEC"),
    ("ordinary_f1", "Ordinary F1"),
]
COHORT_NOTE = (
    "Direct ordinary-F1 ranking withheld: cohort comparisons are per-dataset "
    "cross-group checks, not a claim that one hash is shared across AIVIVN, "
    "VSFC, and VSMEC. The supplied checkpoint reports primary AIVIVN "
    "normalized-test SHA-256 prefix 27DA59A4... and baseline prefix "
    "F6F6A78E...; dataset-specific matched ID/gold proof is pending."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_row(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    matches = [row for row in rows if row["canonical_row"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one canonical row {name!r}, found {len(matches)}")
    return matches[0]


def canonical_row_any(rows: list[dict[str, str]], names: tuple[str, ...]) -> dict[str, str]:
    matches = [row for row in rows if row["canonical_row"] in names]
    if len(matches) != 1:
        raise ValueError(f"Expected one canonical row among {names!r}, found {len(matches)}")
    return matches[0]


def local_row(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    matches = [row for row in rows if row["model"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one local row {name!r}, found {len(matches)}")
    return matches[0]


def canonical_backbone(row: dict[str, str], model: str) -> str:
    recipe = row["actual_recipe"].strip().lower()
    if model == "Azure deployment (gpt-4.1-mini)":
        return "Azure deployment (gpt-4.1-mini; version unrecorded)"
    if "xlmr" in recipe:
        return "XLM-R-large"
    if "phobert" in recipe:
        return "PhoBERT-base"
    raise ValueError(
        f"Cannot infer canonical backbone for {model!r} from actual recipe {row['actual_recipe']!r}; "
        "refusing a default PhoBERT label."
    )


def from_canonical(row: dict[str, str], model: str, scope: str) -> dict[str, str]:
    values: dict[str, str] = {
        "model": model,
        "actual_recipe": row["actual_recipe"],
        "backbone": canonical_backbone(row, model),
        "scope": scope,
        "record_status": f"{row['status']}; PENDING_CROSS_COHORT_VERIFICATION",
        "source": "revision_evidence/q1b_canonical_table.csv",
        "seed_coverage": row["seed_coverage"],
        "seeds": row["seed_coverage"],
        "n": row["n"],
        "identity_or_scope_note": row["identity_or_scope_note"],
        "cohort_comparability_note": COHORT_NOTE,
    }
    for key, _ in DATASETS:
        source_key = {"UIT_VSFC": "vsfc", "UIT_VSMEC": "vsmec", "AIVIVN": "aivivn", "ordinary_f1": "ordinary_f1"}[key]
        mean = row[f"{source_key}_macro_f1_mean"] if source_key != "ordinary_f1" else row["ordinary_f1_mean"]
        sd = row[f"{source_key}_macro_f1_sd"] if source_key != "ordinary_f1" else row["ordinary_f1_sd"]
        values[f"{key}_mean"] = f"{100 * float(mean):.6f}" if mean else ""
        values[f"{key}_sd"] = f"{100 * float(sd):.6f}" if sd else ""
    return values


def from_local(row: dict[str, str], model: str, recipe: str, scope: str) -> dict[str, str]:
    values: dict[str, str] = {
        "model": model,
        "actual_recipe": recipe,
        "backbone": row["backbone"],
        "scope": scope,
        "record_status": "PROVISIONAL_LOCAL_SUBSET; PENDING_CROSS_COHORT_VERIFICATION",
        "source": "figures/source_data/q1b_external_provisional_subset.csv",
        "seed_coverage": row["seeds"],
        "seeds": row["seeds"],
        "n": row["n"],
        "identity_or_scope_note": "Existing manuscript subset retained for layout review; canonical recipe/result closure pending.",
        "cohort_comparability_note": COHORT_NOTE,
    }
    for key, _ in DATASETS:
        values[f"{key}_mean"] = row[f"{key}_mean"]
        values[f"{key}_sd"] = row[f"{key}_sd"]
    return values


def fmt_value(row: dict[str, str], key: str) -> str:
    mean = row[f"{key}_mean"]
    sd = row[f"{key}_sd"]
    if not mean:
        return "N/A (no value)"
    if not sd:
        return f"{float(mean):.1f} (n={row['n']}; SD unavailable)"
    return f"{float(mean):.1f} +/- {float(sd):.1f}"


def main() -> None:
    canonical = read_csv(CANONICAL)
    local = read_csv(LOCAL)
    primary_matches = [row for row in canonical if row["canonical_row"] == "ViPragSent XLM-R-large primary"]
    if primary_matches:
        ours_row = from_canonical(primary_matches[0], "ViPragSent (ours, XLM-R-large)", "primary")
    else:
        ours_row = from_local(
            local_row(local, "ViPragSent (ours, XLM-R-large)"),
            "ViPragSent (ours, XLM-R-large)",
            "Current ours record; recipe closure pending",
            "primary",
        )
    rows = [
        ours_row,
        from_canonical(canonical_row(canonical, "PhoBERT single-task routed"), "PhoBERT single-task routed", "baseline"),
        from_canonical(canonical_row(canonical, "PhoBERT multitask 8-head"), "PhoBERT multitask 8-head", "baseline"),
        from_canonical(canonical_row(canonical, "XLM-R-large multitask 8-head"), "XLM-R-large multitask 8-head", "baseline"),
        from_canonical(canonical_row(canonical, "Sailor-7B SFT"), "Sailor-7B SFT", "baseline"),
        from_canonical(canonical_row(canonical, "Vistral-7B SFT"), "Vistral-7B SFT", "baseline"),
        from_canonical(canonical_row_any(canonical, ("GPT artifact (actual deployment)", "GPT artifact (configured deployment)")), "Azure deployment (gpt-4.1-mini)", "baseline"),
    ]
    primary_rows = [row for row in rows if row["scope"] == "primary"]
    if len(primary_rows) != 1 or primary_rows[0]["backbone"] != "XLM-R-large":
        raise AssertionError("Q1b working draft must contain exactly one primary XLM-R-large row")
    gpt_rows = [row for row in rows if row["model"] == "Azure deployment (gpt-4.1-mini)"]
    if len(gpt_rows) != 1 or not gpt_rows[0]["backbone"].startswith("Azure deployment (gpt-4.1-mini"):
        raise AssertionError("GPT row must retain Azure deployment identity and version-unrecorded note")
    fields = ["model", "actual_recipe", "backbone", "scope", "record_status", "source", "seed_coverage", "seeds", "n", "identity_or_scope_note", "cohort_comparability_note"]
    fields.extend(field for key, _ in DATASETS for field in (f"{key}_mean", f"{key}_sd"))
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    with CONSOLIDATED.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_lines = [
        "# Q1b expanded working draft",
        "",
        "> **WORKING DRAFT - ANALYZED / PENDING CROSS-COHORT VERIFICATION - NOT ACCEPTED FINAL**",
        ">",
        "> This layout combines the extracted canonical rows with the existing local Sailor/Vistral subset; the current canonical primary XLM-R row is used once when available, with the local ours row retained as the fallback. It does not modify the manuscript or accepted PDF. The contextual PhoBERT-backed ViPragSent inventory row is omitted from the main table by scope decision.",
        "",
        "| Model | Actual recipe | n | UIT-VSFC | UIT-VSMEC | AIVIVN | Ordinary F1 (display only) | Working status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table_lines.append(
            "| " + " | ".join(
                [
                    row["model"],
                    row["actual_recipe"],
                    row["n"],
                    fmt_value(row, "UIT_VSFC"),
                    fmt_value(row, "UIT_VSMEC"),
                    fmt_value(row, "AIVIVN"),
                    fmt_value(row, "ordinary_f1"),
                    row["record_status"],
                ]
            ) + " |"
        )
    table_lines.extend(
        [
            "",
            "Values are displayed in percentage points. `+/-` denotes the supplied sample SD; the Azure deployment (gpt-4.1-mini) row has n=1 and therefore no SD estimate; the deployed version is unrecorded. All rows remain pending cross-cohort verification for common task definition, split/gold alignment, and recipe comparability.",
            "**Cohort caveat — direct ordinary-F1 ranking is withheld.** Comparisons are per-dataset cross-group checks, not a claim that one hash is shared across AIVIVN, VSFC, and VSMEC. The supplied checkpoint reports primary AIVIVN normalized-test SHA-256 prefix `27DA59A4...` and baseline prefix `F6F6A78E...`; the aggregate is shown only for layout review until per-dataset matched ID/gold proof or an explicit cohort separation is available.",
            "",
            "## Coverage ledger for unresolved historical rows",
            "",
            "| Historical/reference row | Draft handling | Value status | Reason |",
            "|---|---|---|---|",
            "| Q1b-02 PhoBERT fine-tune | Not added as a duplicate historical label | No value in this historical slot | The extracted actual recipe is retained as `PhoBERT multitask 8-head`; do not silently rewrite it as fine-tune. |",
            "| Q1b-06 GPT-4o-mini zero-shot | Not added | No value | The extracted artifact is GPT-4.1-mini dedicated prompts, identity-mismatched with the GPT-4o reference row. |",
            "| Q1b-07 GPT-4o-mini 8-shot | Not added | No value | No completed comparable 8-shot pack is available in the supplied checkpoint. |",
            "| Q1b-08 ViPragSent full PhoBERT | Omitted from main table | No value in main-table scope | Contextual PhoBERT-backed ViPragSent row is excluded to retain one primary ours row; provenance remains outside this draft. |",
            "",
            "Source hashes are recorded in `q1b_working_draft_manifest.json`. This artifact is for layout review only and is not a scientific approval or final manuscript source.",
        ]
    )
    (OUT_DIR / "q1b_expanded_working_draft.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    ledger = "\n".join(table_lines[table_lines.index("## Coverage ledger for unresolved historical rows") :]) + "\n"
    (OUT_DIR / "q1b_coverage_ledger.md").write_text(ledger, encoding="utf-8")
    manifest = {
        "artifact_status": "WORKING_DRAFT_PENDING_CROSS_COHORT_VERIFICATION",
        "canonical_input": str(CANONICAL),
        "canonical_sha256": sha256(CANONICAL),
        "local_subset_input": str(LOCAL),
        "local_subset_sha256": sha256(LOCAL),
        "output_csv": str(CONSOLIDATED),
        "rows_in_main_draft": [row["model"] for row in rows],
        "omitted_contextual_row": "ViPragSent full PhoBERT inventory",
        "historical_rows_ledgered_without_values": ["Q1b-02", "Q1b-06", "Q1b-07", "Q1b-08"],
        "q2_used": False,
        "ordinary_ranking_status": "WITHHELD_PENDING_MATCHED_ID_GOLD_PROOF",
        "cohort_provenance": {
            "primary_aivivn_normalized_test_sha256_prefix": "27DA59A4...",
            "baseline_normalized_test_sha256_prefix": "F6F6A78E...",
            "hash_comparison_semantics": "per-dataset cross-group matching; not cross-dataset hash equality",
            "baseline_hash_scope_reported": "dataset-specific prefixes; semantic cohort equality under investigation",
            "required_for_final": "matched ID/gold proof or explicit cohort separation",
        },
    }
    (OUT_DIR / "q1b_working_draft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} Q1b working rows to {OUT_DIR}")


if __name__ == "__main__":
    main()

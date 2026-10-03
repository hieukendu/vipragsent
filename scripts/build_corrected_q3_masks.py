"""Materialize the corrected Q3 masks without changing the frozen legacy masks."""

from __future__ import annotations

import argparse
import csv

from _bootstrap import ROOT
from vipragsent.data.loaders import load_vipragsent
from vipragsent.data.masks import (
    EXPECTED_BUDGETS,
    REQUIRED_MASK_COLUMNS,
    read_mask,
    validate_q3_masks,
)
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.rationale_promotion import load_approved_rationales


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/q3_low_resource_sarcasm")
    parser.add_argument("--output", default="data/processed/q3_low_resource_sarcasm_corrected")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = ROOT / args.source
    output = ROOT / args.output
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    train_by_id = {item.sample_id: item for item in bundle.train}
    legacy_report = validate_q3_masks(source, train_by_id, strict_frozen=True)
    rationales = load_approved_rationales(ROOT)
    rationale_ids = {str(sample_id) for sample_id, value in rationales.items() if str(value.get("rationale", "")).strip()}
    positive_ids = {sample_id for sample_id, example in train_by_id.items() if int(example.labels["sarcasm"]) == 1}
    missing = sorted(positive_ids - rationale_ids)
    if missing:
        raise RuntimeError(f"Corrected Q3 requires rationale supervision for every sarcasm-positive train row; missing {len(missing)} IDs")

    output.mkdir(parents=True, exist_ok=True)
    rows_by_budget: dict[str, list[dict[str, str]]] = {}
    for budget in EXPECTED_BUDGETS:
        rows: list[dict[str, str]] = []
        for row in read_mask(source / f"budget_{budget}_masks.csv"):
            corrected = {key: row[key] for key in REQUIRED_MASK_COLUMNS}
            # Preserve the locked positive-only rationale policy: the corrected
            # change adds rationale supervision to out-of-budget positives, but
            # does not introduce rationale supervision for the fixed negatives.
            corrected["rationale_loss_mask"] = corrected["is_sarcasm_positive"]
            rows.append(corrected)
        rows_by_budget[budget] = rows
        path = output / f"budget_{budget}_masks.csv"
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing corrected mask: {path}")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_MASK_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    summary_path = output / "budget_summary.csv"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["budget", "selected_sarcasm_positives", "all_train_negatives_retained", "train_rows_total"], lineterminator="\n")
        writer.writeheader()
        for budget in EXPECTED_BUDGETS:
            rows = rows_by_budget[budget]
            writer.writerow(
                {
                    "budget": budget,
                    "selected_sarcasm_positives": sum(int(row["positive_selected_for_budget"]) for row in rows),
                    "all_train_negatives_retained": sum(int(row["is_sarcasm_positive"]) == 0 for row in rows),
                    "train_rows_total": len(rows),
                }
            )

    readme_path = output / "README.md"
    if readme_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing README: {readme_path}")
    readme_path.write_text(
        "# Corrected Q3 low-resource sarcasm masks\n\n"
        "Derived from the frozen nested positive subsets and fixed negative pool.\n"
        "For every out-of-budget sarcasm-positive row, only `sarcasm_target_mask` is zero.\n"
        "`rationale_loss_mask` remains one for every sarcasm-positive row, including\n"
        "out-of-budget positives; fixed negative rows retain the locked zero mask.\n"
        "Polarity, emotion, and all non-sarcasm pragmatic targets remain active.\n",
        encoding="utf-8",
    )

    print({"status": "PASS", "source": str(source.relative_to(ROOT)), "output": str(output.relative_to(ROOT)), "legacy_source_hashes": {budget: legacy_report["mask_hashes"][budget] for budget in EXPECTED_BUDGETS}, "corrected_hashes": {budget: sha256_file(output / f"budget_{budget}_masks.csv") for budget in EXPECTED_BUDGETS}, "rationale_records": len(rationale_ids)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

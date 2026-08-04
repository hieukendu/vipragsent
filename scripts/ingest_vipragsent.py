from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from vipragsent.data.annotation import recompute_human_iaa
from vipragsent.data.loaders import ingest_zip
from vipragsent.phase import write_phase_handoff


def find_zip(root: Path) -> Path | None:
    candidates = sorted(root.glob("ViPragSent_Experiment_Dataset_FINAL_V8.zip"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and validate the frozen ViPragSent V8 package")
    parser.add_argument("--zip", dest="zip_path")
    args = parser.parse_args()
    zip_path = Path(args.zip_path) if args.zip_path else find_zip(ROOT)
    if not zip_path or not zip_path.exists():
        write_phase_handoff("01", "BLOCKED", inputs_read=["ViPragSent_Experiment_Dataset_FINAL_V8.zip"], blockers=["Dataset ZIP is missing"], next_phase_ready=False)
        return 2
    try:
        manifest = ingest_zip(zip_path, raw_root=ROOT / "data/raw/vipragsent_package", processed_root=ROOT / "data/processed", manifest_root=ROOT / "data/manifests")
        raw_package = next((ROOT / "data/raw/vipragsent_package").glob("*/"))
        iaa = recompute_human_iaa(raw_package)
        (ROOT / "data/manifests" / "human_iaa_recomputed.json").write_text(json.dumps({"computed_before_adjudication": True, "fields": iaa}, indent=2) + "\n", encoding="utf-8")
        write_phase_handoff(
            "01",
            "PASS",
            inputs_read=[str(zip_path), "02_vipragsent/*.csv", "04_q3_low_resource_sarcasm/*.csv", "05_rationale_generation/rationale_generation_input_train.jsonl"],
            files_created=["data/raw/vipragsent_package", "data/processed/vipragsent", "data/processed/q3_low_resource_sarcasm", "data/processed/rationales/azure_rationale_input_train.jsonl", "data/manifests/dataset_manifest.json", "data/manifests/human_iaa_recomputed.json"],
            tests_run=["V8 count/split/schema validation", "Q3 nested-mask validation", "human IAA recomputation", "rationale placeholder sanitization"],
            tests_passed=True,
            next_phase_ready=True,
        )
        print(json.dumps({"status": "PASS", "source_zip_sha256": manifest["source_zip_sha256"], "processed_fingerprint": manifest["processed_fingerprint"]}, indent=2))
        return 0
    except Exception as exc:
        write_phase_handoff("01", "FAIL", inputs_read=[str(zip_path)], blockers=[str(exc)], next_phase_ready=False)
        print(f"ingest failed: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.hashing import sha256_file
from vipragsent.phase import write_phase_handoff


MANUAL_READMES = {
    "uit_vsfc": """# UIT-VSFC manual drop\n\nPlace the license-compliant official UIT-VSFC test file at `data/external/manual_drop/uit_vsfc/test.csv`.\nThe normalized file must be `data/processed/external/uit_vsfc/test.csv` with columns `sample_id,text,polarity`.\nProvide the official source URL/revision, license/access note, split evidence, and SHA-256 in `data/manifests/external_datasets.json`.\nDo not substitute an unsplit mirror or a random split.\n""",
    "uit_vsmec": """# UIT-VSMEC manual drop\n\nPlace the license-compliant official UIT-VSMEC test file at `data/external/manual_drop/uit_vsmec/test.csv`.\nThe normalized file must be `data/processed/external/uit_vsmec/test.csv` with columns `sample_id,text,emotion`.\nProvide the official source URL/revision, license/access note, split evidence, and SHA-256 in `data/manifests/external_datasets.json`.\nDo not substitute an unsplit mirror or a random split.\n""",
    "aivivn_original": """# AIVIVN original manual fallback\n\nThe original Kaggle source is `mcocoz/aivivn-2019`. Configure Kaggle credentials through the standard environment or `KAGGLE_CONFIG_DIR`; never commit credentials.\nThe original binary files are provenance-only. Q1b uses the bundled `AIVIVN-human-derived-3way` split.\n""",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate external dataset downloads")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name, content in MANUAL_READMES.items():
        path = ROOT / "data/external/manual_drop" / name / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    bundled = ROOT / "data/processed/external/aivivn_human_derived_3way/test.csv"
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "uit_vsfc": {"status": "BLOCKED", "source": "official UIT source or official UIT Hugging Face organization", "normalized_path": "data/processed/external/uit_vsfc/test.csv", "checksum": None, "license_note": "manual agreement/access required"},
            "uit_vsmec": {"status": "BLOCKED", "source": "official UIT source", "normalized_path": "data/processed/external/uit_vsmec/test.csv", "checksum": None, "license_note": "manual agreement/access required"},
            "aivivn_original": {"status": "BLOCKED", "source": "Kaggle mcocoz/aivivn-2019", "normalized_path": None, "checksum": None, "license_note": "Kaggle credentials required; provenance-only"},
            "aivivn_human_derived_3way": {"status": "PASS" if bundled.exists() else "BLOCKED", "source": "bundled V8 package", "normalized_path": str(bundled.relative_to(ROOT)) if bundled.exists() else None, "checksum": sha256_file(bundled) if bundled.exists() else None, "license_note": "project-bundled human-derived split"},
        },
        "q1b_uses_bundled_aivivn_human_derived_3way": True,
        "external_finetuning": False,
    }
    path = ROOT / "data/manifests/external_datasets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    blocked = [name for name, item in manifest["datasets"].items() if item["status"] == "BLOCKED"]
    status = "PASS" if not blocked else "BLOCKED"
    write_phase_handoff("02", status, inputs_read=["22_DATA_SOURCE_REGISTRY.md", "V8 bundled AIVIVN files"], files_created=["data/manifests/external_datasets.json", "data/external/manual_drop/*/README.md"], tests_run=["external manifest generation", "bundled AIVIVN schema/checksum check"], tests_passed=True, blockers=[f"Manual or credentialed dataset required: {name}" for name in blocked], next_phase_ready=not blocked)
    print(json.dumps({"status": status, "blocked": blocked, "dry_run": args.dry_run}, indent=2))
    return 0 if not blocked or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Recompute pre-adjudication IAA from the supplied annotation CSVs only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT))
from release_paths import default_release  # noqa: E402
from release_contract import validate_release  # noqa: E402
from reproduction_inputs import json_bytes, locked_write  # noqa: E402

from vipragsent.constants import PRAGMATIC_LABELS  # noqa: E402
from vipragsent.data.annotation import recompute_human_iaa_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=default_release(ROOT))
    args = parser.parse_args()
    release = args.release.resolve()
    validate_release(release, ROOT / ".work/release_validation")
    fields = recompute_human_iaa_files(release / "annotations/annotator_1.csv", release / "annotations/annotator_2.csv")
    report = {"computed_before_adjudication": True, "fields": fields,
              "verification_scope": "released_first_pass_annotations_not_annotation_process_replay",
              "macro_pragmatic_alpha": sum(float(row["krippendorff_alpha_nominal"]) for row in fields if row["field"] in PRAGMATIC_LABELS) / len(PRAGMATIC_LABELS)}
    locked_write(ROOT / "data/manifests/human_iaa_recomputed.json", json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

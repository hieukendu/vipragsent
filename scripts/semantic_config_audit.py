from __future__ import annotations

import json

from _bootstrap import ROOT
from vipragsent.config_validation import validate_config_tree


def main() -> int:
    report = validate_config_tree(ROOT)
    report["full_inference_output_source"] = "classification_heads"
    report["rationale_decoder_enabled_at_inference"] = False
    report["cot_only_output_source"] = "parsed_generated_labels"
    report["phobert_preprocessing"] = "vncorenlp_rdrsegmenter"
    output = ROOT / "reports/semantic_config_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

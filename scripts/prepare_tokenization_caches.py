from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.data.loaders import load_vipragsent
from vipragsent.data.preprocessing import DeterministicSegmenter, PreprocessingSpec, TextPreprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic tokenizer caches")
    parser.add_argument("--backbone", choices=["phobert_base", "xlmr_large", "sailor_7b", "vistral_7b"], default="phobert_base")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    preprocessor = TextPreprocessor(PreprocessingSpec(args.backbone, "vncorenlp_rdrsegmenter" if args.backbone == "phobert_base" else "unicode_nfc", "fixture-v1" if args.fixture else "runtime-v1"), segmenter=DeterministicSegmenter(version="fixture-whitespace" if args.fixture else "vncorenlp-rdrsegmenter-pending"))
    output_root = ROOT / "data/processed/tokenized_text" / args.backbone
    reports = {}
    for split, examples in bundle.splits.items():
        reports[split] = preprocessor.write_cache([{"sample_id": row.sample_id, "text": row.text} for row in examples], output_root / f"{split}.jsonl")
    print(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

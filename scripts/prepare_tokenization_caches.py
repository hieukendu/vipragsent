from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.data.loaders import load_vipragsent
from vipragsent.data.preprocessing import (
    DeterministicSegmenter,
    PreprocessingSpec,
    TextPreprocessor,
    VnCoreNLPSegmenter,
)
from vipragsent.hashing import sha256_file
from vipragsent.models.factory import load_model_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic tokenizer caches")
    parser.add_argument("--backbone", choices=["phobert_base", "xlmr_large", "sailor_7b", "vistral_7b"], default="phobert_base")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    model_spec = load_model_registry(ROOT / "configs/models/model_registry.yaml")[args.backbone]
    segmenter = DeterministicSegmenter() if args.fixture and args.backbone == "phobert_base" else VnCoreNLPSegmenter.from_env() if args.backbone == "phobert_base" else None
    preprocessor = TextPreprocessor(PreprocessingSpec(args.backbone, "vncorenlp_rdrsegmenter" if args.backbone == "phobert_base" else "unicode_nfc", "fixture-v1" if args.fixture else "runtime-v1", tokenizer_revision=model_spec.tokenizer_revision, model_revision=model_spec.revision, execution_mode="fixture" if args.fixture else "production"), segmenter=segmenter)
    output_root = ROOT / "data/processed/tokenized_text" / args.backbone
    reports = {}
    for split, examples in bundle.splits.items():
        reports[split] = preprocessor.write_cache([{"sample_id": row.sample_id, "text": row.text} for row in examples], output_root / f"{split}.jsonl")
    manifest_path = ROOT / "data/manifests/tokenization_cache_manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    backbones = dict(existing.get("backbones", {})) if isinstance(existing.get("backbones", {}), dict) else {}
    backbones[args.backbone] = {
        "model_revision": model_spec.revision,
        "tokenizer_revision": model_spec.tokenizer_revision,
        "preprocessing_name": preprocessor.spec.preprocessing_name,
        "preprocessing_version": preprocessor.spec.preprocessing_version,
        "execution_mode": preprocessor.spec.execution_mode,
        "max_length": preprocessor.spec.max_length,
        "segmenter_version": getattr(preprocessor.segmenter, "version", None),
        "segmenter_resource_checksum": getattr(preprocessor.segmenter, "resource_checksum", None),
        "splits": {
            split: report | {
                "path": (output_root / f"{split}.jsonl").relative_to(ROOT).as_posix(),
                "sha256": sha256_file(output_root / f"{split}.jsonl"),
            }
            for split, report in reports.items()
        },
    }
    execution_modes = {str(item.get("execution_mode")) for item in backbones.values()}
    execution_mode = next(iter(execution_modes)) if len(execution_modes) == 1 else "mixed"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "execution_mode": execution_mode,
            "data_fingerprint": bundle.fingerprint,
            "dataset_split_counts": {split: len(examples) for split, examples in bundle.splits.items()},
            "backbones": {name: backbones[name] for name in sorted(backbones)},
        },
    )
    print(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

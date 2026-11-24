"""Prepare canonical ARR_clean inputs without requiring a saved-artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from release_contract import validate_release
from release_paths import default_release
from reproduction_inputs import prepare_external, prepare_supplementary


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(source: Path, destination: Path) -> int:
    if destination.exists():
        if sha(source) != sha(destination):
            raise RuntimeError(f"Existing file differs (not overwritten): {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return 1


def write_manifest(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Existing runtime manifest differs (not overwritten): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=default_release(root))
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument("--bundle", type=Path, help="Optional separately supplied saved-artifact bundle")
    parser.add_argument("--with-artifacts", action="store_true")
    parser.add_argument("--external-root", type=Path, help="Optional local data root containing normalized external test CSVs")
    args = parser.parse_args()
    release, target = args.release.resolve(), args.output_root.resolve()
    bundle = args.bundle.resolve() if args.bundle else None
    if args.with_artifacts and bundle is None:
        parser.error("Saved predictions are not included. Supply --bundle explicitly for --with-artifacts.")
    if bundle is not None and not (bundle / "runtime_data").is_dir():
        parser.error("The supplied bundle has no runtime_data directory.")
    if args.with_artifacts and not (bundle / "artifact_archive").is_dir():
        parser.error("The supplied bundle has no artifact_archive directory.")
    report = validate_release(release, root / ".work/release_validation")
    count = 0
    mappings = {"q3_masks": "q3_low_resource_sarcasm", "rationales": "rationales"}
    for name in ("train.csv", "dev.csv", "test.csv", "vipragsent_full.csv", "label_schema.json", "split_manifest.csv"):
        count += copy(release / name, target / "data/processed/vipragsent" / name)
    for source, destination in mappings.items():
        for path in sorted((release / source).iterdir()):
            if path.is_file() and path.name != "manifest.json":
                count += copy(path, target / "data/processed" / destination / path.name)
    # Release-byte locks do not reconstruct historical provenance.
    dataset = json.loads((release / "dataset_manifest.json").read_text(encoding="utf-8"))
    for key in ("source_repository", "source_hub_revision", "rationale_source_sha256"):
        dataset.pop(key, None)
    # The loader defines 'active rationale' as the sanitized generator INPUT.
    # Approved generation OUTPUT is a separate file and needs a separate lock.
    dataset["active_rationale_sha256"] = sha(release / "rationales/azure_rationale_input_train.jsonl").upper()
    dataset["active_rationale_path"] = "data/processed/rationales/azure_rationale_input_train.jsonl"
    dataset["approved_generated_rationale_sha256"] = sha(release / "rationales/approved_generated_rationales_train.jsonl").upper()
    dataset["hash_scope"] = "delivered_release_bytes_not_historical_provenance"
    masks = json.loads((release / "q3_masks/manifest.json").read_text(encoding="utf-8"))
    for key in ("source_directory", "source_mask_hashes", "target_directory"):
        masks.pop(key, None)
    masks["mask_hashes"] = {
        budget: sha(release / f"q3_masks/budget_{budget}_masks.csv").upper()
        for budget in ("32", "64", "128", "256", "512", "full")
    }
    masks["hash_scope"] = "delivered_release_bytes_not_historical_provenance"
    write_manifest(target / "data/processed/q3_low_resource_sarcasm/manifest.json", masks)
    write_manifest(target / "data/processed/vipragsent/dataset_manifest.json", dataset)
    write_manifest(target / "data/manifests/dataset_manifest.json", dataset)
    write_manifest(target / "data/manifests/loss_weights_train.json", json.loads((release / "loss_weights_train.json").read_text(encoding="utf-8")))
    write_manifest(target / "data/manifests/release_contract.json", report)
    count += prepare_supplementary(root, release, target)
    if args.external_root is not None:
        count += prepare_external(args.external_root.resolve(), target)
    if bundle is not None:
        for path in (bundle / "runtime_data").rglob("*"):
            rel = path.relative_to(bundle / "runtime_data")
            if path.is_file() and rel.parts[:2] not in {
                ("processed", "vipragsent"), ("processed", "rationales"),
                ("processed", "q3_low_resource_sarcasm"),
                ("manifests", "dataset_manifest.json"), ("manifests", "loss_weights_train.json"),
            }:
                count += copy(path, target / "data" / rel)
        if args.with_artifacts:
            for path in (bundle / "artifact_archive").rglob("*"):
                if path.is_file():
                    count += copy(path, target / path.relative_to(bundle / "artifact_archive"))
    print(json.dumps({"status": "PASS", "files_copied": count,
                      "canonical_data_verified": True,
                      "historical_provenance_verified": False,
                      "saved_artifacts_restored": args.with_artifacts,
                      "external_test_inputs_prepared": args.external_root is not None,
                      "training_or_API_rerun": False}, indent=2))


if __name__ == "__main__":
    main()

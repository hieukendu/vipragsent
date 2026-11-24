"""Validate supplementary runtime inputs without importing workstation evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

PRAGMATIC = ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")
BUDGETS = ("32", "64", "128", "256", "512", "full")
EXTERNAL = {
    "uit_vsfc": (3166, "polarity", {"negative", "neutral", "positive"},
                 "C27B61C70DE1090BD0A12F0F2824148D1691C68FB9CB18A07F23AE5282E75D96"),
    "uit_vsmec": (693, "emotion", {"anger", "disgust", "enjoyment", "fear", "other", "sadness", "surprise"},
                  "1D2E604BD06A4AAAAD33A3B98DE2FB910E9020F4D468EF8BC16DDCDEADB3D955"),
    "aivivn_human_derived_3way": (1609, "label", {"negative", "neutral", "positive"},
                                   "27DA59A452ACC0FA63E26A7F49325DE7D963353E1179C5FE21BC92DB9394EDA8"),
}


def locked_write(path: Path, payload: bytes) -> int:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing runtime input differs (not overwritten): {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return 1


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_fewshot(path: Path, train: dict[str, dict[str, str]], eligible: set[str] | None = None) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    demos = manifest["demonstrations"]
    ids = manifest["sample_ids"]
    if len(ids) != 8 or len(set(ids)) != 8 or ids != [d["sample_id"] for d in demos]:
        raise ValueError(f"Invalid eight-shot IDs: {path.name}")
    if manifest.get("source_split") != "vipragsent_train_only":
        raise ValueError("Demonstrations must be train-only")
    if eligible is not None and not set(ids).issubset(eligible):
        raise ValueError(f"Demonstrations exceed Q3 budget: {path.name}")
    keys = (*PRAGMATIC, "polarity", "emotion")
    for demo in demos:
        row = train.get(demo["sample_id"])
        if row is None or demo["text"] != row["text"] or set(demo["labels"]) != set(keys):
            raise ValueError(f"Demonstration is not an exact train record: {path.name}")
        if any(str(demo["labels"][key]) != row[key] for key in keys):
            raise ValueError(f"Demonstration gold mismatch: {path.name}")
    task = manifest.get("task", "pragmatic")
    if task == "pragmatic":
        if any(int(demos[i]["labels"][key]) != 1 for i, key in enumerate(PRAGMATIC)):
            raise ValueError("Invalid pragmatic positive demonstration")
        for demo, polarity in zip(demos[6:], ("positive", "negative"), strict=True):
            if any(int(demo["labels"][key]) for key in PRAGMATIC) or demo["labels"]["polarity"] != polarity:
                raise ValueError("Invalid ordinary demonstration control")
    else:
        expected = ({"negative": 3, "neutral": 2, "positive": 3} if task == "polarity" else
                    {"anger": 1, "disgust": 1, "enjoyment": 1, "fear": 1, "other": 2, "sadness": 1, "surprise": 1})
        if task not in {"polarity", "emotion"} or {k: sum(d["labels"][task] == k for d in demos) for k in expected} != expected:
            raise ValueError(f"Invalid task demonstration composition: {path.name}")
        if manifest.get("composition") != expected:
            raise ValueError(f"Composition metadata disagrees with demonstrations: {path.name}")
    expected_hash = manifest.get("prompt_hash")
    unhashed = {k: v for k, v in manifest.items() if k != "prompt_hash"}
    actual = hashlib.sha256(json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    if expected_hash != actual:
        raise ValueError(f"Demonstration manifest hash mismatch: {path.name}")


def prepare_supplementary(source: Path, release: Path, target: Path) -> int:
    train = {r["sample_id"]: r for r in csv_rows(release / "train.csv")}
    count = 0
    names = ["pragmatic_v1.json", "polarity_v1.json", "emotion_v1.json"]
    names.extend(f"q3_budget_{budget}_v1.json" for budget in BUDGETS)
    for name in names:
        path = source / "prompts/fewshot" / name
        budget = path.stem.removeprefix("q3_budget_").removesuffix("_v1") if path.name.startswith("q3_budget_") else None
        eligible = None
        if budget is not None:
            eligible = {r["sample_id"] for r in csv_rows(release / f"q3_masks/budget_{budget}_masks.csv")
                        if r["is_sarcasm_positive"] == "0" or r["positive_selected_for_budget"] == "1"}
        validate_fewshot(path, train, eligible)
        count += locked_write(target / "data/manifests/prompts" / path.name, path.read_bytes())
    # A delivered-file lock is not an invented historical approval or API receipt.
    rationale = release / "rationales/approved_generated_rationales_train.jsonl"
    rows = [json.loads(line) for line in rationale.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(train) or any(not r.get("rationale") for r in rows):
        raise ValueError("Released rationale IDs or coverage differ from train")
    lock = {"schema_version": 1, "canonical_file_sha256": hashlib.sha256(rationale.read_bytes()).hexdigest().upper(),
            "successful_count": len(rows), "train_count": len(train), "coverage_rate": 1.0,
            "hash_scope": "delivered_release_bytes_not_historical_approval", "historical_approval_verified": False}
    count += locked_write(target / "data/manifests/approved_generated_rationales_train.json", json_bytes(lock))
    return count


def prepare_external(source: Path, target: Path) -> int:
    """Use explicit local test inputs; verify locked bytes before writing anything."""
    prepared = []
    manifest = {"datasets": {}, "external_finetuning": False, "q1b_uses_bundled_aivivn_human_derived_3way": True,
                "verification_scope": "delivered_test_bytes_and_schema_not_access_receipts"}
    for name, (expected_rows, field, labels, expected_hash) in EXTERNAL.items():
        relative = Path("processed/external") / name / "test.csv"
        payload = (source / relative).read_bytes().replace(b"\r\n", b"\n")
        # Keep AIVIVN's BOM: its accepted test-byte lock includes that header BOM.
        actual = hashlib.sha256(payload).hexdigest().upper()
        if actual != expected_hash:
            raise ValueError(f"Unrecognized external test bytes: {name}")
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline="")))
        ids = [r.get("sample_id", r.get("id")) for r in rows]
        if len(rows) != expected_rows or len(set(ids)) != len(rows) or None in ids or "" in ids:
            raise ValueError(f"Invalid external test count or IDs: {name}")
        if any(r.get(field) not in labels or not r.get("text", r.get("comment", "")) for r in rows):
            raise ValueError(f"Invalid external test labels or text: {name}")
        prepared.append((target / "data" / relative, payload))
        manifest["datasets"][name] = {"status": "PASS", "normalized_path": "data/" + relative.as_posix(),
                                      "checksum": actual, "row_count": len(rows), "split": "test",
                                      "hash_scope": "verified_test_bytes", "redistribution_permission_verified": False}
    # Conflict checks happen for the complete set before any local copy.
    prepared.append((target / "data/manifests/external_datasets.json", json_bytes(manifest)))
    for path, payload in prepared:
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"Existing runtime input differs (not overwritten): {path}")
    return sum(locked_write(path, payload) for path, payload in prepared)

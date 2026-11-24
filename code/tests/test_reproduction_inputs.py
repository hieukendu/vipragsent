from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from build_submission import PENDING_MATERIALS, source_paths
from reproduction_inputs import (
    BUDGETS,
    EXTERNAL,
    PRAGMATIC,
    csv_rows,
    locked_write,
    prepare_external,
    prepare_supplementary,
    validate_fewshot,
)

from vipragsent.azure.prompts import validate_task_demo_manifest
from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS
from vipragsent.data.annotation import recompute_human_iaa_files
from vipragsent.orchestration.rationale_promotion import load_approved_rationales

ROOT = Path(__file__).resolve().parents[1]


def test_supplied_rubric_is_preserved_and_packaged() -> None:
    path = ROOT / "docs/annotation_rubric.md"
    assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == "A8C4F805834E85B8CFAA651F6C07AE79E4ED6B5476C7EE715B67E605B0CFBAEA"
    assert path in source_paths(ROOT)
    assert not any("manual" in name or "rubric" in name for name in PENDING_MATERIALS)
    text = path.read_text(encoding="utf-8")
    for label in (*PRAGMATIC, *POLARITY_LABELS, *EMOTION_LABELS):
        assert label in text


def test_rubric_examples_preserve_released_gold() -> None:
    text = (ROOT / "docs/annotation_rubric.md").read_text(encoding="utf-8")
    examples = list(dict.fromkeys(re.findall(r'> \*\*\u201c(.+?)\u201d\*\*', text)))
    assert len(examples) == len(PRAGMATIC) == 6
    rows = csv_rows(ROOT / "data/processed/vipragsent/vipragsent_full.csv")
    for label, example in zip(PRAGMATIC, examples, strict=True):
        # The sarcasm example is an explicitly documented excerpt, not full text.
        matches = [r for r in rows if r["text"].startswith(example)] if label == "sarcasm" else [r for r in rows if r["text"] == example]
        assert len(matches) == 1 and matches[0][label] == "1"
        if label == "code_switching":
            assert all(matches[0][key] == "1" for key in ("implicit_sentiment", "irony", "code_switching"))


@pytest.mark.parametrize("name", ["pragmatic_v1", "polarity_v1", "emotion_v1", *[f"q3_budget_{b}_v1" for b in BUDGETS]])
def test_fixed_demos_match_train_gold_and_budget(name: str) -> None:
    train = {r["sample_id"]: r for r in csv_rows(ROOT / "data/processed/vipragsent/train.csv")}
    path = ROOT / f"prompts/fewshot/{name}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    eligible = None
    if name.startswith("q3_budget_"):
        budget = name.removeprefix("q3_budget_").removesuffix("_v1")
        eligible = {r["sample_id"] for r in csv_rows(ROOT / f"data/processed/q3_low_resource_sarcasm/budget_{budget}_masks.csv")
                    if r["is_sarcasm_positive"] == "0" or r["positive_selected_for_budget"] == "1"}
    validate_fewshot(path, train, eligible)
    validate_task_demo_manifest(manifest, manifest.get("task", "pragmatic"), q3_eligible_ids=eligible)
    assert (ROOT / f"data/manifests/prompts/{name}.json").read_bytes() == path.read_bytes()


def test_demos_reject_modified_gold(tmp_path: Path) -> None:
    train = {r["sample_id"]: r for r in csv_rows(ROOT / "data/processed/vipragsent/train.csv")}
    manifest = json.loads((ROOT / "prompts/fewshot/pragmatic_v1.json").read_text(encoding="utf-8"))
    manifest["demonstrations"][0]["labels"]["sarcasm"] ^= 1
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="gold mismatch"):
        validate_fewshot(path, train)


def test_delivered_rationales_load_without_fabricated_approval() -> None:
    records = load_approved_rationales(ROOT)
    assert len(records) == 7998
    lock = json.loads((ROOT / "data/manifests/approved_generated_rationales_train.json").read_text(encoding="utf-8"))
    assert lock["historical_approval_verified"] is False
    assert "approval_record_sha256" not in lock


@pytest.mark.parametrize("name", EXTERNAL)
def test_prepared_external_tests_are_exact_locked_partitions(name: str) -> None:
    path = ROOT / f"data/processed/external/{name}/test.csv"
    # These privately available external tests are optional, not archive members.
    if not path.exists():
        pytest.skip("optional external test not supplied")
    expected_rows, field, labels, expected_hash = EXTERNAL[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == expected_hash
    rows = csv_rows(path)
    assert len(rows) == expected_rows
    assert {r[field] for r in rows}.issubset(labels)
    assert not any("processed/external" in p.as_posix() for p in source_paths(ROOT))


def test_external_import_rejects_unlocked_bytes_before_writing(tmp_path: Path) -> None:
    bad = tmp_path / "source/processed/external/uit_vsfc/test.csv"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"sample_id,text,polarity\nchanged,changed,neutral\n")
    destination = tmp_path / "target"
    with pytest.raises(ValueError, match="Unrecognized external test bytes"):
        prepare_external(tmp_path / "source", destination)
    assert not destination.exists()


def test_runtime_input_lock_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "user.json"
    path.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="not overwritten"):
        locked_write(path, b"different")
    assert path.read_bytes() == b"existing"


def test_all_nine_demos_are_required(tmp_path: Path) -> None:
    # Input prep must not silently pass if an entire prompt directory is absent.
    release = tmp_path / "release"
    release.mkdir()
    (release / "train.csv").write_text("sample_id,text\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        prepare_supplementary(tmp_path / "missing_source", release, tmp_path / "target")


def test_released_annotation_iaa_matches_paper_rounding() -> None:
    release = __import__("release_paths").default_release(ROOT)
    rows = recompute_human_iaa_files(release / "annotations/annotator_1.csv", release / "annotations/annotator_2.csv")
    fields = {r["field"]: r for r in rows}
    expected = (0.9181, 0.8957, 0.9923, 0.5746, 0.7630, 0.7599)
    for label, value in zip(PRAGMATIC, expected, strict=True):
        assert fields[label]["n"] == 11997
        assert round(fields[label]["krippendorff_alpha_nominal"], 4) == value
    assert round(fields["polarity"]["cohen_kappa"], 3) == 0.854
    assert round(fields["emotion"]["cohen_kappa"], 3) == 0.759
    assert round(sum(fields[k]["krippendorff_alpha_nominal"] for k in PRAGMATIC) / len(PRAGMATIC), 3) == 0.817


def test_annotation_ids_must_align(tmp_path: Path) -> None:
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    first.write_text("sample_id\none\n", encoding="utf-8")
    second.write_text("sample_id\ntwo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not aligned"):
        recompute_human_iaa_files(first, second)

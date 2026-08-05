from __future__ import annotations

from pathlib import Path

from vipragsent.runtime.batch_probe import probe_physical_batch
from vipragsent.runtime.model_assets import merge_family_manifest, write_family_status


def _write_family(root: Path, family: str, *, fixture: bool = False) -> None:
    write_family_status(root, family, "cache", {"status": "PASS", "revision": "locked"})
    write_family_status(root, family, "smoke", {"status": "PASS", "actual_local_loads": True})
    write_family_status(root, family, "batch", {"status": "PASS", "frozen": True, "successful_batch": 4, "fixture_probe": fixture})


def test_global_model_manifest_requires_real_family_evidence(tmp_path: Path) -> None:
    registry = {"encoder": {"repo_id": "fixture/encoder", "revision": "locked", "tokenizer_revision": "locked"}}
    _write_family(tmp_path, "encoder", fixture=True)
    fixture_manifest = merge_family_manifest(tmp_path, registry)
    assert fixture_manifest["weights_downloaded"] is False
    _write_family(tmp_path, "encoder", fixture=False)
    real_manifest = merge_family_manifest(tmp_path, registry)
    assert real_manifest["weights_downloaded"] is True


def test_batch_probe_records_success_and_oom_failure(tmp_path: Path) -> None:
    def probe(batch: int) -> bool:
        if batch > 4:
            raise RuntimeError("CUDA out of memory")
        return True

    result = probe_physical_batch(
        tmp_path,
        "encoder",
        probe=probe,
        candidate_order=(32, 16, 8, 4),
        hardware_identity="test-gpu",
    )
    assert result["status"] == "PASS"
    assert result["successful_batch"] == 4
    assert any(item["oom"] is True for item in result["failed_candidates"])

    blocked = probe_physical_batch(tmp_path, "blocked", probe=lambda batch: False, candidate_order=(2, 1))
    assert blocked["status"] == "BLOCKED"
    assert blocked["successful_batch"] is None

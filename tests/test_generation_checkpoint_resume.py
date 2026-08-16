from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vipragsent.hashing import sha256_json
from vipragsent.training.generation_checkpoint import (
    GenerationCheckpointError,
    load_generation_checkpoint,
    save_generation_checkpoint,
)


def _provenance() -> dict[str, object]:
    dataset_hash = "A" * 64
    return {
        "model": {"name": "tiny", "revision": "r1"},
        "model_artifact": {"identity": "tiny-model@r1"},
        "tokenizer_artifact": {"identity": "tiny-tokenizer@r1"},
        "dataset": {"identity": "tiny-dataset", "hash": dataset_hash},
        "data_hash": dataset_hash,
        "optimizer": {"name": "AdamW", "revision": "torch"},
        "scheduler": {"name": "linear", "total_steps": 4},
        "rng": {"seed": 17, "algorithm": "torch+numpy+python"},
        "data_order": {"epoch": 1, "indices": [2, 0, 1]},
        "config": {"batch_size": 1, "gradient_accumulation": 1},
        "model_environment": {"device": "cpu", "dtype": "float32"},
    }


def _step(model, optimizer, scheduler, x, y):
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_uninterrupted_and_resume_are_exact(tmp_path: Path) -> None:
    torch.manual_seed(17)
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    first = nn.Linear(2, 1)
    uninterrupted = nn.Linear(2, 1)
    uninterrupted.load_state_dict(first.state_dict())
    resumed = nn.Linear(2, 1)
    resumed.load_state_dict(first.state_dict())
    opt_a = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    opt_b = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    sch_a = torch.optim.lr_scheduler.LinearLR(opt_a, total_iters=4)
    sch_b = torch.optim.lr_scheduler.LinearLR(opt_b, total_iters=4)
    for index in range(4):
        _step(uninterrupted, opt_a, sch_a, x[index:index + 1], y[index:index + 1])
        if index < 2:
            _step(resumed, opt_b, sch_b, x[index:index + 1], y[index:index + 1])
            if index == 1:
                save_generation_checkpoint(tmp_path / "resume.pt", resumed, opt_b, sch_b, {"step": 2}, _provenance())
    loaded = load_generation_checkpoint(tmp_path / "resume.pt", resumed, expected_provenance=_provenance(), optimizer=opt_b, scheduler=sch_b)
    assert loaded.run_state["step"] == 2
    for index in range(2, 4):
        _step(resumed, opt_b, sch_b, x[index:index + 1], y[index:index + 1])
    assert all(torch.equal(a, b) for a, b in zip(uninterrupted.parameters(), resumed.parameters()))


def test_corruption_and_identity_mismatch_are_rejected(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    save_generation_checkpoint(path, model, None, None, {"step": 0}, _provenance())
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(GenerationCheckpointError, match="content hash"):
        load_generation_checkpoint(path, nn.Linear(2, 1), expected_provenance=_provenance())
    save_generation_checkpoint(path, model, None, None, {"step": 0}, _provenance())
    different = _provenance()
    different["model"] = {"name": "other", "revision": "r1"}
    with pytest.raises(GenerationCheckpointError, match="identity mismatch"):
        load_generation_checkpoint(path, nn.Linear(2, 1), expected_provenance=different)


def test_legacy_fixture_requires_explicit_opt_in(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "state": {"step": 1}}, path)
    with pytest.raises(GenerationCheckpointError, match="legacy"):
        load_generation_checkpoint(path, nn.Linear(2, 1))
    loaded = load_generation_checkpoint(path, nn.Linear(2, 1), allow_legacy_fixture=True, fixture_mode=True)
    assert loaded.checkpoint.report.legacy_compatibility is True


def test_legacy_fixture_with_expected_provenance_requires_validated_identity(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "state": {"step": 1}}, path)
    with pytest.raises(GenerationCheckpointError, match="lacks validated provenance"):
        load_generation_checkpoint(
            path,
            nn.Linear(2, 1),
            allow_legacy_fixture=True,
            fixture_mode=True,
            expected_provenance=_provenance(),
        )

    provenance = _provenance()
    torch.save(
        {
            "model": model.state_dict(),
            "state": {"step": 1},
            "metadata": {"provenance": provenance, "provenance_sha256": sha256_json(provenance)},
        },
        path,
    )
    loaded = load_generation_checkpoint(
        path,
        nn.Linear(2, 1),
        allow_legacy_fixture=True,
        fixture_mode=True,
        expected_provenance=provenance,
    )
    assert loaded.checkpoint.report.legacy_compatibility is True


@pytest.mark.parametrize("data_hash", ["not-a-sha", "DATA_TINY", "0" * 63])
def test_production_checkpoint_rejects_non_sha256_dataset_hash(tmp_path: Path, data_hash: str) -> None:
    model = nn.Linear(2, 1)
    provenance = _provenance()
    provenance["data_hash"] = data_hash
    provenance["dataset"] = {"identity": "tiny-dataset", "hash": data_hash}
    with pytest.raises(GenerationCheckpointError, match="canonical SHA-256"):
        save_generation_checkpoint(
            tmp_path / "invalid.pt",
            model,
            None,
            None,
            {"step": 0},
            provenance,
            production_provenance_required=True,
        )


def test_cpu_fixture_may_keep_non_sha256_dataset_hash(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    provenance = _provenance()
    provenance["data_hash"] = "fixture-data"
    provenance["dataset"] = {"identity": "fixture", "hash": "fixture-data"}
    manifest = save_generation_checkpoint(
        tmp_path / "fixture.pt",
        model,
        None,
        None,
        {"step": 0},
        provenance,
        fixture_mode=True,
    )
    assert manifest.provenance["data_hash"] == "fixture-data"

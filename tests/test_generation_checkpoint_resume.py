from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vipragsent.training.generation_checkpoint import (
    GenerationCheckpointError,
    load_generation_checkpoint,
    save_generation_checkpoint,
)


def _provenance() -> dict[str, object]:
    return {
        "model": {"name": "tiny", "revision": "r1"},
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
    loaded = load_generation_checkpoint(path, nn.Linear(2, 1), allow_legacy_fixture=True)
    assert loaded.checkpoint.report.legacy_compatibility is True

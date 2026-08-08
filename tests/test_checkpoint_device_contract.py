from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vipragsent.runtime.device import (
    DeviceContractReporter,
    move_batch_to_model_device,
    place_non_quantized_model,
    resolve_model_input_device,
)
from vipragsent.training.checkpoints import (
    CheckpointContractError,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from vipragsent.training.engine import CheckpointManager, RunState


class _TwoHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({"polarity": nn.Linear(3, 2), "emotion": nn.Linear(3, 4)})
        self.config = SimpleNamespace(active_tasks=("polarity", "emotion"))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.heads["polarity"](values)


class _SerializedQuantMetadataModel(nn.Module):
    def __init__(self, *, quantized: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self._vipragsent_quantized = quantized

    def _save_to_state_dict(self, destination: dict[str, torch.Tensor], prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "weight.absmax"] = torch.ones(1)
        destination[prefix + "weight.quant_state.bitsandbytes__nf4"] = torch.ones(1, dtype=torch.uint8)


class _NestedSerializedQuantMetadataModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _SerializedQuantMetadataModel(quantized=True)


def _payload(model: nn.Module) -> dict[str, object]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return build_checkpoint_payload(model, optimizer, None, nn.Identity(), as_run_state())


def as_run_state() -> dict[str, object]:
    return {"epoch": 1, "best_metric": 0.5, "status": "PASS"}


def test_checkpoint_v2_round_trip(tmp_path: Path) -> None:
    model = _TwoHeadModel()
    path = save_checkpoint(tmp_path / "best.pt", _payload(model))
    loaded = load_checkpoint(path, model, required_head_prefixes=("heads.polarity", "heads.emotion"), report_path=tmp_path / "load.json")
    assert loaded.report.schema_version == 2
    assert loaded.report.legacy_compatibility is False
    assert loaded.report.matched_ratio == 1.0


def test_checkpoint_legacy_fixture_compatibility(tmp_path: Path) -> None:
    model = _TwoHeadModel()
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "state": as_run_state()}, path)
    loaded = load_checkpoint(path, _TwoHeadModel(), allow_legacy_fixture=True)
    assert loaded.report.legacy_compatibility is True
    assert loaded.report.schema_version == 1


def test_checkpoint_missing_model_state_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing.pt"
    torch.save({"schema_version": 2, "metadata": {}}, path)
    with pytest.raises(CheckpointContractError, match="model_state_dict"):
        load_checkpoint(path, _TwoHeadModel(), report_path=tmp_path / "missing-report.json")
    assert '"status": "FAIL"' in (tmp_path / "missing-report.json").read_text(encoding="utf-8")


def test_checkpoint_zero_matching_keys_fails(tmp_path: Path) -> None:
    path = save_checkpoint(
        tmp_path / "zero.pt",
        {
            "schema_version": 2,
            "model_state_dict": {"unrelated.weight": torch.ones(1)},
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "loss_aggregator_state_dict": {},
            "run_state": {},
            "rng_state": {},
            "metadata": {},
        },
    )
    with pytest.raises(CheckpointContractError, match="zero matching"):
        load_checkpoint(path, _TwoHeadModel())


def test_checkpoint_required_head_missing_fails(tmp_path: Path) -> None:
    model = _TwoHeadModel()
    state = {key: value for key, value in model.state_dict().items() if not key.startswith("heads.emotion")}
    path = save_checkpoint(tmp_path / "head.pt", _payload(model) | {"model_state_dict": state})
    with pytest.raises(CheckpointContractError, match="missing model keys|required task heads"):
        load_checkpoint(path, _TwoHeadModel(), required_head_prefixes=("heads.emotion",))


def test_checkpoint_load_report_written(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    path = save_checkpoint(tmp_path / "linear.pt", _payload(model))
    report_path = tmp_path / "reports" / "load.json"
    load_checkpoint(path, model, report_path=report_path)
    report = report_path.read_text(encoding="utf-8")
    assert '"matched_key_count": 2' in report
    assert '"status": "PASS"' in report


def test_quantized_nf4_metadata_loader_mismatch_is_explicitly_tolerated(tmp_path: Path) -> None:
    model = _SerializedQuantMetadataModel(quantized=True)
    path = save_checkpoint(tmp_path / "nf4.pt", _payload(model))
    loaded = load_checkpoint(path, model, report_path=tmp_path / "nf4-load.json")
    assert loaded.report.status == "PASS"
    assert loaded.report.loader_tolerated_unexpected_keys == (
        "weight.absmax",
        "weight.quant_state.bitsandbytes__nf4",
    )


def test_nested_quantized_contract_tolerates_nf4_metadata(tmp_path: Path) -> None:
    model = _NestedSerializedQuantMetadataModel()
    path = save_checkpoint(tmp_path / "nested-nf4.pt", _payload(model))
    loaded = load_checkpoint(path, model, report_path=tmp_path / "nested-nf4-load.json")
    assert loaded.report.status == "PASS"
    assert loaded.report.loader_tolerated_unexpected_keys == (
        "backbone.weight.absmax",
        "backbone.weight.quant_state.bitsandbytes__nf4",
    )


def test_quantized_nf4_metadata_tolerance_requires_quantized_contract(tmp_path: Path) -> None:
    model = _SerializedQuantMetadataModel(quantized=False)
    path = save_checkpoint(tmp_path / "nf4-unmarked.pt", _payload(model))
    with pytest.raises(CheckpointContractError, match="model loader reported keys"):
        load_checkpoint(path, model)


def test_checkpoint_round_trip_prediction_equality(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = nn.Linear(3, 2)
    values = torch.randn(2, 3)
    expected = model(values).detach().clone()
    path = save_checkpoint(tmp_path / "prediction.pt", _payload(model))
    with torch.no_grad():
        model.weight.add_(10.0)
    load_checkpoint(path, model)
    assert torch.equal(expected, model(values).detach())


def test_checkpoint_manager_writes_canonical_v2_payload(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    manager = CheckpointManager(tmp_path / "checkpoints", "run")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = manager.save("best", model, optimizer, None, nn.Identity(), RunState(status="PASS"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert "model_state_dict" in payload
    assert "model" not in payload


def test_custom_executor_moves_batch_to_model_device() -> None:
    model = nn.Linear(2, 2, device="meta")
    batch = {"input_ids": torch.ones(1, 2), "targets": torch.zeros(1, 2), "sample_ids": ["s1"]}
    moved = move_batch_to_model_device(batch, model)
    assert moved["input_ids"].device.type == "meta"
    assert moved["targets"].device.type == "meta"
    assert moved["sample_ids"] == ["s1"]


def test_custom_executor_device_report(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    reporter = DeviceContractReporter(tmp_path / "device.json", model, "cpu", model_family="fixture")
    report = reporter.observe({"input_ids": torch.ones(1, 2)}, loss=torch.tensor(0.5))
    assert report["status"] == "PASS"
    assert (tmp_path / "device.json").exists()


def test_quantized_device_map_not_overridden() -> None:
    model = nn.Linear(2, 2)
    model._vipragsent_quantized = True
    model._vipragsent_qlora_contract = {"selected_device": "cpu", "device_map": {"": "cpu"}}
    assert resolve_model_input_device(model) == torch.device("cpu")
    with pytest.raises(RuntimeError):
        place_non_quantized_model(model, "cpu")

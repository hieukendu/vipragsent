from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.audit_final_runtime_integration import _device_report
from torch import nn

import vipragsent.orchestration.stage_registry as stage_registry
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.data.collation import BatchCollator
from vipragsent.data.loaders import DatasetExample
from vipragsent.data.preprocessing import (
    DummyTokenizer,
    PreprocessingSpec,
    TextPreprocessor,
    VnCoreNLPSegmenter,
)
from vipragsent.evaluation.confidence_intervals import evaluate_q1a_confidence_intervals
from vipragsent.hashing import sha256_file
from vipragsent.models.qlora import build_qlora_backbone
from vipragsent.orchestration.executors.component_bundle import run_component_bundle
from vipragsent.orchestration.executors.external_retention import (
    evaluate_external_retention_from_disk,
)
from vipragsent.orchestration.executors.generation import (
    GenerationExecutor,
    teacher_forced_generation_loss,
)
from vipragsent.orchestration.executors.q4 import resolve_and_extract_q4_source
from vipragsent.orchestration.stage_plans import resolve_stage_plan, validate_stage_plan_registry
from vipragsent.orchestration.status import RuntimeBlocked
from vipragsent.runtime.device import (
    assert_runtime_device_contract,
    move_batch_to_device,
    place_non_quantized_model,
)

ROOT = Path(__file__).resolve().parents[1]


def _labels(index: int = 0) -> dict[str, int | str]:
    return {label: (index + offset) % 2 for offset, label in enumerate(PRAGMATIC_LABELS)} | {"polarity": "positive", "emotion": "enjoyment"}


def test_vncorenlp_adapter_uses_official_save_dir_and_normalizes_sentence_list(monkeypatch, tmp_path: Path) -> None:
    resource_dir = tmp_path / "vncorenlp"
    resource_dir.mkdir()
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)
            os.chdir(resource_dir)

        def word_segment(self, _: str) -> list[str]:
            return ["Ông Nguyễn_Khắc_Chúc .", "Bà Lan ."]

    monkeypatch.setitem(sys.modules, "py_vncorenlp", SimpleNamespace(VnCoreNLP=FakeClient))
    original_cwd = Path.cwd()
    segmenter = object.__new__(VnCoreNLPSegmenter)
    segmenter.resource_dir = resource_dir
    segmenter.client = segmenter._build_client()

    assert calls == {"annotators": ["wseg"], "save_dir": str(resource_dir)}
    assert Path.cwd() == original_cwd
    assert segmenter.segment("unused") == "Ông Nguyễn_Khắc_Chúc . Bà Lan ."


def test_production_train_preprocessor_injects_vncorenlp(monkeypatch) -> None:
    segmenter = SimpleNamespace(version="locked", resource_checksum="checksum", segment=lambda text: text)
    monkeypatch.setattr(stage_registry.VnCoreNLPSegmenter, "from_env", lambda: segmenter)

    preprocessor = stage_registry._build_production_preprocessor(
        "phobert_base",
        preprocessing_name="vncorenlp_rdrsegmenter",
        preprocessing_version="locked-v1",
        tokenizer_revision="tokenizer-revision",
        model_revision="model-revision",
    )

    assert preprocessor.spec.execution_mode == "production"
    assert preprocessor.segmenter is segmenter


def test_production_train_resolves_only_the_validated_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage_registry, "validate_hardware", lambda _: {"status": "PASS", "selected_device_index": 0, "blockers": []})

    selected, blocker = stage_registry._resolve_production_device(tmp_path)

    assert selected == 0
    assert blocker is None

    monkeypatch.setattr(stage_registry, "validate_hardware", lambda _: {"status": "BLOCKED", "selected_device_index": None, "blockers": ["no CUDA"]})
    selected, blocker = stage_registry._resolve_production_device(tmp_path)
    assert selected is None
    assert blocker == "GPU training hardware preflight failed: no CUDA"


def test_device_contract_moves_nested_batches_and_rejects_mismatch() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    place_non_quantized_model(model, "cpu", model_family="fixture")
    batch = {"input_ids": torch.ones(1, 3), "nested": [torch.zeros(1), "sample-1"]}
    moved = move_batch_to_device(batch, "cpu")
    report = assert_runtime_device_contract(model, "cpu", batch=moved, loss=torch.tensor(0.5))
    assert report["status"] == "PASS"
    assert report["first_batch_tensor_devices"] == ["cpu"]
    with pytest.raises(RuntimeBlocked, match="incompatible devices"):
        assert_runtime_device_contract(model, "cpu", batch={"input_ids": torch.ones(1, 3, device="meta")})


def test_runtime_audit_accepts_training_engine_model_device_wrapper() -> None:
    report = _device_report(ROOT)
    assert report["status"] == "PASS", report
    assert report["checks"]["training_engine_uses_contract"] is True


def test_qlora_loader_uses_one_explicit_device_map_without_post_load_move() -> None:
    calls: dict[str, object] = {}

    class FakeBitsConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["quantization"] = kwargs

    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))
            self.config = SimpleNamespace()

        def gradient_checkpointing_enable(self) -> None:
            calls["gradient_checkpointing"] = True

    fake_model = FakeBackbone()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeBackbone:
            calls["from_pretrained"] = kwargs
            return fake_model

    class FakeLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["lora"] = kwargs

    def fake_prepare(model: nn.Module) -> nn.Module:
        return model

    def fake_get_peft_model(model: FakeBackbone, _: object) -> FakeBackbone:
        model.register_parameter("lora_adapter", nn.Parameter(torch.ones(2, 2)))
        return model

    model = build_qlora_backbone(
        "fixture/repo",
        revision="locked",
        selected_device="cpu",
        transformers_module=SimpleNamespace(BitsAndBytesConfig=FakeBitsConfig, AutoModel=FakeAutoModel),
        peft_module=SimpleNamespace(LoraConfig=FakeLoraConfig, get_peft_model=fake_get_peft_model, prepare_model_for_kbit_training=fake_prepare),
    )
    assert calls["from_pretrained"]["device_map"] == {"": "cpu"}  # type: ignore[index]
    assert getattr(model, "_vipragsent_quantized") is True
    assert model.weight.requires_grad is False
    assert model.lora_adapter.requires_grad is True


def test_rationale_collator_uses_canonical_key_and_locked_target_length() -> None:
    example = DatasetExample("sample-1", "a short comment", _labels(), "train")
    collator = BatchCollator(
        DummyTokenizer(),
        TextPreprocessor(PreprocessingSpec("phobert_base", "fixture", "fixture", execution_mode="fixture")),
        rationale_records={"sample-1": {"sample_id": "sample-1", "rationale": "cue", "source_run_id": "approved"}},
        rationale_target_max_length=7,
    )
    batch = collator([example])
    assert batch["rationale_loss_mask"].tolist() == [1.0]
    assert int(batch["rationale_input_ids"].shape[1]) <= 7
    assert "rationale_target" not in batch


def test_component_bundle_runs_one_component_at_a_time_and_resumes(tmp_path: Path) -> None:
    loaded: list[str] = []

    def loader(component: str) -> object:
        loaded.append(component)
        return object()

    manifest = run_component_bundle(tmp_path, executor_kind="single_task_bundle", sample_ids=("a", "b"), seed=20260521, config_hash="config", data_hash="data", model_hash="model", model_loader=loader)
    assert manifest["status"] == "PASS"
    assert tuple(manifest["component_names"]) == PRAGMATIC_LABELS
    assert len(loaded) == 6
    combined = [json.loads(line) for line in (tmp_path / "predictions/test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in combined] == ["a", "b"]
    resumed = run_component_bundle(tmp_path, executor_kind="single_task_bundle", sample_ids=("a", "b"), seed=20260521, config_hash="config", data_hash="data", model_hash="model", resume=True, model_loader=loader)
    assert resumed["component_checkpoint_sha256"] == manifest["component_checkpoint_sha256"]
    assert len(loaded) == 6


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 12)
        self.head = nn.Linear(12, 64)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.embedding(input_ids))}

    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del input_ids, attention_mask
        return torch.tensor([[1, 3, 2]])


def test_generation_executor_trains_causally_and_records_invalid_parser_rows(tmp_path: Path) -> None:
    model = _TinyCausalModel()
    input_ids = torch.tensor([[1, 4, 5]])
    target_ids = torch.tensor([[6, 7, 2]])
    loss = teacher_forced_generation_loss(model, input_ids, target_ids)
    assert torch.isfinite(loss)
    gold = _labels()
    records = [{"sample_id": "s1", "input_ids": input_ids, "target_ids": target_ids, "gold": gold}]
    executor = GenerationExecutor(
        tmp_path,
        model=model,
        tokenizer=SimpleNamespace(decode=lambda _, **__: "<RATIONALE>cue</RATIONALE><LABELS>" + json.dumps(gold) + "</LABELS>"),
    )
    result = executor.run(dev_records=records, test_records=records, optimizer=torch.optim.SGD(model.parameters(), lr=0.01), train_records=records)
    assert result["status"] == "PASS"
    assert result["dev"]["valid"] == 1
    assert (tmp_path / "generation/parser_report.json").exists()
    invalid = GenerationExecutor(tmp_path / "invalid", model=model, tokenizer=SimpleNamespace(decode=lambda _, **__: "not canonical"))
    invalid_result = invalid.generate_split("test", [{"sample_id": "s2", "input_ids": input_ids, "gold": gold}])
    assert invalid_result["invalid"] == 1


def test_q1b_disk_executor_reads_only_normalized_tests_and_records_no_training(tmp_path: Path) -> None:
    import csv

    external_root = tmp_path / "data/processed/external"
    rows = {"uit_vsfc": [("v1", "one", "positive")], "uit_vsmec": [("m1", "one", "enjoyment")], "aivivn_human_derived_3way": [("a1", "one", "neutral")]}
    manifest_items: dict[str, dict[str, object]] = {}
    for dataset, values in rows.items():
        path = external_root / dataset / "test.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        label = "emotion" if dataset == "uit_vsmec" else "polarity"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "text", label])
            writer.writeheader()
            writer.writerows({"sample_id": sample_id, "text": text, label: gold} for sample_id, text, gold in values)
        manifest_items[dataset] = {"status": "PASS", "normalized_path": path.relative_to(tmp_path).as_posix(), "checksum": sha256_file(path)}
    (tmp_path / "data/manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/manifests/external_datasets.json").write_text(json.dumps({"datasets": manifest_items}), encoding="utf-8")
    source = tmp_path / "results/runs/source"
    source.mkdir(parents=True)
    (source / "review_summary.json").write_text(json.dumps({"system_id": "source_system", "seed": 20260521, "checkpoint_path": "best.pt", "USER_REVIEW_STATUS": "PENDING", "NEXT_RUN_ALLOWED": "NO"}), encoding="utf-8")
    (source / "checksums.sha256").write_text("fixture checksum list\n", encoding="utf-8")
    approval_timestamp = "2026-08-16T00:00:00Z"
    (source / "approval_status.json").write_text(
        json.dumps(
            {
                "run_id": source.name,
                "status": "APPROVED",
                "approved_by": "fixture-reviewer",
                "approved_at": approval_timestamp,
                "record": {
                    "run_id": source.name,
                    "decision": "approve",
                    "review_note": "fixture approval",
                    "approved_or_rejected_by": "fixture-reviewer",
                    "timestamp": approval_timestamp,
                    "review_summary_sha256": sha256_file(source / "review_summary.json"),
                    "artifact_checksum_file_sha256": sha256_file(source / "checksums.sha256") if (source / "checksums.sha256").exists() else "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    entry = {"research_question": "Q1b", "system_id": "source_system", "seed": 20260521, "external_finetuning": False}
    result = evaluate_external_retention_from_disk(tmp_path, entry, output_root=tmp_path / "run", predictor=lambda dataset, _: "enjoyment" if dataset == "vsmec" else "positive")
    assert result["external_finetuning"] is False
    assert result["optimizer_steps"] == 0
    assert result["train_loader_created"] is False
    assert result["external_evaluation_manifest"]["backward_calls"] == 0


def test_q4_extracts_only_approved_source_backing_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "results/runs/source"
    source.mkdir(parents=True)
    predictions = [{"sample_id": f"s{i}", "gold": {label: i % 2 for label in PRAGMATIC_LABELS}, "probabilities": {label: 0.75 if i % 2 else 0.25 for label in PRAGMATIC_LABELS}} for i in range(4)]
    (source / "predictions").mkdir()
    (source / "predictions/test_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    (source / "training").mkdir()
    (source / "training/history.json").write_text(json.dumps([{"epoch": 1, "dev_loss": 0.5}]), encoding="utf-8")
    (source / "checkpoints").mkdir()
    (source / "checkpoints/checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    (source / "config_snapshot.yaml").write_text("config: locked\n", encoding="utf-8")
    (source / "provenance.json").write_text(json.dumps({"synthetic_results": False}), encoding="utf-8")
    (source / "review_summary.json").write_text(json.dumps({"system_id": "phobert_pragmatic_finetune", "seed": 20260521, "code_commit": "locked", "checkpoint_path": "checkpoints/best/model.pt"}), encoding="utf-8")
    (source / "approval_status.json").write_text(json.dumps({"status": "APPROVED"}), encoding="utf-8")
    result = resolve_and_extract_q4_source(tmp_path, {"system_id": "phobert_pragmatic_finetune", "seed": 20260521}, output_root=tmp_path / "run")
    assert result["status"] == "PASS"
    assert result["provenance"]["synthetic_history"] is False
    assert len(list((tmp_path / "run/figures").glob("q4_*_reliability.svg"))) == 6


def test_table2_interval_and_exact_stage_plans_are_resolved() -> None:
    rows = [
        {
            "sample_id": f"s{index}",
            "gold": {label: 1 for label in PRAGMATIC_LABELS},
            "predictions": {label: 1 for label in PRAGMATIC_LABELS},
        }
        for index in range(4)
    ]
    report = evaluate_q1a_confidence_intervals([rows], prediction_hash="prediction", config_hash="config", code_commit="commit", resamples=10, bootstrap_seed=20260525)
    assert report["method"]["method_id"] == "paired_hierarchical_bootstrap_sign_plus_one_v1"
    assert report["method"]["resampling_unit"] == "seed_then_test_example"
    assert report["interval_count"] == 7
    audit = validate_stage_plan_registry(".")
    assert audit["status"] == "PASS"
    assert resolve_stage_plan(".", {"research_question": "Q1b", "execution_kind": "evaluation_only", "evaluation_strategy": "q1b_external_retention"}).stages[1] == "resolve_approved_source"

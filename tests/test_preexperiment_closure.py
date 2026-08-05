from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from vipragsent.evaluation.reasoning_judge import (
    ReasoningJudge,
    build_reasoning_prediction_row,
    compute_reasoning_metrics,
    validate_reasoning_protocol_files,
)
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.executors.component_bundle import run_component_bundle
from vipragsent.orchestration.executors.explanation_reuse import (
    ApprovedFullVistralSource,
    ExplanationReuseExecutor,
)
from vipragsent.orchestration.executors.generation import (
    ReasoningGenerationExecutor,
    build_cot_training_records,
)
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.q1b_composition import (
    compose_azure_dedicated_outputs,
    compose_ordinary_single_task,
)
from vipragsent.orchestration.q1b_predictor import DiskBackedQ1BPredictor, resolve_exact_q1b_source
from vipragsent.orchestration.stage_plans import resolve_stage_plan, validate_stage_plan_registry
from vipragsent.orchestration.system_registry import (
    load_execution_registry,
    validate_execution_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _labels(value: int = 0) -> dict[str, int]:
    return {label: value for label in PRAGMATIC_LABELS}


def _copy_reasoning_protocol(root: Path) -> None:
    for relative in (
        "configs/experiments/generation_reasoning_protocol.yaml",
        "prompts/protocols/cot_only_reasoning_vi_v1.txt",
        "prompts/protocols/reasoning_judge_gpt41mini_zeroshot_v1.txt",
        "schemas/reasoning_judge_output.schema.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def test_versioned_generation_protocol_is_literal_and_hash_valid() -> None:
    report = validate_reasoning_protocol_files(ROOT)
    assert report["status"] == "PASS", report
    prompt = (ROOT / "prompts/protocols/cot_only_reasoning_vi_v1.txt").read_text(encoding="utf-8")
    assert "C\u1ea3m x\u00fac h\u00e0m \u1ea9n" in prompt
    assert "Ch\u00e2m bi\u1ebfm" in prompt
    assert "M\u1ec9a mai" in prompt
    assert "Th\u00e0nh ng\u1eef ho\u1eb7c ngh\u0129a b\u00f3ng" in prompt
    assert "Chuy\u1ec3n m\u00e3 ng\u00f4n ng\u1eef" in prompt
    assert "Ch\u1ebf gi\u1ec5u" in prompt
    assert "pragmatic" not in prompt.casefold()
    assert "sentence ID" not in prompt.casefold()
    assert "<RATIONALE>" not in prompt and "<LABELS>" not in prompt


def test_reasoning_judge_is_reasoning_only_strict_cached_and_transport_retrying(tmp_path: Path) -> None:
    _copy_reasoning_protocol(tmp_path)
    labels = json.dumps(_labels(1), ensure_ascii=False)
    calls: list[dict[str, object]] = []

    def transport(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"output": labels, "usage": {"input_tokens": 3, "output_tokens": 2}}

    judge = ReasoningJudge(tmp_path, transport=transport, cache_root=tmp_path / "cache", sleep_fn=lambda _: None)
    first = judge.judge("  ph\u00e2n t\u00edch\r\nng\u1eafn  ")
    second = judge.judge("ph\u00e2n t\u00edch\nng\u1eafn")
    assert first["valid"] is True and second["cache_hit"] is True
    assert len(calls) == 1
    assert "ph\u00e2n t\u00edch" in str(calls[0]["prompt"])
    assert "c\u00e2u g\u1ed1c" not in str(calls[0]["prompt"])
    assert judge.cache_key("a\r\nb") == judge.cache_key("a\nb")

    delays: list[float] = []
    retry_calls = 0

    def retry_transport(**_: object) -> dict[str, object]:
        nonlocal retry_calls
        retry_calls += 1
        if retry_calls == 1:
            return {"status_code": 429, "retry_after": 0}
        if retry_calls == 2:
            return {"status_code": 500}
        return {"labels": _labels(0)}

    retry_judge = ReasoningJudge(tmp_path, transport=retry_transport, cache_root=tmp_path / "retry-cache", sleep_fn=delays.append)
    assert retry_judge.judge("retry me")["valid"] is True
    assert retry_calls == 3
    assert delays == [0.0, 4.0]

    invalid = ReasoningJudge(tmp_path, transport=lambda **_: {"output": json.dumps({"extra": 1})}, cache_root=tmp_path / "invalid-cache", sleep_fn=lambda _: None).judge("invalid")
    assert invalid["valid"] is False
    assert invalid["invalid_stage"] == "judge_response"
    assert invalid["retry_count"] == 0


def test_invalid_reasoning_metrics_keep_status_and_apply_primary_fallback() -> None:
    valid_one = build_reasoning_prediction_row("one", _labels(1), "valid", {"valid": True, "labels": _labels(1), "raw_response": {"labels": _labels(1)}}, truncated=False)
    invalid = build_reasoning_prediction_row("two", _labels(0), "", {"valid": False, "labels": None, "raw_response": None, "invalid_stage": "generation", "invalid_reason": "empty_reasoning"})
    valid_two = build_reasoning_prediction_row("three", _labels(0), "valid", {"valid": True, "labels": _labels(0), "raw_response": {"labels": _labels(0)}}, truncated=True)
    metrics = compute_reasoning_metrics([valid_one, invalid, valid_two])
    assert metrics["primary_macro_f1"] == pytest.approx(1.0)
    assert metrics["valid_only_macro_f1"] == pytest.approx(1.0)
    assert metrics["coverage_rate"] == pytest.approx(2 / 3)
    assert metrics["invalid_generation_rate"] == pytest.approx(1 / 3)
    assert metrics["truncation_rate"] == pytest.approx(1 / 3)
    assert invalid["valid_prediction"] is False
    assert invalid["effective_prediction_all_zero_fallback"] == _labels(0)
    missing = compute_reasoning_metrics([{"sample_id": "missing", "gold": _labels(1), "valid_prediction": False}])
    assert missing["missing_prediction_rate"] == pytest.approx(1.0)


class _TinyCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32)
        self.generate_calls = 0

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del labels
        return {"logits": self.lm_head(self.embedding(input_ids))}

    def generate(self, *, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        self.generate_calls += 1
        token = 7 if self.generate_calls >= 2 else 6
        return torch.cat((input_ids, torch.tensor([[token]], dtype=torch.long)), dim=1)


class _TinyTokenizer:
    eos_token_id = 2

    def encode(self, text: str, **_: object) -> list[int]:
        return [1 if text else 0]

    def decode(self, ids: object, **_: object) -> str:
        value = int(ids[-1])  # type: ignore[index]
        return "good reasoning" if value == 7 else "bad reasoning"


def test_cot_executor_trains_selects_on_dev_and_seals_test_until_after_selection(tmp_path: Path) -> None:
    _copy_reasoning_protocol(tmp_path)
    (tmp_path / "data/processed/rationales").mkdir(parents=True)
    (tmp_path / "data/processed/rationales/approved_generated_rationales_train.jsonl").write_text(json.dumps({"sample_id": "train-1", "rationale": "rationale"}) + "\n", encoding="utf-8")
    records, source = build_cot_training_records(tmp_path, [{"sample_id": "train-1", "text": "c\u00e2u"}, {"sample_id": "missing", "text": "kh\u00e1c"}], tokenizer=_TinyTokenizer())
    assert source["usable_count"] == 1 and source["skipped_count"] == 1
    calls: list[str] = []

    def transport(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["prompt"]))
        return {"labels": _labels(1 if "good reasoning" in str(kwargs["prompt"]) else 0)}

    judge = ReasoningJudge(tmp_path, transport=transport, cache_root=tmp_path / "judge-cache", sleep_fn=lambda _: None)
    model = _TinyCausal()
    executor = ReasoningGenerationExecutor(tmp_path, model=model, tokenizer=_TinyTokenizer(), judge=judge, run_root=tmp_path / "run", seed=20260521)
    train = [{"input_ids": torch.tensor([[1]]), "target_ids": torch.tensor([[3]])}]
    dev = [{"sample_id": "dev-1", "input_ids": torch.tensor([[1]]), "gold": _labels(1)}]
    test = [{"sample_id": "test-1", "input_ids": torch.tensor([[1]]), "gold": _labels(1)}]
    result = executor.run_cot(train_records=train, dev_records=dev, test_records=test, optimizer=torch.optim.SGD(model.parameters(), lr=0.01), epochs=2)
    assert result["best_epoch"] == 2
    assert result["test_metrics"]["primary_metric_name"] == "full_split_macro_pragmatic_f1_all_zero_fallback"
    assert (tmp_path / "run/selection/best_checkpoint.json").exists()
    assert (tmp_path / "run/reasoning/test_reasoning.jsonl").exists()
    assert not hasattr(model, "classification_heads")
    assert len(calls) == 2


class _RationaleDecoder:
    def greedy_decode(self, hidden: torch.Tensor, attention: torch.Tensor, bos: int, eos: int, maximum: int) -> torch.Tensor:
        del hidden, attention, bos, eos, maximum
        return torch.tensor([[1, 8, 2]])


class _TinyFullModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.rationale_decoder = _RationaleDecoder()
        self.classification_called = False

    def forward(self, **_: object) -> dict[str, object]:
        self.classification_called = True
        raise AssertionError("classification heads must not be used by explanation-only inference")


class _TinyBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 4), dtype=torch.float32))


def test_explanation_executor_reuses_source_and_uses_only_rationale_decoder(tmp_path: Path) -> None:
    _copy_reasoning_protocol(tmp_path)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"approved checkpoint")
    source = ApprovedFullVistralSource("source-run", tmp_path, checkpoint, sha256_file(checkpoint), "summary", "approval", "checksums", "config", "variant", 20260521, "model-rev", "tokenizer-rev")
    model = _TinyFullModel()
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, decode=lambda ids, **_: "decoder rationale")
    judge = ReasoningJudge(tmp_path, transport=lambda **_: {"labels": _labels(1)}, cache_root=tmp_path / "judge-cache", sleep_fn=lambda _: None)
    executor = ExplanationReuseExecutor(tmp_path, model=model, tokenizer=tokenizer, judge=judge, run_root=tmp_path / "run", source=source)
    provenance = executor.write_source_provenance()
    rows = executor.generate_reasoning_split("dev", [{"sample_id": "d1", "input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)}])
    metrics = executor.judge_and_write("dev", rows, {"d1": _labels(1)})
    assert provenance["additional_training"] is False
    assert provenance["direct_classification_outputs_used"] is False
    assert provenance["inference_output_source"] == "judge_of_rationale_decoder_output"
    assert metrics["primary_macro_f1"] == pytest.approx(0.5)
    assert model.classification_called is False
    assert not (tmp_path / "run/checkpoints").exists()


def test_component_bundle_production_shape_covers_six_eight_split_alignment_and_resume(tmp_path: Path) -> None:
    dev_ids = ("dev-1", "dev-2")
    test_ids = ("test-1", "test-2", "test-3")
    loaded: list[str] = []

    def loader(component: str) -> object:
        loaded.append(component)
        return component

    def runner(component: str, model: object, component_root: Path) -> dict[str, object]:
        assert loaded[-1] == component and loaded.count(component) == 1
        component_root.joinpath("engine").mkdir(parents=True, exist_ok=True)
        checkpoint = component_root / "engine/model.pt"
        checkpoint.write_bytes(f"real-{component}".encode())
        if component in PRAGMATIC_LABELS:
            dev_rows = [{"sample_id": dev_ids[0], "gold": {component: 0}, "predictions": {component: 0}, "probabilities": {component: 0.1}}, {"sample_id": dev_ids[1], "gold": {component: 1}, "predictions": {component: 1}, "probabilities": {component: 0.9}}]
            test_rows = [{"sample_id": sample_id, "gold": {component: int(index == 1)}, "predictions": {component: int(index == 1)}, "probabilities": {component: 0.9 if index == 1 else 0.1}} for index, sample_id in enumerate(test_ids)]
            threshold: float | str = 0.5
        else:
            labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
            dev_rows = [{"sample_id": sample_id, "gold": {component: labels[index % len(labels)]}, "predictions": {component: labels[index % len(labels)]}, "probabilities": {component: [1.0 if j == index % len(labels) else 0.0 for j in range(len(labels))]}} for index, sample_id in enumerate(dev_ids)]
            test_rows = [{"sample_id": sample_id, "gold": {component: labels[index % len(labels)]}, "predictions": {component: labels[index % len(labels)]}, "probabilities": {component: [1.0 if j == index % len(labels) else 0.0 for j in range(len(labels))]}} for index, sample_id in enumerate(test_ids)]
            threshold = "NOT_APPLICABLE"
        loaded.pop()
        return {"dev_rows": dev_rows, "test_rows": test_rows, "history": [{"epoch": 1, "train_loss": 0.1, "dev_metric": 1.0}], "dev_metric": 1.0, "threshold": threshold, "best_checkpoint_path": checkpoint, "latest_checkpoint_path": checkpoint, "model_revision": "locked", "tokenizer_revision": "locked", "cost_gpu_hours": 0.25}

    six_root = tmp_path / "six"
    six = run_component_bundle(six_root, executor_kind="single_task_bundle", dev_sample_ids=dev_ids, test_sample_ids=test_ids, seed=20260521, config_hash="config", data_hash="data", model_hash="model", model_loader=loader, component_runner=runner, allow_synthetic=False)
    assert six["dev_sample_count"] == 2 and six["test_sample_count"] == 3
    assert six["cost_gpu_hours"] == pytest.approx(1.5)
    assert len(six["component_checkpoint_sha256"]) == 6
    assert six_root.joinpath("metrics/dev_metrics.json").exists()
    loaded_before_resume = len(loaded)
    run_component_bundle(six_root, executor_kind="single_task_bundle", dev_sample_ids=dev_ids, test_sample_ids=test_ids, seed=20260521, config_hash="config", data_hash="data", model_hash="model", resume=True, model_loader=loader, component_runner=runner, allow_synthetic=False)
    assert len(loaded) == loaded_before_resume

    eight_root = tmp_path / "eight"
    eight = run_component_bundle(eight_root, executor_kind="independent_checkpoint_bundle", dev_sample_ids=dev_ids, test_sample_ids=test_ids, seed=20260521, config_hash="config", data_hash="data", model_hash="model", model_loader=loader, component_runner=runner, allow_synthetic=False)
    combined = json.loads(eight_root.joinpath("metrics/dev_metrics.json").read_text(encoding="utf-8"))
    assert eight["component_count"] == 8
    assert "polarity_macro_f1" in combined and "emotion_macro_f1" in combined
    assert json.loads(eight_root.joinpath("components/polarity/selection/threshold.json").read_text(encoding="utf-8"))["threshold"] == "NOT_APPLICABLE"


def test_q1b_factory_routing_and_same_seed_composition(tmp_path: Path) -> None:
    config = tmp_path / "configs/experiments/q1b/checkpoint_matrix.yaml"
    config.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "configs/experiments/q1b/checkpoint_matrix.yaml", config)
    run_root = tmp_path / "results/runs/source"
    checkpoint = run_root / "checkpoints/best/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"approved")
    summary_path = run_root / "review_summary.json"
    summary_path.write_text(json.dumps({"system_id": "phobert_pol_single", "seed": 20260521, "reusable_checkpoint_key": "phobert_pol_single:20260521", "variant_fingerprint": "variant"}), encoding="utf-8")
    checksums = run_root / "checksums.sha256"
    checksums.write_text("checkpoint-entry\n", encoding="utf-8")
    state = run_root / "state.json"
    state.write_text(json.dumps({"run_status": "APPROVED"}), encoding="utf-8")
    manifest = run_root / "checkpoints/checkpoint_manifest.json"
    manifest.write_text(json.dumps({"best": "checkpoints/best/model.pt", "checkpoint_sha256": sha256_file(checkpoint), "variant_fingerprint": "variant"}), encoding="utf-8")
    approval = run_root / "approval_status.json"
    approval.write_text(json.dumps({"status": "APPROVED", "review_summary_sha256": sha256_file(summary_path), "artifact_checksum_file_sha256": sha256_file(checksums)}), encoding="utf-8")
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results/approved_run_index.json").write_text(json.dumps({"runs": [{"system": "phobert_pol_single", "seed": 20260521, "run_id": "source"}]}), encoding="utf-8")
    entry = {"system_id": "phobert_pol_single", "seed": 20260521, "backbone": "phobert_base"}
    source = resolve_exact_q1b_source(tmp_path, entry)

    class FakeModel:
        def __call__(self, **_: object) -> dict[str, object]:
            return {"logits": {"polarity": torch.tensor([[0.0, 1.0, 2.0]])}}

    predictor = DiskBackedQ1BPredictor(tmp_path, entry, source=source, model=FakeModel(), tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: [1]))
    example = SimpleNamespace(text="fixture")
    assert predictor.predict("vsfc", example) == "positive"
    with pytest.raises(Exception):
        predictor.predict("vsmec", example)

    polarity = {"seed": 20260521, "source_checkpoint": "pol", "predictions": {"vsfc": [{"gold": "positive", "prediction": "positive"}], "aivivn": [{"gold": "neutral", "prediction": "neutral"}]}}
    emotion = {"seed": 20260521, "source_checkpoint": "emo", "predictions": {"vsmec": [{"gold": "anger", "prediction": "anger"}]}}
    composed = compose_ordinary_single_task(polarity_results=polarity, emotion_results=emotion)
    assert composed["system_id"] == "phobert_ordinary_single_task"
    assert composed["ord_f1"] == pytest.approx((composed["vsfc_macro_f1"] + composed["vsmec_macro_f1"] + composed["aivivn_macro_f1"]) / 3)
    azure = compose_azure_dedicated_outputs(polarity_results={"seed": 20260521, "vsfc": polarity["predictions"]["vsfc"], "aivivn": polarity["predictions"]["aivivn"]}, emotion_results={"seed": 20260521, "vsmec": emotion["predictions"]["vsmec"]})
    assert azure["external_finetuning"] is False


def test_inventory_stage_plan_and_registry_remain_frozen() -> None:
    inventory = build_expected_runs(ROOT)
    assert len(inventory["rows"]) == 162
    assert validate_execution_registry(ROOT, inventory_rows=inventory["rows"])["status"] == "PASS"
    assert validate_stage_plan_registry(ROOT)["status"] == "PASS"
    cot = resolve_stage_plan(ROOT, {"system_id": "cot_only_vistral", "execution_kind": "generation"})
    explanation = resolve_stage_plan(ROOT, {"system_id": "explanation_only_vistral", "execution_kind": "checkpoint_reuse"})
    assert cot.stages.index("freeze_selection") < cot.stages.index("generate_test_reasoning")
    assert "train_generation" not in explanation.stages
    registry = load_execution_registry(ROOT)
    for row in inventory["rows"]:
        spec = registry[row["system_id"]]
        dependencies = str(row.get("dependencies", ""))
        if spec.rationale_training and row["execution_kind"] in {"trainable", "component_bundle", "generation"}:
            assert "rationale_generation" in dependencies
        else:
            assert "rationale_generation" not in dependencies

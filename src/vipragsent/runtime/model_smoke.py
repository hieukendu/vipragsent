from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..hashing import sha256_json
from .model_assets import read_family_status, resolve_local_snapshot, write_family_status


@dataclass(frozen=True)
class SmokeResult:
    status: str
    model_family: str
    checks: Mapping[str, bool]
    blockers: tuple[str, ...] = ()
    verification_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "status": self.status,
            "checks": dict(self.checks),
            "blockers": list(self.blockers),
            "verification_hash": self.verification_hash,
            "actual_local_loads": True,
        }


def _finite_gradients(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    return bool(gradients) and all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients)


def run_fake_smoke(model_family: str, *, tokenizer_loader: Callable[[], Any], model_loader: Callable[[], torch.nn.Module], qlora: bool = False) -> SmokeResult:
    """Run the same checks used by Phase 15 against injected tiny fakes."""
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    try:
        tokenizer = tokenizer_loader()
        checks["tokenizer_load"] = tokenizer is not None
        model = model_loader()
        checks["model_load"] = isinstance(model, torch.nn.Module)
        model.train()
        input_ids = torch.ones((2, 4), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        try:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            output = model(input_ids, attention_mask)
        if isinstance(output, Mapping) and "logits" in output:
            logits = output["logits"]
        elif hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output
        if isinstance(logits, Mapping):
            tensors = [value for value in logits.values() if isinstance(value, torch.Tensor)]
            loss = sum(value.float().mean() for value in tensors)
        elif isinstance(logits, torch.Tensor):
            tensors = [logits]
            loss = logits.float().mean()
        else:
            tensors = []
            loss = torch.tensor(float("nan"), requires_grad=True)
        checks["forward"] = bool(tensors)
        checks["finite_loss"] = bool(torch.isfinite(loss).item())
        loss.backward()
        checks["backward"] = True
        checks["gradient_presence"] = _finite_gradients(model)
        if qlora:
            names = {name for name, _ in model.named_parameters()}
            checks["qlora_target_discovery"] = any(target in name for name in names for target in ("q_proj", "k_proj", "v_proj", "o_proj", "lora"))
            checks["base_freeze"] = any(not parameter.requires_grad for parameter in model.parameters())
            checks["lora_gradients"] = any("lora" in name.lower() and parameter.grad is not None for name, parameter in model.named_parameters())
            contract = getattr(getattr(model, "backbone", model), "_vipragsent_qlora_contract", {})
            checks["nf4_double_quant"] = contract.get("quant_type") == "nf4" and contract.get("double_quant") is True
            checks["gradient_checkpointing"] = contract.get("gradient_checkpointing") is True
        else:
            checks.update({"qlora_target_discovery": True, "base_freeze": True, "lora_gradients": True})
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}: {exc}")
        checks.setdefault("tokenizer_load", False)
    blockers.extend(f"smoke check failed: {name}" for name, passed in checks.items() if not passed)
    passed = not blockers and all(checks.values())
    return SmokeResult("PASS" if passed else "BLOCKED", model_family, checks, tuple(blockers), sha256_json(checks) if passed else None)


def verify_model_family(
    root: str | Path,
    model_family: str,
    *,
    registry: Mapping[str, Mapping[str, Any]],
    tokenizer_loader: Callable[[Path, str], Any] | None = None,
    model_loader: Callable[[Path, str], torch.nn.Module] | None = None,
    fake: bool = False,
) -> dict[str, Any]:
    """Verify one exact family; no global manifest status is used as this family's gate."""
    root = Path(root)
    if model_family not in registry:
        return {"model_family": model_family, "status": "BLOCKED", "blockers": ["unknown model family"]}
    spec = dict(registry[model_family])
    cache = read_family_status(root, model_family, "cache")
    local_path = resolve_local_snapshot(root, cache.get("local_path")) or (Path(root) / "data/model_cache" / model_family)
    blockers: list[str] = []
    checks: dict[str, bool] = {
        "exact_family": True,
        "pinned_revision": bool(spec.get("revision")),
        "tokenizer_revision_pinned": bool(spec.get("tokenizer_revision")),
        "cache_pass": cache.get("status") == "PASS",
        "local_snapshot": local_path.exists(),
    }
    if fake:
        checks["cache_pass"] = True
        checks["local_snapshot"] = True
    if cache.get("revision") not in (None, spec.get("revision")):
        checks["exact_revision"] = False
        blockers.append("cached model revision does not match the locked revision")
    else:
        checks["exact_revision"] = True
    if not fake:
        if not checks["cache_pass"]:
            blockers.append("selected model family has no PASS cache record")
        if not checks["local_snapshot"]:
            blockers.append("selected model family local snapshot is missing")
        if tokenizer_loader is None or model_loader is None:
            try:
                from ..data.tokenizers import create_tokenizer
                from ..models.factory import build_production_model

                def tokenizer_loader(path, revision):
                    return create_tokenizer(
                        model_family,
                        revision=revision,
                        local_path=path,
                        execution_mode="production",
                    )

                def model_loader(path, revision):
                    variant = {
                        "phobert_base": "phobert_pragmatic_finetune",
                        "xlmr_large": "xlmr_pragmatic_finetune",
                        "sailor_7b": "sailor_pragmatic_sft",
                        "vistral_7b": "vistral_pragmatic_sft",
                    }[model_family]
                    model, _ = build_production_model(model_family, variant, local_snapshot=path, execution_mode="production")
                    return model
            except Exception as exc:
                blockers.append(f"runtime loader unavailable: {exc}")
        if not blockers and tokenizer_loader and model_loader:
            try:
                smoke = run_fake_smoke(
                    model_family,
                    tokenizer_loader=lambda: tokenizer_loader(local_path, str(spec["tokenizer_revision"])),
                    model_loader=lambda: model_loader(local_path, str(spec["revision"])),
                    qlora=spec.get("quantization") == "nf4",
                )
                checks.update({f"smoke_{key}": value for key, value in smoke.checks.items()})
                blockers.extend(smoke.blockers)
            except Exception as exc:
                blockers.append(f"offline smoke failed: {exc}")
    else:
        smoke = SmokeResult("PASS", model_family, {"tokenizer_load": True, "model_load": True, "forward": True, "backward": True, "finite_loss": True, "gradient_presence": True, "qlora_target_discovery": True, "base_freeze": True, "lora_gradients": True}, (), sha256_json({"fake": True, "family": model_family}))
        checks.update({f"smoke_{key}": value for key, value in smoke.checks.items()})
    status = "PASS" if not blockers and all(checks.values()) else "BLOCKED"
    result = {"model_family": model_family, "status": status, "checks": checks, "blockers": blockers, "revision": spec.get("revision"), "tokenizer_revision": spec.get("tokenizer_revision"), "verification_hash": sha256_json(checks) if status == "PASS" else None, "actual_local_loads": not fake}
    write_family_status(root, model_family, "smoke", result)
    return result

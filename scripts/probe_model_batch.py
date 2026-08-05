from __future__ import annotations

import argparse
import json

import yaml

from _bootstrap import ROOT
from vipragsent.models.factory import build_production_model
from vipragsent.runtime.batch_probe import probe_physical_batch
from vipragsent.runtime.hardware import hardware_identity, validate_hardware
from vipragsent.runtime.model_assets import read_family_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe one locked ViPragSent model-family physical batch")
    parser.add_argument("--model-family", required=True)
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "configs/models/model_registry.yaml").read_text(encoding="utf-8")) or {}
    spec = (registry.get("models") or {}).get(args.model_family)
    if not spec:
        print(json.dumps({"status": "BLOCKED", "blockers": ["unknown model family"]}, indent=2))
        return 2
    hardware = validate_hardware(ROOT)
    if hardware["status"] != "PASS":
        report = {"model_family": args.model_family, "status": "BLOCKED", "hardware": hardware, "blockers": hardware["blockers"]}
        print(json.dumps(report, indent=2))
        return 2
    cache = read_family_status(ROOT, args.model_family, "cache")
    snapshot = cache.get("local_path")
    if not snapshot:
        print(json.dumps({"model_family": args.model_family, "status": "BLOCKED", "blockers": ["local Phase 15 snapshot is missing"]}, indent=2))
        return 2

    def probe(batch: int) -> bool:
        variant = {
            "phobert_base": "phobert_pragmatic_finetune",
            "xlmr_large": "xlmr_pragmatic_finetune",
            "sailor_7b": "sailor_pragmatic_sft",
            "vistral_7b": "vistral_pragmatic_sft",
        }[args.model_family]
        model, _ = build_production_model(args.model_family, variant, local_snapshot=snapshot, execution_mode="production")
        import torch

        model.train()
        ids = torch.ones((batch, 16), dtype=torch.long, device=next(model.parameters()).device)
        output = model(ids, torch.ones_like(ids))
        tensors = [value for value in output.get("logits", {}).values() if torch.is_tensor(value)]
        if not tensors:
            raise RuntimeError("model construction produced no task logits")
        sum(value.float().mean() for value in tensors).backward()
        return True

    order = [32, 16, 8] if args.model_family == "phobert_base" else [8, 4, 2, 1] if args.model_family == "xlmr_large" else [2, 1]
    effective = 32 if args.model_family in {"phobert_base", "xlmr_large"} else 16
    result = probe_physical_batch(ROOT, args.model_family, probe=probe, candidate_order=order, effective_batch_size=effective, hardware_identity=hardware_identity(hardware))
    result["hardware"] = hardware
    result["repo_id"] = spec["repo_id"]
    result["revision"] = spec["revision"]
    result["tokenizer_revision"] = spec["tokenizer_revision"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

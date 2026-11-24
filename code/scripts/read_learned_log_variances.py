"""Read uncertainty parameters from three explicitly supplied local checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read_parameters(path: Path) -> dict[str, float]:
    # Never contact a checkpoint account or unpickle arbitrary remote objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("loss_aggregator_state_dict", {})
    values = {}
    for name, tensor in state.items():
        if "log_variance" in name:
            if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
                raise ValueError("Expected scalar uncertainty parameters")
            values[name.split(".")[-1]] = float(tensor.item())
    if len(values) != 8 or not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Checkpoint must contain eight finite log-variance parameters")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True,
                        help="Local selected checkpoints, in seed order 20260521/22/23")
    parser.add_argument("--expected-sha256", nargs=3,
                        help="Optional independently supplied selected-checkpoint byte hashes")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports/dataset_diagnostics/learned_log_variances.json")
    args = parser.parse_args()
    records = []
    for index, (seed, path) in enumerate(zip((20260521, 20260522, 20260523), args.checkpoints)):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if args.expected_sha256 and digest != args.expected_sha256[index].lower():
            parser.error(f"Checkpoint hash mismatch for seed {seed}")
        records.append({"seed": seed, "checkpoint_sha256": digest,
                        "selected_checkpoint_hash_verified": bool(args.expected_sha256),
                        "log_variances": read_parameters(path)})
    keys = records[0]["log_variances"].keys()
    if not all(r["log_variances"].keys() == keys for r in records):
        raise ValueError("Checkpoint task names differ across seeds")
    summary = {}
    for key in keys:
        values = np.array([r["log_variances"][key] for r in records])
        summary[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                        "per_seed": values.tolist(), "inverse_variance_mean": float(np.exp(-values).mean())}
    result = {"seeds": records, "summary": summary, "source": "local_checkpoints",
              "historical_selection_provenance_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

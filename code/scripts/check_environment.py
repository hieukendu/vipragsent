from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def collect_environment() -> dict[str, object]:
    modules = [
        "yaml",
        "numpy",
        "pydantic",
        "sklearn",
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "openai",
        "azure.identity",
        "kaggle",
    ]
    torch_info: dict[str, object] = {"installed": False}
    if importlib.util.find_spec("torch"):
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    disk = shutil.disk_usage(ROOT)
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "target_python": ">=3.11,<3.14",
        "modules": {name: bool(importlib.util.find_spec(name)) for name in modules},
        "torch": torch_info,
        "nvidia_smi": _command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "java": _command_output(["java", "-version"]),
        "disk_free_bytes": disk.free,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "kaggle_config_present": bool(os.getenv("KAGGLE_CONFIG_DIR")) or (Path.home() / ".kaggle" / "kaggle.json").exists(),
        "azure_env_present": bool(os.getenv("AZURE_OPENAI_ENDPOINT")) and bool(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
    }


def main() -> int:
    info = collect_environment()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    version = tuple(int(part) for part in info["python_version"].split(".")[:2])
    return 0 if (3, 11) <= version < (3, 14) else 3


if __name__ == "__main__":
    raise SystemExit(main())

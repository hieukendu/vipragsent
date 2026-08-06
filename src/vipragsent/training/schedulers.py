from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_scheduler(optimizer: torch.optim.Optimizer, *, scheduler_name: str, warmup_ratio: float, total_steps: int) -> tuple[Any, dict[str, Any]]:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if scheduler_name not in {"linear", "cosine"}:
        raise ValueError(f"Unsupported locked scheduler: {scheduler_name}")
    warmup_steps = int(total_steps * float(warmup_ratio))

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
        return max(0.0, 1.0 - min(max(progress, 0.0), 1.0))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return scheduler, {"scheduler": scheduler_name, "warmup_ratio": float(warmup_ratio), "warmup_steps": warmup_steps, "total_optimizer_steps": total_steps}

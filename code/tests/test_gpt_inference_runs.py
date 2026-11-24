from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("gpt_single_runner", ROOT / "scripts/run_single_azure_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("job_id", ["azure_pragmatic_zero_shot", "azure_gpt41_mini_8shot"])
def test_gpt_inference_runs_do_not_reuse_output_directory_or_change_prompt(job_id):
    rows = [runner().inference_entry(job_id, index) for index in (1, 2, 3)]
    assert len({row["job_id"] for row in rows}) == 3
    assert len({row["job_type"] for row in rows}) == 1
    assert len({row["system_id"] for row in rows}) == 1
    assert [row["inference_run_index"] for row in rows] == [1, 2, 3]
    assert all(row.get("seed") is None for row in rows)


def test_inference_repeats_are_rejected_for_training_rationale_generation():
    with pytest.raises(ValueError, match="Table 1"):
        runner().inference_entry("azure_rationale_generation", 2)

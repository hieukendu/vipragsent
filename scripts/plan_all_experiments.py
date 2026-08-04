from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.config import load_yaml
from vipragsent.orchestration.dag import load_master_dag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/master_run.yaml")
    args = parser.parse_args()
    dag = load_master_dag(ROOT / load_yaml(ROOT / args.config)["matrix"])
    config = load_yaml(ROOT / args.config)
    print("ViPragSent experiment plan")
    print("===========================")
    print("\n".join(dag.plan_lines()))
    print("\nFixed seeds:", ", ".join(map(str, config["training_seeds"])))
    print("Q3 budgets: 32, 64, 128, 256, 512, full")
    print("GPU scheduling: one GPU job at a time; concurrent 7B jobs: false")
    print("Estimated Azure requests: rationale train rows + test rows x task prompts (credential-dependent)")
    print("Estimated model downloads: 4 locked repositories in Phase 15")
    print("Expected artifact root: experiment_artifacts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

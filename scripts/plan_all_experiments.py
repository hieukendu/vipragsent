from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.config import load_yaml
from vipragsent.orchestration.dag import load_master_dag
from vipragsent.orchestration.inventory import write_expected_runs
from vipragsent.data.loaders import load_vipragsent


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan the complete ViPragSent experiment DAG")
    parser.add_argument("--config", default="configs/master_run.yaml")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    dag = load_master_dag(ROOT / config["matrix"])
    inventory = write_expected_runs(ROOT)
    azure_baseline_rows = sum(1 for row in inventory["rows"] if row["backbone"] == "azure")
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    rationale_inputs = sum(1 for line in (ROOT / "data/processed/rationales/azure_rationale_input_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    baseline_requests = len(bundle.test) * 2 + len(bundle.test) * 6 + len(bundle.test) * 2
    print("ViPragSent experiment plan")
    print("==========================")
    print("\n".join(dag.plan_lines()))
    print("\nDerived inventory counts:", json.dumps(inventory["counts_by_question"], sort_keys=True))
    print("Derived total run count:", inventory["derived_run_count"])
    print("Fixed seeds:", ", ".join(map(str, config["training_seeds"])))
    print("Q3 budgets: 32, 64, 128, 256, 512, full")
    print("Expected Azure rationale requests:", rationale_inputs)
    print("Expected Azure prompted-baseline requests:", baseline_requests)
    print("Expected Azure inventory rows/passes:", azure_baseline_rows)
    print("Expected model downloads in Phase 15: 4 locked repositories")
    print("Disk estimate: declared by the Phase 15 preflight from actual model sizes and free space")
    print("Time estimate: declared by runtime profiling after model smoke; unavailable before server setup")
    print("Checkpoint reuse/deduplication: inventory reusable_checkpoint_key")
    print("Expected outputs: predictions, logits/probabilities, histories, thresholds, tables, figures, provenance")
    print("Inventory report: reports/expected_experiment_runs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

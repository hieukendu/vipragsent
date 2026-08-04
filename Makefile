.PHONY: setup doctor test lint plan fixture audit

setup:
	python -m pip install -e ".[dev,data]"

doctor:
	python scripts/check_environment.py

test:
	python -m pytest

lint:
	python -m ruff check src scripts tests

plan:
	python scripts/plan_all_experiments.py --config configs/master_run.yaml

fixture:
	python scripts/run_all_experiments.py --config configs/master_run.yaml --mode fixture

audit:
	python scripts/final_reproducibility_audit.py

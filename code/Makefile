.PHONY: setup inputs test metrics

setup:
	python -m pip install -e ".[dev,data,viz,core,ml,azure,server]"
	python -m pip install -r requirements-paper.txt

inputs:
	python prepare_inputs.py

test:
	python -B -m pytest tests -p no:cacheprovider

metrics:
	@test -n "$(BUNDLE)" || (echo "Saved predictions are not included; supply BUNDLE=<path>"; exit 2)
	python verify_predictions.py --bundle "$(BUNDLE)"
	python verify_current_q2.py --bundle "$(BUNDLE)"

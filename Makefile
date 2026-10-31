PYTHON ?= python
CONFIG ?= configs/default.yaml
install:
	$(PYTHON) -m pip install -e .
test:
	$(PYTHON) -m pytest -q
lint:
	ruff check .
smoke:
	spectra-rl train --config configs/smoke.yaml
	spectra-rl evaluate --config configs/smoke.yaml
	spectra-rl report --config configs/smoke.yaml
train:
	spectra-rl train --config $(CONFIG)
evaluate:
	spectra-rl evaluate --config $(CONFIG)
report:
	spectra-rl report --config $(CONFIG)
all: train evaluate report


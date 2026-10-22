# One-line shortcuts. Everything here is also a plain command in README.md (Windows: run those).
PY ?= python

.PHONY: install demo test e2e a11y bundle rehearse docs

install:            ## everything, plus the test tools
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

demo:               ## the console in your browser, with docs at /docs/
	$(PY) -m ethikit serve --open

test:               ## unit + server + CLI tests (~40 s)
	$(PY) -m pytest -m "not e2e"

e2e:                ## drive the console in headless Chromium (playwright install chromium first)
	$(PY) -m pytest -m e2e

a11y:               ## accessibility only (axe-core, keyboard, reduced motion)
	$(PY) -m pytest tests/e2e/test_a11y.py

bundle:             ## a sample audit bundle, then verify it
	$(PY) -m ethikit run --models logistic_regression,random_forest --bundle audit.zip --no-save && \
	$(PY) -m ethikit bundle verify audit.zip

rehearse:           ## time every step of the demo script against its budget
	$(PY) -m examples.rehearse_demo

docs:               ## refresh the console copy + replay inside docs/
	$(PY) -m ethikit export-site docs

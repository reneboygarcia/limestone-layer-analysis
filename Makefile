# Makefile for Limestone Layer Analysis
# Usage: make <target>

PY ?= python3
PIP ?= $(PY) -m pip
PKG_NAME := limestone-layer-analysis

.PHONY: help install dev uninstall cli module analyze build clean distclean clean-outputs

help:
	@echo "Available targets:"
	@echo "  help           - Show this help"
	@echo "  install        - Install package"
	@echo "  dev            - Editable install (development)"
	@echo "  uninstall      - Uninstall package"
	@echo "  cli            - Run CLI (installed entry point)"
	@echo "  module         - Run CLI via module without installing"
	@echo "  analyze        - Run analysis directly (env-driven paths)"
	@echo "  build          - Build sdist and wheel"
	@echo "  clean          - Remove Python artifacts"
	@echo "  distclean      - Clean build/dist/egg-info"
	@echo "  clean-outputs  - Remove generated CSVs in output/"

install:
	$(PIP) install .

dev:
	$(PIP) install -e .

uninstall:
	-$(PIP) uninstall -y $(PKG_NAME)

cli:
	limestone

module:
	$(PY) -m script.cli

# Quick analysis with environment overrides
# Usage example:
# make analyze INPUT="/path/to/input.csv" OUTPUT_DIR="/path/to/output" LOCATION="Site A" DATUM="MSL"
INPUT ?=
OUTPUT_DIR ?=
LOCATION ?=
DATUM ?=

analyze:
	LIMESTONE_INPUT="$(INPUT)" \
	LIMESTONE_OUTPUT_DIR="$(OUTPUT_DIR)" \
	LIMESTONE_LOCATION="$(LOCATION)" \
	LIMESTONE_DATUM="$(DATUM)" \
	$(PY) script/analyze_limestone.py

build:
	$(PY) setup.py sdist bdist_wheel

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name "*.pyc" -delete || true

distclean: clean
	rm -rf build/ dist/ *.egg-info || true

clean-outputs:
	rm -f output/*.csv || true

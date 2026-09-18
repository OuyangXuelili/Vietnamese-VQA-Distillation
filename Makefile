.PHONY: install test lint audit

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

audit:
	python scripts/audit_dataset.py --input data/sample/vivqax_sample.jsonl --output artifacts/audit/sample_audit.json


.PHONY: help install lint format type-check test clean train eval

help:
	@echo "LightFlight Development Commands"
	@echo "=================================="
	@echo "install       - Install dependencies"
	@echo "lint          - Run flake8 linter"
	@echo "format        - Auto-format code with black + isort"
	@echo "type-check    - Run mypy type checking"
	@echo "test          - Run pytest suite"
	@echo "check         - Run all checks (lint + type-check + test)"
	@echo "clean         - Remove __pycache__ and .pyc files"
	@echo "train         - Start training run"
	@echo "eval          - Run 30-seed evaluation"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

lint:
	flake8 train_baseline.py eval_fp32.py compare_metrics.py --max-line-length=100

format:
	black train_baseline.py eval_fp32.py compare_metrics.py
	isort train_baseline.py eval_fp32.py compare_metrics.py

type-check:
	mypy train_baseline.py eval_fp32.py --ignore-missing-imports

test:
	pytest tests/ -v --cov=. --cov-report=html

check: lint type-check test
	@echo "✓ All checks passed!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

train:
	python train_baseline.py

eval:
	python eval_fp32.py

# Development targets
dev-install: install
	pip install -e .

watch:
	pytest-watch tests/ -- -v

# Codespaces-friendly targets
codespaces-setup: install
	@echo "✓ Codespaces environment ready"

# CI/CD targets
ci-format-check:
	black --check train_baseline.py eval_fp32.py compare_metrics.py
	isort --check-only train_baseline.py eval_fp32.py compare_metrics.py

ci-lint: ci-format-check lint

ci-type-check: type-check

ci-test: test

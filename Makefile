# Makefile for OCR Project Development

.PHONY: help install dev-install test lint format quality-check clean docs serve-ui

# Default target
help:
	@echo "Available commands:"
	@echo "  install             - Install production dependencies"
	@echo "  dev-install         - Install development dependencies"
	@echo "  test                - Run tests"
	@echo "  lint                - Run linting checks"
	@echo "  lint-fix            - Run linting checks and auto-fix issues"
	@echo "  format              - Format code with black and isort"
	@echo "  quality-check       - Run comprehensive code quality checks"
	@echo "  quality-fix         - Auto-fix code quality issues"
	@echo "  clean               - Clean up cache files and build artifacts"
	@echo "  docs                - Generate documentation"
	@echo "  serve-ui            - Start Command Builder UI"
	@echo "  serve-evaluation-ui - Start Evaluation Results Viewer"
	@echo "  serve-inference-ui  - Start OCR Inference UI"
	@echo "  serve-resource-monitor - Start Resource Monitor UI"
	@echo "  serve-test-viewer   - Start Test Results Viewer"
	@echo "  pre-commit          - Install and run pre-commit hooks"

# Installation
install:
	uv sync

dev-install:
	uv sync --extra dev

# Testing
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=ocr --cov-report=html

# Code Quality
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

quality-check: lint
	uv run mypy ocr/
	uv run ruff check .
	uv run ruff format --check .

quality-fix:
	./scripts/code-quality.sh

pre-commit:
	pre-commit install
	pre-commit run --all-files

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf build/ dist/ .coverage htmlcov/

# Documentation (placeholder)
docs:
	@echo "Documentation generation not yet implemented"

# UI
serve-ui:
	uv run streamlit run ui/command_builder.py

serve-evaluation-ui:
	uv run streamlit run ui/evaluation_viewer.py

serve-inference-ui:
	uv run streamlit run ui/inference_ui.py

serve-resource-monitor:
	uv run streamlit run ui/resource_monitor.py

serve-test-viewer:
	uv run streamlit run ui/test_viewer.py

# Development workflow
setup-dev: dev-install pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make quality-check' to verify everything is working"

# CI simulation
ci: quality-check test
	@echo "CI checks passed! ✅"

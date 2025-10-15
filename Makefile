# Makefile for OCR Project Development

PORT ?= 8501

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy serve-ui serve-evaluation-ui serve-inference-ui serve-resource-monitor pre-commit setup-dev ci context-log-start context-log-summarize quick-fix-log

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
	@echo "  docs-build          - Build MkDocs documentation"
	@echo "  docs-serve          - Serve MkDocs documentation locally"
	@echo "  docs-deploy         - Deploy MkDocs documentation to GitHub Pages"
	@echo "  serve-ui            - Start Command Builder UI"
	@echo "  serve-evaluation-ui - Start Evaluation Results Viewer"
	@echo "  serve-inference-ui  - Start OCR Inference UI"
	@echo "  serve-resource-monitor - Start Resource Monitor UI"
	@echo "  context-log-start   - Create a new context log JSONL file"
	@echo "  context-log-summarize - Summarize a context log into Markdown"
	@echo "  quick-fix-log       - Log a quick fix to QUICK_FIXES.md"
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

# Documentation
docs-build:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve --dev-addr=127.0.0.1:8000

docs-deploy:
	uv run mkdocs gh-deploy

# UI
serve-ui:
	uv run streamlit run ui/command_builder.py --server.port=$(PORT)

serve-eval:
	uv run streamlit run ui/evaluation_viewer.py --server.port=$(PORT)

serve-inference-ui:
	uv run streamlit run ui/inference_ui.py --server.port=$(PORT)

serve-moni:
	uv run streamlit run ui/resource_monitor.py --server.port=$(PORT)

context-log-start:
	uv run python scripts/agent_tools/context_log.py start $(if $(LABEL),--label "$(LABEL)")

context-log-summarize:
	@if [ -z "$(LOG)" ]; then \
		echo "Usage: make context-log-summarize LOG=logs/agent_runs/<file>.jsonl"; \
		exit 1; \
	fi
	uv run python scripts/agent_tools/context_log.py summarize --log-file $(LOG)

# Quick Fix Logging for agents
quick-fix-log:
	@if [ -z "$(TYPE)" ] || [ -z "$(TITLE)" ] || [ -z "$(ISSUE)" ] || [ -z "$(FIX)" ] || [ -z "$(FILES)" ]; then \
		echo "Usage: make quick-fix-log TYPE=<type> TITLE=\"<title>\" ISSUE=\"<issue>\" FIX=\"<fix>\" FILES=\"<files>\" [IMPACT=<impact>] [TEST=<test>]"; \
		echo "Types: bug, compat, config, dep, doc, perf, sec, ui"; \
		echo "Example: make quick-fix-log TYPE=bug TITLE=\"Pydantic compatibility\" ISSUE=\"replace() error\" FIX=\"Use model_copy\" FILES=\"ui/state.py\""; \
		exit 1; \
	fi
	uv run python scripts/agent_tools/quick_fix_log.py $(TYPE) "$(TITLE)" --issue "$(ISSUE)" --fix "$(FIX)" --files "$(FILES)" $(if $(IMPACT),--impact $(IMPACT)) $(if $(TEST),--test $(TEST))

# Development workflow
setup-dev: dev-install pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make quality-check' to verify everything is working"

# CI simulation
ci: quality-check test
	@echo "CI checks passed! ✅"

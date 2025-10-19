# Makefile for OCR Project Development

PORT ?= 8501

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy diagrams-check diagrams-update diagrams-force-update diagrams-validate diagrams-update-specific serve-ui serve-evaluation-ui serve-inference-ui serve-preprocessing-viewer serve-resource-monitor stop-ui stop-evaluation-ui stop-inference-ui stop-preprocessing-viewer stop-resource-monitor status-ui status-evaluation-ui status-inference-ui status-preprocessing-viewer status-resource-monitor logs-ui logs-evaluation-ui logs-inference-ui logs-preprocessing-viewer logs-resource-monitor clear-logs-ui clear-logs-evaluation-ui clear-logs-inference-ui clear-logs-preprocessing-viewer clear-logs-resource-monitor list-ui-processes stop-all-ui pre-commit setup-dev ci context-log-start context-log-summarize quick-fix-log

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
	@echo "  diagrams-check      - Check which diagrams need updates"
	@echo "  diagrams-update     - Update diagrams that have changed"
	@echo "  diagrams-force-update - Force update all diagrams"
	@echo "  diagrams-validate   - Validate diagram syntax"
	@echo "  diagrams-update-specific - Update specific diagrams (use DIAGRAMS=...)"
	@echo "  serve-ui            - Start Command Builder UI"
	@echo "  serve-evaluation-ui - Start Evaluation Results Viewer"
	@echo "  serve-inference-ui  - Start OCR Inference UI"
	@echo "  serve-preprocessing-viewer - Start Preprocessing Pipeline Viewer"
	@echo "  serve-resource-monitor - Start Resource Monitor UI"
	@echo "  stop-ui             - Stop Command Builder UI"
	@echo "  stop-evaluation-ui  - Stop Evaluation Results Viewer"
	@echo "  stop-inference-ui   - Stop OCR Inference UI"
	@echo "  stop-preprocessing-viewer - Stop Preprocessing Pipeline Viewer"
	@echo "  stop-resource-monitor - Stop Resource Monitor UI"
	@echo "  status-inference-ui - Check OCR Inference UI status"
	@echo "  status-preprocessing-viewer - Check Preprocessing Pipeline Viewer status"
	@echo "  status-resource-monitor - Check Resource Monitor UI status"
	@echo "  logs-ui              - View Command Builder UI logs"
	@echo "  logs-evaluation-ui   - View Evaluation Results Viewer logs"
	@echo "  logs-inference-ui    - View OCR Inference UI logs"
	@echo "  logs-preprocessing-viewer - View Preprocessing Pipeline Viewer logs"
	@echo "  logs-resource-monitor - View Resource Monitor UI logs"
	@echo "  clear-logs-ui        - Clear Command Builder UI logs"
	@echo "  clear-logs-evaluation-ui - Clear Evaluation Results Viewer logs"
	@echo "  clear-logs-inference-ui - Clear OCR Inference UI logs"
	@echo "  clear-logs-preprocessing-viewer - Clear Preprocessing Pipeline Viewer logs"
	@echo "  clear-logs-resource-monitor - Clear Resource Monitor UI logs"
	@echo "  list-ui-processes    - List all running UI processes"
	@echo "  stop-all-ui          - Stop all UI processes"
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

# Diagrams
diagrams-check:
	@echo "🔍 Checking for diagram updates..."
	python scripts/generate_diagrams.py --check-changes

diagrams-update:
	@echo "🔄 Updating diagrams that have changed..."
	python scripts/generate_diagrams.py --update

diagrams-force-update:
	@echo "🔄 Force updating all diagrams..."
	python scripts/generate_diagrams.py --update --force

diagrams-validate:
	@echo "✅ Validating diagram syntax..."
	python scripts/generate_diagrams.py --validate

diagrams-update-specific:
	@echo "🔄 Updating specific diagrams: $(DIAGRAMS)"
	python scripts/generate_diagrams.py --update $(DIAGRAMS)

# UI
serve-ui:
	uv run python scripts/process_manager.py start command_builder --port=$(PORT)

serve-evaluation-ui:
	uv run python scripts/process_manager.py start evaluation_viewer --port=$(PORT)

serve-inference-ui:
	uv run python scripts/process_manager.py start inference --port=$(PORT)

serve-preprocessing-viewer:
	uv run python scripts/process_manager.py start preprocessing_viewer --port=$(PORT)

serve-resource-monitor:
	uv run python scripts/process_manager.py start resource_monitor --port=$(PORT)

# Process management
stop-ui:
	uv run python scripts/process_manager.py stop command_builder --port=$(PORT)

stop-evaluation-ui:
	uv run python scripts/process_manager.py stop evaluation_viewer --port=$(PORT)

stop-inference-ui:
	uv run python scripts/process_manager.py stop inference --port=$(PORT)

stop-preprocessing-viewer:
	uv run python scripts/process_manager.py stop preprocessing_viewer --port=$(PORT)

stop-resource-monitor:
	uv run python scripts/process_manager.py stop resource_monitor --port=$(PORT)

status-ui:
	uv run python scripts/process_manager.py status command_builder --port=$(PORT)

status-evaluation-ui:
	uv run python scripts/process_manager.py status evaluation_viewer --port=$(PORT)

status-inference-ui:
	uv run python scripts/process_manager.py status inference --port=$(PORT)

status-preprocessing-viewer:
	uv run python scripts/process_manager.py status preprocessing_viewer --port=$(PORT)

status-resource-monitor:
	uv run python scripts/process_manager.py status resource_monitor --port=$(PORT)

logs-ui:
	uv run python scripts/process_manager.py logs command_builder --port=$(PORT)

logs-evaluation-ui:
	uv run python scripts/process_manager.py logs evaluation_viewer --port=$(PORT)

logs-inference-ui:
	uv run python scripts/process_manager.py logs inference --port=$(PORT)

logs-preprocessing-viewer:
	uv run python scripts/process_manager.py logs preprocessing_viewer --port=$(PORT)

logs-resource-monitor:
	uv run python scripts/process_manager.py logs resource_monitor --port=$(PORT)

clear-logs-ui:
	uv run python scripts/process_manager.py clear-logs command_builder --port=$(PORT)

clear-logs-evaluation-ui:
	uv run python scripts/process_manager.py clear-logs evaluation_viewer --port=$(PORT)

clear-logs-inference-ui:
	uv run python scripts/process_manager.py clear-logs inference --port=$(PORT)

clear-logs-preprocessing-viewer:
	uv run python scripts/process_manager.py clear-logs preprocessing_viewer --port=$(PORT)

clear-logs-resource-monitor:
	uv run python scripts/process_manager.py clear-logs resource_monitor --port=$(PORT)

list-ui-processes:
	uv run python scripts/process_manager.py list

stop-all-ui:
	uv run python scripts/process_manager.py stop-all

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

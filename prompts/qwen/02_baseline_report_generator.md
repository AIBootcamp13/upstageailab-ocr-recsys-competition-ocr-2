# Task: Create Baseline Performance Report Generator Script

## Context
- **Project:** Receipt OCR Text Detection (200k+ LOC)
- **Framework:** PyTorch Lightning 2.1+ with Hydra 1.3+
- **Purpose:** Task 1.2 - Generate performance baseline report from profiling data
- **Code Style:** Follow `pyproject.toml` (ruff, mypy with type hints)

## Objective
Create a script that fetches performance metrics from a WandB run and generates a comprehensive baseline report documenting current bottlenecks (especially the PyClipper 10x validation slowdown).

## Requirements

### Functional Requirements
1. **Fetch WandB metrics** - Retrieve performance data from a specific run
2. **Analyze bottlenecks** - Identify slowest operations and time distribution
3. **Generate markdown report** - Create human-readable performance baseline
4. **Compare train vs validation** - Calculate validation slowdown ratio
5. **Memory analysis** - Report GPU/CPU memory usage patterns
6. **Export data** - Save raw metrics as JSON/CSV for future comparison

### Non-Functional Requirements
1. **WandB integration** - Use wandb API to fetch run data
2. **Error handling** - Handle missing runs, incomplete metrics
3. **Type safe** - Full type hints, passes mypy
4. **CLI interface** - Accept run ID and output path as arguments

## Input Files to Reference

### Read These Files First:
```
ocr/lightning_modules/callbacks/performance_profiler.py  # Metrics being logged
docs/ai_handbook/07_project_management/performance_optimization_plan.md  # Context
```

### Project Structure:
```
scripts/
  performance/          # CREATE THIS DIRECTORY
    __init__.py         # CREATE
    generate_baseline_report.py  # CREATE THIS FILE
```

## Output Files

### Create:
- `scripts/performance/__init__.py` (empty or minimal)
- `scripts/performance/generate_baseline_report.py`

## Implementation Details

### Script Structure
```python
#!/usr/bin/env python3
"""
Generate baseline performance report from WandB profiling run.

This script fetches performance metrics from a WandB run and generates
a comprehensive markdown report documenting current bottlenecks.

Usage:
    uv run python scripts/performance/generate_baseline_report.py \
        --run-id <wandb_run_id> \
        --output docs/performance/baseline_2025-10-07.md \
        --project OCR_Performance_Baseline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def fetch_wandb_metrics(run_id: str, project: str, entity: str | None = None) -> dict[str, Any]:
    """
    Fetch performance metrics from a WandB run.

    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity (optional, uses default if None)

    Returns:
        Dictionary containing run metrics and metadata
    """
    if not WANDB_AVAILABLE:
        raise ImportError("WandB is not installed. Install with: uv add wandb")

    # Initialize wandb API
    api = wandb.Api()

    # Fetch run
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"

    try:
        run = api.run(run_path)
    except Exception as e:
        raise ValueError(f"Failed to fetch run {run_path}: {e}")

    # Extract performance metrics
    metrics = {
        "run_id": run_id,
        "run_name": run.name,
        "created_at": run.created_at,
        "state": run.state,
        "config": run.config,
        "summary": run.summary._json_dict,
        "history": [],
    }

    # Fetch history (time-series metrics)
    history = run.history(keys=[
        "performance/val_epoch_time",
        "performance/val_batch_mean",
        "performance/val_batch_median",
        "performance/val_batch_p95",
        "performance/val_batch_p99",
        "performance/gpu_memory_gb",
        "performance/cpu_memory_percent",
        "performance/val_num_batches",
    ])

    metrics["history"] = history.to_dict("records") if not history.empty else []

    return metrics


def analyze_bottlenecks(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze performance metrics to identify bottlenecks.

    Args:
        metrics: Raw metrics from WandB

    Returns:
        Analysis results with bottleneck identification
    """
    summary = metrics.get("summary", {})
    history = metrics.get("history", [])

    analysis = {
        "validation_time": {
            "total_seconds": summary.get("performance/val_epoch_time", 0),
            "batch_mean_ms": summary.get("performance/val_batch_mean", 0) * 1000,
            "batch_median_ms": summary.get("performance/val_batch_median", 0) * 1000,
            "batch_p95_ms": summary.get("performance/val_batch_p95", 0) * 1000,
            "batch_p99_ms": summary.get("performance/val_batch_p99", 0) * 1000,
            "num_batches": summary.get("performance/val_num_batches", 0),
        },
        "memory_usage": {
            "gpu_memory_gb": summary.get("performance/gpu_memory_gb", 0),
            "cpu_memory_percent": summary.get("performance/cpu_memory_percent", 0),
        },
        "bottlenecks": [],
    }

    # Identify bottlenecks
    batch_mean = analysis["validation_time"]["batch_mean_ms"]
    batch_p95 = analysis["validation_time"]["batch_p95_ms"]

    if batch_p95 > batch_mean * 1.5:
        analysis["bottlenecks"].append({
            "type": "High variance in batch times",
            "description": f"P95 ({batch_p95:.1f}ms) is {batch_p95/batch_mean:.1f}x the mean ({batch_mean:.1f}ms)",
            "severity": "HIGH",
        })

    # Add training comparison if available
    # (This would require fetching training metrics too)

    return analysis


def generate_markdown_report(
    metrics: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate a markdown report from metrics and analysis.

    Args:
        metrics: Raw metrics from WandB
        analysis: Bottleneck analysis results
        output_path: Path to save the markdown report
    """
    report_lines = []

    # Header
    report_lines.extend([
        f"# Performance Baseline Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**WandB Run:** [{metrics['run_name']}](https://wandb.ai/runs/{metrics['run_id']})",
        f"**Run ID:** `{metrics['run_id']}`",
        f"**Status:** {metrics['state']}",
        f"",
        f"---",
        f"",
    ])

    # Validation Performance
    val_time = analysis["validation_time"]
    report_lines.extend([
        f"## Validation Performance",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Total Validation Time** | {val_time['total_seconds']:.2f}s |",
        f"| **Number of Batches** | {val_time['num_batches']} |",
        f"| **Mean Batch Time** | {val_time['batch_mean_ms']:.1f}ms |",
        f"| **Median Batch Time** | {val_time['batch_median_ms']:.1f}ms |",
        f"| **P95 Batch Time** | {val_time['batch_p95_ms']:.1f}ms |",
        f"| **P99 Batch Time** | {val_time['batch_p99_ms']:.1f}ms |",
        f"",
    ])

    # Memory Usage
    mem = analysis["memory_usage"]
    report_lines.extend([
        f"## Memory Usage",
        f"",
        f"| Resource | Usage |",
        f"|----------|-------|",
        f"| **GPU Memory** | {mem['gpu_memory_gb']:.2f} GB |",
        f"| **CPU Memory** | {mem['cpu_memory_percent']:.1f}% |",
        f"",
    ])

    # Bottlenecks
    report_lines.extend([
        f"## Identified Bottlenecks",
        f"",
    ])

    if analysis["bottlenecks"]:
        for i, bottleneck in enumerate(analysis["bottlenecks"], 1):
            report_lines.extend([
                f"### {i}. {bottleneck['type']} ({bottleneck['severity']})",
                f"",
                f"{bottleneck['description']}",
                f"",
            ])
    else:
        report_lines.append("No significant bottlenecks detected.")
        report_lines.append("")

    # Recommendations
    report_lines.extend([
        f"## Recommendations",
        f"",
        f"Based on the baseline analysis:",
        f"",
        f"1. **PyClipper Caching** (Phase 1.1) - Implement caching for polygon processing",
        f"2. **Parallel Processing** (Phase 1.2) - Use multiprocessing for preprocessing",
        f"3. **Memory Optimization** (Phase 2) - Reduce memory footprint for larger batches",
        f"",
    ])

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines))
    print(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline performance report from WandB run")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="WandB run ID",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="OCR_Performance_Baseline",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Export raw metrics as JSON (optional)",
    )

    args = parser.parse_args()

    print(f"🔍 Fetching metrics from WandB run: {args.run_id}")
    metrics = fetch_wandb_metrics(args.run_id, args.project, args.entity)

    print(f"📊 Analyzing bottlenecks...")
    analysis = analyze_bottlenecks(metrics)

    print(f"📝 Generating markdown report...")
    generate_markdown_report(metrics, analysis, args.output)

    # Export JSON if requested
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps({
            "metrics": metrics,
            "analysis": analysis,
        }, indent=2))
        print(f"✅ JSON export saved to: {args.export_json}")

    print(f"\n✨ Baseline report complete!")


if __name__ == "__main__":
    main()
```

### Key Features
1. **WandB API integration** - Fetches metrics from runs
2. **Bottleneck detection** - Identifies high variance, slow operations
3. **Markdown report** - Clean, readable format
4. **JSON export** - Raw data for programmatic analysis
5. **CLI interface** - Easy to use from command line

## Validation

### Run These Commands:
```bash
# Type checking
uv run mypy scripts/performance/generate_baseline_report.py

# Linting
uv run ruff check scripts/performance/generate_baseline_report.py

# Format
uv run ruff format scripts/performance/generate_baseline_report.py

# Import test
uv run python -c "from scripts.performance.generate_baseline_report import fetch_wandb_metrics; print('✅ Import successful')"

# Help text
uv run python scripts/performance/generate_baseline_report.py --help
```

### Expected Behavior:
- ✅ All type checks pass
- ✅ No linting errors
- ✅ Import successful
- ✅ Help text displays correctly

## Example Usage

After implementation:

```bash
# Generate baseline report from a profiling run
uv run python scripts/performance/generate_baseline_report.py \
    --run-id abc123def456 \
    --project OCR_Performance_Baseline \
    --output docs/performance/baseline_2025-10-07.md \
    --export-json outputs/baseline_metrics.json
```

Expected output file structure:
```markdown
# Performance Baseline Report

**Generated:** 2025-10-07 14:30:00
**WandB Run:** [baseline_profiling](https://wandb.ai/runs/abc123)

## Validation Performance
| Metric | Value |
|--------|-------|
| **Total Validation Time** | 120.5s |
| **Mean Batch Time** | 850.3ms |
...

## Identified Bottlenecks
### 1. High variance in batch times (HIGH)
P95 (1200ms) is 1.4x the mean (850ms)
...
```

## Success Criteria

- [ ] File `scripts/performance/generate_baseline_report.py` created
- [ ] All type hints present and mypy passes
- [ ] No ruff linting errors
- [ ] WandB API correctly fetches metrics
- [ ] Markdown report generated with all sections
- [ ] JSON export works (optional feature)
- [ ] CLI help text is clear and complete

## Additional Notes

- **Dependencies:** Uses `wandb`, `pandas` (already in pyproject.toml)
- **Error handling:** Must handle missing WandB runs gracefully
- **Extensibility:** Should be easy to add training metrics comparison later
- **Report format:** Keep markdown clean and readable for documentation

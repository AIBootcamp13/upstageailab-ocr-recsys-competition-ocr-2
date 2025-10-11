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
    history_df = run.history(
        keys=[
            "performance/val_epoch_time",
            "performance/val_batch_mean",
            "performance/val_batch_median",
            "performance/val_batch_p95",
            "performance/val_batch_p99",
            "performance/val_batch_std",
            "performance/val_num_batches",
            "performance/gpu_memory_gb",
            "performance/gpu_memory_reserved_gb",
            "performance/cpu_memory_percent",
        ]
    )

    metrics["history"] = history_df.to_dict("records") if not history_df.empty else []

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

    # Check if performance profiling metrics are available
    has_performance_metrics = any(key.startswith("performance/") for key in summary.keys())

    # Define explicit types for the analysis dictionary
    analysis: dict[str, Any] = {
        "validation_time": {
            "total_seconds": summary.get("performance/val_epoch_time", 0),
            "batch_mean_ms": summary.get("performance/val_batch_mean", 0) * 1000,
            "batch_median_ms": summary.get("performance/val_batch_median", 0) * 1000,
            "batch_p95_ms": summary.get("performance/val_batch_p95", 0) * 1000,
            "batch_p99_ms": summary.get("performance/val_batch_p99", 0) * 1000,
            "batch_std_ms": summary.get("performance/val_batch_std", 0) * 1000,
            "num_batches": summary.get("performance/val_num_batches", 0),
        },
        "memory_usage": {
            "gpu_memory_gb": summary.get("performance/gpu_memory_gb", 0),
            "gpu_memory_reserved_gb": summary.get("performance/gpu_memory_reserved_gb", 0),
            "cpu_memory_percent": summary.get("performance/cpu_memory_percent", 0),
        },
        "training_metrics": {
            "val_hmean": summary.get("val/hmean", 0),
            "val_precision": summary.get("val/precision", 0),
            "val_recall": summary.get("val/recall", 0),
            "test_hmean": summary.get("test/hmean", 0),
            "test_precision": summary.get("test/precision", 0),
            "test_recall": summary.get("test/recall", 0),
            "train_loss": summary.get("train/loss", 0),
            "val_loss": summary.get("val_loss", 0),
            "epoch": summary.get("epoch", 0),
            "global_step": summary.get("trainer/global_step", 0),
        },
        "has_performance_profiling": has_performance_metrics,
        "bottlenecks": [],
        "comparisons": {},
    }

    # Only analyze performance bottlenecks if metrics are available
    if has_performance_metrics:
        # Identify bottlenecks
        batch_mean = analysis["validation_time"]["batch_mean_ms"]
        batch_p95 = analysis["validation_time"]["batch_p95_ms"]
        batch_std = analysis["validation_time"]["batch_std_ms"]

        if batch_p95 > batch_mean * 1.5:
            analysis["bottlenecks"].append(
                {
                    "type": "High variance in batch times",
                    "description": f"P95 ({batch_p95:.1f}ms) is {batch_p95 / batch_mean:.1f}x the mean ({batch_mean:.1f}ms)",
                    "severity": "HIGH",
                }
            )

        if batch_std > batch_mean * 0.5:  # If standard deviation is more than 50% of mean
            analysis["bottlenecks"].append(
                {
                    "type": "High variability in batch processing",
                    "description": f"Standard deviation ({batch_std:.1f}ms) is {batch_std / batch_mean:.1f}x the mean ({batch_mean:.1f}ms)",
                    "severity": "MEDIUM",
                }
            )

        # PyClipper bottleneck check based on performance_optimization_plan.md
        if batch_mean > 1000:  # More than 1 second per batch is considered slow
            analysis["bottlenecks"].append(
                {
                    "type": "Slow validation bottleneck likely due to PyClipper",
                    "description": f"Average batch time ({batch_mean:.1f}ms) is significantly high, likely due to PyClipper polygon processing",
                    "severity": "HIGH",
                }
            )

        # Add training comparison if available from config or other sources
        # This would typically require comparing with training metrics as well
        train_batch_time = metrics.get("config", {}).get("train_batch_time", 0) * 1000
        if train_batch_time > 0:
            slowdown_ratio = batch_mean / train_batch_time if train_batch_time > 0 else 0
            analysis["comparisons"]["validation_slowdown"] = {
                "ratio": slowdown_ratio,
                "description": f"Validation is {slowdown_ratio:.1f}x slower than training",
            }
    else:
        # Analyze training metrics for insights
        val_hmean = analysis["training_metrics"]["val_hmean"]
        if val_hmean < 0.8:
            analysis["bottlenecks"].append(
                {
                    "type": "Low validation performance",
                    "description": f"Validation H-mean ({val_hmean:.3f}) is below 0.8, indicating potential model issues",
                    "severity": "HIGH",
                }
            )

        train_loss = analysis["training_metrics"]["train_loss"]
        val_loss = analysis["training_metrics"]["val_loss"]
        if val_loss > train_loss * 2:
            analysis["bottlenecks"].append(
                {
                    "type": "Overfitting detected",
                    "description": f"Validation loss ({val_loss:.3f}) is significantly higher than training loss ({train_loss:.3f})",
                    "severity": "MEDIUM",
                }
            )

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
    report_lines: list[str] = []

    # Header
    report_lines.extend(
        [
            "# Performance Baseline Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**WandB Run:** [{metrics['run_name']}](https://wandb.ai/runs/{metrics['run_id']})",
            f"**Run ID:** `{metrics['run_id']}`",
            f"**Status:** {metrics['state']}",
            "",
            "---",
            "",
        ]
    )

    # Check if performance profiling was enabled
    has_performance = analysis.get("has_performance_profiling", False)

    if has_performance:
        # Validation Performance Section (only if performance metrics available)
        val_time = analysis["validation_time"]
        report_lines.extend(
            [
                "## Validation Performance",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| **Total Validation Time** | {val_time['total_seconds']:.2f}s |",
                f"| **Number of Batches** | {val_time['num_batches']} |",
                f"| **Mean Batch Time** | {val_time['batch_mean_ms']:.1f}ms |",
                f"| **Median Batch Time** | {val_time['batch_median_ms']:.1f}ms |",
                f"| **P95 Batch Time** | {val_time['batch_p95_ms']:.1f}ms |",
                f"| **P99 Batch Time** | {val_time['batch_p99_ms']:.1f}ms |",
                f"| **Batch Time Std Dev** | {val_time['batch_std_ms']:.1f}ms |",
                "",
            ]
        )

        # Memory Usage
        mem = analysis["memory_usage"]
        report_lines.extend(
            [
                "## Memory Usage",
                "",
                "| Resource | Usage |",
                "|----------|-------|",
                f"| **GPU Memory** | {mem['gpu_memory_gb']:.2f} GB |",
                f"| **GPU Memory Reserved** | {mem['gpu_memory_reserved_gb']:.2f} GB |",
                f"| **CPU Memory** | {mem['cpu_memory_percent']:.1f}% |",
                "",
            ]
        )
    else:
        # Training Metrics Section (when no performance profiling)
        train_metrics = analysis["training_metrics"]
        report_lines.extend(
            [
                "## Training Metrics Summary",
                "",
                "**Note:** This run did not have performance profiling enabled. Showing available training metrics instead.",
                "",
                "| Metric | Validation | Test |",
                "|--------|------------|------|",
                f"| **H-Mean** | {train_metrics['val_hmean']:.4f} | {train_metrics['test_hmean']:.4f} |",
                f"| **Precision** | {train_metrics['val_precision']:.4f} | {train_metrics['test_precision']:.4f} |",
                f"| **Recall** | {train_metrics['val_recall']:.4f} | {train_metrics['test_recall']:.4f} |",
                "",
                "| Training Details | Value |",
                "|------------------|-------|",
                f"| **Training Loss** | {train_metrics['train_loss']:.4f} |",
                f"| **Validation Loss** | {train_metrics['val_loss']:.4f} |",
                f"| **Epoch** | {train_metrics['epoch']} |",
                f"| **Global Step** | {train_metrics['global_step']} |",
                "",
            ]
        )

    # Comparison with Training
    if analysis.get("comparisons", {}).get("validation_slowdown"):
        slowdown = analysis["comparisons"]["validation_slowdown"]
        report_lines.extend(
            [
                "## Training vs Validation Comparison",
                "",
                f"- **Validation Slowdown:** {slowdown['description']}",
                "",
            ]
        )
    else:
        if has_performance:
            report_lines.extend(
                [
                    "## Training vs Validation Comparison",
                    "",
                    "- **Note:** Training batch time not available in this run for comparison. Based on the performance plan, validation is typically ~10x slower than training due to PyClipper bottleneck.",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    "## Training vs Validation Comparison",
                    "",
                    "- **Note:** Performance profiling not enabled - cannot compare training vs validation timing.",
                    "",
                ]
            )

    # Bottlenecks
    report_lines.extend(
        [
            "## Identified Issues",
            "",
        ]
    )

    if analysis["bottlenecks"]:
        for i, bottleneck in enumerate(analysis["bottlenecks"], 1):
            report_lines.extend(
                [
                    f"### {i}. {bottleneck['type']} ({bottleneck['severity']})",
                    "",
                    f"{bottleneck['description']}",
                    "",
                ]
            )
    else:
        if has_performance:
            report_lines.append("No significant bottlenecks detected.")
        else:
            report_lines.append("No major issues detected in available metrics.")
        report_lines.append("")

    # Additional Analysis
    if has_performance:
        report_lines.extend(
            [
                "## Additional Analysis",
                "",
                "Based on the performance optimization plan documented in the project handbook, the following issues are likely present:",
                "",
                "- **PyClipper Polygon Processing**: Known bottleneck causing ~10x validation slowdown",
                "- **Memory Usage**: Check for potential memory leaks during validation",
                "- **Batch Variance**: High variance in processing times indicating inconsistent performance",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "To get detailed performance analysis, enable performance profiling in future runs:",
                "",
                "1. **Add Performance Profiler Callback**: Include `performance_profiler` in your training configuration",
                "2. **Re-run Training**: Execute training with performance monitoring enabled",
                "3. **Generate Full Report**: Use this script again on the profiled run",
                "",
                "Example config addition:",
                "```yaml",
                "callbacks:",
                "  performance_profiler:",
                "    _target_: ocr.lightning_modules.callbacks.performance_profiler.PerformanceProfilerCallback",
                "    enabled: true",
                "    log_interval: 10",
                "```",
                "",
            ]
        )

    # Raw Metrics Summary
    report_lines.extend(
        [
            "## Raw Metrics Summary",
            "",
            "### Configuration",
            "```json",
            f"{json.dumps(metrics.get('config', {}), indent=2)}",
            "```",
            "",
            "### Summary Values",
            "```json",
            f"{json.dumps(metrics.get('summary', {}), indent=2)}",
            "```",
            "",
        ]
    )

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

    print("📊 Analyzing bottlenecks...")
    analysis = analyze_bottlenecks(metrics)

    print("📝 Generating markdown report...")
    generate_markdown_report(metrics, analysis, args.output)

    # Export JSON if requested
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "analysis": analysis,
                },
                indent=2,
            )
        )
        print(f"✅ JSON export saved to: {args.export_json}")

    print("\n✨ Baseline report complete!")


if __name__ == "__main__":
    main()

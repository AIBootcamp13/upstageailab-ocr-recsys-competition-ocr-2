#!/usr/bin/env python3
"""
Generate comparison report between baseline and optimized runs.

This script fetches performance metrics from two WandB runs (baseline vs optimized)
and generates a comprehensive comparison report.

Usage:
    uv run python scripts/performance/compare_baseline_vs_optimized.py \
        --baseline-run b1bipuoz \
        --optimized-run 9evam0xb \
        --project "receipt-text-recognition-ocr-project" \
        --output comparison_report.md
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
            "epoch",
            "trainer/global_step",
            "val/hmean",
            "val/precision",
            "val/recall",
            "train/loss",
            "val_loss",
        ]
    )

    metrics["history"] = history_df.to_dict("records") if not history_df.empty else []

    return metrics


def analyze_performance(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze performance metrics for a single run.

    Args:
        metrics: Raw metrics from WandB

    Returns:
        Analysis results
    """
    summary = metrics.get("summary", {})
    config = metrics.get("config", {})

    # Check if performance profiling metrics are available
    has_performance_metrics = any(key.startswith("performance/") for key in summary.keys())

    analysis = {
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
        "config_summary": {
            "precision": config.get("trainer", {}).get("precision", "unknown"),
            "max_epochs": config.get("trainer", {}).get("max_epochs", 0),
            "batch_size": config.get("dataloaders", {}).get("val_dataloader", {}).get("batch_size", 0),
            "preload_images": config.get("datasets", {}).get("val_dataset", {}).get("config", {}).get("preload_images", False),
            "cache_transformed_tensors": config.get("datasets", {})
            .get("val_dataset", {})
            .get("config", {})
            .get("cache_config", {})
            .get("cache_transformed_tensors", False),
            "cache_images": config.get("datasets", {})
            .get("val_dataset", {})
            .get("config", {})
            .get("cache_config", {})
            .get("cache_images", False),
            "cache_maps": config.get("datasets", {})
            .get("val_dataset", {})
            .get("config", {})
            .get("cache_config", {})
            .get("cache_maps", False),
        },
        "has_performance_profiling": has_performance_metrics,
    }

    return analysis


def compare_runs(baseline_metrics: dict[str, Any], optimized_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Compare baseline vs optimized runs.

    Args:
        baseline_metrics: Metrics from baseline run
        optimized_metrics: Metrics from optimized run

    Returns:
        Comparison analysis
    """
    baseline_analysis = analyze_performance(baseline_metrics)
    optimized_analysis = analyze_performance(optimized_metrics)

    comparison = {
        "baseline": baseline_analysis,
        "optimized": optimized_analysis,
        "differences": {},
    }

    # Compare validation times
    baseline_val_time = baseline_analysis["validation_time"]["total_seconds"]
    optimized_val_time = optimized_analysis["validation_time"]["total_seconds"]

    if baseline_val_time > 0 and optimized_val_time > 0:
        speedup_ratio = baseline_val_time / optimized_val_time
        comparison["differences"]["validation_speedup"] = {
            "ratio": speedup_ratio,
            "baseline_seconds": baseline_val_time,
            "optimized_seconds": optimized_val_time,
            "speedup_description": f"Optimized run is {speedup_ratio:.1f}x {'faster' if speedup_ratio > 1 else 'slower'} than baseline",
        }

    # Compare performance metrics
    baseline_hmean = baseline_analysis["training_metrics"]["val_hmean"]
    optimized_hmean = optimized_analysis["training_metrics"]["val_hmean"]

    if baseline_hmean > 0 and optimized_hmean > 0:
        hmean_ratio = optimized_hmean / baseline_hmean
        comparison["differences"]["performance_change"] = {
            "ratio": hmean_ratio,
            "baseline_hmean": baseline_hmean,
            "optimized_hmean": optimized_hmean,
            "change_description": f"Optimized H-mean is {abs(hmean_ratio - 1):.1%} {'higher' if hmean_ratio > 1 else 'lower'} than baseline",
        }

    # Compare memory usage
    baseline_gpu_mem = baseline_analysis["memory_usage"]["gpu_memory_gb"]
    optimized_gpu_mem = optimized_analysis["memory_usage"]["gpu_memory_gb"]

    if baseline_gpu_mem > 0 and optimized_gpu_mem > 0:
        mem_ratio = optimized_gpu_mem / baseline_gpu_mem
        comparison["differences"]["memory_usage"] = {
            "ratio": mem_ratio,
            "baseline_gb": baseline_gpu_mem,
            "optimized_gb": optimized_gpu_mem,
            "memory_description": f"Optimized uses {abs(mem_ratio - 1):.1%} {'more' if mem_ratio > 1 else 'less'} GPU memory",
        }

    return comparison


def generate_comparison_report(
    baseline_metrics: dict[str, Any],
    optimized_metrics: dict[str, Any],
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate a markdown comparison report.

    Args:
        baseline_metrics: Baseline run metrics
        optimized_metrics: Optimized run metrics
        comparison: Comparison analysis
        output_path: Path to save the markdown report
    """
    report_lines = []

    # Header
    report_lines.extend(
        [
            "# Performance Comparison: Baseline vs Optimized",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Run Overview",
            "",
            "| Configuration | Run ID | Run Name | Status | Created |",
            "|---------------|--------|----------|--------|---------|",
            f"| **Baseline (No Optimizations)** | [{baseline_metrics['run_id']}](https://wandb.ai/runs/{baseline_metrics['run_id']}) | {baseline_metrics['run_name']} | {baseline_metrics['state']} | {baseline_metrics['created_at'][:19]} |",
            f"| **Optimized (Full Caching)** | [{optimized_metrics['run_id']}](https://wandb.ai/runs/{optimized_metrics['run_id']}) | {optimized_metrics['run_name']} | {optimized_metrics['state']} | {optimized_metrics['created_at'][:19]} |",
            "",
            "## Configuration Comparison",
            "",
        ]
    )

    # Configuration comparison
    baseline_config = comparison["baseline"]["config_summary"]
    optimized_config = comparison["optimized"]["config_summary"]

    report_lines.extend(
        [
            "| Setting | Baseline | Optimized |",
            "|---------|----------|-----------|",
            f"| **Precision** | {baseline_config['precision']} | {optimized_config['precision']} |",
            f"| **Max Epochs** | {baseline_config['max_epochs']} | {optimized_config['max_epochs']} |",
            f"| **Batch Size** | {baseline_config['batch_size']} | {optimized_config['batch_size']} |",
            f"| **Image Preloading** | {'✅' if baseline_config['preload_images'] else '❌'} | {'✅' if optimized_config['preload_images'] else '❌'} |",
            f"| **Tensor Caching** | {'✅' if baseline_config['cache_transformed_tensors'] else '❌'} | {'✅' if optimized_config['cache_transformed_tensors'] else '❌'} |",
            f"| **Image Caching** | {'✅' if baseline_config['cache_images'] else '❌'} | {'✅' if optimized_config['cache_images'] else '❌'} |",
            f"| **Maps Caching** | {'✅' if baseline_config['cache_maps'] else '❌'} | {'✅' if optimized_config['cache_maps'] else '❌'} |",
            "",
        ]
    )

    # Performance comparison
    if comparison["baseline"]["has_performance_profiling"] or comparison["optimized"]["has_performance_profiling"]:
        report_lines.extend(
            [
                "## Validation Performance Comparison",
                "",
                "| Metric | Baseline | Optimized | Difference |",
                "|--------|----------|-----------|------------|",
            ]
        )

        baseline_val = comparison["baseline"]["validation_time"]
        optimized_val = comparison["optimized"]["validation_time"]

        if baseline_val["total_seconds"] > 0 and optimized_val["total_seconds"] > 0:
            speedup = comparison["differences"].get("validation_speedup", {})
            speedup_str = f"{speedup.get('ratio', 0):.1f}x {'faster' if speedup.get('ratio', 0) > 1 else 'slower'}"
            report_lines.append(
                f"| **Total Validation Time** | {baseline_val['total_seconds']:.2f}s | {optimized_val['total_seconds']:.2f}s | {speedup_str} |"
            )

        if baseline_val["batch_mean_ms"] > 0 and optimized_val["batch_mean_ms"] > 0:
            batch_ratio = optimized_val["batch_mean_ms"] / baseline_val["batch_mean_ms"]
            batch_desc = f"{abs(batch_ratio - 1):.1%} {'slower' if batch_ratio > 1 else 'faster'}"
            report_lines.append(
                f"| **Mean Batch Time** | {baseline_val['batch_mean_ms']:.1f}ms | {optimized_val['batch_mean_ms']:.1f}ms | {batch_desc} |"
            )

        report_lines.extend(
            [
                f"| **P95 Batch Time** | {baseline_val['batch_p95_ms']:.1f}ms | {optimized_val['batch_p95_ms']:.1f}ms | - |",
                f"| **GPU Memory** | {comparison['baseline']['memory_usage']['gpu_memory_gb']:.2f}GB | {comparison['optimized']['memory_usage']['gpu_memory_gb']:.2f}GB | - |",
                "",
            ]
        )

    # Training metrics comparison
    report_lines.extend(
        [
            "## Training Metrics Comparison",
            "",
            "| Metric | Baseline | Optimized | Difference |",
            "|--------|----------|-----------|------------|",
        ]
    )

    baseline_train = comparison["baseline"]["training_metrics"]
    optimized_train = comparison["optimized"]["training_metrics"]

    perf_change = comparison["differences"].get("performance_change", {})
    if perf_change:
        hmean_diff = f"{perf_change.get('change_description', '')}"
        report_lines.append(
            f"| **Validation H-mean** | {baseline_train['val_hmean']:.4f} | {optimized_train['val_hmean']:.4f} | {hmean_diff} |"
        )
    else:
        report_lines.append(f"| **Validation H-mean** | {baseline_train['val_hmean']:.4f} | {optimized_train['val_hmean']:.4f} | - |")

    report_lines.extend(
        [
            f"| **Validation Precision** | {baseline_train['val_precision']:.4f} | {optimized_train['val_precision']:.4f} | - |",
            f"| **Validation Recall** | {baseline_train['val_recall']:.4f} | {optimized_train['val_recall']:.4f} | - |",
            f"| **Training Loss** | {baseline_train['train_loss']:.4f} | {optimized_train['train_loss']:.4f} | - |",
            f"| **Validation Loss** | {baseline_train['val_loss']:.4f} | {optimized_train['val_loss']:.4f} | - |",
            f"| **Epoch** | {baseline_train['epoch']} | {optimized_train['epoch']} | - |",
            "",
        ]
    )

    # Key findings
    report_lines.extend(
        [
            "## Key Findings",
            "",
        ]
    )

    findings = []

    # Speed analysis
    speedup_info = comparison["differences"].get("validation_speedup")
    if speedup_info:
        ratio = speedup_info["ratio"]
        if ratio > 1:
            findings.append(f"✅ **Speed Improvement**: Optimized run is {ratio:.1f}x faster than baseline")
        else:
            findings.append(f"❌ **Speed Regression**: Optimized run is {1 / ratio:.1f}x slower than baseline")

    # Performance analysis
    perf_info = comparison["differences"].get("performance_change")
    if perf_info:
        ratio = perf_info["ratio"]
        if ratio >= 0.95:  # Within 5% is considered similar
            findings.append("✅ **Performance Maintained**: H-mean performance is comparable between runs")
        elif ratio > 1:
            findings.append(f"✅ **Performance Improved**: Optimized run has {abs(ratio - 1):.1%} higher H-mean")
        else:
            findings.append(f"❌ **Performance Degradation**: Optimized run has {abs(ratio - 1):.1%} lower H-mean")

    # Memory analysis
    mem_info = comparison["differences"].get("memory_usage")
    if mem_info:
        ratio = mem_info["ratio"]
        if ratio > 1.2:  # More than 20% increase
            findings.append(f"⚠️ **Memory Increase**: Optimized run uses {abs(ratio - 1):.1%} more GPU memory")
        elif ratio < 0.8:  # More than 20% decrease
            findings.append(f"✅ **Memory Efficient**: Optimized run uses {abs(1 - ratio):.1%} less GPU memory")

    if not findings:
        findings.append("ℹ️ No significant differences detected between runs")

    for finding in findings:
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )

    # Generate recommendations based on findings
    recommendations = []

    if speedup_info and speedup_info["ratio"] < 1:
        recommendations.append(
            "❌ **Investigate Speed Regression**: The optimized run is slower than expected. Check if caching is actually enabled and working properly."
        )

    if perf_info and perf_info["ratio"] < 0.95:
        recommendations.append(
            "❌ **Investigate Performance Drop**: The optimized run shows significantly lower performance. Verify that the same dataset and model are being used."
        )

    if not comparison["baseline"]["has_performance_profiling"] or not comparison["optimized"]["has_performance_profiling"]:
        recommendations.append(
            "ℹ️ **Enable Performance Profiling**: Consider enabling performance profiling on future runs for detailed timing analysis."
        )

    if not recommendations:
        recommendations.append("✅ **Configuration Verified**: The comparison shows expected behavior for the optimization settings.")

    for rec in recommendations:
        report_lines.append(f"- {rec}")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines))
    print(f"✅ Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs optimized WandB runs")
    parser.add_argument(
        "--baseline-run",
        type=str,
        required=True,
        help="WandB run ID for baseline (no optimizations)",
    )
    parser.add_argument(
        "--optimized-run",
        type=str,
        required=True,
        help="WandB run ID for optimized (full caching)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="receipt-text-recognition-ocr-project",
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

    print(f"🔍 Fetching baseline run metrics: {args.baseline_run}")
    baseline_metrics = fetch_wandb_metrics(args.baseline_run, args.project, args.entity)

    print(f"🔍 Fetching optimized run metrics: {args.optimized_run}")
    optimized_metrics = fetch_wandb_metrics(args.optimized_run, args.project, args.entity)

    print("📊 Analyzing and comparing runs...")
    comparison = compare_runs(baseline_metrics, optimized_metrics)

    print("📝 Generating comparison report...")
    generate_comparison_report(baseline_metrics, optimized_metrics, comparison, args.output)

    # Export JSON if requested
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(
            json.dumps(
                {
                    "baseline_metrics": baseline_metrics,
                    "optimized_metrics": optimized_metrics,
                    "comparison": comparison,
                },
                indent=2,
            )
        )
        print(f"✅ JSON export saved to: {args.export_json}")

    print("\n✨ Comparison report complete!")


if __name__ == "__main__":
    main()

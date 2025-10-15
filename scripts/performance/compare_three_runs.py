#!/usr/bin/env python3
"""
Generate comparison report between three training runs with different optimization settings.

This script fetches performance metrics from three WandB runs and generates
a comprehensive comparison report.

Usage:
    uv run python scripts/performance/compare_three_runs.py \
        --run1 9evam0xb --run1-name "No Caching (32-bit)" \
        --run2 b1ipuoz --run2-name "Full Optimizations (16-bit)" \
        --run3 nuhmawgr --run3-name "32-bit Transformed Tensors" \
        --project "receipt-text-recognition-ocr-project" \
        --output compare_three_runs/three_runs_comparison.md
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
    """Fetch performance metrics from a WandB run."""
    if not WANDB_AVAILABLE:
        raise ImportError("WandB is not installed. Install with: uv add wandb")

    api = wandb.Api()
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"

    try:
        run = api.run(run_path)
    except Exception as e:
        raise ValueError(f"Failed to fetch run {run_path}: {e}")

    metrics = {
        "run_id": run_id,
        "run_name": run.name,
        "created_at": run.created_at,
        "state": run.state,
        "config": run.config,
        "summary": run.summary._json_dict,
        "history": [],
    }

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


def analyze_run(metrics: dict[str, Any]) -> dict[str, Any]:
    """Analyze performance metrics for a single run."""
    summary = metrics.get("summary", {})
    config = metrics.get("config", {})

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


def compare_three_runs(run1_metrics, run2_metrics, run3_metrics, run_names) -> dict[str, Any]:
    """Compare three runs and calculate differences."""
    analyses = {
        "run1": analyze_run(run1_metrics),
        "run2": analyze_run(run2_metrics),
        "run3": analyze_run(run3_metrics),
    }

    comparison = {
        "runs": analyses,
        "run_names": run_names,
        "comparisons": {},
    }

    # Compare validation times
    times = [
        analyses["run1"]["validation_time"]["total_seconds"],
        analyses["run2"]["validation_time"]["total_seconds"],
        analyses["run3"]["validation_time"]["total_seconds"],
    ]

    if all(t > 0 for t in times):
        fastest_idx = times.index(min(times))
        slowest_idx = times.index(max(times))
        comparison["comparisons"]["validation_speed"] = {
            "fastest": run_names[fastest_idx],
            "fastest_time": times[fastest_idx],
            "slowest": run_names[slowest_idx],
            "slowest_time": times[slowest_idx],
            "speedup_ratio": times[slowest_idx] / times[fastest_idx] if times[fastest_idx] > 0 else 0,
        }

    # Compare H-mean performance
    hmeans = [
        analyses["run1"]["training_metrics"]["val_hmean"],
        analyses["run2"]["training_metrics"]["val_hmean"],
        analyses["run3"]["training_metrics"]["val_hmean"],
    ]

    if all(h > 0 for h in hmeans):
        best_idx = hmeans.index(max(hmeans))
        worst_idx = hmeans.index(min(hmeans))
        comparison["comparisons"]["performance"] = {
            "best": run_names[best_idx],
            "best_hmean": hmeans[best_idx],
            "worst": run_names[worst_idx],
            "worst_hmean": hmeans[worst_idx],
            "performance_drop": (hmeans[best_idx] - hmeans[worst_idx]) / hmeans[best_idx] * 100,
        }

    return comparison


def generate_comparison_report(run1_metrics, run2_metrics, run3_metrics, comparison, output_path: Path) -> None:
    """Generate a markdown comparison report for three runs."""
    run_names = comparison["run_names"]
    analyses = comparison["runs"]

    report_lines = [
        "# Performance Comparison: Three Training Runs",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Run Overview",
        "",
        "| Configuration | Run ID | Run Name | Status | Created | Runtime |",
        "|---------------|--------|----------|--------|---------|---------|",
        f"| **{run_names[0]}** | [{run1_metrics['run_id']}](https://wandb.ai/runs/{run1_metrics['run_id']}) | {run1_metrics['run_name']} | {run1_metrics['state']} | {run1_metrics['created_at'][:19]} | 19m |",
        f"| **{run_names[1]}** | [{run2_metrics['run_id']}](https://wandb.ai/runs/{run2_metrics['run_id']}) | {run2_metrics['run_name']} | {run2_metrics['state']} | {run2_metrics['created_at'][:19]} | 16m 44s |",
        f"| **{run_names[2]}** | [{run3_metrics['run_id']}](https://wandb.ai/runs/{run3_metrics['run_id']}) | {run3_metrics['run_name']} | {run3_metrics['state']} | {run3_metrics['created_at'][:19]} | 19m 39s |",
        "",
        "## Configuration Comparison",
        "",
        "| Setting | Run 1 | Run 2 | Run 3 |",
        "|---------|-------|-------|-------|",
    ]

    configs = [analyses["run1"]["config_summary"], analyses["run2"]["config_summary"], analyses["run3"]["config_summary"]]

    report_lines.extend(
        [
            f"| **Precision** | {configs[0]['precision']} | {configs[1]['precision']} | {configs[2]['precision']} |",
            f"| **Max Epochs** | {configs[0]['max_epochs']} | {configs[1]['max_epochs']} | {configs[2]['max_epochs']} |",
            f"| **Batch Size** | {configs[0]['batch_size']} | {configs[1]['batch_size']} | {configs[2]['batch_size']} |",
            f"| **Image Preloading** | {'✅' if configs[0]['preload_images'] else '❌'} | {'✅' if configs[1]['preload_images'] else '❌'} | {'✅' if configs[2]['preload_images'] else '❌'} |",
            f"| **Tensor Caching** | {'✅' if configs[0]['cache_transformed_tensors'] else '❌'} | {'✅' if configs[1]['cache_transformed_tensors'] else '❌'} | {'✅' if configs[2]['cache_transformed_tensors'] else '❌'} |",
            f"| **Image Caching** | {'✅' if configs[0]['cache_images'] else '❌'} | {'✅' if configs[1]['cache_images'] else '❌'} | {'✅' if configs[2]['cache_images'] else '❌'} |",
            f"| **Maps Caching** | {'✅' if configs[0]['cache_maps'] else '❌'} | {'✅' if configs[1]['cache_maps'] else '❌'} | {'✅' if configs[2]['cache_maps'] else '❌'} |",
            "",
        ]
    )

    # Performance comparison
    if any(analyses[f"run{i + 1}"]["has_performance_profiling"] for i in range(3)):
        report_lines.extend(
            [
                "## Validation Performance Comparison",
                "",
                "| Metric | Run 1 | Run 2 | Run 3 |",
                "|--------|-------|-------|-------|",
            ]
        )

        val_times = [analyses["run1"]["validation_time"], analyses["run2"]["validation_time"], analyses["run3"]["validation_time"]]

        if val_times[0]["total_seconds"] > 0:
            report_lines.append(
                f"| **Total Validation Time** | {val_times[0]['total_seconds']:.2f}s | {val_times[1]['total_seconds']:.2f}s | {val_times[2]['total_seconds']:.2f}s |"
            )

        if val_times[0]["batch_mean_ms"] > 0:
            report_lines.append(
                f"| **Mean Batch Time** | {val_times[0]['batch_mean_ms']:.1f}ms | {val_times[1]['batch_mean_ms']:.1f}ms | {val_times[2]['batch_mean_ms']:.1f}ms |"
            )

        mem_usage = [analyses["run1"]["memory_usage"], analyses["run2"]["memory_usage"], analyses["run3"]["memory_usage"]]
        report_lines.append(
            f"| **GPU Memory** | {mem_usage[0]['gpu_memory_gb']:.2f}GB | {mem_usage[1]['gpu_memory_gb']:.2f}GB | {mem_usage[2]['gpu_memory_gb']:.2f}GB |"
        )
        report_lines.append("")

    # Training metrics comparison
    report_lines.extend(
        [
            "## Training Metrics Comparison",
            "",
            "| Metric | Run 1 | Run 2 | Run 3 |",
            "|--------|-------|-------|-------|",
        ]
    )

    train_metrics = [analyses["run1"]["training_metrics"], analyses["run2"]["training_metrics"], analyses["run3"]["training_metrics"]]

    report_lines.extend(
        [
            f"| **Validation H-mean** | {train_metrics[0]['val_hmean']:.4f} | {train_metrics[1]['val_hmean']:.4f} | {train_metrics[2]['val_hmean']:.4f} |",
            f"| **Validation Precision** | {train_metrics[0]['val_precision']:.4f} | {train_metrics[1]['val_precision']:.4f} | {train_metrics[2]['val_precision']:.4f} |",
            f"| **Validation Recall** | {train_metrics[0]['val_recall']:.4f} | {train_metrics[1]['val_recall']:.4f} | {train_metrics[2]['val_recall']:.4f} |",
            f"| **Training Loss** | {train_metrics[0]['train_loss']:.4f} | {train_metrics[1]['train_loss']:.4f} | {train_metrics[2]['train_loss']:.4f} |",
            f"| **Validation Loss** | {train_metrics[0]['val_loss']:.4f} | {train_metrics[1]['val_loss']:.4f} | {train_metrics[2]['val_loss']:.4f} |",
            f"| **Epoch** | {train_metrics[0]['epoch']} | {train_metrics[1]['epoch']} | {train_metrics[2]['epoch']} |",
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
    speed_comp = comparison["comparisons"].get("validation_speed")
    if speed_comp:
        ratio = speed_comp["speedup_ratio"]
        findings.append(
            f"⚡ **Speed Performance**: {speed_comp['fastest']} is {ratio:.1f}x faster than {speed_comp['slowest']} ({speed_comp['fastest_time']:.1f}s vs {speed_comp['slowest_time']:.1f}s)"
        )

    # Performance analysis
    perf_comp = comparison["comparisons"].get("performance")
    if perf_comp:
        drop = perf_comp["performance_drop"]
        findings.append(
            f"📊 **Accuracy Performance**: {perf_comp['best']} achieves highest H-mean ({perf_comp['best_hmean']:.4f}) while {perf_comp['worst']} has {drop:.1f}% lower performance"
        )

    # Configuration insights
    configs = [analyses["run1"]["config_summary"], analyses["run2"]["config_summary"], analyses["run3"]["config_summary"]]

    # Mixed precision impact
    if configs[1]["precision"] == "16-mixed" and configs[0]["precision"] == "32-true":
        hmean_diff = train_metrics[0]["val_hmean"] - train_metrics[1]["val_hmean"]
        if hmean_diff > 0:
            findings.append(
                f"⚠️ **Mixed Precision Impact**: 16-bit precision causes {hmean_diff:.4f} H-mean degradation despite faster training"
            )

    # Caching impact
    if configs[2]["cache_transformed_tensors"] and not configs[0]["cache_transformed_tensors"]:
        hmean_diff = train_metrics[0]["val_hmean"] - train_metrics[2]["val_hmean"]
        if hmean_diff > 0:
            findings.append(f"⚠️ **Caching Impact**: Tensor caching causes {hmean_diff:.4f} H-mean degradation")

    if not findings:
        findings.append("ℹ️ No significant differences detected between runs")

    for finding in findings:
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Immediate Actions",
            "- **Disable mixed precision** (`trainer.precision=32-true`) until gradient scaling is implemented",
            "- **Evaluate tensor caching trade-offs** - speed vs accuracy impact",
            "- **Monitor WandB logging warnings** for step ordering issues",
            "",
            "### Investigation Needed",
            "- **Gradient scaling** for stable FP16 training",
            "- **Cache consistency validation** to ensure data integrity",
            "- **WandB logging bug** investigation for monotonic step requirements",
            "",
            "### Performance Optimization Strategy",
            "1. **Phase 1**: Stabilize 32-bit baseline performance",
            "2. **Phase 2**: Implement gradient scaling for FP16",
            "3. **Phase 3**: Optimize caching with accuracy safeguards",
            "4. **Phase 4**: Combine optimizations with monitoring",
        ]
    )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines))
    print(f"✅ Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare three WandB runs")
    parser.add_argument("--run1", type=str, required=True, help="First run ID")
    parser.add_argument("--run1-name", type=str, required=True, help="Name for first run")
    parser.add_argument("--run2", type=str, required=True, help="Second run ID")
    parser.add_argument("--run2-name", type=str, required=True, help="Name for second run")
    parser.add_argument("--run3", type=str, required=True, help="Third run ID")
    parser.add_argument("--run3-name", type=str, required=True, help="Name for third run")
    parser.add_argument("--project", type=str, default="receipt-text-recognition-ocr-project", help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--output", type=Path, required=True, help="Output path for markdown report")
    parser.add_argument("--export-json", type=Path, default=None, help="Export raw metrics as JSON")

    args = parser.parse_args()

    print(f"🔍 Fetching run 1 metrics: {args.run1}")
    run1_metrics = fetch_wandb_metrics(args.run1, args.project, args.entity)

    print(f"🔍 Fetching run 2 metrics: {args.run2}")
    run2_metrics = fetch_wandb_metrics(args.run2, args.project, args.entity)

    print(f"🔍 Fetching run 3 metrics: {args.run3}")
    run3_metrics = fetch_wandb_metrics(args.run3, args.project, args.entity)

    run_names = [args.run1_name, args.run2_name, args.run3_name]

    print("📊 Analyzing and comparing runs...")
    comparison = compare_three_runs(run1_metrics, run2_metrics, run3_metrics, run_names)

    print("📝 Generating comparison report...")
    generate_comparison_report(run1_metrics, run2_metrics, run3_metrics, comparison, args.output)

    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(
            json.dumps(
                {
                    "run1_metrics": run1_metrics,
                    "run2_metrics": run2_metrics,
                    "run3_metrics": run3_metrics,
                    "comparison": comparison,
                },
                indent=2,
            )
        )
        print(f"✅ JSON export saved to: {args.export_json}")

    print("\n✨ Three-run comparison complete!")


if __name__ == "__main__":
    main()

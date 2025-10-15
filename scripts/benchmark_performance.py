#!/usr/bin/env python3
"""
Performance Benchmark Script
Compares baseline vs optimized configurations for OCR training pipeline.
"""

import subprocess
import sys
import time
from pathlib import Path

# Common test parameters
COMMON_PARAMS = [
    "trainer.max_epochs=3",
    "trainer.limit_val_batches=50",
    "logger.wandb.enabled=false",
    "seed=42",
]

BENCHMARKS = {
    "baseline": {
        "name": "Baseline (32-bit, no caching)",
        "params": [
            *COMMON_PARAMS,
            "exp_name=benchmark_baseline",
            "trainer.precision=32-true",
            "datasets.val_dataset.config.preload_images=false",
            "datasets.val_dataset.config.cache_config.cache_transformed_tensors=false",
        ],
    },
    "optimized": {
        "name": "Full Optimizations (16-bit + all caching)",
        "params": [
            *COMMON_PARAMS,
            "exp_name=benchmark_optimized",
            "trainer.precision=16-mixed",
            # Caching already enabled in configs/data/base.yaml
        ],
    },
}


def run_benchmark(name: str, config: dict) -> dict:
    """Run a single benchmark and return timing results."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: {config['name']}")
    print(f"{'=' * 80}\n")

    cmd = ["uv", "run", "python", "runners/train.py"] + config["params"]
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        end_time = time.time()
        elapsed = end_time - start_time

        return {
            "name": name,
            "display_name": config["name"],
            "elapsed_time": elapsed,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "display_name": config["name"],
            "elapsed_time": 600,
            "success": False,
            "error": "Timeout after 10 minutes",
        }
    except Exception as e:
        return {
            "name": name,
            "display_name": config["name"],
            "elapsed_time": 0,
            "success": False,
            "error": str(e),
        }


def extract_metrics(output: str) -> dict:
    """Extract key metrics from training output."""
    metrics = {}

    # Look for epoch timing
    import re

    epoch_pattern = r"Epoch (\d+)/\d+.*?(\d+:\d+:\d+)"
    epochs = re.findall(epoch_pattern, output)
    if epochs:
        metrics["epochs"] = epochs

    # Look for cache statistics
    cache_pattern = r"Cache Statistics - Hits: (\d+), Misses: (\d+), Hit Rate: ([\d.]+)%"
    cache_match = re.search(cache_pattern, output)
    if cache_match:
        metrics["cache_hits"] = int(cache_match.group(1))
        metrics["cache_misses"] = int(cache_match.group(2))
        metrics["cache_hit_rate"] = float(cache_match.group(3))

    # Look for preloading
    preload_pattern = r"Preloaded (\d+)/(\d+) images"
    preload_match = re.search(preload_pattern, output)
    if preload_match:
        metrics["images_preloaded"] = int(preload_match.group(1))

    return metrics


def print_results(results: list):
    """Print formatted benchmark results."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 80}\n")

    if not results:
        print("No results to display")
        return

    baseline = next((r for r in results if r["name"] == "baseline"), None)
    optimized = next((r for r in results if r["name"] == "optimized"), None)

    for result in results:
        print(f"\n{result['display_name']}")
        print("-" * 80)
        print(f"  Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        print(f"  Total Time: {result['elapsed_time']:.2f}s")

        if result["success"] and "stdout" in result:
            metrics = extract_metrics(result["stdout"])
            if metrics:
                print("  Metrics:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")

    # Calculate speedup
    if baseline and optimized and baseline["success"] and optimized["success"]:
        speedup = baseline["elapsed_time"] / optimized["elapsed_time"]
        time_saved = baseline["elapsed_time"] - optimized["elapsed_time"]

        print(f"\n{'=' * 80}")
        print("PERFORMANCE COMPARISON")
        print(f"{'=' * 80}")
        print(f"  Baseline Time:   {baseline['elapsed_time']:.2f}s")
        print(f"  Optimized Time:  {optimized['elapsed_time']:.2f}s")
        print(f"  Time Saved:      {time_saved:.2f}s ({time_saved / baseline['elapsed_time'] * 100:.1f}%)")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"{'=' * 80}\n")


def main():
    """Run all benchmarks and report results."""
    print("OCR Training Performance Benchmark")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []

    # Run benchmarks
    for name, config in BENCHMARKS.items():
        result = run_benchmark(name, config)
        results.append(result)

        # Save individual logs
        log_dir = Path("outputs") / f"benchmark_{name}"
        log_dir.mkdir(parents=True, exist_ok=True)

        if "stdout" in result:
            (log_dir / "stdout.log").write_text(result["stdout"])
        if "stderr" in result:
            (log_dir / "stderr.log").write_text(result["stderr"])

    # Print summary
    print_results(results)

    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Return exit code based on success
    if all(r["success"] for r in results):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

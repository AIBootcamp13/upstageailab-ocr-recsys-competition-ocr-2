#!/usr/bin/env python3
"""
Performance Impact Measurement Script

This script measures the actual performance impact of enabling polygon cache
and performance callbacks.
"""

import subprocess
import time
from pathlib import Path


def run_training_test(config_overrides: list[str], description: str) -> dict:
    """Run a training test and measure performance."""
    print(f"📊 Measuring performance for: {description}")

    start_time = time.time()

    cmd = [
        "uv",
        "run",
        "python",
        "runners/train.py",
        "--config-name=performance_test",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=5",  # Small but meaningful batch
        "trainer.limit_val_batches=3",
        "trainer.enable_progress_bar=false",
    ] + config_overrides

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    total_time = time.time() - start_time

    success = result.returncode == 0

    # Extract some basic metrics from output
    output = result.stdout + result.stderr

    metrics = {"success": success, "total_time": total_time, "exit_code": result.returncode}

    # Look for validation metrics
    if "val_hmean:" in output:
        try:
            # Extract hmean value
            lines = output.split("\n")
            for line in lines:
                if "val_hmean:" in line:
                    value_str = line.split("val_hmean:")[1].split()[0]
                    metrics["val_hmean"] = float(value_str)
                    break
        except (ValueError, IndexError):
            pass

    return metrics


def main():
    """Measure performance impact of different features."""
    print("🚀 Measuring performance impact of optimization features...\n")

    # Test configurations
    tests = [
        {"overrides": [], "description": "Baseline (no optimizations)"},
        {"overrides": ["data.polygon_cache.enabled=true"], "description": "Polygon cache enabled"},
        {
            "overrides": ["+callbacks.performance_profiler=ocr.lightning_modules.callbacks.PerformanceProfilerCallback"],
            "description": "Performance profiler callback",
        },
        {
            "overrides": [
                "data.polygon_cache.enabled=true",
                "+callbacks.performance_profiler=ocr.lightning_modules.callbacks.PerformanceProfilerCallback",
            ],
            "description": "Both optimizations enabled",
        },
    ]

    results = {}

    # Run each test
    for test in tests:
        desc = test["description"]
        overrides = test["overrides"]

        metrics = run_training_test(overrides, desc)
        results[desc] = metrics

        status = "✅ PASS" if metrics["success"] else "❌ FAIL"
        print(f"{status} - {desc}: {metrics['total_time']:.2f}s")
    print("\n📈 Performance Analysis:")

    if all(r["success"] for r in results.values()):
        baseline_time = results["Baseline (no optimizations)"]["total_time"]

        print("🎉 All tests passed!")

        for desc, metrics in results.items():
            if desc != "Baseline (no optimizations)":
                speedup = baseline_time / metrics["total_time"]
                overhead = ((metrics["total_time"] - baseline_time) / baseline_time) * 100
                print(f"  {desc}: {speedup:.2f}x speedup, {overhead:.1f}% overhead")
        # Check if polygon cache provides benefit
        cache_time = results["Polygon cache enabled"]["total_time"]
        if cache_time < baseline_time:
            improvement = ((baseline_time - cache_time) / baseline_time) * 100
            print(f"✅ Polygon cache provides {improvement:.1f}% performance improvement")
        else:
            overhead = ((cache_time - baseline_time) / baseline_time) * 100
            print(f"⚠️  Polygon cache has {overhead:.1f}% overhead")
    else:
        print("⚠️  Some tests failed. Cannot perform performance analysis.")
        failed_tests = [desc for desc, metrics in results.items() if not metrics["success"]]
        print(f"Failed tests: {', '.join(failed_tests)}")

    # Recommendations
    print("\n💡 Recommendations:")
    if results.get("Polygon cache enabled", {}).get("success", False):
        cache_time = results["Polygon cache enabled"]["total_time"]
        baseline_time = results["Baseline (no optimizations)"]["total_time"]
        if cache_time <= baseline_time * 1.05:  # Within 5% overhead
            print("✅ Polygon cache: SAFE TO ENABLE (minimal overhead)")
        else:
            print("⚠️  Polygon cache: REVIEW BEFORE ENABLING (significant overhead)")

    if results.get("Performance profiler callback", {}).get("success", False):
        callback_time = results["Performance profiler callback"]["total_time"]
        baseline_time = results["Baseline (no optimizations)"]["total_time"]
        if callback_time <= baseline_time * 1.10:  # Within 10% overhead
            print("✅ Performance profiler: SAFE TO ENABLE (acceptable overhead)")
        else:
            print("⚠️  Performance profiler: REVIEW BEFORE ENABLING (high overhead)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance Features Testing Script

This script provides isolated testing of performance optimization features
to ensure they don't break core functionality before re-enabling them.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of a performance feature test."""

    feature_name: str
    success: bool
    duration: float
    metrics: dict[str, Any]
    error_message: str = ""
    config_used: dict[str, Any] | None = None


class PerformanceTester:
    """Isolated performance features testing."""

    def __init__(self, base_config: str = "performance_test"):
        self.base_config = base_config
        self.results: list[TestResult] = []
        self.test_dir = Path("performance_test_results")
        self.test_dir.mkdir(exist_ok=True)

    def run_command(self, cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        return subprocess.run(cmd, capture_output=True, text=True, env=full_env, cwd=Path.cwd())

    def test_baseline(self) -> TestResult:
        """Test baseline performance without any optimizations."""
        print("🧪 Testing baseline (no performance features)...")

        start_time = time.time()

        # Run training with baseline config
        cmd = [
            "uv",
            "run",
            "python",
            "runners/train.py",
            f"--config-name={self.base_config}",
            "performance_test.polygon_cache=false",
            "callbacks={}",
        ]

        result = self.run_command(cmd)
        duration = time.time() - start_time

        success = result.returncode == 0
        error_msg = result.stderr if not success else ""

        # Extract basic metrics from output
        metrics = self._extract_metrics(result.stdout)

        return TestResult(
            feature_name="baseline",
            success=success,
            duration=duration,
            metrics=metrics,
            error_message=error_msg,
            config_used={"polygon_cache": False, "callbacks": {}},
        )

    def test_polygon_cache(self) -> TestResult:
        """Test polygon caching feature."""
        print("🧪 Testing polygon cache...")

        start_time = time.time()

        # Run training with polygon cache enabled
        cmd = [
            "uv",
            "run",
            "python",
            "runners/train.py",
            f"--config-name={self.base_config}",
            "performance_test.polygon_cache=true",
            "data.polygon_cache.enabled=true",
            "callbacks={}",
        ]

        result = self.run_command(cmd)
        duration = time.time() - start_time

        success = result.returncode == 0
        error_msg = result.stderr if not success else ""

        metrics = self._extract_metrics(result.stdout)

        return TestResult(
            feature_name="polygon_cache",
            success=success,
            duration=duration,
            metrics=metrics,
            error_message=error_msg,
            config_used={"polygon_cache": True, "data.polygon_cache.enabled": True},
        )

    def test_throughput_callback(self) -> TestResult:
        """Test throughput monitoring callback."""
        print("🧪 Testing throughput monitor callback...")

        start_time = time.time()

        # Run training with throughput callback
        cmd = [
            "uv",
            "run",
            "python",
            "runners/train.py",
            f"--config-name={self.base_config}",
            "performance_test.polygon_cache=false",
            "callbacks.throughput_monitor=ocr.lightning_modules.callbacks.throughput_monitor.ThroughputMonitor",
        ]

        result = self.run_command(cmd)
        duration = time.time() - start_time

        success = result.returncode == 0
        error_msg = result.stderr if not success else ""

        metrics = self._extract_metrics(result.stdout)

        return TestResult(
            feature_name="throughput_monitor",
            success=success,
            duration=duration,
            metrics=metrics,
            error_message=error_msg,
            config_used={"callbacks": {"throughput_monitor": "ThroughputMonitor"}},
        )

    def test_profiler_callback(self) -> TestResult:
        """Test profiler callback."""
        print("🧪 Testing profiler callback...")

        start_time = time.time()

        # Run training with profiler callback
        cmd = [
            "uv",
            "run",
            "python",
            "runners/train.py",
            f"--config-name={self.base_config}",
            "performance_test.polygon_cache=false",
            "callbacks.profiler=ocr.lightning_modules.callbacks.profiler.ProfilerCallback",
        ]

        result = self.run_command(cmd)
        duration = time.time() - start_time

        success = result.returncode == 0
        error_msg = result.stderr if not success else ""

        metrics = self._extract_metrics(result.stdout)

        return TestResult(
            feature_name="profiler",
            success=success,
            duration=duration,
            metrics=metrics,
            error_message=error_msg,
            config_used={"callbacks": {"profiler": "ProfilerCallback"}},
        )

    def _extract_metrics(self, output: str) -> dict[str, Any]:
        """Extract basic metrics from training output."""
        metrics = {}

        # Look for validation metrics
        if "val_hmean" in output:
            # Extract hmean value (rough parsing)
            lines = output.split("\n")
            for line in lines:
                if "val_hmean:" in line:
                    try:
                        # Extract number after "val_hmean:"
                        parts = line.split("val_hmean:")
                        if len(parts) > 1:
                            value_str = parts[1].split()[0]
                            metrics["val_hmean"] = float(value_str)
                    except (ValueError, IndexError):
                        pass

        # Look for training time
        if "epoch" in output.lower():
            metrics["completed_epochs"] = 1  # Since we limit to 1 epoch

        return metrics

    def run_all_tests(self) -> list[TestResult]:
        """Run all performance feature tests."""
        print("🚀 Starting performance features assessment...")

        tests = [
            self.test_baseline,
            self.test_polygon_cache,
            self.test_throughput_callback,
            self.test_profiler_callback,
        ]

        for test_func in tests:
            try:
                result = test_func()
                self.results.append(result)
                print(f"✅ {result.feature_name}: {'PASS' if result.success else 'FAIL'} ({result.duration:.2f}s)")
                if not result.success:
                    print(f"   Error: {result.error_message[:200]}...")
            except Exception as e:
                error_result = TestResult(
                    feature_name=test_func.__name__.replace("test_", ""), success=False, duration=0.0, metrics={}, error_message=str(e)
                )
                self.results.append(error_result)
                print(f"❌ {error_result.feature_name}: FAIL - Exception: {str(e)}")

        return self.results

    def save_results(self):
        """Save test results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.test_dir / f"performance_test_results_{timestamp}.json"

        results_data = {
            "timestamp": timestamp,
            "tests": [asdict(result) for result in self.results],
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "total_duration": sum(r.duration for r in self.results),
            },
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"📊 Results saved to: {results_file}")
        return results_file


def main():
    """Main entry point."""
    tester = PerformanceTester()

    try:
        results = tester.run_all_tests()
        # Print summary
        passed = sum(1 for r in results if r.success)
        total = len(results)

        print(f"\n📈 Summary: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All performance features tests passed!")
        else:
            print("⚠️  Some tests failed. Check results file for details.")
            failed_tests = [r.feature_name for r in results if not r.success]
            print(f"Failed tests: {', '.join(failed_tests)}")

    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

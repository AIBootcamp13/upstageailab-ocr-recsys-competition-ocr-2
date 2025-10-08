#!/usr/bin/env python3
"""
Quick Performance Features Validation Script

This script quickly validates that performance optimization features
can be enabled without breaking config loading or basic functionality.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            timeout=60,  # 60 second timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def test_config_loading():
    """Test that configs load without errors."""
    print("🧪 Testing config loading...")

    # Test baseline config
    success, output = run_command(
        [
            "uv",
            "run",
            "python",
            "-c",
            """
import os
os.chdir('/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2')
from hydra import compose, initialize_config_dir
initialize_config_dir(config_dir='configs')
cfg = compose(config_name='performance_test')
print('Config loaded successfully')
        """,
        ]
    )

    if not success:
        print(f"❌ Config loading failed: {output}")
        return False

    print("✅ Config loading successful")
    return True


def test_polygon_cache_config():
    """Test polygon cache config can be enabled."""
    print("🧪 Testing polygon cache config...")

    success, output = run_command(
        [
            "uv",
            "run",
            "python",
            "-c",
            """
import os
os.chdir('/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2')
from hydra import compose, initialize_config_dir
initialize_config_dir(config_dir='configs')
cfg = compose(config_name='performance_test', overrides=['data.polygon_cache.enabled=true'])
print('Polygon cache config loaded successfully')
        """,
        ]
    )

    if not success:
        print(f"❌ Polygon cache config failed: {output}")
        return False

    print("✅ Polygon cache config successful")
    return True


def test_throughput_callback_import():
    """Test that callback imports work (skip throughput monitor as it doesn't exist)."""
    print("🧪 Testing callback imports...")

    success, output = run_command(
        [
            "uv",
            "run",
            "python",
            "-c",
            """
import os
os.chdir('/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2')
try:
    from ocr.lightning_modules.callbacks import PerformanceProfilerCallback
    print('PerformanceProfilerCallback imported successfully')
except ImportError as e:
    print(f'Import failed: {e}')
    exit(1)
        """,
        ]
    )

    if not success:
        print(f"❌ Callback import failed: {output}")
        return False

    print("✅ Callback import successful")
    return True


def test_performance_callback():
    """Test that performance profiler callback can be enabled."""
    print("🧪 Testing performance profiler callback...")

    success, output = run_command(
        [
            "uv",
            "run",
            "python",
            "runners/train.py",
            "--config-name=performance_test",
            "+callbacks.performance_profiler=ocr.lightning_modules.callbacks.PerformanceProfilerCallback",
            "trainer.max_epochs=1",
            "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=1",
            "trainer.enable_progress_bar=false",
        ]
    )

    if not success:
        print(f"❌ Performance callback test failed: {output[-500:]}")  # Last 500 chars
        return False

    print("✅ Performance callback test successful")
    return True


def test_polygon_cache_training():
    """Test that polygon cache can be enabled without breaking training."""
    print("🧪 Testing polygon cache in training...")

    success, output = run_command(
        [
            "uv",
            "run",
            "python",
            "runners/train.py",
            "--config-name=performance_test",
            "data.polygon_cache.enabled=true",
            "trainer.max_epochs=1",
            "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=1",
            "trainer.enable_progress_bar=false",
        ]
    )

    if not success:
        print(f"❌ Polygon cache training failed: {output[-500:]}")  # Last 500 chars
        return False

    print("✅ Polygon cache training successful")
    return True


def main():
    """Run all validation tests."""
    print("🚀 Running quick performance features validation...\n")

    tests = [
        test_config_loading,
        test_polygon_cache_config,
        test_throughput_callback_import,
        test_performance_callback,
        test_polygon_cache_training,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            results.append(False)

    print(f"\n📈 Summary: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All performance features validation passed!")
        return 0
    else:
        print("⚠️  Some validations failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

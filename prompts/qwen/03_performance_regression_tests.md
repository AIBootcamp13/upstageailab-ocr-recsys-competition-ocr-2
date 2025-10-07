# Task: Create Performance Regression Test Suite

## Context
- **Project:** Receipt OCR Text Detection (200k+ LOC)
- **Framework:** PyTorch Lightning 2.1+ with Hydra 1.3+ and pytest
- **Purpose:** Task 1.3 - Create automated regression tests to catch performance degradation
- **Code Style:** Follow `pyproject.toml` (ruff, mypy with type hints)

## Objective
Create a pytest-based test suite that validates performance metrics don't regress beyond acceptable thresholds. Tests should run in CI and fail if validation time increases >10% or memory usage exceeds limits.

## Requirements

### Functional Requirements
1. **Validation time regression** - Ensure validation doesn't exceed baseline + 10%
2. **Memory usage limits** - GPU <80% available, CPU <80% available
3. **Cache hit rate** - After warmup, cache hit rate >80% (when caching enabled)
4. **Batch throughput** - Maintain minimum samples/sec threshold
5. **Load from baseline** - Read baseline metrics from JSON file

### Non-Functional Requirements
1. **CI integration** - Can run in GitHub Actions
2. **Configurable thresholds** - SLOs defined in config file
3. **Clear failure messages** - Show actual vs expected when failing
4. **Type safe** - Full type hints, passes mypy

## Input Files to Reference

### Read These Files First:
```
tests/performance/test_polygon_caching.py                # Existing performance tests
tests/integration/test_performance_profiler.py           # Profiler tests pattern
ocr/lightning_modules/callbacks/performance_profiler.py  # Metrics available
docs/ai_handbook/07_project_management/performance_optimization_plan.md  # SLOs
```

### Project Structure:
```
tests/
  performance/
    __init__.py                    # Already exists
    test_polygon_caching.py        # Already exists
    test_regression.py             # CREATE THIS FILE
    baselines/                     # CREATE THIS DIRECTORY
      baseline_metrics.json        # Will be created by baseline script
      thresholds.yaml              # CREATE THIS FILE
```

## Output Files

### Create:
- `tests/performance/test_regression.py`
- `tests/performance/baselines/thresholds.yaml`
- `.github/workflows/performance-regression.yml` (CI config)

## Implementation Details

### 1. Thresholds Configuration
```yaml
# tests/performance/baselines/thresholds.yaml
# Performance SLOs (Service Level Objectives)

validation:
  # Validation time must not exceed baseline by this percentage
  time_regression_threshold_pct: 10

  # Absolute maximum validation time (fallback if no baseline)
  max_time_seconds: 300

  # Minimum batch throughput
  min_samples_per_second: 50

memory:
  # GPU memory usage limits
  max_gpu_memory_pct: 80
  max_gpu_memory_gb: 12

  # CPU memory usage limits
  max_cpu_memory_pct: 80

cache:
  # Cache performance (when caching is enabled)
  min_hit_rate_pct: 80
  warmup_batches: 10  # Ignore first N batches for hit rate

tolerance:
  # Measurement noise tolerance
  measurement_noise_pct: 2
```

### 2. Regression Test Suite
```python
# tests/performance/test_regression.py
"""
Performance regression test suite.

These tests ensure that performance optimizations don't regress and
that the system maintains acceptable performance characteristics.

Usage:
    # Run all performance tests
    pytest tests/performance/test_regression.py -v

    # Run with baseline comparison
    pytest tests/performance/test_regression.py -v \
        --baseline tests/performance/baselines/baseline_metrics.json

    # Run in CI
    pytest tests/performance/test_regression.py -v --ci
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from ocr.lightning_modules.callbacks import PerformanceProfilerCallback


def load_thresholds() -> dict[str, Any]:
    """Load performance thresholds from config."""
    threshold_file = Path(__file__).parent / "baselines" / "thresholds.yaml"

    if not threshold_file.exists():
        # Use default thresholds if file doesn't exist
        return {
            "validation": {
                "time_regression_threshold_pct": 10,
                "max_time_seconds": 300,
                "min_samples_per_second": 50,
            },
            "memory": {
                "max_gpu_memory_pct": 80,
                "max_gpu_memory_gb": 12,
                "max_cpu_memory_pct": 80,
            },
            "cache": {
                "min_hit_rate_pct": 80,
                "warmup_batches": 10,
            },
            "tolerance": {
                "measurement_noise_pct": 2,
            },
        }

    with open(threshold_file) as f:
        return yaml.safe_load(f)


def load_baseline(baseline_path: str | None) -> dict[str, Any] | None:
    """Load baseline metrics from JSON file."""
    if baseline_path is None:
        return None

    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        return None

    with open(baseline_file) as f:
        return json.load(f)


def pytest_addoption(parser):
    """Add pytest command-line options."""
    parser.addoption(
        "--baseline",
        action="store",
        default=None,
        help="Path to baseline metrics JSON file",
    )
    parser.addoption(
        "--ci",
        action="store_true",
        default=False,
        help="Running in CI environment",
    )


@pytest.fixture
def thresholds():
    """Load performance thresholds."""
    return load_thresholds()


@pytest.fixture
def baseline(request):
    """Load baseline metrics if provided."""
    baseline_path = request.config.getoption("--baseline")
    return load_baseline(baseline_path)


@pytest.fixture
def dummy_model():
    """Create a minimal model for testing."""
    class DummyModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def validation_step(self, batch, batch_idx):
            x, y = batch
            loss = torch.nn.functional.mse_loss(self(x), y)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

    return DummyModel()


@pytest.fixture
def dummy_dataloader():
    """Create a minimal dataloader."""
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=10)


class TestValidationPerformance:
    """Test suite for validation performance regression."""

    def test_validation_time_within_threshold(
        self, dummy_model, dummy_dataloader, thresholds, baseline, tmp_path
    ):
        """Test that validation time doesn't exceed baseline + threshold."""
        profiler = PerformanceProfilerCallback(enabled=True)

        trainer = Trainer(
            max_epochs=1,
            callbacks=[profiler],
            default_root_dir=tmp_path,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        trainer.validate(dummy_model, dummy_dataloader)

        # Get actual validation time
        actual_time = sum(profiler.validation_batch_times)

        if baseline is not None:
            # Compare to baseline
            baseline_time = baseline.get("analysis", {}).get("validation_time", {}).get("total_seconds", 0)
            threshold_pct = thresholds["validation"]["time_regression_threshold_pct"]
            max_allowed_time = baseline_time * (1 + threshold_pct / 100)

            assert actual_time <= max_allowed_time, (
                f"Validation time regression detected!\n"
                f"  Baseline: {baseline_time:.2f}s\n"
                f"  Current:  {actual_time:.2f}s\n"
                f"  Allowed:  {max_allowed_time:.2f}s (baseline + {threshold_pct}%)\n"
                f"  Exceeded by: {(actual_time - max_allowed_time):.2f}s"
            )
        else:
            # Use absolute threshold
            max_time = thresholds["validation"]["max_time_seconds"]
            assert actual_time <= max_time, (
                f"Validation time exceeds absolute limit!\n"
                f"  Current: {actual_time:.2f}s\n"
                f"  Limit:   {max_time:.2f}s"
            )

    def test_batch_time_variance(
        self, dummy_model, dummy_dataloader, thresholds, tmp_path
    ):
        """Test that batch time variance is acceptable."""
        import numpy as np

        profiler = PerformanceProfilerCallback(enabled=True)

        trainer = Trainer(
            max_epochs=1,
            callbacks=[profiler],
            default_root_dir=tmp_path,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        trainer.validate(dummy_model, dummy_dataloader)

        batch_times = np.array(profiler.validation_batch_times)
        mean_time = np.mean(batch_times)
        p95_time = np.percentile(batch_times, 95)

        # P95 shouldn't be more than 2x the mean (indicates consistent performance)
        max_p95_ratio = 2.0
        actual_ratio = p95_time / mean_time

        assert actual_ratio <= max_p95_ratio, (
            f"High variance in batch times detected!\n"
            f"  Mean:  {mean_time*1000:.1f}ms\n"
            f"  P95:   {p95_time*1000:.1f}ms\n"
            f"  Ratio: {actual_ratio:.2f}x (should be <{max_p95_ratio}x)"
        )


class TestMemoryUsage:
    """Test suite for memory usage regression."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_within_limit(
        self, dummy_model, dummy_dataloader, thresholds, tmp_path
    ):
        """Test that GPU memory usage stays within limits."""
        profiler = PerformanceProfilerCallback(enabled=True, profile_memory=True)

        trainer = Trainer(
            max_epochs=1,
            callbacks=[profiler],
            default_root_dir=tmp_path,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="gpu",
            devices=1,
        )

        trainer.validate(dummy_model, dummy_dataloader)

        # Get GPU memory usage
        gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_pct = (gpu_memory_gb / gpu_total_gb) * 100

        max_pct = thresholds["memory"]["max_gpu_memory_pct"]
        max_gb = thresholds["memory"]["max_gpu_memory_gb"]

        assert gpu_memory_pct <= max_pct, (
            f"GPU memory usage exceeds limit!\n"
            f"  Current: {gpu_memory_pct:.1f}% ({gpu_memory_gb:.2f}GB)\n"
            f"  Limit:   {max_pct}% ({max_gb}GB)"
        )

    def test_cpu_memory_within_limit(
        self, dummy_model, dummy_dataloader, thresholds, tmp_path
    ):
        """Test that CPU memory usage stays within limits."""
        import psutil

        profiler = PerformanceProfilerCallback(enabled=True, profile_memory=True)

        trainer = Trainer(
            max_epochs=1,
            callbacks=[profiler],
            default_root_dir=tmp_path,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        trainer.validate(dummy_model, dummy_dataloader)

        cpu_memory_pct = psutil.virtual_memory().percent
        max_pct = thresholds["memory"]["max_cpu_memory_pct"]

        assert cpu_memory_pct <= max_pct, (
            f"CPU memory usage exceeds limit!\n"
            f"  Current: {cpu_memory_pct:.1f}%\n"
            f"  Limit:   {max_pct}%"
        )


class TestCachePerformance:
    """Test suite for cache performance (when caching is enabled)."""

    @pytest.mark.skip(reason="Cache not yet implemented")
    def test_cache_hit_rate(self, thresholds):
        """Test that cache hit rate meets minimum after warmup."""
        # This test will be implemented once PolygonCache is added
        pass
```

### 3. CI Workflow
```yaml
# .github/workflows/performance-regression.yml
name: Performance Regression Tests

on:
  pull_request:
    branches: [ main, develop ]
  push:
    branches: [ main, develop ]

jobs:
  performance-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: uv sync

    - name: Run performance regression tests
      run: |
        uv run pytest tests/performance/test_regression.py -v --ci \
          --baseline tests/performance/baselines/baseline_metrics.json

    - name: Upload performance report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: performance-report.html
```

## Validation

### Run These Commands:
```bash
# Type checking
uv run mypy tests/performance/test_regression.py

# Linting
uv run ruff check tests/performance/test_regression.py

# Format
uv run ruff format tests/performance/test_regression.py

# Run tests (without baseline)
uv run pytest tests/performance/test_regression.py -v

# Run tests with baseline (once baseline exists)
uv run pytest tests/performance/test_regression.py -v \
  --baseline tests/performance/baselines/baseline_metrics.json
```

### Expected Behavior:
- ✅ All type checks pass
- ✅ No linting errors
- ✅ Tests run successfully
- ✅ Clear failure messages when thresholds exceeded

## Example Usage

After implementation:

```bash
# Run regression tests locally
uv run pytest tests/performance/test_regression.py -v

# Run with baseline comparison
uv run pytest tests/performance/test_regression.py -v \
  --baseline outputs/baseline_metrics.json

# Run in CI mode (stricter)
uv run pytest tests/performance/test_regression.py -v --ci
```

## Success Criteria

- [ ] File `tests/performance/test_regression.py` created
- [ ] File `tests/performance/baselines/thresholds.yaml` created
- [ ] File `.github/workflows/performance-regression.yml` created
- [ ] All type hints present and mypy passes
- [ ] No ruff linting errors
- [ ] Tests run without errors
- [ ] Clear, actionable failure messages
- [ ] Thresholds are configurable via YAML

## Additional Notes

- **Dependencies:** Uses `pytest`, `pyyaml` (already in pyproject.toml)
- **Baseline file:** Will be created by `generate_baseline_report.py` script
- **CI integration:** Should fail PRs if performance regresses >10%
- **Extensibility:** Easy to add new metrics and thresholds

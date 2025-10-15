#!/usr/bin/env python
"""Benchmark validation performance with and without RAM caching."""

import subprocess
import time


def run_validation(with_preload=True):
    """Run validation and measure time."""
    config_override = f"datasets.val_dataset.preload_maps={str(with_preload).lower()}"

    cmd = [
        "uv",
        "run",
        "python",
        "runners/train.py",
        "trainer.limit_train_batches=0",
        "trainer.limit_val_batches=100",  # 100 batches for meaningful comparison
        "trainer.max_epochs=1",
        config_override,
    ]

    print(f"\n{'=' * 60}")
    print(f"Running validation with preload_maps={with_preload}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return None

    return elapsed


def main():
    print("Validation Performance Benchmark")
    print("=" * 60)

    # Warm-up run (to load models, etc.)
    print("\nWarm-up run...")
    run_validation(with_preload=True)

    # Benchmark without RAM preloading (disk I/O)
    time_no_preload = run_validation(with_preload=False)

    # Benchmark with RAM preloading
    time_with_preload = run_validation(with_preload=True)

    # Results
    if time_no_preload and time_with_preload:
        speedup = time_no_preload / time_with_preload
        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print(f"Without RAM preloading: {time_no_preload:.2f}s")
        print(f"With RAM preloading:    {time_with_preload:.2f}s")
        print(f"Speedup:                {speedup:.2f}x")
        print(f"{'=' * 60}\n")

        if speedup < 1.5:
            print("⚠️  Speedup is less than 1.5x. The bottleneck may be elsewhere.")
        elif speedup < 3:
            print("✓  Moderate speedup achieved. Consider further optimizations.")
        else:
            print("✅ Significant speedup achieved!")


if __name__ == "__main__":
    main()

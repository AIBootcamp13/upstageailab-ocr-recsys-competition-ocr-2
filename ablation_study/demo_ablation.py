#!/usr/bin/env python3
"""
Example: Quick Ablation Study Demo

This script demonstrates how to run a simple ablation study
without wandb for testing purposes.
"""

import subprocess
from pathlib import Path


def demo_ablation():
    """Run a quick ablation study demo."""
    print("🔬 Running Ablation Study Demo")
    print("=" * 50)

    # Example: Learning rate sweep with minimal epochs
    print("📊 Demo: Learning Rate Ablation (3 configurations, 2 epochs each)")

    # Create a minimal config for demo
    demo_config = """
training:
  learning_rate: [1e-3, 5e-4, 1e-4]
trainer:
  max_epochs: 2
wandb: false  # Disable wandb for demo
experiment_tag: "demo_lr_ablation"
"""

    # Save demo config
    config_path = Path("configs/ablation/demo.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        f.write(demo_config)

    print("✅ Created demo config")

    # Run the ablation (without -m for sequential execution in demo)
    print("🚀 Running experiments...")
    cmd = [
        "python",
        "run_ablation.py",
        "+ablation=demo",
        "experiment_tag=demo_lr_ablation",
    ]

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Demo experiments completed!")
            print("\nNext steps for full ablation studies:")
            print("1. Enable wandb: Set wandb=true in configs")
            print("2. Use multirun: Add -m flag for parallel execution")
            print("3. Collect results: python collect_results.py --project OCR_Ablation --tag demo_lr_ablation")
            print(
                "4. Generate table: python generate_ablation_table.py --input results.csv --ablation-type learning_rate --metric val/hmean"
            )
        else:
            print("❌ Demo failed")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])

    except Exception as e:
        print(f"❌ Error: {e}")

    # Cleanup
    if config_path.exists():
        config_path.unlink()
        print("🧹 Cleaned up demo config")


if __name__ == "__main__":
    demo_ablation()

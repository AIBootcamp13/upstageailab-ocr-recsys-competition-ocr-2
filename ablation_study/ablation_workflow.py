#!/usr/bin/env python3
"""
Ablation Study Workflow Manager

This script provides a complete workflow for running ablation studies:
1. Execute multiple experiments
2. Collect results
3. Generate comparison tables
4. Create visualizations

Usage:
    # Complete learning rate ablation workflow
    python ablation_workflow.py --ablation learning_rate --tag lr_study

    # Custom ablation with specific configs
    python ablation_workflow.py --ablation custom --configs "training.learning_rate=[1e-3,5e-4]" --tag custom_study
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def run_experiments(ablation_type: str, tag: str, configs: list = None):
    """Run ablation experiments using Hydra multirun."""
    print(f"🚀 Starting {ablation_type} ablation experiments...")

    # Build command
    cmd = [
        "python",
        "run_ablation.py",
        f"+ablation={ablation_type}",
        f"experiment_tag={tag}",
    ]

    if configs:
        cmd.extend(configs)

    # Add multirun flag for parallel execution
    cmd.extend(["-m"])  # Hydra multirun flag

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("✅ Experiments completed successfully!")
            return True
        else:
            print("❌ Experiments failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        return False


def collect_results(project: str, tag: str, output_file: str):
    """Collect results from wandb."""
    print(f"📊 Collecting results from wandb project: {project}")

    cmd = [
        "python",
        "collect_results.py",
        "--project",
        project,
        "--tag",
        tag,
        "--output",
        output_file,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("✅ Results collected successfully!")
            return True
        else:
            print("❌ Failed to collect results!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error collecting results: {e}")
        return False


def generate_table(
    input_file: str,
    ablation_type: str,
    metric: str,
    output_md: str = None,
    output_latex: str = None,
):
    """Generate ablation study table."""
    print(f"📋 Generating ablation table from {input_file}")

    cmd = [
        "python",
        "generate_ablation_table.py",
        "--input",
        input_file,
        "--ablation-type",
        ablation_type,
        "--metric",
        metric,
    ]

    if output_md:
        cmd.extend(["--output-md", output_md])
    if output_latex:
        cmd.extend(["--output-latex", output_latex])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("✅ Table generated successfully!")
            return True
        else:
            print("❌ Failed to generate table!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error generating table: {e}")
        return False


def create_visualization(input_file: str, ablation_type: str, metric: str, output_file: str):
    """Create visualization of ablation results."""
    print("📈 Creating visualization...")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        df = pd.read_csv(input_file)

        # Create appropriate plot based on ablation type
        plt.figure(figsize=(10, 6))

        if ablation_type == "learning_rate":
            x_col = "training.learning_rate"
            x_label = "Learning Rate"
            plt.xscale("log")
        elif ablation_type == "batch_size":
            x_col = "data.batch_size"
            x_label = "Batch Size"
        elif ablation_type == "model":
            x_col = "model.backbone.name"
            x_label = "Model Architecture"
        else:
            x_col = df.select_dtypes(include=[np.number]).columns[0]
            x_label = x_col

        if x_col in df.columns and metric in df.columns:
            sns.scatterplot(data=df, x=x_col, y=metric, s=100, alpha=0.7)
            sns.lineplot(data=df, x=x_col, y=metric, alpha=0.5)

            plt.xlabel(x_label)
            plt.ylabel(metric.replace("val/", "").replace("test/", "").title())
            plt.title(f"Ablation Study: {ablation_type.replace('_', ' ').title()}")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"✅ Visualization saved to {output_file}")
            return True
        else:
            print("❌ Required columns not found for visualization")
            return False

    except ImportError:
        print("⚠️  Matplotlib/seaborn not available, skipping visualization")
        return False
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Complete ablation study workflow")
    parser.add_argument(
        "--ablation",
        required=True,
        choices=["learning_rate", "batch_size", "model_architecture", "custom"],
        help="Type of ablation study",
    )
    parser.add_argument("--tag", required=True, help="Experiment tag for tracking")
    parser.add_argument("--configs", nargs="+", help="Additional config overrides for custom ablation")
    parser.add_argument("--project", default="OCR_Ablation", help="wandb project name")
    parser.add_argument("--metric", default="val/hmean", help="Primary metric for comparison")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip running experiments")
    parser.add_argument("--skip-collection", action="store_true", help="Skip result collection")
    parser.add_argument("--skip-table", action="store_true", help="Skip table generation")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # Define output files
    results_file = f"ablation_results_{args.tag}.csv"
    table_md_file = f"ablation_table_{args.tag}.md"
    table_latex_file = f"ablation_table_{args.tag}.tex"
    viz_file = f"ablation_plot_{args.tag}.png"

    # Step 1: Run experiments
    if not args.skip_experiments:
        print("\n" + "=" * 60)
        print("STEP 1: RUNNING EXPERIMENTS")
        print("=" * 60)

        success = run_experiments(args.ablation, args.tag, args.configs)
        if not success:
            print("❌ Workflow failed at experiment stage")
            sys.exit(1)

        # Wait a bit for wandb to sync
        print("⏳ Waiting for wandb sync...")
        time.sleep(10)

    # Step 2: Collect results
    if not args.skip_collection:
        print("\n" + "=" * 60)
        print("STEP 2: COLLECTING RESULTS")
        print("=" * 60)

        success = collect_results(args.project, args.tag, results_file)
        if not success:
            print("❌ Workflow failed at collection stage")
            sys.exit(1)

    # Step 3: Generate table
    if not args.skip_table:
        print("\n" + "=" * 60)
        print("STEP 3: GENERATING TABLES")
        print("=" * 60)

        success = generate_table(results_file, args.ablation, args.metric, table_md_file, table_latex_file)
        if not success:
            print("❌ Workflow failed at table generation stage")
            sys.exit(1)

    # Step 4: Create visualization
    if not args.skip_viz:
        print("\n" + "=" * 60)
        print("STEP 4: CREATING VISUALIZATION")
        print("=" * 60)

        create_visualization(results_file, args.ablation, args.metric, viz_file)

    print("\n" + "=" * 60)
    print("🎉 ABLATION STUDY WORKFLOW COMPLETED!")
    print("=" * 60)
    print(f"📊 Results: {results_file}")
    print(f"📋 Table: {table_md_file}")
    print(f"📈 Plot: {viz_file}")
    print("\nNext steps:")
    print("1. Review the results in wandb dashboard")
    print("2. Analyze the generated table and plot")
    print("3. Document findings in your research notes")


if __name__ == "__main__":
    main()

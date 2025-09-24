#!/usr/bin/env python3
"""
Ablation Study Results Collector

This script collects results from wandb experiments and organizes them for analysis.

Usage:
    # Collect results from a specific project
    python collect_results.py --project "OCR_Ablation" --tag "lr_ablation"

    # Collect all results from project
    python collect_results.py --project "OCR_Ablation"

    # Save to specific file
    python collect_results.py --project "OCR_Ablation" --output results.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import wandb
from wandb.apis.public import Runs


def collect_wandb_results(project_name: str, tag: str = None, entity: str = None) -> pd.DataFrame:
    """
    Collect results from wandb experiments.

    Args:
        project_name: Name of the wandb project
        tag: Optional tag to filter runs
        entity: wandb entity (username/team)

    Returns:
        DataFrame with experiment results
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get runs from project
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)

    results = []

    for run in runs:
        # Filter by tag if specified
        if tag and tag not in run.tags:
            continue

        # Extract config and metrics
        config = run.config

        # Get summary metrics (final values)
        summary = run.summary

        # Get system metrics if available
        system_metrics = {}
        if run.system_metrics:
            # Get final values for key metrics
            system_metrics = {
                'gpu_memory_used': run.system_metrics.get('_gpu_memory_used', [0])[-1] if '_gpu_memory_used' in run.system_metrics else 0,
                'cpu_percent': run.system_metrics.get('_cpu_percent', [0])[-1] if '_cpu_percent' in run.system_metrics else 0,
            }

        # Combine all data
        result = {
            'run_id': run.id,
            'run_name': run.name,
            'status': run.state,
            'created_at': run.created_at,
            'duration': (run.heartbeat_at - run.created_at).total_seconds() if run.heartbeat_at else None,
            **config,  # Flatten config
            **summary,  # Flatten summary metrics
            **system_metrics,
        }

        results.append(result)

    return pd.DataFrame(results)


def save_results_to_csv(df: pd.DataFrame, output_path: str, sort_by: str = None):
    """
    Save results to CSV with proper formatting.

    Args:
        df: DataFrame with results
        output_path: Path to save CSV
        sort_by: Column to sort by (optional)
    """
    # Sort if requested
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Collected {len(df)} experiments")


def print_summary_table(df: pd.DataFrame, group_by: str = None, metrics: list = None):
    """
    Print a summary table of results.

    Args:
        df: DataFrame with results
        group_by: Column to group by for comparison
        metrics: List of metrics to display
    """
    if metrics is None:
        # Auto-detect common metrics
        common_metrics = ['val/recall', 'val/precision', 'val/hmean', 'test/recall', 'test/precision', 'test/hmean']
        metrics = [m for m in common_metrics if m in df.columns]

    if group_by and group_by in df.columns:
        # Grouped summary
        grouped = df.groupby(group_by)[metrics].agg(['mean', 'std', 'max', 'min'])
        print(f"\nSummary by {group_by}:")
        print(grouped.round(4))
    else:
        # Overall summary
        summary = df[metrics].agg(['mean', 'std', 'max', 'min'])
        print("\nOverall Summary:")
        print(summary.round(4))

    # Show top performers
    if metrics:
        best_metric = metrics[0]  # Use first metric for ranking
        if best_metric in df.columns:
            print(f"\nTop 5 by {best_metric}:")
            top_runs = df.nlargest(5, best_metric)[['run_name', best_metric] + metrics[1:]]
            print(top_runs.round(4))


def main():
    parser = argparse.ArgumentParser(description="Collect ablation study results from wandb")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", help="wandb entity (username/team)")
    parser.add_argument("--tag", help="Filter runs by tag")
    parser.add_argument("--output", default="ablation_results.csv", help="Output CSV file")
    parser.add_argument("--sort-by", help="Column to sort results by")
    parser.add_argument("--group-by", help="Column to group results by for summary")
    parser.add_argument("--metrics", nargs="+", help="Metrics to include in summary")

    args = parser.parse_args()

    print(f"Collecting results from project: {args.project}")
    if args.tag:
        print(f"Filtering by tag: {args.tag}")

    # Collect results
    df = collect_wandb_results(args.project, args.tag, args.entity)

    if df.empty:
        print("No results found!")
        return

    # Save to CSV
    save_results_to_csv(df, args.output, args.sort_by)

    # Print summary
    print_summary_table(df, args.group_by, args.metrics)


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/collect_results.py
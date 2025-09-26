#!/usr/bin/env python3
"""
Ablation Study Table Generator

This script generates publication-ready tables from ablation study results.

Usage:
    # Generate learning rate ablation table
    python generate_ablation_table.py --input results.csv --ablation-type learning_rate --metric val/hmean

    # Generate model comparison table
    python generate_ablation_table.py --input results.csv --ablation-type model --metric val/hmean --baseline resnet18

    # Generate custom table with specific columns
    python generate_ablation_table.py --input results.csv --columns "model.backbone.name,training.learning_rate,val/recall,val/precision,val/hmean"
"""

import argparse

import numpy as np
import pandas as pd


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(csv_path)


def extract_config_value(config_str: str, key: str):
    """Extract nested config value from flattened column name."""
    # Handle nested keys like "model.backbone.name"
    try:
        # This is a simplified extraction - in practice you might need more robust parsing
        if key in config_str:
            # For now, return the key itself as the column represents the config value
            return config_str
    except:
        return config_str
    return config_str


def create_ablation_table(
    df: pd.DataFrame,
    ablation_type: str,
    metric: str,
    baseline: str = None,
    columns: list = None,
) -> pd.DataFrame:
    """
    Create a formatted ablation table.

    Args:
        df: DataFrame with experiment results
        ablation_type: Type of ablation ('learning_rate', 'batch_size', 'model', etc.)
        metric: Primary metric to compare
        baseline: Baseline configuration for relative comparison
        columns: Specific columns to include

    Returns:
        Formatted DataFrame for the ablation table
    """

    # Define column mappings based on ablation type
    column_mappings = {
        "learning_rate": {
            "config_col": "training.learning_rate",
            "display_name": "Learning Rate",
            "format_func": lambda x: f"{x:.0e}",
        },
        "batch_size": {
            "config_col": "data.batch_size",
            "display_name": "Batch Size",
            "format_func": str,
        },
        "model": {
            "config_col": "model.backbone.name",
            "display_name": "Model",
            "format_func": str,
        },
    }

    if ablation_type not in column_mappings:
        # Custom ablation - use provided columns
        if not columns:
            raise ValueError("For custom ablation types, specify --columns")

        # Use first column as the varying parameter
        config_col = columns[0]
        display_name = config_col.split(".")[-1].title()
        format_func = str
    else:
        mapping = column_mappings[ablation_type]
        config_col = mapping["config_col"]
        display_name = mapping["display_name"]
        format_func = mapping["format_func"]

    # Filter to relevant columns
    if columns:
        available_columns = [col for col in columns if col in df.columns]
    else:
        available_columns = [
            config_col,
            metric,
            "val/recall",
            "val/precision",
            "duration",
        ]

    # Keep only available columns
    available_columns = [col for col in available_columns if col in df.columns]
    table_df = df[available_columns].copy()

    # Rename columns for display
    column_rename = {config_col: display_name}
    if metric in table_df.columns:
        column_rename[metric] = f"{metric} (Primary)"

    table_df = table_df.rename(columns=column_rename)

    # Format the config column
    if display_name in table_df.columns:
        table_df[display_name] = table_df[display_name].apply(format_func)

    # Add relative improvement if baseline is specified
    if baseline and metric in df.columns:
        baseline_value = df[df[config_col] == baseline][metric].mean()
        if not np.isnan(baseline_value):
            table_df["Improvement"] = ((table_df[metric] - baseline_value) / baseline_value * 100).round(2)
            table_df["Improvement"] = table_df["Improvement"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

    # Sort by primary metric (descending)
    if metric in table_df.columns:
        table_df = table_df.sort_values(metric, ascending=False)

    return table_df


def save_table_markdown(table_df: pd.DataFrame, output_path: str):
    """Save table as markdown format."""
    markdown_table = table_df.to_markdown(index=False, floatfmt=".4f")
    with open(output_path, "w") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(markdown_table)
    print(f"Markdown table saved to {output_path}")


def save_table_latex(table_df: pd.DataFrame, output_path: str, caption: str = "Ablation Study Results"):
    """Save table as LaTeX format."""
    latex_table = table_df.to_latex(index=False, float_format="%.4f", caption=caption, label="tab:ablation")
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {output_path}")


def print_table(table_df: pd.DataFrame):
    """Print formatted table to console."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(table_df.to_string(index=False, float_format="%.4f"))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study tables")
    parser.add_argument("--input", required=True, help="Input CSV file with results")
    parser.add_argument(
        "--ablation-type",
        required=True,
        choices=["learning_rate", "batch_size", "model", "custom"],
        help="Type of ablation study",
    )
    parser.add_argument("--metric", required=True, help="Primary metric to compare")
    parser.add_argument("--baseline", help="Baseline configuration for relative comparison")
    parser.add_argument("--columns", nargs="+", help="Specific columns to include (for custom ablation)")
    parser.add_argument("--output-md", help="Output markdown file")
    parser.add_argument("--output-latex", help="Output LaTeX file")
    parser.add_argument("--caption", default="Ablation Study Results", help="LaTeX table caption")

    args = parser.parse_args()

    # Load results
    df = load_results(args.input)
    print(f"Loaded {len(df)} experiments from {args.input}")

    # Create ablation table
    table_df = create_ablation_table(df, args.ablation_type, args.metric, args.baseline, args.columns)

    # Print to console
    print_table(table_df)

    # Save formats
    if args.output_md:
        save_table_markdown(table_df, args.output_md)

    if args.output_latex:
        save_table_latex(table_df, args.output_latex, args.caption)


if __name__ == "__main__":
    main()

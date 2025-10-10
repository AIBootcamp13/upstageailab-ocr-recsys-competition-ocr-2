#!/usr/bin/env python3
"""
Experiment Analysis Script for DBNet++ vs CRAFT Training Runs

This script processes Wandb CSV exports to analyze training metrics,
detect performance anomalies, and generate summaries for AI analysis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_wandb_csvs(run_folder: str) -> dict:
    """Load all Wandb CSV files from the run folder."""
    folder = Path(run_folder)
    data = {}

    for csv_file in folder.glob("*.csv"):
        df = pd.read_csv(csv_file)
        # Extract metric name from column headers
        for col in df.columns:
            if "val/" in col and not col.endswith("__MIN") and not col.endswith("__MAX"):
                metric = col.split(" - ")[-1]
                data[metric] = df[["trainer/global_step", col]].rename(columns={col: "value"})
                break

    return data


def detect_anomalies(metrics: dict, threshold: float = 0.1) -> list:
    """Detect significant drops in metrics."""
    anomalies = []

    for metric_name, df in metrics.items():
        if "recall" in metric_name.lower() or "hmean" in metric_name.lower():
            values = df["value"].values
            steps = df["trainer/global_step"].values

            for i in range(1, len(values)):
                drop = values[i - 1] - values[i]
                if drop > threshold:
                    anomalies.append(
                        {
                            "step": int(steps[i]),
                            "metric": metric_name,
                            "drop": round(drop, 3),
                            "from": round(values[i - 1], 3),
                            "to": round(values[i], 3),
                        }
                    )

    return anomalies


def generate_summary(metrics: dict, anomalies: list) -> dict:
    """Generate a concise summary for AI analysis."""
    summary = {
        "model_config": {"architecture": "dbnetpp", "encoder": "resnet50", "batch_size": 8, "learning_rate": "1e-3"},
        "performance_metrics": {},
        "data_insights": {"total_steps": 0, "suspected_bad_batches": len(anomalies), "drop_frequency": f"{len(anomalies)} drops detected"},
        "anomalies": anomalies,
    }

    for metric_name, df in metrics.items():
        values = df["value"].values
        summary["performance_metrics"][f"best_{metric_name}"] = round(np.max(values), 4)
        summary["performance_metrics"][f"final_{metric_name}"] = round(values[-1], 4)
        summary["performance_metrics"][f"mean_{metric_name}"] = round(np.mean(values), 4)

    summary["data_insights"]["total_steps"] = int(df["trainer/global_step"].max())

    return summary


def main():
    run_folder = "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/wchoi189_dbnetpp-resnet18-unet-db-head-db-loss-bs8-lr1e-3_hmean0.898"

    # Load data
    metrics = load_wandb_csvs(run_folder)

    # Detect anomalies
    anomalies = detect_anomalies(metrics, threshold=0.05)  # Lower threshold for sensitivity

    # Generate summary
    summary = generate_summary(metrics, anomalies)

    # Print summary
    print(json.dumps(summary, indent=2))

    # Save to file
    with open("experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

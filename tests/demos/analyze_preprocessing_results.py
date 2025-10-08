#!/usr/bin/env python3
"""
Analysis script for preprocessing test results.

Provides tools to analyze, compare, and visualize preprocessing test results.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class ResultsAnalyzer:
    """Analyzer for preprocessing test results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.consolidated_results = None
        self.comparison_report = None

    def load_results(self) -> None:
        """Load all result files."""
        # Load consolidated results
        consolidated_file = self.results_dir / "consolidated_results.json"
        if consolidated_file.exists():
            with open(consolidated_file) as f:
                self.consolidated_results = json.load(f)

        # Load comparison report
        comparison_file = self.results_dir / "comparison_report.json"
        if comparison_file.exists():
            with open(comparison_file) as f:
                self.comparison_report = json.load(f)

    def print_summary_report(self) -> None:
        """Print a summary report of all configurations."""
        if not self.comparison_report:
            print("No comparison report found. Run load_results() first.")
            return

        print("=== Preprocessing Test Results Summary ===\n")

        print(f"Total configurations tested: {self.comparison_report['test_summary']['total_configurations']}")
        print(f"Samples per configuration: {self.comparison_report['test_summary']['total_samples_per_config']}\n")

        # Create a table-like output
        print("Configuration Performance:")
        print("-" * 80)
        print("<25")
        print("-" * 80)

        for config in self.comparison_report["configuration_comparison"]:
            success_rate = config["detection_success_rate"] * 100
            avg_time = config["avg_processing_time"] * 1000  # Convert to ms
            print(
                f"{config['config_name']:<25} {success_rate:>8.1f}% {avg_time:>8.1f}ms "
                f"{config['successful_detections']:>8} {config['orientation_corrections']:>8}"
            )

        print()

    def analyze_detection_differences(self) -> dict[str, Any]:
        """Analyze differences between detection methods."""
        if not self.consolidated_results:
            return {}

        analysis = {
            "method_success_rates": {},
            "method_avg_times": {},
            "orientation_corrections_by_method": {},
            "corner_detection_patterns": {},
        }

        method_stats = {}

        for result in self.consolidated_results:
            for sample in result["samples"]:
                method = sample.get("method", "failed")
                if method not in method_stats:
                    method_stats[method] = {"total_samples": 0, "successful_samples": 0, "total_time": 0.0, "orientation_corrections": 0}

                method_stats[method]["total_samples"] += 1
                method_stats[method]["total_time"] += sample["processing_time"]

                if sample["detection_success"]:
                    method_stats[method]["successful_samples"] += 1

                if "orientation" in sample.get("metadata", {}):
                    method_stats[method]["orientation_corrections"] += 1

        # Calculate rates
        for method, stats in method_stats.items():
            analysis["method_success_rates"][method] = stats["successful_samples"] / stats["total_samples"]
            analysis["method_avg_times"][method] = stats["total_time"] / stats["total_samples"]
            analysis["orientation_corrections_by_method"][method] = stats["orientation_corrections"]

        return analysis

    def create_performance_comparison_chart(self, output_path: Path | None = None) -> None:
        """Create a performance comparison chart."""
        if not self.comparison_report:
            print("No comparison report available")
            return

        configs = [c["config_name"] for c in self.comparison_report["configuration_comparison"]]
        success_rates = [c["detection_success_rate"] * 100 for c in self.comparison_report["configuration_comparison"]]
        avg_times = [c["avg_processing_time"] * 1000 for c in self.comparison_report["configuration_comparison"]]  # ms

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Success rate chart
        bars1 = ax1.bar(configs, success_rates, color="skyblue")
        ax1.set_title("Detection Success Rate by Configuration")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_xlabel("Configuration")
        ax1.tick_params(axis="x", rotation=45)
        for bar, rate in zip(bars1, success_rates, strict=True):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, ".1f", ha="center", va="bottom")

        # Processing time chart
        bars2 = ax2.bar(configs, avg_times, color="lightgreen")
        ax2.set_title("Average Processing Time by Configuration")
        ax2.set_ylabel("Time (ms)")
        ax2.set_xlabel("Configuration")
        ax2.tick_params(axis="x", rotation=45)
        for bar, time_val in zip(bars2, avg_times, strict=True):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, ".1f", ha="center", va="bottom")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Performance chart saved to {output_path}")
        else:
            plt.show()

    def analyze_sample_variability(self) -> dict[str, Any]:
        """Analyze how different samples perform across configurations."""
        if not self.consolidated_results:
            return {}

        sample_performance = {}
        num_samples = len(self.consolidated_results[0]["samples"]) if self.consolidated_results else 0

        for sample_idx in range(num_samples):
            sample_results = []
            for result in self.consolidated_results:
                if sample_idx < len(result["samples"]):
                    sample_result = result["samples"][sample_idx]
                    sample_results.append(
                        {
                            "config": result["config_name"],
                            "success": sample_result["detection_success"],
                            "method": sample_result.get("method"),
                            "time": sample_result["processing_time"],
                        }
                    )

            if sample_results:
                sample_performance[f"sample_{sample_idx}"] = sample_results

        return sample_performance

    def export_detailed_report(self, output_path: Path) -> None:
        """Export a detailed analysis report."""
        if not self.consolidated_results:
            print("No results to export")
            return

        report = {
            "summary": self.comparison_report,
            "method_analysis": self.analyze_detection_differences(),
            "sample_analysis": self.analyze_sample_variability(),
            "recommendations": self.generate_recommendations(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Detailed report exported to {output_path}")

    def generate_recommendations(self) -> list[str]:
        """Generate recommendations based on results."""
        recommendations: list[str] = []

        if not self.comparison_report or not self.consolidated_results:
            return recommendations

        # Find best performing configuration
        best_config = max(
            self.comparison_report["configuration_comparison"], key=lambda x: (x["detection_success_rate"], -x["avg_processing_time"])
        )

        recommendations.append(f"Best overall configuration: {best_config['config_name']} .1f.1f")

        # Check for configurations with high success but slow performance
        for config in self.comparison_report["configuration_comparison"]:
            if config["detection_success_rate"] > 0.8 and config["avg_processing_time"] > 0.5:
                recommendations.append(f"Consider optimizing {config['config_name']} - high success rate but slow processing")

        # Check for configurations with orientation corrections
        orientation_configs = [c for c in self.consolidated_results if c["summary"]["orientation_corrections"] > 0]
        if orientation_configs:
            recommendations.append(
                "Orientation correction is active in some configurations. Monitor angle correction values for effectiveness."
            )

        return recommendations

    def show_sample_details(self, sample_idx: int) -> None:
        """Show detailed results for a specific sample."""
        if not self.consolidated_results:
            print("No results loaded")
            return

        print(f"=== Sample {sample_idx} Details ===\n")

        for result in self.consolidated_results:
            if sample_idx >= len(result["samples"]):
                continue

            sample = result["samples"][sample_idx]
            config_name = result["config_name"]

            print(f"Configuration: {config_name}")
            print(f"  Success: {sample['detection_success']}")
            print(f"  Method: {sample.get('method', 'N/A')}")
            print(".3f")
            print(f"  Original shape: {sample['original_shape']}")
            print(f"  Final shape: {sample['final_shape']}")

            if "corners" in sample and sample["corners"] is not None:
                corners = np.array(sample["corners"])
                print(f"  Corners detected: {len(corners)} points")
                print(f"  Corner coordinates: {corners.tolist()}")

            if "metadata" in sample and "orientation" in sample["metadata"]:
                orientation = sample["metadata"]["orientation"]
                print(f"  Orientation correction: {orientation}")

            print()


def main():
    parser = argparse.ArgumentParser(description="Analyze preprocessing test results")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing test results")
    parser.add_argument("--summary", action="store_true", help="Print summary report")
    parser.add_argument("--chart", type=Path, help="Create performance comparison chart")
    parser.add_argument("--export-report", type=Path, help="Export detailed analysis report")
    parser.add_argument("--sample-details", type=int, help="Show details for specific sample index")

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.load_results()

    if args.summary:
        analyzer.print_summary_report()

    if args.chart:
        analyzer.create_performance_comparison_chart(args.chart)

    if args.export_report:
        analyzer.export_detailed_report(args.export_report)

    if args.sample_details is not None:
        analyzer.show_sample_details(args.sample_details)

    # If no specific action requested, show summary
    if not any([args.summary, args.chart, args.export_report, args.sample_details is not None]):
        analyzer.print_summary_report()


if __name__ == "__main__":
    main()

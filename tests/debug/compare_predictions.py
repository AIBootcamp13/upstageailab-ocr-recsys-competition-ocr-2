#!/usr/bin/env python3
"""
Compare OCR predictions from different preprocessing methods.
"""

import os

import pandas as pd


def load_csv_results(csv_path):
    """Load CSV results and return as DataFrame."""
    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].astype(str)
    return df


def compare_predictions(csv_files):
    """Compare predictions across different preprocessing methods."""
    results = {}

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = load_csv_results(csv_file)
            method_name = csv_file.replace("submissions_epoch22_", "").replace(".csv", "")
            results[method_name] = df
            print(f"✅ Loaded {method_name}: {len(df)} predictions")
        else:
            print(f"❌ Missing: {csv_file}")

    return results


def analyze_differences(results):
    """Analyze differences between preprocessing methods."""
    if len(results) < 2:
        print("Need at least 2 result sets to compare")
        return

    methods = list(results.keys())

    print(f"\n📊 Comparing {len(methods)} preprocessing methods")
    print("=" * 60)

    # Compare detection counts
    print("\n🔢 Detection Counts:")
    for method, df in results.items():
        total_polygons = df["polygons"].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0).sum()
        empty_predictions = (df["polygons"].isna() | (df["polygons"] == "")).sum()
        print(f"  {method}: {total_polygons} total polygons, {empty_predictions} empty predictions")

    # Compare specific differences
    if len(methods) >= 2:
        print("\nDetailed Comparison:")
        for i in range(1, len(methods)):
            method1, method2 = methods[0], methods[i]
            df1, df2 = results[method1], results[method2]

            # Merge on filename
            merged = pd.merge(df1, df2, on="filename", suffixes=(f"_{method1}", f"_{method2}"))

            # Count differences
            different_predictions = 0
            for _, row in merged.iterrows():
                poly1 = str(row[f"polygons_{method1}"]) if pd.notna(row[f"polygons_{method1}"]) else ""
                poly2 = str(row[f"polygons_{method2}"]) if pd.notna(row[f"polygons_{method2}"]) else ""
                if poly1 != poly2:
                    different_predictions += 1

            print(
                f"  {method1} vs {method2}: {different_predictions}/{len(merged)} images differ ({different_predictions/len(merged)*100:.1f}%)"
            )


def main():
    csv_files = [
        "submissions_epoch22_corrected_gray.csv",
        "submissions_epoch22_hmean0922_gray.csv",
        "submissions_epoch22_hmean0922_normal.csv",
    ]

    print("🔬 OCR Prediction Comparison Tool")
    print("=" * 40)

    results = compare_predictions(csv_files)
    analyze_differences(results)

    print("\nRecommendations:")
    if len(results) >= 2:
        # Simple heuristic: prefer method with most detections
        best_method = max(
            results.keys(), key=lambda m: results[m]["polygons"].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0).sum()
        )
        print(f"  Highest detection count: {best_method}")

        # Check for consistency
        methods = list(results.keys())
        if len(methods) >= 2:
            df1, df2 = results[methods[0]], results[methods[1]]
            merged = pd.merge(df1, df2, on="filename", suffixes=(f"_{methods[0]}", f"_{methods[1]}"))
            same_predictions = sum(
                1 for _, row in merged.iterrows() if str(row[f"polygons_{methods[0]}"]) == str(row[f"polygons_{methods[1]}"])
            )
            consistency = same_predictions / len(merged) * 100
            print(f"  Consistency between {methods[0]} and {methods[1]}: {consistency:.1f}%")


if __name__ == "__main__":
    main()

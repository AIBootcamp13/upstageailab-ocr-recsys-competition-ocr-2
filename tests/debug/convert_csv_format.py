#!/usr/bin/env python3
"""
Convert OCR prediction CSV files to the required submission format.
Format: filename,polygons,avg_confidence
Where polygons are flattened coordinates separated by | for multiple words.
"""

import ast
import csv
from collections import defaultdict
from pathlib import Path


def parse_polygon_string(polygon_str):
    """Parse polygon string from CSV and flatten to x y x y ... format."""
    try:
        # Parse the string representation of list of lists
        polygon_list = ast.literal_eval(polygon_str)
        # Flatten to x y x y ... format
        flattened = []
        for point in polygon_list:
            flattened.extend([str(int(point[0])), str(int(point[1]))])
        return " ".join(flattened)
    except (ValueError, SyntaxError, IndexError) as e:
        print(f"Error parsing polygon: {polygon_str}, error: {e}")
        return ""


def convert_csv_format(input_file, output_file, max_polygon_length=1000):
    """Convert CSV from individual word format to image-level format."""
    print(f"Processing {input_file}...")

    # Group data by filename
    image_data = defaultdict(lambda: {"polygons": [], "confidences": []})

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["image_filename"]
            polygon_str = parse_polygon_string(row["points"])
            confidence = row.get("confidence", "").strip()

            if polygon_str:  # Only add if polygon parsing succeeded
                image_data[filename]["polygons"].append(polygon_str)
                if confidence:
                    try:
                        image_data[filename]["confidences"].append(float(confidence))
                    except ValueError:
                        pass  # Skip invalid confidence values

    # Write output in new format
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "polygons", "avg_confidence"])

        for filename, data in sorted(image_data.items()):
            polygons = "|".join(data["polygons"])

            # Limit polygon string length to prevent extremely long lines
            if len(polygons) > max_polygon_length:
                polygons = polygons[:max_polygon_length] + "...[TRUNCATED]"
                print(f"Warning: Truncated polygons for {filename} (was {len('|'.join(data['polygons']))} chars)")

            # Calculate average confidence
            if data["confidences"]:
                avg_confidence = sum(data["confidences"]) / len(data["confidences"])
                avg_confidence = f"{avg_confidence:.6f}"
            else:
                avg_confidence = ""

            writer.writerow([filename, polygons, avg_confidence])

    print(f"Converted {len(image_data)} images to {output_file}")


def convert_csv_format_no_truncate(input_file, output_file):
    """Convert CSV from individual word format to image-level format without truncation."""
    print(f"Processing {input_file}...")

    # Group data by filename
    image_data = defaultdict(lambda: {"polygons": [], "confidences": []})

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["image_filename"]
            polygon_str = parse_polygon_string(row["points"])
            confidence = row.get("confidence", "").strip()

            if polygon_str:  # Only add if polygon parsing succeeded
                image_data[filename]["polygons"].append(polygon_str)
                if confidence:
                    try:
                        image_data[filename]["confidences"].append(float(confidence))
                    except ValueError:
                        pass  # Skip invalid confidence values

    # Write output in new format (no confidence column for submission)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "polygons"])

        for filename, data in sorted(image_data.items()):
            polygons = "|".join(data["polygons"])
            # No truncation - keep full polygon data
            writer.writerow([filename, polygons])

    print(f"Converted {len(image_data)} images to {output_file} (no truncation, no confidence)")


def main():
    # Convert no-confidence file without truncation for submission
    if Path("submission_grays_no_conf.csv").exists():
        convert_csv_format_no_truncate("submission_grays_no_conf.csv", "submission_grays_no_conf_submission.csv")

    # Convert other files with truncation (for internal use)
    files_to_convert = [
        ("submission_grays_conf.csv", "submission_grays_conf_formatted.csv"),
        ("submission_gray2_bn0.2.csv", "submission_gray2_bn0.2_formatted.csv"),
    ]

    for input_file, output_file in files_to_convert:
        if Path(input_file).exists():
            convert_csv_format(input_file, output_file, max_polygon_length=2000)  # Limit to prevent pipe breaks
        else:
            print(f"Warning: {input_file} not found, skipping...")

    print("All conversions completed!")


if __name__ == "__main__":
    main()

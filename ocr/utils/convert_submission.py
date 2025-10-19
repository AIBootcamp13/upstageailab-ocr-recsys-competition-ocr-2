import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def convert_json_to_csv(json_path, output_path, include_confidence=False, force=False):
    # Validate input file exists
    json_file_path = Path(json_path)
    if not json_file_path.exists():
        print(f"Error: Input JSON file '{json_path}' does not exist.")
        return None

    if not json_file_path.is_file():
        print(f"Error: '{json_path}' is not a file.")
        return None

    # Check if CSV file already exists
    csv_file = Path(output_path)
    if csv_file.exists() and not force:
        print(f"Error: Output file '{csv_file}' already exists. Use --force to overwrite.")
        return None

    try:
        with open(json_path, encoding="utf-8") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: Could not find JSON file '{json_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_path}': {e}")
        return None
    except PermissionError:
        print(f"Error: Permission denied reading file '{json_path}'")
        return None
    except Exception as e:
        print(f"Error: Unexpected error reading file '{json_path}': {e}")
        return None

    if "images" not in data:
        print(f"Error: The JSON file '{json_path}' doesn't contain the required 'images' key.")
        return None

    print(f"Processing {len(data['images'])} images from '{json_path}'...")

    rows = []
    for filename, content in data["images"].items():
        if "words" not in content:
            print(f"Warning: '{filename}' doesn't contain the 'words' key. Skipping.")
            continue

        polygons = []
        confidences = []
        for idx, word in content["words"].items():
            if "points" not in word:
                print(f"Warning: '{idx}' in '{filename}' doesn't contain the 'points' key. Skipping word.")
                continue

            points = word["points"]
            if not points or len(points) == 0:
                print(f"Warning: No points found in '{idx}' of '{filename}'. Skipping word.")
                continue

            try:
                polygon = " ".join([" ".join(map(str, point)) for point in points])
                polygons.append(polygon)

                # Extract confidence if available and requested
                if include_confidence:
                    confidence = word.get("confidence", 1.0)  # Default to 1.0 if not present
                    confidences.append(confidence)
            except Exception as e:
                print(f"Warning: Error processing word '{idx}' in '{filename}': {e}. Skipping word.")
                continue

        if not polygons:
            print(f"Warning: No valid polygons found for '{filename}'. Skipping image.")
            continue

        polygons_str = "|".join(polygons)
        if include_confidence and confidences:
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
            rows.append([filename, polygons_str, avg_confidence])
        else:
            rows.append([filename, polygons_str])

    if not rows:
        print("Error: No valid data found to convert.")
        return None

    columns = ["filename", "polygons"]
    if include_confidence:
        columns.append("avg_confidence")

    try:
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_path, index=False)
        print(f"Successfully wrote {len(rows)} rows to '{output_path}'")
    except Exception as e:
        print(f"Error: Failed to write CSV file '{output_path}': {e}")
        return None

    return len(rows), output_path


def convert():
    parser = argparse.ArgumentParser(description="Convert JSON to CSV")
    parser.add_argument("-J", "--json_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument(
        "-O",
        "--output_path",
        type=str,
        required=True,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--include_confidence",
        action="store_true",
        help="Include confidence scores in the CSV output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file without prompting",
    )

    args = parser.parse_args()

    result = convert_json_to_csv(args.json_path, args.output_path, args.include_confidence, args.force)
    if result:
        num_rows, output_file = result
        print(f"Conversion completed: {num_rows} rows written to '{output_file}'")
        return 0
    else:
        print("Conversion failed.")
        return 1


if __name__ == "__main__":
    exit_code = convert()
    sys.exit(exit_code)

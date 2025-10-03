import argparse
import json
from pathlib import Path

import pandas as pd


def convert_json_to_csv(json_path, output_path, include_confidence=False):
    # Check if CSV file already exists
    csv_file = Path(output_path)
    if csv_file.exists():
        response = input(f"The file '{csv_file}' already exists. Do you want to overwrite it? (yes/No): ").strip().lower()
        if response != "yes":
            print("Conversion cancelled.")
            return None

    with open(json_path) as json_file:
        data = json.load(json_file)
    assert "images" in data, "The JSON file doesn't contain the 'images' key."

    rows = []
    for filename, content in data["images"].items():
        assert "words" in content, f"The '{filename}' doesn't contain the 'words' key."

        polygons = []
        confidences = []
        for idx, word in content["words"].items():
            assert "points" in word, f"'{idx}' in '{filename}' doesn't contain the 'points' key."

            points = word["points"]
            assert len(points) > 0, f"No points found in '{idx}' of '{filename}'."

            polygon = " ".join([" ".join(map(str, point)) for point in points])
            polygons.append(polygon)

            # Extract confidence if available and requested
            if include_confidence:
                confidence = word.get("confidence", 1.0)  # Default to 1.0 if not present
                confidences.append(confidence)

        polygons_str = "|".join(polygons)
        if include_confidence and confidences:
            avg_confidence = sum(confidences) / len(confidences)
            rows.append([filename, polygons_str, avg_confidence])
        else:
            rows.append([filename, polygons_str])

    columns = ["filename", "polygons"]
    if include_confidence:
        columns.append("avg_confidence")

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

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

    args = parser.parse_args()

    result = convert_json_to_csv(args.json_path, args.output_path, args.include_confidence)
    if result:
        num_rows, output_file = result
        print(f"Successfully converted {num_rows} rows to '{output_file}'")


if __name__ == "__main__":
    convert()

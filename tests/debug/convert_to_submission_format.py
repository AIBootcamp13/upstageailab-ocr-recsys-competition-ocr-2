import ast
import csv
import sys


def convert_csv_format(input_file, output_file):
    """Convert OCR prediction CSV to competition submission format."""

    # Read the input CSV
    data = {}
    with open(input_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["image_filename"]

            # Parse the points string (it's a string representation of a list of lists)
            points_str = row["points"].strip()
            try:
                # Convert string to actual list of lists
                points = ast.literal_eval(points_str)
                if isinstance(points, list) and len(points) > 0:
                    # Convert to space-separated coordinates
                    coords = []
                    for point in points:
                        if isinstance(point, list) and len(point) >= 2:
                            coords.extend([str(point[0]), str(point[1])])
                    polygon_str = " ".join(coords)
                else:
                    continue
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse points for {filename}: {points_str}")
                continue

            # Group polygons by filename
            if filename not in data:
                data[filename] = []
            data[filename].append(polygon_str)

    # Write the output CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "polygons"])

        for filename, polygons in data.items():
            # Join multiple polygons with |
            polygons_str = "|".join(polygons)
            writer.writerow([filename, polygons_str])

    print(f"Converted {len(data)} images with {sum(len(p) for p in data.values())} total polygons")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_submission_format.py <input_csv> <output_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_csv_format(input_file, output_file)

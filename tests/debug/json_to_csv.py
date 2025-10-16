import csv
import json
import sys


def json_to_csv(json_file_path, csv_file_path):
    """Convert OCR submission JSON to CSV format."""
    print(f"Loading JSON file: {json_file_path}")

    with open(json_file_path) as f:
        data = json.load(f)

    print(f"Processing {len(data['images'])} images...")

    # Prepare CSV data
    csv_data = []
    headers = ["image_filename", "word_id", "points", "confidence"]

    for image_filename, image_data in data["images"].items():
        for word_id, word_data in image_data["words"].items():
            row = {
                "image_filename": image_filename,
                "word_id": word_id,
                "points": str(word_data["points"]),
                "confidence": word_data.get("confidence", ""),
            }
            csv_data.append(row)

    print(f"Writing {len(csv_data)} rows to CSV: {csv_file_path}")

    # Write to CSV
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)

    print("Conversion completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_csv.py <input_json> <output_csv>")
        sys.exit(1)

    json_file = sys.argv[1]
    csv_file = sys.argv[2]

    json_to_csv(json_file, csv_file)

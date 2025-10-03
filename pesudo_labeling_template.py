import json
import os
from collections import OrderedDict

import numpy as np
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from tqdm import tqdm

# NOTE: This script requires 'python-doctr[tf]' and 'tensorflow'
# Add them to your requirements.txt and run 'uv sync'


def generate_pseudo_labels():
    """
    Uses a pre-trained doctr model to generate bounding box labels for a
    directory of images and saves them in the format expected by the project.
    """
    # --- CONFIGURATION ---
    # 1. Point this to the folder with your 3000 unlabeled images
    IMAGE_DIR = "data/datasets/images/unlabeled"
    # 2. This is where the final JSON labels will be saved
    OUTPUT_JSON_PATH = "data/datasets/jsons/train_pseudo.json"
    # 3. Choose a pre-trained model from doctr
    # Options: 'db_resnet50', 'db_mobilenet_v3_large', 'linknet_resnet18'
    # 'db_resnet50' is a strong default choice.
    MODEL_ARCH = "db_resnet50"
    # --- END CONFIGURATION ---

    print(f"Loading '{MODEL_ARCH}' model...")
    # Load the pre-trained text detection model
    # The model will be downloaded automatically on first run
    predictor = detection_predictor(arch=MODEL_ARCH, pretrained=True)

    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Image directory not found at '{IMAGE_DIR}'")
        print("Please create it and place your images inside.")
        # Create a dummy folder to prevent crashing
        os.makedirs(IMAGE_DIR, exist_ok=True)
        return

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print(f"Warning: No images found in '{IMAGE_DIR}'")
        return

    print(f"Found {len(image_files)} images to label.")

    # This will be the final JSON structure
    output_data = OrderedDict(images=OrderedDict())

    # Process images in batches for efficiency
    batch_size = 16
    for i in tqdm(range(0, len(image_files), batch_size), desc="Generating Labels"):
        batch_files = image_files[i : i + batch_size]
        batch_paths = [os.path.join(IMAGE_DIR, fname) for fname in batch_files]

        # Load documents and run prediction
        docs = [DocumentFile.from_images(p) for p in batch_paths]
        results = predictor(docs)

        for fname, result in zip(batch_files, results, strict=False):
            # The structure your project expects
            image_entry = OrderedDict(words=OrderedDict())

            # The model returns boxes with relative coordinates (0-1).
            # We need to convert them to absolute pixel coordinates.
            img_height, img_width = result.pages[0].dimensions
            boxes = result.pages[0].blocks

            for word_idx, box in enumerate(boxes):
                for line in box.lines:
                    for word in line.words:
                        # Get absolute pixel coordinates
                        abs_coords = word.geometry

                        # Convert (xmin, ymin, xmax, ymax) to a 4-point polygon
                        # The format is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        p1 = [abs_coords[0][0] * img_width, abs_coords[0][1] * img_height]
                        p2 = [abs_coords[1][0] * img_width, abs_coords[0][1] * img_height]
                        p3 = [abs_coords[1][0] * img_width, abs_coords[1][1] * img_height]
                        p4 = [abs_coords[0][0] * img_width, abs_coords[1][1] * img_height]

                        polygon_points = [p1, p2, p3, p4]

                        # Ensure points are integers as per the project's format
                        polygon_points_int = np.round(polygon_points).astype(int).tolist()

                        # Add to the dictionary with a formatted key like "0001"
                        word_key = f"{word_idx + 1:04d}"
                        image_entry["words"][word_key] = OrderedDict(points=polygon_points_int)

            output_data["images"][fname] = image_entry

    # Save the final JSON file
    print(f"Saving {len(output_data['images'])} labeled images to '{OUTPUT_JSON_PATH}'...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print("Pseudo-labeling complete!")
    print(
        f"Next steps: \n1. (Recommended) Review '{OUTPUT_JSON_PATH}' with a tool like Label Studio. \n2. Use the generated JSON to train your model."
    )


if __name__ == "__main__":
    generate_pseudo_labels()

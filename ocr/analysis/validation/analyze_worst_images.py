# scripts/analyze_worst_images.py

import argparse
import json
import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_bboxes_on_image(image: np.ndarray, gt_boxes: Any, pred_boxes: Any) -> np.ndarray:
    vis_image = image.copy()
    # Draw ground truth boxes in GREEN
    for bbox in gt_boxes:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    # Draw predicted boxes in RED
    for bbox in pred_boxes:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


def analyze_worst_images(results_json_path: str, image_dir: str, top_n: int = 10):
    print(f"🔍 Analyzing results from: {results_json_path}")
    if not os.path.exists(results_json_path):
        print(f"❌ Error: Results file not found at '{results_json_path}'")
        return

    with open(results_json_path) as f:
        per_sample_results = json.load(f)

    results_list = [{"filename": fname, **metrics} for fname, metrics in per_sample_results.items()]
    df = pd.DataFrame(results_list)
    worst_df = df.sort_values(by="hmean", ascending=True).head(top_n)

    print(f"\n--- Top {top_n} Worst Performing Images ---")
    print(worst_df[["filename", "hmean", "precision", "recall"]].to_string())

    plt.figure(figsize=(20, top_n * 5))
    for i, row in enumerate(worst_df.itertuples()):
        image_path = os.path.join(image_dir, str(row.filename))
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping.")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to read image at {image_path}. Skipping.")
            continue
        # Normalize boxes in case they are stored as JSON strings, bytes, tuples, or numpy arrays
        gt_boxes: Any = row.gt_bboxes
        det_boxes: Any = row.det_bboxes
        if isinstance(gt_boxes, str | bytes):
            try:
                gt_boxes = json.loads(gt_boxes)
            except Exception:
                gt_boxes = []
        if isinstance(det_boxes, str | bytes):
            try:
                det_boxes = json.loads(det_boxes)
            except Exception:
                det_boxes = []
        if isinstance(gt_boxes, np.ndarray):
            gt_boxes = gt_boxes.tolist()
        if isinstance(det_boxes, np.ndarray):
            det_boxes = det_boxes.tolist()
        if not isinstance(gt_boxes, list):
            gt_boxes = list(gt_boxes) if isinstance(gt_boxes, tuple | set) else []
        if not isinstance(det_boxes, list):
            det_boxes = list(det_boxes) if isinstance(det_boxes, tuple | set) else []
        vis_image = draw_bboxes_on_image(image, gt_boxes, det_boxes)
        ax = plt.subplot(top_n // 2 + 1, 2, i + 1)
        ax = plt.subplot(top_n // 2 + 1, 2, i + 1)
        ax.imshow(vis_image)
        ax.set_title(f"{row.filename}\nH-Mean: {row.hmean:.3f} | P: {row.precision:.3f} | R: {row.recall:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize worst-performing images.")
    parser.add_argument("results_json", type=str, help="Path to 'per_sample_results.json'.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/ICDAR17_Korean/images",
        help="Path to validation images.",
    )
    parser.add_argument("--top_n", type=int, default=10, help="Number of worst images to display.")
    args = parser.parse_args()
    analyze_worst_images(args.results_json, args.image_dir, args.top_n)

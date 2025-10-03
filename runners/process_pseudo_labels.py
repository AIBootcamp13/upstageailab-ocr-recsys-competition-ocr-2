#!/usr/bin/env python3
"""
Upstage Document Parse API Pipeline for Pseudo-Label Image Processing

This script processes existing pseudo-label images by:
1. Converting PNG images to WebP format for efficiency
2. Calling Upstage Document Parse API to extract text and bounding boxes
3. Converting API responses to CLEval-compatible JSON annotations

Usage:
    python runners/process_pseudo_labels.py --dataset cord-v2 --limit 10 --dry-run
    python runners/process_pseudo_labels.py --dataset all --convert-webp-only
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for the processing pipeline"""

    upstage_api_key: str
    input_base_dir: str
    output_base_dir: str
    max_workers: int = 4
    rate_limit_delay: float = 0.1  # seconds between API calls
    webp_quality: int = 85


class UpstageDocumentParser:
    """Handles Upstage Document Parse API interactions"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.upstage.ai/v1/document-ai/layout-analysis"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def process_image(self, image_path: str) -> dict | None:
        """Process a single image with Upstage API"""
        try:
            with open(image_path, "rb") as f:
                files = {"document": f}
                data = {"ocr": True}  # Enable OCR for better text extraction

                response = requests.post(self.base_url, headers=self.headers, files=files, data=data, timeout=30)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"API error for {image_path}: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None


class ImageConverter:
    """Handles image format conversion"""

    @staticmethod
    def convert_to_webp(input_path: str, output_path: str, quality: int = 85) -> bool:
        """Convert image to WebP format"""
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if necessary (WebP doesn't support all modes)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # Save as WebP
                img.save(output_path, "WEBP", quality=quality, optimize=True)
                return True
        except Exception as e:
            logger.error(f"Error converting {input_path} to WebP: {str(e)}")
            return False


class CLEvalAnnotationConverter:
    """Converts Upstage API responses to CLEval-compatible format"""

    @staticmethod
    def convert_response_to_cleval(api_response: dict, image_filename: str) -> dict:
        """Convert Upstage API response to CLEval annotation format"""
        annotations = []

        if "elements" not in api_response:
            logger.warning(f"No elements found in API response for {image_filename}")
            return {"annotations": annotations}

        for element in api_response["elements"]:
            if "bounding_box" not in element or "text" not in element:
                continue

            # Convert bounding box to polygon format
            bbox = element["bounding_box"]
            if len(bbox) >= 4:
                # CLEval expects polygons as list of [x,y] coordinates
                polygon = [[point["x"], point["y"]] for point in bbox]

                annotation = {
                    "polygon": polygon,
                    "text": element["text"],
                    "confidence": element.get("confidence", 1.0),
                    "category": element.get("category", "text"),
                }
                annotations.append(annotation)

        return {"annotations": annotations}


class PseudoLabelProcessor:
    """Main processor for pseudo-label images"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.parser = UpstageDocumentParser(config.upstage_api_key)
        self.converter = ImageConverter()

    def get_image_paths(self, dataset: str) -> list[Path]:
        """Get all image paths for a dataset"""
        image_dir = Path(self.config.input_base_dir) / dataset / "images"
        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return []

        return list(image_dir.glob("*.png"))

    def process_single_image(self, image_path: Path, output_dir: Path, convert_webp: bool = True) -> bool:
        """Process a single image end-to-end"""
        try:
            # Convert to WebP if requested
            if convert_webp:
                webp_path = output_dir / "images_webp" / f"{image_path.stem}.webp"
                webp_path.parent.mkdir(parents=True, exist_ok=True)

                if not self.converter.convert_to_webp(str(image_path), str(webp_path), self.config.webp_quality):
                    return False

                # Use WebP for API processing
                api_image_path = webp_path
            else:
                api_image_path = image_path

            # Call Upstage API
            time.sleep(self.config.rate_limit_delay)  # Rate limiting
            api_response = self.parser.process_image(str(api_image_path))

            if api_response is None:
                return False

            # Convert to CLEval format
            cleval_annotation = CLEvalAnnotationConverter.convert_response_to_cleval(api_response, image_path.name)

            # Save annotation
            annotation_path = output_dir / "annotations" / f"{image_path.stem}.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)

            with open(annotation_path, "w", encoding="utf-8") as f:
                json.dump(cleval_annotation, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False

    def process_dataset(
        self, dataset: str, limit: int | None = None, dry_run: bool = False, convert_webp_only: bool = False
    ) -> dict[str, int]:
        """Process all images in a dataset"""
        image_paths = self.get_image_paths(dataset)
        if not image_paths:
            return {"total": 0, "processed": 0, "failed": 0}

        if limit:
            image_paths = image_paths[:limit]

        output_dir = Path(self.config.output_base_dir) / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"total": len(image_paths), "processed": 0, "failed": 0}

        if dry_run:
            logger.info(f"Dry run: Would process {len(image_paths)} images from {dataset}")
            return stats

        if convert_webp_only:
            logger.info(f"Converting {len(image_paths)} images to WebP format only")
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for image_path in image_paths:
                    webp_path = output_dir / "images_webp" / f"{image_path.stem}.webp"
                    webp_path.parent.mkdir(parents=True, exist_ok=True)
                    future = executor.submit(self.converter.convert_to_webp, str(image_path), str(webp_path), self.config.webp_quality)
                    futures.append(future)

                for future in tqdm(futures, desc=f"Converting {dataset} to WebP"):
                    if future.result():
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
        else:
            logger.info(f"Processing {len(image_paths)} images from {dataset}")
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for image_path in image_paths:
                    future = executor.submit(self.process_single_image, image_path, output_dir, convert_webp=True)
                    futures.append(future)

                for future in tqdm(futures, desc=f"Processing {dataset}"):
                    if future.result():
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1

        return stats


def main():
    parser = argparse.ArgumentParser(description="Process pseudo-label images with Upstage API")
    parser.add_argument("--dataset", choices=["cord-v2", "sroie", "wildreceipt", "all"], default="cord-v2", help="Dataset to process")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without processing")
    parser.add_argument("--convert-webp-only", action="store_true", help="Only convert images to WebP without API processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent workers")
    parser.add_argument("--output-dir", default="outputs/upstage_processed", help="Output directory for processed data")

    args = parser.parse_args()

    # Load configuration
    upstage_key = os.getenv("UPSTAGE_API_KEY")
    if not upstage_key:
        logger.error("UPSTAGE_API_KEY environment variable not set")
        return 1

    config = ProcessingConfig(
        upstage_api_key=upstage_key,
        input_base_dir="/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/pseudo_label",
        output_base_dir=args.output_dir,
        max_workers=args.max_workers,
    )

    processor = PseudoLabelProcessor(config)

    datasets = ["cord-v2", "sroie", "wildreceipt"] if args.dataset == "all" else [args.dataset]

    total_stats = {"total": 0, "processed": 0, "failed": 0}

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        stats = processor.process_dataset(dataset=dataset, limit=args.limit, dry_run=args.dry_run, convert_webp_only=args.convert_webp_only)

        for key in total_stats:
            total_stats[key] += stats[key]

        logger.info(f"Dataset {dataset}: {stats['processed']}/{stats['total']} processed, {stats['failed']} failed")

    logger.info(f"Total: {total_stats['processed']}/{total_stats['total']} processed, {total_stats['failed']} failed")

    if not args.dry_run and not args.convert_webp_only:
        cost_estimate = total_stats["processed"] * 0.01  # $0.01 per page
        logger.info(f"Estimated cost: ${cost_estimate:.2f} ({total_stats['processed']} pages × $0.01)")

    return 0


if __name__ == "__main__":
    exit(main())

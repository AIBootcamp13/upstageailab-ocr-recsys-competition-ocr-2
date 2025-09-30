from __future__ import annotations

"""Generate docTR preprocessing demo artifacts.

This utility synthesizes a simple document-style image, runs the
DocumentPreprocessor with docTR features enabled and disabled, and
exports before/after images alongside processing metadata. The outputs
are stored under ``docs/ai_handbook/02_protocols/outputs/doctr_preprocessing``.

Run with:
    uv run python scripts/generate_doctr_demo.py
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr.datasets.preprocessing import DOCTR_AVAILABLE, DocumentPreprocessor

OUTPUT_DIR = Path("docs/ai_handbook/02_protocols/outputs/doctr_preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("generate_doctr_demo")


def _synthesize_document(width: int = 720, height: int = 512) -> np.ndarray:
    """Create a synthetic document with skew, borders, and text blocks."""

    canvas = np.full((height, width, 3), 240, dtype=np.uint8)

    # Draw a dark border to mimic scanning artefacts
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (30, 30, 30), thickness=12)

    # Create a rotated rectangle representing the document
    rect_center = (int(width * 0.5), int(height * 0.5))
    rotation_matrix = cv2.getRotationMatrix2D(rect_center, angle=-8, scale=1.0)

    doc_mask = np.ones((height, width), dtype=np.uint8) * 255
    doc_mask = cv2.warpAffine(doc_mask, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    doc_mask = cv2.GaussianBlur(doc_mask, (3, 3), 0)

    base_doc = np.full_like(canvas, 252)
    canvas = np.where(doc_mask[..., None] > 128, base_doc, canvas)

    # Draw text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, text in enumerate(
        [
            "docTR Preprocessing Demo",
            "Orientation, rcrops, padding cleanup",
            "Synthetic content for UI previews",
            "Toggle preprocessing in the Streamlit app",
        ]
    ):
        position = (int(width * 0.18), int(height * 0.35) + idx * 40)
        cv2.putText(canvas, text, position, font, 0.9, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # Add a skewed signature block
    signature_box = np.array(
        [
            [int(width * 0.62), int(height * 0.72)],
            [int(width * 0.86), int(height * 0.67)],
            [int(width * 0.88), int(height * 0.82)],
            [int(width * 0.64), int(height * 0.86)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, signature_box, (220, 220, 220))
    cv2.polylines(canvas, [signature_box], True, (80, 80, 80), thickness=2, lineType=cv2.LINE_AA)

    return canvas


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f"Failed to persist image at {path}")


def _serialize_metadata(path: Path, metadata: dict[str, Any]) -> None:
    prepared = _convert_metadata(metadata)
    path.write_text(json.dumps(prepared, indent=2, sort_keys=True), encoding="utf-8")


def _convert_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return float(value)
    if isinstance(value, dict):
        return {key: _convert_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_metadata(item) for item in value]
    return value


def _run_preprocessor(enable_doctr: bool, image: np.ndarray) -> dict[str, Any]:
    kwargs = {
        "enable_document_detection": True,
        "enable_perspective_correction": True,
        "enable_enhancement": True,
        "enhancement_method": "office_lens",
        "enable_text_enhancement": True,
        "target_size": (640, 640),
        "enable_orientation_correction": enable_doctr,
        "orientation_angle_threshold": 1.0,
        "orientation_expand_canvas": True,
        "orientation_preserve_original_shape": False,
        "use_doctr_geometry": enable_doctr,
        "doctr_assume_horizontal": False,
        "enable_padding_cleanup": enable_doctr,
    }
    preprocessor = DocumentPreprocessor(**kwargs)
    return preprocessor(image)


def main() -> None:
    LOGGER.info("Generating docTR preprocessing demo artifacts (docTR available=%s)", DOCTR_AVAILABLE)

    synthetic = _synthesize_document()

    base_path = OUTPUT_DIR
    original_path = base_path / "demo_original.png"
    _save_image(original_path, synthetic)

    outputs: dict[str, dict[str, Any]] = {
        "original": {
            "path": str(original_path),
            "metadata": {
                "shape": list(synthetic.shape),
                "description": "Synthetic document used for docTR preprocessing demo",
            },
        }
    }

    for mode, enable_doctr in (("opencv", False), ("doctr", True)):
        LOGGER.info("Running preprocessor mode=%s", mode)
        try:
            result = _run_preprocessor(enable_doctr=enable_doctr, image=synthetic.copy())
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Preprocessor failed for mode=%s: %s", mode, exc)
            outputs[mode] = {"error": str(exc)}
            continue

        processed_image = result["image"]
        metadata = result.get("metadata", {})

        image_path = base_path / f"demo_{mode}.png"
        _save_image(image_path, processed_image)

        metadata_path = base_path / f"demo_{mode}.metadata.json"
        _serialize_metadata(metadata_path, metadata)

        outputs[mode] = {
            "path": str(image_path),
            "metadata_path": str(metadata_path),
            "metadata": _convert_metadata(metadata),
        }

    manifest_path = base_path / "manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info("Artifacts written to %s", base_path)


if __name__ == "__main__":
    main()

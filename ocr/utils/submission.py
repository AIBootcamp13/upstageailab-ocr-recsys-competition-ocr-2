"""Utilities for handling OCR prediction submissions."""

from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


class SubmissionWriter:
    """Handles formatting and saving OCR predictions to submission files.

    This class encapsulates the logic for converting prediction data into
    the expected JSON format and saving it to disk.
    """

    def __init__(self, config: Any):
        """Initialize the writer with configuration.

        Args:
            config: Configuration object containing paths and settings
        """
        self.config = config

    def save(self, predict_outputs: dict[str, Any]) -> None:
        """Save prediction outputs to a submission file.

        Args:
            predict_outputs: Dictionary mapping filenames to prediction data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = Path(f"{self.config.paths.submission_dir}") / f"{timestamp}.json"
        submission_file.parent.mkdir(parents=True, exist_ok=True)

        submission = self._format_submission(predict_outputs)

        with submission_file.open("w") as fp:
            if getattr(self.config, "minified_json", False):
                json.dump(submission, fp, indent=None, separators=(",", ":"))
            else:
                json.dump(submission, fp, indent=4)

    def _format_submission(self, predict_outputs: dict[str, Any]) -> dict[str, Any]:
        """Format prediction outputs into submission JSON structure.

        Args:
            predict_outputs: Raw prediction outputs from predict_step

        Returns:
            Formatted submission dictionary
        """
        submission: OrderedDict[str, Any] = OrderedDict(images=OrderedDict())
        include_confidence = getattr(self.config, "include_confidence", False)

        for filename, pred_data in predict_outputs.items():
            if include_confidence and isinstance(pred_data, dict):
                boxes = pred_data["boxes"]
                scores = pred_data["scores"]
            else:
                boxes = pred_data
                scores = None

            # Format each word/box
            words = OrderedDict()
            for idx, box in enumerate(boxes):
                points = box.tolist() if hasattr(box, "tolist") else box
                word_data = OrderedDict(points=points)
                if include_confidence and scores is not None:
                    word_data["confidence"] = float(scores[idx])
                words[f"{idx + 1:04}"] = word_data

            # Add to submission
            submission["images"][filename] = OrderedDict(words=words)

        return submission

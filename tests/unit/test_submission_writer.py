"""Unit tests for SubmissionWriter service.

Tests CSV and JSON output format compatibility with competition requirements.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from ui.apps.inference.models.data_contracts import InferenceResult, Predictions, PreprocessingInfo
from ui.apps.inference.services.submission_writer import SubmissionEntry, SubmissionWriter


@pytest.fixture
def sample_results():
    """Create sample inference results for testing."""
    # Create a small dummy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    return [
        InferenceResult(
            filename="image001.jpg",
            success=True,
            image=dummy_image,
            predictions=Predictions(
                # Competition format: space-separated coordinates
                polygons="10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250",
                texts=["Sample", "Text"],
                confidences=[0.95, 0.87],
            ),
            preprocessing=PreprocessingInfo(enabled=False),
        ),
        InferenceResult(
            filename="image002.jpg",
            success=True,
            image=dummy_image,
            predictions=Predictions(
                polygons="20 60 120 60 120 160 20 160",
                texts=["Another"],
                confidences=[0.92],
            ),
            preprocessing=PreprocessingInfo(enabled=False),
        ),
        InferenceResult(
            filename="image003.jpg",
            success=False,
            image=dummy_image,
            predictions=Predictions(polygons="", texts=[], confidences=[]),
            preprocessing=PreprocessingInfo(enabled=False),
            error="Failed to process",
        ),
    ]


def test_submission_entry_model():
    """Test SubmissionEntry Pydantic model."""
    entry = SubmissionEntry(
        filename="test.jpg",
        polygons="10 10 90 10 90 90 10 90",  # Space-separated coordinates
        confidence=0.95,
    )
    assert entry.filename == "test.jpg"
    assert entry.polygons == "10 10 90 10 90 90 10 90"
    assert entry.confidence == 0.95

    # Test without confidence
    entry_no_conf = SubmissionEntry(
        filename="test.jpg",
        polygons="10 10 90 10 90 90 10 90",
    )
    assert entry_no_conf.confidence is None


def test_submission_entry_to_dict():
    """Test SubmissionEntry to_dict conversion."""
    # With confidence
    entry = SubmissionEntry(
        filename="test.jpg",
        polygons="10 10 90 10 90 90 10 90",  # Space-separated
        confidence=0.95,
    )
    result = entry.to_dict()
    assert result == {
        "filename": "test.jpg",
        "polygons": "10 10 90 10 90 90 10 90",
        "confidence": 0.95,
    }

    # Without confidence
    entry_no_conf = SubmissionEntry(
        filename="test.jpg",
        polygons="10 10 90 10 90 90 10 90",
    )
    result_no_conf = entry_no_conf.to_dict()
    assert result_no_conf == {
        "filename": "test.jpg",
        "polygons": "10 10 90 10 90 90 10 90",
    }


def test_write_csv_format(sample_results):
    """Test CSV output matches competition format: filename,polygons[,confidence]."""
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.csv"

        # Test without confidence
        SubmissionWriter.write_csv(sample_results, output_path, include_confidence=False)

        # Read and verify format
        with output_path.open("r") as f:
            lines = f.readlines()

        assert len(lines) == 4  # Header + 3 results
        assert lines[0].strip() == "filename,polygons"
        assert lines[1].startswith("image001.jpg,")
        assert lines[2].startswith("image002.jpg,")
        assert lines[3].startswith("image003.jpg,")  # Failed result with empty polygons

        # Verify polygons are preserved (space-separated)
        assert "10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250" in lines[1]


def test_write_csv_with_confidence(sample_results):
    """Test CSV output with confidence scores as third column."""
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output_conf.csv"

        SubmissionWriter.write_csv(sample_results, output_path, include_confidence=True)

        with output_path.open("r") as f:
            lines = f.readlines()

        assert len(lines) == 4  # Header + 3 results
        assert lines[0].strip() == "filename,polygons,confidence"

        # Check confidence values (average of [0.95, 0.87] = 0.91)
        parts1 = lines[1].strip().split(",")
        assert len(parts1) >= 3
        assert parts1[0] == "image001.jpg"
        assert float(parts1[-1]) == pytest.approx(0.91, rel=0.01)

        # Check second image (confidence 0.92)
        parts2 = lines[2].strip().split(",")
        assert float(parts2[-1]) == pytest.approx(0.92, rel=0.01)

        # Check failed result has empty confidence
        parts3 = lines[3].strip().split(",")
        assert parts3[-1] == ""  # Empty confidence for failed result


def test_write_json_format(sample_results):
    """Test JSON output format."""
    import json

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.json"

        SubmissionWriter.write_json(sample_results, output_path, include_confidence=False)

        with output_path.open("r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 3

        # Check first entry (space-separated coordinates)
        assert data[0]["filename"] == "image001.jpg"
        assert data[0]["polygons"] == "10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250"
        assert "confidence" not in data[0]  # No confidence when not requested

        # Check failed entry
        assert data[2]["filename"] == "image003.jpg"
        assert data[2]["polygons"] == ""


def test_write_json_with_confidence(sample_results):
    """Test JSON output with confidence scores."""
    import json

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output_conf.json"

        SubmissionWriter.write_json(sample_results, output_path, include_confidence=True)

        with output_path.open("r") as f:
            data = json.load(f)

        # Check confidence values
        assert "confidence" in data[0]
        assert data[0]["confidence"] == pytest.approx(0.91, rel=0.01)  # Average of [0.95, 0.87]

        assert "confidence" in data[1]
        assert data[1]["confidence"] == pytest.approx(0.92, rel=0.01)

        # Failed result should not have confidence
        assert "confidence" not in data[2] or data[2]["confidence"] is None


def test_write_batch_results(sample_results):
    """Test write_batch_results writes both formats correctly."""
    with TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "output.json"
        csv_path = Path(tmpdir) / "output.csv"

        written_files = SubmissionWriter.write_batch_results(
            sample_results,
            json_path=json_path,
            csv_path=csv_path,
            include_confidence=True,
        )

        assert "json" in written_files
        assert "csv" in written_files
        assert written_files["json"] == json_path
        assert written_files["csv"] == csv_path
        assert json_path.exists()
        assert csv_path.exists()


def test_generate_summary_stats(sample_results):
    """Test summary statistics generation."""
    stats = SubmissionWriter.generate_summary_stats(sample_results)

    assert stats["total_images"] == 3
    assert stats["successful"] == 2
    assert stats["failed"] == 1
    assert stats["success_rate"] == "66.7%"
    assert stats["total_polygons_detected"] == 3  # 2 from first + 1 from second
    assert stats["total_texts_detected"] == 3  # 2 + 1
    assert stats["avg_polygons_per_image"] == "1.5"  # 3 polygons / 2 successful

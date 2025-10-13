"""Contract compliance tests for preprocessing module."""

import numpy as np
import pytest

from ocr.datasets.preprocessing.contracts import (
    ContractEnforcer,
    DetectionResultContract,
    ImageInputContract,
    PreprocessingResultContract,
    validate_image_input_with_fallback,
)


class TestContractCompliance:
    """Test contract compliance across all interfaces."""

    def test_image_input_contract_valid(self):
        """Test ImageInputContract with valid inputs."""
        valid_images = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8),
            np.random.randint(0, 255, (200, 300, 4), dtype=np.uint8),
        ]

        for image in valid_images:
            contract = ImageInputContract(image=image)
            assert contract.image is image

    def test_image_input_contract_invalid(self):
        """Test ImageInputContract with invalid inputs."""
        invalid_images = [
            np.array([]),  # Empty array
            np.array([1, 2, 3]),  # 1D array
            np.random.randint(0, 255, (100,)),  # 1D array
            "not an array",  # Wrong type
            None,  # None
        ]

        for image in invalid_images:
            with pytest.raises(ValueError):
                ImageInputContract(image=image)

    def test_preprocessing_result_contract_valid(self):
        """Test PreprocessingResultContract with valid results."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        metadata = {"steps": ["enhancement"], "processing_time": 0.5}

        contract = PreprocessingResultContract(image=image, metadata=metadata)
        assert contract.image is image
        assert contract.metadata == metadata

    def test_detection_result_contract_valid(self):
        """Test DetectionResultContract with valid results."""
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])

        # Test with all fields
        contract = DetectionResultContract(corners=corners, confidence=0.95, method="test_method")
        assert contract.corners is not None and np.array_equal(contract.corners, corners)
        assert contract.confidence == 0.95
        assert contract.method == "test_method"

        # Test with minimal fields
        contract = DetectionResultContract(corners=corners)
        assert contract.corners is not None and np.array_equal(contract.corners, corners)
        assert contract.confidence is None
        assert contract.method is None

    def test_contract_enforcer_validation(self):
        """Test ContractEnforcer validation methods."""
        valid_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = {"image": valid_image, "metadata": {"test": True}}

        # Should not raise
        ContractEnforcer.validate_image_input_contract(valid_image)
        ContractEnforcer.validate_preprocessing_result_contract(result)

        # Should raise for invalid inputs
        with pytest.raises(ValueError):
            ContractEnforcer.validate_image_input_contract(np.array([]))

        with pytest.raises(ValueError):
            ContractEnforcer.validate_preprocessing_result_contract({"invalid": True})

    def test_validation_decorator_fallback(self):
        """Test validation decorator fallback behavior."""

        class TestProcessor:
            @validate_image_input_with_fallback
            def process(self, image):
                return {"image": image, "metadata": {"processed": True}}

        processor = TestProcessor()

        # Valid input should work normally
        valid_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = processor.process(valid_image)
        assert result["metadata"]["processed"] is True

        # Invalid input should trigger fallback
        invalid_image = np.array([])
        result = processor.process(invalid_image)
        assert "error" in result["metadata"]
        assert result["metadata"]["processing_steps"] == ["fallback"]


if __name__ == "__main__":
    pytest.main([__file__])

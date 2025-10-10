import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from scripts.preprocess_maps import preprocess, validate_generated_maps


class TestPreprocessFunction:
    """Tests for the preprocess function in preprocess_maps.py"""

    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_image_path = Path(self.temp_dir) / "images"
        self.mock_image_path.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_validation_missing_datasets(self):
        """Test that function raises error when 'datasets' key is missing from config"""
        cfg = DictConfig({})
        dataset_key = "train_dataset"

        with pytest.raises(ValueError, match="Dataset 'train_dataset' not found in config.datasets"):
            preprocess(cfg, dataset_key)

    def test_config_validation_missing_dataset_key(self):
        """Test that function raises error when specific dataset key is missing"""
        cfg = DictConfig({"datasets": {}, "collate_fn": {}})
        dataset_key = "train_dataset"

        with pytest.raises(ValueError, match="Dataset 'train_dataset' not found in config.datasets"):
            preprocess(cfg, dataset_key)

    def test_config_validation_missing_collate_fn(self):
        """Test that function raises error when 'collate_fn' key is missing from config"""
        cfg = DictConfig({"datasets": {"train_dataset": {}}})
        dataset_key = "train_dataset"

        with pytest.raises(ValueError, match="collate_fn not found in config"):
            preprocess(cfg, dataset_key)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_instantiate_dataset_and_collate_fn(self, mock_instantiate):
        """Test that dataset and collate function are instantiated correctly"""
        # Mock the OCRDataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=2)

        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Verify instantiation calls
        assert mock_instantiate.call_count == 2
        mock_instantiate.assert_any_call(cfg.datasets[dataset_key])
        mock_instantiate.assert_any_call(cfg.collate_fn)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_output_directory_creation(self, mock_instantiate):
        """Test that output directory is correctly created"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Check that output directory was created
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        assert expected_output_dir.exists()

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_successful_preprocessing(self, mock_instantiate):
        """Test successful preprocessing of valid samples"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=2)

        # Create different samples for each index request
        def get_sample(index):
            if index == 0:
                return {
                    "image": torch.rand(3, 224, 224),
                    "image_filename": f"test_image_{index}.jpg",
                    "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
                }
            elif index == 1:
                return {
                    "image": torch.rand(3, 224, 224),
                    "image_filename": f"test_image_{index}.jpg",
                    "polygons": [np.array([[[30, 30], [40, 30], [40, 40], [30, 40]]])],
                }
            else:
                raise IndexError("Index out of range")

        mock_dataset.__getitem__ = Mock(side_effect=get_sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Check that both samples were processed successfully
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 2
        assert mock_collate_fn.make_prob_thresh_map.call_count == 2

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_samples_with_no_polygons_skipped(self, mock_instantiate):
        """Test that samples with no valid polygons are skipped"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=2)

        # Create samples: one with polygons, one without
        def get_sample(index):
            if index == 0:
                return {
                    "image": torch.rand(3, 224, 224),
                    "image_filename": f"test_image_{index}.jpg",
                    "polygons": [],  # No polygons
                }
            elif index == 1:
                return {
                    "image": torch.rand(3, 224, 224),
                    "image_filename": f"test_image_{index}.jpg",
                    "polygons": [np.array([[[30, 30], [40, 30], [40, 40], [30, 40]]])],  # Has polygons
                }
            else:
                raise IndexError("Index out of range")

        mock_dataset.__getitem__ = Mock(side_effect=get_sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Only the sample with polygons should be processed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1  # Only one file since one sample had no polygons
        assert mock_collate_fn.make_prob_thresh_map.call_count == 1

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_invalid_image_types_handled(self, mock_instantiate):
        """Test that invalid image types are handled with proper error"""
        # Mock the dataset
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        sample = {
            "image": "invalid_image_type",
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)  # Should handle the error gracefully and continue

        # No output files should be created since image processing failed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 0

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_limit_samples_config(self, mock_instantiate):
        """Test that sample limit is respected when specified in config"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=10)  # Dataset has 10 samples

        def get_sample(index):
            return {
                "image": torch.rand(3, 224, 224),
                "image_filename": f"test_image_{index}.jpg",
                "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
            }

        mock_dataset.__getitem__ = Mock(side_effect=get_sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {
                    "train_num_samples": 3  # Limit to 3 samples
                },
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Only the limited number of samples should be processed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 3
        assert mock_collate_fn.make_prob_thresh_map.call_count == 3

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_sample_limit_larger_than_dataset(self, mock_instantiate):
        """Test that when limit is larger than dataset size, all samples are processed"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=2)  # Dataset has 2 samples

        def get_sample(index):
            return {
                "image": torch.rand(3, 224, 224),
                "image_filename": f"test_image_{index}.jpg",
                "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
            }

        mock_dataset.__getitem__ = Mock(side_effect=get_sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {
                    "train_num_samples": 10  # Limit to 10 samples (more than dataset size)
                },
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # All dataset samples should be processed (not more than dataset size)
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 2  # Same as dataset size
        assert mock_collate_fn.make_prob_thresh_map.call_count == 2

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_numpy_image_conversion(self, mock_instantiate):
        """Test processing of numpy array images"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        # Create a sample with numpy array image
        numpy_image = np.random.rand(224, 224, 3)  # HWC format
        sample = {
            "image": numpy_image,
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Verify that the numpy image was properly converted and processed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1
        assert mock_collate_fn.make_prob_thresh_map.call_count == 1

        # Check that the maps were made with the expected shape
        args, kwargs = mock_collate_fn.make_prob_thresh_map.call_args
        # The image should be converted to tensor format (C, H, W)
        assert args[0].shape == (3, 224, 224)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_grayscale_numpy_image_conversion(self, mock_instantiate):
        """Test processing of grayscale numpy array images"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        # Create a sample with grayscale numpy array image (H, W)
        grayscale_image = np.random.rand(224, 224)  # HW format
        sample = {
            "image": grayscale_image,
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Verify that the grayscale numpy image was properly converted and processed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1
        assert mock_collate_fn.make_prob_thresh_map.call_count == 1

        # Check that the maps were made with the expected shape
        args, kwargs = mock_collate_fn.make_prob_thresh_map.call_args
        # The grayscale image should be converted to tensor format (1, H, W)
        assert args[0].shape == (1, 224, 224)


class TestPreprocessFunctionErrorHandling:
    """Tests for error handling in the preprocess function"""

    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_image_path = Path(self.temp_dir) / "images"
        self.mock_image_path.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_dataset_initialization_failure(self, mock_instantiate):
        """Test that initialization failures are properly handled"""
        # Mock the instantiation to raise an exception
        mock_instantiate.side_effect = Exception("Instantiation failed")

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        with pytest.raises(Exception, match="Instantiation failed"):
            preprocess(cfg, dataset_key)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_error_processing_sample_continues(self, mock_instantiate):
        """Test that when an error occurs during sample processing, it continues with other samples"""
        # Create a dataset that has one good sample and one that causes an error
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=2)

        def get_sample(index):
            if index == 0:
                # Good sample
                return {
                    "image": torch.rand(3, 224, 224),
                    "image_filename": f"test_image_{index}.jpg",
                    "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
                }
            elif index == 1:
                # Sample that will cause an error during processing
                return {
                    "image": "invalid_type",
                    "image_filename": f"test_image_{index}.jpg",
                    "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
                }
            else:
                raise IndexError("Index out of range")

        mock_dataset.__getitem__ = Mock(side_effect=get_sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)  # Should not raise an exception

        # Only the good sample should be processed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        # Only 1 file should be created since the second sample caused an error
        assert len(output_files) == 1
        # Only the first sample should have called make_prob_thresh_map
        assert mock_collate_fn.make_prob_thresh_map.call_count == 1

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_map_generation_failure_continues(self, mock_instantiate):
        """Test that when map generation fails for a sample, processing continues"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=2)

        def get_sample(index):
            return {
                "image": torch.rand(3, 224, 224),
                "image_filename": f"test_image_{index}.jpg",
                "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
            }

        mock_dataset.__getitem__ = Mock(side_effect=get_sample)

        # Make the first call to make_prob_thresh_map succeed, second fail
        call_count = 0

        def mock_make_prob_thresh_map(image_tensor, valid_polygons, filename):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                return {"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
            else:
                # Second call raises an exception
                raise Exception("Map generation failed")

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(side_effect=mock_make_prob_thresh_map)

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)  # Should not raise an exception

        # Only the first sample should be processed successfully
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        # Only 1 file should be created since the second sample caused an error in map generation
        assert len(output_files) == 1
        assert mock_collate_fn.make_prob_thresh_map.call_count == 2  # Called for both samples

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_sanity_check_on_generated_files(self, mock_instantiate):
        """Test that the sanity check is performed on generated files"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={
                "prob_map": np.random.rand(224, 224),  # 2D array
                "thresh_map": np.random.rand(224, 224),  # 2D array
            }
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Verify preprocessing completed successfully
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1

    def test_non_iterable_polygons_handled(self):
        """Test that non-iterable polygons are handled with a warning"""
        with patch("scripts.preprocess_maps.hydra.utils.instantiate") as mock_instantiate:
            # Mock the dataset and collate function
            mock_dataset = Mock()
            mock_dataset.image_path = str(self.mock_image_path)
            mock_dataset.__len__ = Mock(return_value=1)

            # Sample with non-iterable polygons
            sample = {
                "image": torch.rand(3, 224, 224),
                "image_filename": "test_image.jpg",
                "polygons": "not_an_iterable",  # This is not iterable
            }
            mock_dataset.__getitem__ = Mock(return_value=sample)

            mock_collate_fn = Mock()
            # make_prob_thresh_map should not be called because polygons are invalid
            mock_collate_fn.make_prob_thresh_map = Mock(
                return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
            )

            mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

            cfg = DictConfig(
                {
                    "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                    "collate_fn": {"_target_": "some.collate.function"},
                    "data": {},
                }
            )

            dataset_key = "train_dataset"

            preprocess(cfg, dataset_key)

            # No files should be created because the polygons are not iterable
            expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
            output_files = list(expected_output_dir.glob("*.npz"))
            assert len(output_files) == 0
            # make_prob_thresh_map should not be called due to invalid polygons
            mock_collate_fn.make_prob_thresh_map.assert_not_called()


class TestPreprocessConfigValidation:
    """Tests for config validation in the preprocess function"""

    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_image_path = Path(self.temp_dir) / "images"
        self.mock_image_path.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_missing_datasets_key(self):
        """Test that the function raises an error when datasets key is missing"""
        cfg = DictConfig({})
        dataset_key = "train_dataset"

        with pytest.raises(ValueError, match="Dataset 'train_dataset' not found in config.datasets"):
            preprocess(cfg, dataset_key)

    def test_config_missing_specific_dataset(self):
        """Test that the function raises an error when specific dataset key is missing"""
        cfg = DictConfig(
            {"datasets": {"val_dataset": {"_target_": "some.dataset.class"}}, "collate_fn": {"_target_": "some.collate.class"}}
        )
        dataset_key = "train_dataset"  # This key doesn't exist in datasets

        with pytest.raises(ValueError, match="Dataset 'train_dataset' not found in config.datasets"):
            preprocess(cfg, dataset_key)

    def test_config_missing_collate_fn(self):
        """Test that the function raises an error when collate_fn is missing"""
        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "some.dataset.class"}}
                # Missing collate_fn
            }
        )
        dataset_key = "train_dataset"

        with pytest.raises(ValueError, match="collate_fn not found in config"):
            preprocess(cfg, dataset_key)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_valid_config_structure_passes(self, mock_instantiate):
        """Test that valid config structure processes successfully"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        # Should not raise any exceptions
        preprocess(cfg, dataset_key)

        # Verify output files were created
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1


class TestPreprocessEdgeCases:
    """Tests for edge cases in the preprocess function"""

    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_image_path = Path(self.temp_dir) / "images"
        self.mock_image_path.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_dataset_with_zero_samples(self, mock_instantiate):
        """Test preprocessing when dataset has zero samples"""
        # Mock the dataset with zero length
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=0)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # No files should be created since no samples to process
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 0

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_dataset_with_invalid_numpy_shapes(self, mock_instantiate):
        """Test handling of numpy arrays with invalid shapes"""
        # Mock the dataset with image that has an invalid shape
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        # Create a sample with a numpy array that has an unsupported shape
        invalid_shape_image = np.random.rand(5, 3)  # 2D array, but not image format
        sample = {
            "image": invalid_shape_image,
            "image_filename": "test_image.jpg",
            "polygons": [np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]])],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)  # Should handle the error and continue

        # No output files should be created since image processing failed
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 0

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_samples_with_empty_polygons_list(self, mock_instantiate):
        """Test that samples with empty polygon lists are skipped"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [],  # Empty polygons list
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # No output files should be created since there are no valid polygons
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 0

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_polygons_with_invalid_ndim(self, mock_instantiate):
        """Test that polygons with invalid dimensions are filtered out"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        # Sample with polygons that have invalid dimensions (not 3D)
        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [
                np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),  # 2D array, not 3D
                np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]]),  # Valid 3D array
            ],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Should process successfully with only the valid polygon
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1

    @patch("scripts.preprocess_maps.hydra.utils.instantiate")
    def test_polygons_with_insufficient_vertices(self, mock_instantiate):
        """Test that polygons with insufficient vertices are filtered out"""
        # Mock the dataset and collate function
        mock_dataset = Mock()
        mock_dataset.image_path = str(self.mock_image_path)
        mock_dataset.__len__ = Mock(return_value=1)

        # Sample with polygons that have insufficient vertices (< 3 points)
        sample = {
            "image": torch.rand(3, 224, 224),
            "image_filename": "test_image.jpg",
            "polygons": [
                np.array([[[10, 10], [20, 10]]]),  # Only 2 vertices, need at least 3
                np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]]),  # Valid 4 vertices
            ],
        }
        mock_dataset.__getitem__ = Mock(return_value=sample)

        mock_collate_fn = Mock()
        mock_collate_fn.make_prob_thresh_map = Mock(
            return_value={"prob_map": np.random.rand(224, 224), "thresh_map": np.random.rand(224, 224)}
        )

        mock_instantiate.side_effect = [mock_dataset, mock_collate_fn]

        cfg = DictConfig(
            {
                "datasets": {"train_dataset": {"_target_": "ocr.datasets.base.OCRDataset"}},
                "collate_fn": {"_target_": "some.collate.function"},
                "data": {},
            }
        )

        dataset_key = "train_dataset"

        preprocess(cfg, dataset_key)

        # Should process successfully with only the valid polygon
        expected_output_dir = Path(mock_dataset.image_path).parent / f"{Path(mock_dataset.image_path).name}_maps"
        output_files = list(expected_output_dir.glob("*.npz"))
        assert len(output_files) == 1


class TestValidateGeneratedMaps:
    """Tests for the validate_generated_maps function"""

    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "maps"
        self.output_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_maps_pass_validation(self):
        """Test that valid maps pass validation"""
        # Create valid map files
        prob_map = np.random.rand(1, 100, 100).astype(np.float32)
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)

        np.savez(self.output_dir / "test1.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Should not raise an exception
        validate_generated_maps(self.output_dir, 1)

    def test_invalid_shape_fails_validation(self):
        """Test that maps with invalid shapes fail validation"""
        # Create invalid map (missing channel dimension)
        prob_map = np.random.rand(100, 100).astype(np.float32)
        thresh_map = np.random.rand(100, 100).astype(np.float32)

        np.savez(self.output_dir / "test1.npz", prob_map=prob_map, thresh_map=thresh_map)

        with pytest.raises(ValueError, match="should be 3D"):
            validate_generated_maps(self.output_dir, 1)

    def test_mismatched_shapes_fail_validation(self):
        """Test that maps with mismatched shapes fail validation"""
        # Create maps with different shapes
        prob_map = np.random.rand(1, 100, 100).astype(np.float32)
        thresh_map = np.random.rand(1, 50, 50).astype(np.float32)

        np.savez(self.output_dir / "test1.npz", prob_map=prob_map, thresh_map=thresh_map)

        with pytest.raises(ValueError, match="prob_map.*!=.*thresh_map"):
            validate_generated_maps(self.output_dir, 1)

    def test_nan_values_fail_validation(self):
        """Test that maps with NaN values fail validation"""
        prob_map = np.full((1, 100, 100), np.nan, dtype=np.float32)
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)

        np.savez(self.output_dir / "test1.npz", prob_map=prob_map, thresh_map=thresh_map)

        with pytest.raises(ValueError, match="NaN"):
            validate_generated_maps(self.output_dir, 1)

    def test_out_of_range_values_fail_validation(self):
        """Test that maps with out-of-range values fail validation"""
        prob_map = np.full((1, 100, 100), 2.0, dtype=np.float32)  # > 1.0
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)

        np.savez(self.output_dir / "test1.npz", prob_map=prob_map, thresh_map=thresh_map)

        with pytest.raises(ValueError, match="out of range"):
            validate_generated_maps(self.output_dir, 1)

    def test_missing_file_fails_validation(self):
        """Test that missing files fail validation"""
        with pytest.raises(ValueError, match="Expected 1 map files, found 0"):
            validate_generated_maps(self.output_dir, 1)

    def test_wrong_dtype_fails_validation(self):
        """Test that maps with wrong dtype fail validation"""
        prob_map = np.random.rand(1, 100, 100).astype(np.float64)  # Wrong dtype
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)

        np.savez(self.output_dir / "test1.npz", prob_map=prob_map, thresh_map=thresh_map)

        with pytest.raises(ValueError, match="should be float32"):
            validate_generated_maps(self.output_dir, 1)

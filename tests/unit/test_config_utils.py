"""Unit tests for config_utils module."""

import numpy as np
from omegaconf import OmegaConf

from ocr.lightning_modules.utils.config_utils import extract_metric_kwargs, extract_normalize_stats


class TestExtractMetricKwargs:
    """Tests for extract_metric_kwargs function."""

    def test_none_input_returns_empty_dict(self):
        """Test that None input returns an empty dictionary."""
        result = extract_metric_kwargs(None)
        assert result == {}

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns an empty dictionary."""
        config = OmegaConf.create({})
        result = extract_metric_kwargs(config)
        assert result == {}

    def test_config_with_target_field_excludes_target(self):
        """Test that _target_ field is excluded from the result."""
        config = OmegaConf.create({"_target_": "some.target.class", "param1": "value1", "param2": 42})
        result = extract_metric_kwargs(config)
        expected = {"param1": "value1", "param2": 42}
        assert result == expected

    def test_config_without_target_field_preserves_all_fields(self):
        """Test that config without _target_ preserves all fields."""
        config = OmegaConf.create({"param1": "value1", "param2": 42, "param3": [1, 2, 3]})
        result = extract_metric_kwargs(config)
        expected = {"param1": "value1", "param2": 42, "param3": [1, 2, 3]}
        assert result == expected

    def test_non_dict_container_returns_empty_dict(self):
        """Test that non-dict container returns empty dict."""
        config = OmegaConf.create([1, 2, 3])  # This creates a ListConfig
        result = extract_metric_kwargs(config)
        assert result == {}

    def test_nested_config_flattens_to_dict(self):
        """Test that nested config is properly flattened to a dict."""
        config = OmegaConf.create({"level1": {"level2": "value", "num": 123}, "top_level": "test"})
        result = extract_metric_kwargs(config)
        expected = {"level1": {"level2": "value", "num": 123}, "top_level": "test"}
        assert result == expected


class TestExtractNormalizeStats:
    """Tests for extract_normalize_stats function."""

    def test_none_config_returns_none_tuple(self):
        """Test that None config returns (None, None)."""
        result = extract_normalize_stats(None)
        assert result == (None, None)

    def test_config_without_transforms_returns_none_tuple(self):
        """Test that config without transforms returns (None, None)."""
        config = OmegaConf.create({"other_field": "value"})
        result = extract_normalize_stats(config)
        assert result == (None, None)

    def test_config_with_empty_transforms_returns_none_tuple(self):
        """Test that config with empty transforms returns (None, None)."""
        config = OmegaConf.create({"transforms": {}})
        result = extract_normalize_stats(config)
        assert result == (None, None)

    def test_normalize_transform_with_valid_mean_std_returns_arrays(self):
        """Test that valid Normalize transform returns mean and std arrays."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]}
                        ]
                    }
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_normalize_transform_with_single_channel_values(self):
        """Test that Normalize transform works with single channel values."""
        config = OmegaConf.create(
            {"transforms": {"val_transform": {"transforms": [{"_target_": "albumentations.Normalize", "mean": [0.5], "std": [0.5]}]}}}
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5], dtype=np.float32)
        expected_std = np.array([0.5], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_non_normalize_transform_ignored(self):
        """Test that non-Normalize transforms are ignored."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [
                            {"_target_": "albumentations.Resize", "height": 224, "width": 224},
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]},
                        ]
                    }
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_normalize_transform_missing_mean_or_std_ignored(self):
        """Test that Normalize transform without mean or std is ignored."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [
                            {
                                "_target_": "albumentations.Normalize",
                                "mean": [0.5900, 0.5610, 0.5292],
                                # Missing std
                            },
                            {
                                "_target_": "albumentations.Normalize",
                                "std": [0.2136, 0.2290, 0.2511],
                                # Missing mean
                            },
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]},
                        ]
                    }
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_normalize_transform_with_mismatched_mean_std_sizes_ignored(self):
        """Test that Normalize transform with mismatched mean/std sizes is ignored."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [
                            {
                                "_target_": "albumentations.Normalize",
                                "mean": [0.485, 0.456],  # 2 elements
                                "std": [0.2136, 0.2290, 0.2511],  # 3 elements
                            },
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]},
                        ]
                    }
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_normalize_transform_with_invalid_array_conversion_ignored(self):
        """Test that Normalize transform with invalid values is ignored."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [
                            {"_target_": "albumentations.Normalize", "mean": ["invalid", "values"], "std": [0.2136, 0.2290, 0.2511]},
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]},
                        ]
                    }
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_normalize_transform_with_4_channels_ignored(self):
        """Test that Normalize transform with 4 channels (not 1 or 3) is ignored."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [
                            {
                                "_target_": "albumentations.Normalize",
                                "mean": [0.485, 0.456, 0.406, 0.5],  # 4 channels
                                "std": [0.229, 0.224, 0.225, 0.5],  # 4 channels
                            },
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]},
                        ]
                    }
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_all_transform_types_searched(self):
        """Test that all transform types (train, val, test, predict) are searched."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {"transforms": []},
                    "val_transform": {"transforms": []},
                    "test_transform": {
                        "transforms": [
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]}
                        ]
                    },
                    "predict_transform": {"transforms": []},
                }
            }
        )
        result = extract_normalize_stats(config)
        expected_mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
        expected_std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

    def test_first_valid_normalize_transform_returned(self):
        """Test that the first valid Normalize transform is returned."""
        config = OmegaConf.create(
            {
                "transforms": {
                    "train_transform": {
                        "transforms": [{"_target_": "albumentations.Normalize", "mean": [0.1, 0.1, 0.1], "std": [0.1, 0.1, 0.1]}]
                    },
                    "val_transform": {
                        "transforms": [
                            {"_target_": "albumentations.Normalize", "mean": [0.5900, 0.5610, 0.5292], "std": [0.2136, 0.2290, 0.2511]}
                        ]
                    },
                }
            }
        )
        result = extract_normalize_stats(config)
        # Should return the first one found (from train_transform)
        expected_mean = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        expected_std = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        assert np.array_equal(result[0], expected_mean)
        assert np.array_equal(result[1], expected_std)

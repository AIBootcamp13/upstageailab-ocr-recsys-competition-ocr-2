import os
import tempfile
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from ocr.datasets.base import OCRDataset
from ocr.datasets.db_collate_fn import DBCollateFN


def test_ocr_dataset_initialization():
    """Test OCRDataset initialization with and without annotations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock image file
        img_path = os.path.join(temp_dir, "test.jpg")
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        # Create a mock transform
        transform = Mock()
        transform.return_value = {"image": torch.rand(3, 100, 100), "polygons": [], "inverse_matrix": np.eye(3)}

        # Test initialization without annotations
        dataset = OCRDataset(image_path=temp_dir, annotation_path=None, transform=transform)
        assert len(dataset) == 1
        assert list(dataset.anns.keys())[0] == "test.jpg"

        # Test initialization with annotation file
        ann_path = os.path.join(temp_dir, "annotations.json")
        with open(ann_path, "w") as f:
            f.write('{"images": {"test.jpg": {"words": {"word1": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}}}}}')

        dataset_with_anns = OCRDataset(image_path=temp_dir, annotation_path=ann_path, transform=transform)
        assert len(dataset_with_anns) == 1


def test_ocr_dataset_getitem_map_loading():
    """Test loading of pre-processed probability and threshold maps in OCRDataset"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a subdirectory to match the expected structure
        sub_dir = os.path.join(temp_dir, "test_images")
        os.makedirs(sub_dir)

        # Create a mock image file
        img_path = os.path.join(sub_dir, "test.jpg")
        img = Image.new("RGB", (200, 100))
        img.save(img_path)

        # Create maps directory and .npz file
        maps_dir = os.path.join(temp_dir, "test_images_maps")  # Follows the naming convention: {image_dir_name}_maps
        os.makedirs(maps_dir)
        map_path = os.path.join(maps_dir, "test.npz")  # Uses stem of image filename without extension

        # Create mock probability and threshold maps
        prob_map = np.random.rand(100, 200).astype(np.float32)
        thresh_map = np.random.rand(100, 200).astype(np.float32)
        np.savez_compressed(map_path, prob_map=prob_map, thresh_map=thresh_map)

        # Create a mock transform
        transform = Mock()
        transform.return_value = {"image": torch.rand(3, 100, 200), "polygons": [], "inverse_matrix": np.eye(3)}

        # Create dataset with image path that contains maps
        dataset = OCRDataset(image_path=sub_dir, annotation_path=None, transform=transform, load_maps=True)

        # Get item and check if maps were loaded
        item = dataset[0]
        assert "prob_map" in item
        assert "thresh_map" in item
        np.testing.assert_array_equal(item["prob_map"], prob_map)
        np.testing.assert_array_equal(item["thresh_map"], thresh_map)


def test_ocr_dataset_getitem_fallback():
    """Test fallback when maps are not found"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock image file
        img_path = os.path.join(temp_dir, "test.jpg")
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        # Create a mock transform
        transform = Mock()
        transform.return_value = {"image": torch.rand(3, 100, 100), "polygons": [], "inverse_matrix": np.eye(3)}

        # Create dataset with image path that does not contain maps
        dataset = OCRDataset(image_path=temp_dir, annotation_path=None, transform=transform)

        # Get item and check that maps are not loaded
        item = dataset[0]
        # Maps would be handled by collate function in this case
        assert "image" in item
        assert "image_filename" in item
        assert item["image_filename"] == "test.jpg"


def test_dbcollatefn_init():
    """Test DBCollateFN initialization"""
    collate_fn = DBCollateFN(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)
    assert collate_fn.shrink_ratio == 0.4
    assert collate_fn.thresh_min == 0.3
    assert collate_fn.thresh_max == 0.7
    assert not collate_fn.inference_mode


def test_dbcollatefn_inference_mode():
    """Test DBCollateFN in inference mode"""
    collate_fn = DBCollateFN()
    collate_fn.inference_mode = True

    # Create mock batch data without maps
    batch = []
    for i in range(2):
        batch_item = OrderedDict(
            image=torch.rand(3, 100, 100),
            image_filename=f"test_{i}.jpg",
            image_path=f"path/to/test_{i}.jpg",
            inverse_matrix=np.eye(3),
            raw_size=(100, 100),
            orientation=1,
            shape=(100, 100),
        )
        batch.append(batch_item)

    collated = collate_fn(batch)

    # Check that only basic items are included in inference mode
    assert "images" in collated
    assert "image_filename" in collated
    assert "image_path" in collated
    assert "inverse_matrix" in collated
    assert "raw_size" in collated
    assert "orientation" in collated
    assert "canonical_size" in collated

    # Check that maps are not included in inference mode
    assert "polygons" not in collated
    assert "prob_maps" not in collated
    assert "thresh_maps" not in collated


def test_dbcollatefn_with_preloaded_maps():
    """Test DBCollateFN with pre-loaded probability and threshold maps"""
    collate_fn = DBCollateFN()

    # Create mock batch data with pre-processed maps
    batch = []
    for i in range(2):
        prob_map = np.random.rand(100, 100).astype(np.float32)
        thresh_map = np.random.rand(100, 100).astype(np.float32)
        polygon = np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]], dtype=np.float32)  # (1, n_points, 2)
        polygons = [polygon]

        batch_item = OrderedDict(
            image=torch.rand(3, 100, 100),
            image_filename=f"test_{i}.jpg",
            image_path=f"path/to/test_{i}.jpg",
            inverse_matrix=np.eye(3),
            raw_size=(100, 100),
            orientation=1,
            shape=(100, 100),
            prob_map=prob_map,
            thresh_map=thresh_map,
            polygons=polygons,
        )
        batch.append(batch_item)

    collated = collate_fn(batch)

    # Check that all required fields are present
    assert "images" in collated
    assert "image_filename" in collated
    assert "image_path" in collated
    assert "inverse_matrix" in collated
    assert "raw_size" in collated
    assert "orientation" in collated
    assert "canonical_size" in collated
    assert "polygons" in collated
    assert "prob_maps" in collated
    assert "thresh_maps" in collated

    # Check tensor dimensions
    # Note: Pre-loaded maps now get the channel dimension added to match loss expectations
    assert collated["images"].shape == (2, 3, 100, 100)
    assert collated["prob_maps"].shape == (2, 1, 100, 100)  # Channel dimension added: (B, 1, H, W)
    assert collated["thresh_maps"].shape == (2, 1, 100, 100)  # Channel dimension added: (B, 1, H, W)


def test_dbcollatefn_with_tensor_maps():
    """Test DBCollateFN when maps are already tensors"""
    collate_fn = DBCollateFN()

    # Create mock batch data with tensor maps
    batch = []
    for i in range(2):
        prob_map = torch.rand(1, 100, 100)  # Already a tensor
        thresh_map = torch.rand(1, 100, 100)  # Already a tensor
        polygons = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]

        batch_item = OrderedDict(
            image=torch.rand(3, 100, 100),
            image_filename=f"test_{i}.jpg",
            image_path=f"path/to/test_{i}.jpg",
            inverse_matrix=np.eye(3),
            raw_size=(100, 100),
            orientation=1,
            shape=(100, 100),
            prob_map=prob_map,
            thresh_map=thresh_map,
            polygons=polygons,
        )
        batch.append(batch_item)

    collated = collate_fn(batch)

    # Check that maps are still tensors after collation
    assert collated["prob_maps"].shape == (2, 1, 100, 100)
    assert collated["thresh_maps"].shape == (2, 1, 100, 100)
    assert isinstance(collated["prob_maps"], torch.Tensor)
    assert isinstance(collated["thresh_maps"], torch.Tensor)


def test_dbcollatefn_on_the_fly_maps():
    """Test DBCollateFN generating maps on-the-fly when they're not pre-loaded"""
    collate_fn = DBCollateFN()

    # Create mock batch data without pre-processed maps
    batch = []
    for i in range(2):
        # Format polygon correctly as a 3D array with shape (1, n_points, 2) to match expected format
        polygon = np.array([[[20, 20], [80, 20], [80, 80], [20, 80]]], dtype=np.float32)  # (1, n_points, 2)
        polygons = [polygon]

        batch_item = OrderedDict(
            image=torch.rand(3, 100, 100),
            image_filename=f"test_{i}.jpg",
            image_path=f"path/to/test_{i}.jpg",
            inverse_matrix=np.eye(3),
            raw_size=(100, 100),
            orientation=1,
            shape=(100, 100),
            polygons=polygons,
        )
        # Intentionally not including prob_map and thresh_map to trigger on-the-fly generation
        batch.append(batch_item)

    collated = collate_fn(batch)

    # Check that all required fields are present
    assert "images" in collated
    assert "image_filename" in collated
    assert "image_path" in collated
    assert "inverse_matrix" in collated
    assert "raw_size" in collated
    assert "orientation" in collated
    assert "canonical_size" in collated
    assert "polygons" in collated
    assert "prob_maps" in collated
    assert "thresh_maps" in collated

    # Check tensor dimensions
    assert collated["images"].shape == (2, 3, 100, 100)
    assert collated["prob_maps"].shape == (2, 1, 100, 100)
    assert collated["thresh_maps"].shape == (2, 1, 100, 100)


def test_dbcollatefn_make_prob_thresh_map():
    """Test the make_prob_thresh_map method directly"""
    collate_fn = DBCollateFN(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)

    # Create a test image and polygon
    image = torch.rand(3, 100, 100)
    polygon = np.array([[[20, 20], [80, 20], [80, 80], [20, 80]]], dtype=np.float32)  # (1, n_points, 2)
    polygons = [polygon]
    filename = "test.jpg"

    result = collate_fn.make_prob_thresh_map(image, polygons, filename)

    # Check that the result has the expected fields
    assert "prob_map" in result
    assert "thresh_map" in result

    # Check the shapes
    assert result["prob_map"].shape == (100, 100)
    assert result["thresh_map"].shape == (100, 100)

    # Check the value ranges
    assert result["prob_map"].min() >= 0.0
    assert result["prob_map"].max() <= 1.0
    assert result["thresh_map"].min() >= 0.3  # thresh_min
    # Allow for small floating point precision differences
    assert result["thresh_map"].max() <= 0.700001  # thresh_max with small tolerance


def test_dbcollatefn_with_degenerate_polygons():
    """Test DBCollateFN handling of degenerate polygons"""
    collate_fn = DBCollateFN()

    # Create mock batch data with degenerate polygons (less than 3 points)
    batch = []
    for i in range(2):
        # Create polygons with fewer than 3 points which should be skipped
        polygons = [
            np.array([[[10, 10]]], dtype=np.float32),  # Only 1 point - degenerate (1, 1, 2)
            np.array([[[10, 10], [20, 10]]], dtype=np.float32),  # Only 2 points - degenerate (1, 2, 2)
            np.array([[[20, 20], [80, 20], [80, 80], [20, 80]]], dtype=np.float32),  # Valid 4-point polygon (1, 4, 2)
        ]

        batch_item = OrderedDict(
            image=torch.rand(3, 100, 100),
            image_filename=f"test_{i}.jpg",
            image_path=f"path/to/test_{i}.jpg",
            inverse_matrix=np.eye(3),
            raw_size=(100, 100),
            orientation=1,
            shape=(100, 100),
            polygons=polygons,
        )
        batch.append(batch_item)

    collated = collate_fn(batch)

    # Check that the result is still valid despite degenerate polygons
    assert collated["images"].shape == (2, 3, 100, 100)
    assert collated["prob_maps"].shape == (2, 1, 100, 100)
    assert collated["thresh_maps"].shape == (2, 1, 100, 100)


def test_dbcollatefn_distance_method():
    """Test the distance calculation method"""
    collate_fn = DBCollateFN()

    # Create coordinate grids
    xs = np.array([[0, 1, 2], [0, 1, 2]])
    ys = np.array([[0, 0, 0], [1, 1, 1]])

    # Define two points for a line
    point_1 = np.array([0, 0])
    point_2 = np.array([2, 1])

    distances = collate_fn.distance(xs, ys, point_1, point_2)

    # Check that distances has correct shape
    assert distances.shape == (2, 3)

    # Check that all distances are non-negative
    assert np.all(distances >= 0)


if __name__ == "__main__":
    test_ocr_dataset_initialization()
    test_ocr_dataset_getitem_map_loading()
    test_ocr_dataset_getitem_fallback()
    test_dbcollatefn_init()
    test_dbcollatefn_inference_mode()
    test_dbcollatefn_with_preloaded_maps()
    test_dbcollatefn_with_tensor_maps()
    test_dbcollatefn_on_the_fly_maps()
    test_dbcollatefn_make_prob_thresh_map()
    test_dbcollatefn_with_degenerate_polygons()
    test_dbcollatefn_distance_method()
    print("All tests passed!")

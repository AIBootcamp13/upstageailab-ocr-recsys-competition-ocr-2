"""Test to diagnose collate function polygon handling."""

import numpy as np
import torch

from ocr.datasets.db_collate_fn import DBCollateFN


def test_polygon_shapes():
    """Test collate function with different polygon shapes."""

    collate_fn = DBCollateFN()

    # Simulate an image (3, 640, 640)
    image = torch.rand(3, 640, 640)

    # Test Case 1: Polygon with batch dimension (1, 4, 2)
    print("=" * 80)
    print("Test Case 1: Polygon with shape (1, 4, 2) - WITH batch dimension")
    print("=" * 80)
    poly_with_batch = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]], dtype=np.float32)
    print(f"Polygon shape: {poly_with_batch.shape}")
    print(f"Polygon: {poly_with_batch}")

    batch_1 = [
        {
            "image": image,
            "polygons": [poly_with_batch],
            "image_filename": "test1.jpg",
            "image_path": "/path/to/test1.jpg",
            "inverse_matrix": np.eye(3),
            "shape": (640, 640),
        }
    ]

    try:
        result = collate_fn(batch_1)
        print("✅ SUCCESS - Collate function handled polygon with batch dimension")
        print(f"Prob map shape: {result['prob_maps'].shape}")
        print(f"Thresh map shape: {result['thresh_maps'].shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Test Case 2: Polygon WITHOUT batch dimension (4, 2)
    print("\n" + "=" * 80)
    print("Test Case 2: Polygon with shape (4, 2) - WITHOUT batch dimension")
    print("=" * 80)
    poly_without_batch = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    print(f"Polygon shape: {poly_without_batch.shape}")
    print(f"Polygon: {poly_without_batch}")

    batch_2 = [
        {
            "image": image,
            "polygons": [poly_without_batch],
            "image_filename": "test2.jpg",
            "image_path": "/path/to/test2.jpg",
            "inverse_matrix": np.eye(3),
            "shape": (640, 640),
        }
    ]

    try:
        result = collate_fn(batch_2)
        print("✅ SUCCESS - Collate function handled polygon without batch dimension")
        print(f"Prob map shape: {result['prob_maps'].shape}")
        print(f"Thresh map shape: {result['thresh_maps'].shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Test Case 3: Multiple polygons without batch dimension
    print("\n" + "=" * 80)
    print("Test Case 3: Multiple polygons with shape (4, 2) each")
    print("=" * 80)
    poly1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    poly2 = np.array([[300, 300], [400, 300], [400, 400], [300, 400]], dtype=np.float32)
    print(f"Polygon 1 shape: {poly1.shape}")
    print(f"Polygon 2 shape: {poly2.shape}")

    batch_3 = [
        {
            "image": image,
            "polygons": [poly1, poly2],
            "image_filename": "test3.jpg",
            "image_path": "/path/to/test3.jpg",
            "inverse_matrix": np.eye(3),
            "shape": (640, 640),
        }
    ]

    try:
        result = collate_fn(batch_3)
        print("✅ SUCCESS - Collate function handled multiple polygons")
        print(f"Prob map shape: {result['prob_maps'].shape}")
        print(f"Thresh map shape: {result['thresh_maps'].shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_polygon_shapes()

#!/usr/bin/env python3
"""Test script for Wandb fallback functionality in checkpoint catalog.

This script verifies that the Wandb fallback integration works correctly.
"""

import sys
from pathlib import Path

# Test imports
try:
    from ui.apps.inference.services.checkpoint import (
        WandbClient,
        extract_run_id_from_checkpoint,
        get_wandb_client,
    )

    print("✓ Successfully imported Wandb client components")
except ImportError as exc:
    print(f"✗ Failed to import Wandb client: {exc}")
    sys.exit(1)


def test_wandb_client_initialization():
    """Test Wandb client initialization."""
    print("\n=== Test 1: Wandb Client Initialization ===")

    try:
        client = WandbClient()
        print("✓ WandbClient initialized")
        print(f"  - Available: {client._is_available}")

        # Test singleton pattern
        get_wandb_client()
        print("✓ get_wandb_client() works")

        return True

    except Exception as exc:
        print(f"✗ Failed to initialize client: {exc}")
        return False


def test_extract_run_id():
    """Test run ID extraction from checkpoint paths."""
    print("\n=== Test 2: Run ID Extraction ===")

    # Create a fake checkpoint path
    test_path = Path("/fake/outputs/test-exp/checkpoints/epoch=10.ckpt")

    try:
        run_id = extract_run_id_from_checkpoint(test_path)
        print("✓ extract_run_id_from_checkpoint executed")
        print(f"  - Run ID: {run_id if run_id else 'None (expected for fake path)'}")

        return True

    except Exception as exc:
        print(f"✗ Failed to extract run ID: {exc}")
        return False


def test_wandb_metadata_construction():
    """Test metadata construction from Wandb (with mock data)."""
    print("\n=== Test 3: Metadata Construction ===")

    try:
        client = get_wandb_client()

        # Test with fake run ID (should gracefully fail)
        fake_checkpoint = Path("/fake/checkpoint.ckpt")
        metadata = client.get_metadata_from_wandb("fake/project/run123", fake_checkpoint)

        print("✓ get_metadata_from_wandb executed")
        print(f"  - Result: {metadata if metadata else 'None (expected for fake run)'}")

        return True

    except Exception as exc:
        print(f"✗ Failed to construct metadata: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_cache_functionality():
    """Test cache clearing functionality."""
    print("\n=== Test 4: Cache Functionality ===")

    try:
        client = get_wandb_client()
        client.clear_cache()
        print("✓ Cache cleared successfully")

        return True

    except Exception as exc:
        print(f"✗ Failed to clear cache: {exc}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Wandb Fallback Integration Tests")
    print("=" * 60)

    results = {
        "Client Initialization": test_wandb_client_initialization(),
        "Run ID Extraction": test_extract_run_id(),
        "Metadata Construction": test_wandb_metadata_construction(),
        "Cache Functionality": test_cache_functionality(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)

    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

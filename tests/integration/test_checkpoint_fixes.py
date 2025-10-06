#!/usr/bin/env python3
"""
Test script to verify checkpoint catalog fixes.

This script tests the issues that were resolved:
1. Descriptive checkpoint naming (no more "unknown")
2. Schema validation compatibility (no channel mismatches)
3. Complete metadata extraction (architecture, backbone, decoder info)

Usage: python test_checkpoint_fixes.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.apps.inference.models.checkpoint import CheckpointMetadata
from ui.apps.inference.models.config import PathConfig
from ui.apps.inference.services.checkpoint_catalog import CatalogOptions, build_catalog


def test_checkpoint_catalog() -> tuple[list[CheckpointMetadata], list[str]]:
    """Test the checkpoint catalog and return results with any issues found."""
    print("🔍 Testing Checkpoint Catalog Fixes")
    print("=" * 50)

    # Create path config
    paths = PathConfig(outputs_dir=Path("outputs"), hydra_config_filenames=["config.yaml", "train.yaml"])

    # Build catalog options
    options = CatalogOptions.from_paths(paths)

    # Build catalog
    checkpoints = build_catalog(options)

    issues = []

    print(f"📊 Found {len(checkpoints)} checkpoints")
    print()

    # Test 1: Descriptive naming (no "unknown")
    print("1️⃣ Testing Descriptive Checkpoint Naming")
    print("-" * 40)

    unknown_count = 0
    for cp in checkpoints:
        display_name = cp.to_display_option()
        if "unknown" in display_name.lower():
            unknown_count += 1
            issues.append(f"❌ Checkpoint has 'unknown' in display name: {display_name}")
            print(f"   ❌ {display_name}")
        else:
            print(f"   ✅ {display_name}")

    if unknown_count == 0:
        print("✅ All checkpoints have descriptive names!")
    else:
        print(f"❌ {unknown_count} checkpoints still have 'unknown' in their names")
    print()

    # Test 2: Schema validation (no issues)
    print("2️⃣ Testing Schema Validation")
    print("-" * 40)

    validation_issues = 0
    for cp in checkpoints:
        if cp.issues:
            validation_issues += 1
            for issue in cp.issues:
                issues.append(f"❌ Schema validation issue for {cp.to_display_option()}: {issue}")
                print(f"   ❌ {cp.to_display_option()}: {issue}")
        else:
            print(f"   ✅ {cp.to_display_option()} - {cp.schema_family_id}")

    if validation_issues == 0:
        print("✅ All checkpoints pass schema validation!")
    else:
        print(f"❌ {validation_issues} checkpoints have schema validation issues")
    print()

    # Test 3: Complete metadata
    print("3️⃣ Testing Complete Metadata Extraction")
    print("-" * 40)

    metadata_issues = 0
    for cp in checkpoints:
        # Check required fields
        missing_fields = []

        if not cp.architecture or cp.architecture == "unknown":
            missing_fields.append("architecture")
        if not cp.backbone or cp.backbone == "unknown":
            missing_fields.append("backbone")

        # Check decoder metadata (allow some flexibility for different architectures)
        decoder_complete = (
            cp.decoder.inner_channels is not None
            or cp.decoder.output_channels is not None
            or cp.decoder.in_channels  # Allow empty list for some architectures
        )

        if not decoder_complete:
            missing_fields.append("decoder_info")

        if missing_fields:
            metadata_issues += 1
            issues.append(f"❌ Incomplete metadata for {cp.to_display_option()}: missing {', '.join(missing_fields)}")
            print(f"   ❌ {cp.to_display_option()}: missing {', '.join(missing_fields)}")
        else:
            decoder_info = f"decoder(in={len(cp.decoder.in_channels) if cp.decoder.in_channels else 0}, inner={cp.decoder.inner_channels}, out={cp.decoder.output_channels})"
            print(f"   ✅ {cp.to_display_option()}: {cp.architecture}/{cp.backbone}, {decoder_info}")

    if metadata_issues == 0:
        print("✅ All checkpoints have complete metadata!")
    else:
        print(f"❌ {metadata_issues} checkpoints have incomplete metadata")
    print()

    return checkpoints, issues


def test_ui_compatibility() -> list[str]:
    """Test that checkpoints are compatible with the UI inference interface."""
    print("4️⃣ Testing UI Compatibility")
    print("-" * 40)

    issues = []

    try:
        from ui.apps.inference.models.config import PathConfig
        from ui.apps.inference.services.checkpoint_catalog import CatalogOptions, build_catalog

        paths = PathConfig(outputs_dir=Path("outputs"), hydra_config_filenames=["config.yaml", "train.yaml"])
        options = CatalogOptions.from_paths(paths)
        checkpoints = build_catalog(options)

        # Check that we can create display options for all checkpoints
        display_options = []
        for cp in checkpoints:
            try:
                display_option = cp.to_display_option()
                display_options.append(display_option)
                print(f"   ✅ {display_option}")
            except Exception as e:
                issues.append(f"❌ Failed to create display option: {e}")
                print(f"   ❌ Display option failed: {e}")

        if len(display_options) == len(checkpoints):
            print("✅ All checkpoints are UI-compatible!")
        else:
            print(f"❌ {len(checkpoints) - len(display_options)} checkpoints are not UI-compatible")

    except Exception as e:
        issues.append(f"❌ UI compatibility test failed: {e}")
        print(f"❌ UI compatibility test failed: {e}")

    print()
    return issues


def main():
    """Run all tests and report results."""
    print("🧪 Checkpoint Catalog Fix Verification")
    print("=" * 60)
    print()

    all_issues = []

    # Run tests
    checkpoints, catalog_issues = test_checkpoint_catalog()
    all_issues.extend(catalog_issues)

    ui_issues = test_ui_compatibility()
    all_issues.extend(ui_issues)

    # Summary
    print("📋 TEST SUMMARY")
    print("=" * 60)

    if not all_issues:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Checkpoint catalog fixes are working correctly")
        print("✅ All checkpoints have descriptive names")
        print("✅ All checkpoints pass schema validation")
        print("✅ All checkpoints have complete metadata")
        print("✅ All checkpoints are UI-compatible")
        return 0
    else:
        print(f"❌ {len(all_issues)} issues found:")
        for issue in all_issues:
            print(f"   {issue}")
        print()
        print("🔧 Please review the issues above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

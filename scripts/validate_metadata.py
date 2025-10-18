#!/usr/bin/env python
"""Validate checkpoint metadata files.

This script validates .metadata.yaml files for checkpoints to ensure they
conform to the CheckpointMetadataV1 schema and meet business requirements.

Usage:
    # Validate all checkpoints in outputs directory
    python scripts/validate_metadata.py --outputs-dir outputs/

    # Validate specific experiment
    python scripts/validate_metadata.py --exp-dir outputs/my_experiment/

    # Validate single checkpoint
    python scripts/validate_metadata.py --checkpoint outputs/my_experiment/checkpoints/best.ckpt

    # Verbose output (show each checkpoint)
    python scripts/validate_metadata.py --outputs-dir outputs/ --verbose

    # Show only errors
    python scripts/validate_metadata.py --outputs-dir outputs/ --errors-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.apps.inference.services.checkpoint.validator import MetadataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for validation tool.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Validate checkpoint metadata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Outputs directory to search for checkpoints (recursive)",
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        help="Single experiment directory to validate",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Single checkpoint file to validate",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recursively search subdirectories",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show validation result for each checkpoint",
    )

    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show errors (suppress success messages)",
    )

    parser.add_argument(
        "--schema-version",
        default="1.0",
        help="Target schema version to validate against (default: 1.0)",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.errors_only:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not any([args.outputs_dir, args.exp_dir, args.checkpoint]):
        parser.error("Must specify one of: --outputs-dir, --exp-dir, or --checkpoint")
        return 1

    # Create validator
    validator = MetadataValidator(schema_version=args.schema_version)

    try:
        if args.checkpoint:
            # Validate single checkpoint
            if not args.checkpoint.exists():
                LOGGER.error("Checkpoint file does not exist: %s", args.checkpoint)
                return 1

            result = validator.validate_checkpoint_file(args.checkpoint)

            if result.is_valid:
                LOGGER.info("✓ Metadata valid: %s", args.checkpoint)
                return 0
            else:
                LOGGER.error("✗ Validation failed: %s", result.error)
                return 1

        elif args.exp_dir or args.outputs_dir:
            # Validate directory
            target_dir = args.exp_dir or args.outputs_dir
            recursive = not args.no_recursive

            report = validator.validate_directory(
                target_dir,
                recursive=recursive,
                verbose=args.verbose and not args.errors_only,
            )

            # Print summary
            print()  # Empty line before summary
            print(report.summary())

            # Show detailed errors if requested
            if args.verbose and report.invalid > 0:
                print("\nDetailed Errors:")
                print("=" * 60)
                for result in report.results:
                    if not result.is_valid and result.error_type != "missing":
                        print(f"\n{result.checkpoint_path}:")
                        print(f"  Type: {result.error_type}")
                        print(f"  Error: {result.error}")

            # Show missing files if requested
            if args.verbose and report.missing > 0:
                print("\nMissing Metadata Files:")
                print("=" * 60)
                for result in report.results:
                    if result.error_type == "missing":
                        print(f"  {result.checkpoint_path}")

            # Return non-zero if any validation failures
            if report.invalid > 0 or report.missing > 0:
                return 1

            return 0

    except KeyboardInterrupt:
        LOGGER.warning("\nValidation interrupted by user")
        return 130

    except Exception as exc:
        LOGGER.error("Validation failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

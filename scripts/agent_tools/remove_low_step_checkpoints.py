#!/usr/bin/env python3
"""
Script to find and optionally remove checkpoints with less than 100 steps.
"""

import argparse
import re
from pathlib import Path


def extract_step_from_filename(filename: str) -> int:
    """
    Extract step number from checkpoint filename or parent directory name.

    Supports various patterns:
    - epoch-9-step-1030.ckpt
    - epoch_epoch=00-step_step=0001-loss_val_loss=61.8077.ckpt
    - resnet18_bs12_epoch_epoch=0_step_step=13.ckpt
    - epoch-0-step-52.ckpt
    - epoch=0-step=50.ckpt
    - Files in directories like epoch_epoch=01-step_step=0546-loss_val/loss=1.6217.ckpt
    """
    # Check if this is a file in a directory with step info
    import os

    dirname = os.path.dirname(filename)
    dirname_basename = os.path.basename(dirname)

    # If the directory name contains step info, use that
    if match := re.search(r"step[=_-](\d+)", dirname_basename):
        return int(match.group(1))

    # Otherwise, check the filename itself
    basename = os.path.basename(filename).replace(".ckpt", "")
    if match := re.search(r"step[=_-](\d+)", basename):
        return int(match.group(1))

    # If no step found, return 0 (will be considered < 100)
    return 0


def find_checkpoints_less_than_100_steps(outputs_dir: Path) -> list[tuple[Path, int]]:
    """
    Find all checkpoint files with step count < 100.

    Returns list of (filepath, step_count) tuples.
    """
    checkpoints_to_remove = []

    for ckpt_file in outputs_dir.rglob("*.ckpt"):
        step_count = extract_step_from_filename(str(ckpt_file))
        if step_count < 100:
            checkpoints_to_remove.append((ckpt_file, step_count))

    return checkpoints_to_remove


def main():
    parser = argparse.ArgumentParser(description="Find and optionally remove checkpoints with < 100 steps")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Outputs directory to search")

    args = parser.parse_args()

    if not args.outputs_dir.exists():
        print(f"Error: {args.outputs_dir} does not exist")
        return

    print(f"Searching for checkpoints with < 100 steps in {args.outputs_dir}")
    print("=" * 60)

    checkpoints_to_remove = find_checkpoints_less_than_100_steps(args.outputs_dir)

    if not checkpoints_to_remove:
        print("No checkpoints found with < 100 steps.")
        return

    # Sort by step count for better display
    checkpoints_to_remove.sort(key=lambda x: x[1])

    total_size = 0
    for ckpt_path, step_count in checkpoints_to_remove:
        try:
            size = ckpt_path.stat().st_size
            total_size += size
            print(f"{step_count:2d} steps: {ckpt_path} ({size / (1024 * 1024):.1f} MB)")
        except OSError:
            print(f"{step_count:2d} steps: {ckpt_path} (size unknown)")

    print("=" * 60)
    print(f"Total: {len(checkpoints_to_remove)} checkpoints, {total_size / (1024 * 1024):.2f} MB")

    if not args.dry_run:
        response = input("\nAre you sure you want to delete these checkpoints? (yes/no): ")
        if response.lower() == "yes":
            deleted_count = 0
            for ckpt_path, _ in checkpoints_to_remove:
                try:
                    ckpt_path.unlink()
                    deleted_count += 1
                    print(f"Deleted: {ckpt_path}")
                except OSError as e:
                    print(f"Error deleting {ckpt_path}: {e}")

            print(f"\nSuccessfully deleted {deleted_count} checkpoints")
        else:
            print("Operation cancelled.")
    else:
        print("\nThis was a dry run. Use --dry-run=false to actually delete the files.")


if __name__ == "__main__":
    main()

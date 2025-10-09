#!/usr/bin/env python3
"""
Script to find and optionally remove checkpoints with less than 5 epochs.
"""

import argparse
import re
from pathlib import Path


def extract_epoch_from_filename(filename: str) -> int | None:
    """
    Extract epoch number from checkpoint filename or parent directory name.

    Supports various patterns:
    - epoch-9-step-1030.ckpt
    - epoch_epoch=00-step_step=0001-loss_val_loss=61.8077.ckpt
    - resnet18_bs12_epoch_epoch=0_step_step=13.ckpt
    - epoch-0-step-52.ckpt
    - epoch=0-step=50.ckpt
    - Files in directories like epoch_epoch=01-step_step=0546-loss_val/loss=1.6217.ckpt
    """
    # Check if this is a file in a directory with epoch info
    import os

    dirname = os.path.dirname(filename)
    dirname_basename = os.path.basename(dirname)

    # If the directory name contains epoch info, use that
    epoch_pattern = re.compile(r"epoch[=\-_](?P<epoch>\d+)")
    if match := epoch_pattern.search(dirname_basename):
        return int(match.group("epoch"))

    # Otherwise, check the filename itself
    basename = os.path.basename(filename).replace(".ckpt", "")
    if match := epoch_pattern.search(basename):
        return int(match.group("epoch"))

    # If no epoch found, return None
    return None


def find_checkpoints_less_than_5_epochs(outputs_dir: Path) -> list[tuple[Path, int | None]]:
    """
    Find all checkpoint files with epoch count < 5 or None.

    Returns list of (filepath, epoch_count) tuples.
    """
    checkpoints_to_remove = []

    for ckpt_file in outputs_dir.rglob("*.ckpt"):
        epoch_count = extract_epoch_from_filename(str(ckpt_file))
        if epoch_count is None or epoch_count < 5:
            checkpoints_to_remove.append((ckpt_file, epoch_count))

    return checkpoints_to_remove


def main():
    parser = argparse.ArgumentParser(description="Find and optionally remove checkpoints with < 5 epochs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--yes", action="store_true", help="Automatically confirm deletion without prompting")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Outputs directory to search")

    args = parser.parse_args()

    if not args.outputs_dir.exists():
        print(f"Error: {args.outputs_dir} does not exist")
        return

    print(f"Searching for checkpoints with < 5 epochs in {args.outputs_dir}")
    print("=" * 60)

    checkpoints_to_remove = find_checkpoints_less_than_5_epochs(args.outputs_dir)

    if not checkpoints_to_remove:
        print("No checkpoints found with < 5 epochs.")
        return

    # Sort by epoch count for better display
    checkpoints_to_remove.sort(key=lambda x: (x[1] is None, x[1] or 0))

    total_size = 0
    for ckpt_path, epoch_count in checkpoints_to_remove:
        try:
            size = ckpt_path.stat().st_size
            total_size += size
            epoch_str = "None" if epoch_count is None else str(epoch_count)
            print(f"Epoch {epoch_str}: {ckpt_path} ({size / (1024 * 1024):.1f} MB)")
        except OSError:
            epoch_str = "None" if epoch_count is None else str(epoch_count)
            print(f"Epoch {epoch_str}: {ckpt_path} (size unknown)")

    print("=" * 60)
    print(f"Total: {len(checkpoints_to_remove)} checkpoints, {total_size / (1024 * 1024):.2f} MB")

    if args.dry_run:
        print("\nThis was a dry run. Use --dry-run=false to actually delete the files.")
        return

    if not args.yes:
        response = input("\nAre you sure you want to delete these checkpoints? (yes/no): ")
        if response.lower() != "yes":
            print("Operation cancelled.")
            return

    deleted_count = 0
    for ckpt_path, _ in checkpoints_to_remove:
        try:
            ckpt_path.unlink()
            deleted_count += 1
            print(f"Deleted: {ckpt_path}")
        except OSError as e:
            print(f"Error deleting {ckpt_path}: {e}")

    print(f"\nSuccessfully deleted {deleted_count} checkpoints")


if __name__ == "__main__":
    main()

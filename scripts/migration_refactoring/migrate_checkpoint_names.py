#!/usr/bin/env python3
"""
Script to migrate checkpoint filenames from old formats to the new filesystem-safe format.

Old formats:
- epoch_epoch=02-step_step=0819-loss_val_loss=1.5191.ckpt
- epoch-9-step-1030.ckpt
- epoch_epoch=01-step_step=0546-loss_val/loss=1.6217.ckpt
- epoch_0_step_273.ckpt

New format: epoch_{epoch:02d}_step_{step:06d}.ckpt
"""

import argparse
import re
from pathlib import Path


def parse_checkpoint_info(filename: str) -> tuple[int, int] | None:
    """
    Parse epoch and step information from various checkpoint filename formats.

    Returns (epoch, step) tuple or None if parsing fails.
    """
    # Remove .ckpt extension
    stem = filename.replace(".ckpt", "")

    # Pattern 1: epoch_epoch=XX-step_step=YYYY format
    if pattern1 := re.search(r"epoch[=_-](\d+).*step[=_-](\d+)", stem):
        epoch = int(pattern1.group(1))
        step = int(pattern1.group(2))
        return epoch, step

    # Pattern 2: epoch-XX-step-YYYY format
    if pattern2 := re.search(r"epoch[=_-](\d+).*step[=_-](\d+)", stem):
        epoch = int(pattern2.group(1))
        step = int(pattern2.group(2))
        return epoch, step

    # Pattern 3: epoch_XX_step_YYYY format (already new format)
    if pattern3 := re.search(r"epoch_(\d+)_step_(\d+)", stem):
        epoch = int(pattern3.group(1))
        step = int(pattern3.group(2))
        return epoch, step

    # Pattern 4: Look for any epoch and step mentions (fallback)
    epoch_match = re.search(r"epoch[=_-](\d+)", stem)
    step_match = re.search(r"step[=_-](\d+)", stem)

    if epoch_match and step_match:
        epoch = int(epoch_match.group(1))
        step = int(step_match.group(1))
        return epoch, step

    # Special case: files that start with just "loss=" - these are likely truncated
    # and should be skipped as they don't contain enough information
    if stem.startswith("loss=") or len(stem) < 5:
        return None

    return None


def generate_new_filename(epoch: int, step: int) -> str:
    """Generate new checkpoint filename in the standardized format."""
    return f"epoch_{epoch:02d}_step_{step:06d}.ckpt"


def migrate_checkpoint(checkpoint_path: Path, dry_run: bool = True) -> bool:
    """
    Migrate a single checkpoint to the new naming format.

    Returns True if migration was successful or not needed.
    """
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return False

    current_name = checkpoint_path.name
    info = parse_checkpoint_info(current_name)

    if info is None:
        print(f"Warning: Could not parse checkpoint info from: {current_name}")
        return False

    epoch, step = info
    new_name = generate_new_filename(epoch, step)

    if current_name == new_name:
        print(f"Already in correct format: {current_name}")
        return True

    new_path = checkpoint_path.parent / new_name

    if new_path.exists():
        print(f"Warning: Target filename already exists: {new_path}")
        return False

    if dry_run:
        print(f"Would rename: {current_name} -> {new_name}")
    else:
        try:
            checkpoint_path.rename(new_path)
            print(f"Renamed: {current_name} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {current_name}: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoint filenames to new format")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Root outputs directory to scan for checkpoints")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be renamed without actually doing it")
    parser.add_argument("--execute", action="store_true", help="Actually perform the renaming (overrides dry-run)")

    args = parser.parse_args()

    if args.execute:
        args.dry_run = False

    if not args.outputs_dir.exists():
        print(f"Outputs directory not found: {args.outputs_dir}")
        return

    print(f"Scanning for checkpoints in: {args.outputs_dir}")
    print(f"Dry run: {args.dry_run}")

    checkpoints = list(args.outputs_dir.rglob("*.ckpt"))
    print(f"Found {len(checkpoints)} checkpoint files")

    success_count = 0
    error_count = 0

    for checkpoint_path in checkpoints:
        if migrate_checkpoint(checkpoint_path, args.dry_run):
            success_count += 1
        else:
            error_count += 1

    print("\nMigration complete:")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")

    if args.dry_run:
        print("\nThis was a dry run. Use --execute to perform actual renaming.")


if __name__ == "__main__":
    main()

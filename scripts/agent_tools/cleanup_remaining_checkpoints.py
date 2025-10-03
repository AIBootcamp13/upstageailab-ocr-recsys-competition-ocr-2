#!/usr/bin/env python3
"""
Enhanced script to clean up checkpoint naming issues:
1. Rename checkpoint files that contain special characters like '='
2. Handle various naming formats and edge cases
3. Move epoch 0 checkpoints to a temporary folder for review before deletion
"""

import re
import shutil
from pathlib import Path


def parse_checkpoint_info(filename: str) -> tuple[int, int] | None:
    """
    Parse epoch and step information from checkpoint filenames with various formats.

    Handles formats like:
    - epoch_epoch=02-step_step=0819-loss_val_loss=1.5191.ckpt
    - resnet18_bs16_epoch_epoch=7_step_step=1640.ckpt
    - epoch_02_step_0819.ckpt
    - epoch-2-step-819.ckpt
    """
    stem = filename.replace(".ckpt", "")

    # Try different patterns to extract epoch and step
    patterns = [
        r"epoch[=_-](\d+).*step[=_-](\d+)",  # epoch=02-step=0819
        r"epoch[=_-](\d+).*step[=_-](\d+)",  # epoch_02_step_0819
        r"epoch[=_-](\d+)",  # epoch=02 (if no step)
    ]

    for pattern in patterns:
        epoch_match = re.search(pattern, stem)
        if epoch_match:
            epoch = int(epoch_match.group(1))

            # Try to find step
            step_match = re.search(r"step[=_-](\d+)", stem)
            step = int(step_match.group(1)) if step_match else 0

            return epoch, step

    return None


def rename_problematic_checkpoints():
    """Rename checkpoint files that contain problematic characters or naming."""
    # Check both outputs and lightning_logs directories
    search_dirs = [Path("outputs"), Path("lightning_logs")]
    renamed_count = 0

    # Characters that are problematic in filenames
    problematic_chars = ["=", " ", ":", "*", "?", '"', "<", ">", "|"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for ckpt_path in search_dir.rglob("*.ckpt"):
            filename = ckpt_path.name

            # Check if filename contains problematic characters
            has_problematic_chars = any(char in filename for char in problematic_chars)

            # Also check for non-standard naming patterns
            is_standard_format = re.match(r"^epoch_\d+_step_\d+\.ckpt$", filename)

            if not has_problematic_chars and is_standard_format:
                continue

            print(f"Found problematic checkpoint: {ckpt_path}")

            # Parse the checkpoint info
            info = parse_checkpoint_info(filename)
            if not info:
                print(f"Warning: Could not parse info from: {filename}")
                continue

            epoch, step = info

            # Create new filename in the standard format
            new_filename = f"epoch_{epoch:02d}_step_{step:06d}.ckpt"
            new_path = ckpt_path.parent / new_filename

            # Check if target already exists
            if new_path.exists():
                print(f"Warning: Target already exists: {new_path}")
                continue

            # Rename the file
            try:
                ckpt_path.rename(new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    print(f"\nRenamed {renamed_count} checkpoint files.")


def move_epoch0_checkpoints():
    """Move only epoch 0 checkpoints to a temporary folder for review."""
    # Check both outputs and lightning_logs directories
    search_dirs = [Path("outputs"), Path("lightning_logs")]
    epoch0_dir = Path("outputs/epoch0_checkpoints")
    epoch0_dir.mkdir(exist_ok=True)

    epoch0_pattern = re.compile(r"epoch[=_-](\d+)")
    moved_count = 0

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for ckpt_path in search_dir.rglob("*.ckpt"):
            # Skip files already in the epoch0_checkpoints folder
            if "epoch0_checkpoints" in str(ckpt_path):
                continue

            filename = ckpt_path.name

            # Extract epoch number
            epoch_match = epoch0_pattern.search(filename)
            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))

            # Only move epoch 0 checkpoints
            if epoch == 0:
                dst_path = epoch0_dir / filename
                try:
                    shutil.move(str(ckpt_path), str(dst_path))
                    print(f"Moved: {ckpt_path} -> {dst_path}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {ckpt_path}: {e}")

    print(f"Total epoch 0 checkpoints moved: {moved_count}")


def main():
    print("=== Enhanced Checkpoint Cleanup Script ===")
    print()

    print("1. Renaming checkpoint files with problematic characters...")
    rename_problematic_checkpoints()

    print("\n2. Moving epoch 0 checkpoints to temporary folder...")
    move_epoch0_checkpoints()

    print("\n=== Cleanup Complete ===")
    print("Review the files in outputs/epoch0_checkpoints/ before deletion.")
    print("Run: rm -rf outputs/epoch0_checkpoints/  (when ready to delete)")


if __name__ == "__main__":
    main()

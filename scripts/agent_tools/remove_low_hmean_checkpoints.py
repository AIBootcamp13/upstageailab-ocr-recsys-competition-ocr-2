#!/usr/bin/env python3
"""
Script to find and remove checkpoints with low hmean scores (< threshold).
Keeps checkpoints with hmean >= threshold or no hmean information (for manual review).
"""

import argparse
import re
from pathlib import Path


def extract_hmean_from_filename(filename: str) -> float | None:
    """
    Extract hmean (harmonic mean) score from checkpoint filename or parent directory name.

    Supports various patterns:
    - val/hmean=0.8920-best.ckpt
    - hmean=0.7990-best.ckpt
    - hmean=0.0000-best.ckpt
    """
    import os

    dirname = os.path.dirname(filename)
    dirname_basename = os.path.basename(dirname)

    # Check directory name first (for val/hmean=X.XXXX-best.ckpt pattern)
    hmean_pattern = re.compile(r"hmean[=:](\d+\.\d+)")
    if match := hmean_pattern.search(dirname_basename):
        return float(match.group(1))

    # Otherwise, check the filename itself
    basename = os.path.basename(filename).replace(".ckpt", "")
    if match := hmean_pattern.search(basename):
        return float(match.group(1))

    # If no hmean found, return None
    return None


def find_low_hmean_checkpoints(outputs_dir: Path, min_hmean: float = 0.80) -> list[tuple[Path, float]]:
    """
    Find all checkpoint files with hmean < min_hmean.
    Keeps checkpoints with hmean >= min_hmean or no hmean information (for manual review).

    Returns list of (filepath, hmean_score) tuples to remove.
    """
    checkpoints_to_remove = []

    for ckpt_file in outputs_dir.rglob("*.ckpt"):
        hmean_score = extract_hmean_from_filename(str(ckpt_file))
        # Only remove checkpoints that have hmean scores AND those scores are below threshold
        if hmean_score is not None and hmean_score < min_hmean:
            checkpoints_to_remove.append((ckpt_file, hmean_score))

    return checkpoints_to_remove
    """
    Find all checkpoint files with hmean < 0.80.
    Keeps checkpoints with hmean >= 0.80 or no hmean information.

    Returns list of (filepath, hmean_score) tuples to remove.
    """
    checkpoints_to_remove = []

    for ckpt_file in outputs_dir.rglob("*.ckpt"):
        hmean_score = extract_hmean_from_filename(str(ckpt_file))
        if hmean_score is not None and hmean_score < 0.80:
            checkpoints_to_remove.append((ckpt_file, hmean_score))

    return checkpoints_to_remove


def main():
    parser = argparse.ArgumentParser(description="Find and remove checkpoints with hmean < threshold")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--yes", action="store_true", help="Automatically confirm deletion without prompting")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Outputs directory to search")
    parser.add_argument("--min-hmean", type=float, default=0.80, help="Minimum hmean score to keep (default: 0.80)")

    args = parser.parse_args()

    if not args.outputs_dir.exists():
        print(f"Error: {args.outputs_dir} does not exist")
        return

    print(f"Searching for checkpoints with hmean < {args.min_hmean} in {args.outputs_dir}")
    print("=" * 60)

    checkpoints_to_remove = find_low_hmean_checkpoints(args.outputs_dir, args.min_hmean)

    if not checkpoints_to_remove:
        print(f"No checkpoints found with hmean < {args.min_hmean}.")
        return

    # Sort by hmean score for better display (lowest first)
    checkpoints_to_remove.sort(key=lambda x: x[1] or 0)

    total_size = 0
    for ckpt_path, hmean_score in checkpoints_to_remove:
        try:
            size = ckpt_path.stat().st_size
            total_size += size
            hmean_str = "None" if hmean_score is None else f"{hmean_score:.4f}"
            print(f"hmean {hmean_str}: {ckpt_path} ({size / (1024 * 1024):.1f} MB)")
        except OSError:
            hmean_str = "None" if hmean_score is None else f"{hmean_score:.4f}"
            print(f"hmean {hmean_str}: {ckpt_path} (size unknown)")

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

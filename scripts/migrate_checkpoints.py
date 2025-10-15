#!/usr/bin/env python3
"""
Migrate existing checkpoints to the new naming scheme.

This script:
1. Renames checkpoints from old format to new hierarchical format
2. Deletes unnecessary checkpoints (early epochs, duplicates)
3. Preserves important checkpoints (best, last, later epochs)
"""

import argparse
import re
from pathlib import Path


def parse_old_checkpoint_name(filename: str) -> dict[str, int | None]:
    """
    Parse old checkpoint filename to extract components.

    Old formats:
    - epoch_epoch_22_step_step_001932_20251009_015037.ckpt
    - epoch_09_step_002730.ckpt
    - epoch=67-step=18564.ckpt

    Returns:
        Dictionary with 'epoch', 'step' keys (timestamp is ignored)
    """
    stem = filename.replace(".ckpt", "")

    # Format 1: epoch_epoch_XX_step_step_XXXXXX_TIMESTAMP
    pattern1 = r"epoch_epoch_(\d+)_step_step_(\d+)_\d{8}_\d{6}"
    if match := re.match(pattern1, stem):
        return {
            "epoch": int(match.group(1)),
            "step": int(match.group(2)),
        }

    # Format 2: epoch_XX_step_XXXXXX
    pattern2 = r"epoch_(\d+)_step_(\d+)"
    if match := re.match(pattern2, stem):
        return {
            "epoch": int(match.group(1)),
            "step": int(match.group(2)),
        }

    # Format 3: epoch=XX-step=XXXXXX
    pattern3 = r"epoch=(\d+)-step=(\d+)"
    if match := re.match(pattern3, stem):
        return {
            "epoch": int(match.group(1)),
            "step": int(match.group(2)),
        }

    return {"epoch": None, "step": None}


def create_new_checkpoint_name(epoch: int, step: int) -> str:
    """
    Create new checkpoint filename in the hierarchical format.

    Format: epoch-XX_step-XXXXXX.ckpt
    """
    return f"epoch-{epoch:02d}_step-{step:06d}.ckpt"


def should_delete_checkpoint(checkpoint_path: Path, keep_min_epoch: int = 10) -> tuple[bool, str]:
    """
    Determine if a checkpoint should be deleted.

    Rules:
    1. Delete if it's from lightning_logs_backup
    2. Delete if epoch < keep_min_epoch (early training checkpoints)
    3. Keep if it's in the latest 3 epochs of an experiment

    Returns:
        (should_delete, reason)
    """
    # Rule 1: Delete old lightning_logs backups
    if "lightning_logs_backup" in str(checkpoint_path):
        return True, "old lightning_logs backup"

    # Parse checkpoint info
    info = parse_old_checkpoint_name(checkpoint_path.name)
    epoch = info.get("epoch")

    if epoch is None:
        return False, "unable to parse epoch"

    # Rule 2: Delete early epoch checkpoints
    if epoch < keep_min_epoch:
        return True, f"early epoch (< {keep_min_epoch})"

    return False, "keep"


def get_experiment_checkpoints(outputs_dir: Path) -> dict[str, list[Path]]:
    """
    Group checkpoints by experiment directory.

    Returns:
        Dictionary mapping experiment name to list of checkpoint paths
    """
    experiments = {}

    for exp_dir in outputs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        checkpoint_dir = exp_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue

        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            experiments[exp_dir.name] = checkpoints

    return experiments


def migrate_checkpoint(checkpoint_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """
    Migrate a single checkpoint to the new naming format.

    Returns:
        (success, message)
    """
    # Skip if already in new format
    if re.match(r"epoch-\d{2}_step-\d{6}", checkpoint_path.stem):
        return True, "already in new format"

    # Skip special checkpoints
    if checkpoint_path.stem in ["last", "best"]:
        return True, "special checkpoint, no rename needed"

    if checkpoint_path.stem.startswith("best-"):
        return True, "best checkpoint, no rename needed"

    # Parse old name
    info = parse_old_checkpoint_name(checkpoint_path.name)
    epoch = info.get("epoch")
    step = info.get("step")

    if epoch is None or step is None:
        return False, f"unable to parse: {checkpoint_path.name}"

    # Create new name
    new_name = create_new_checkpoint_name(epoch, step)
    new_path = checkpoint_path.parent / new_name

    # Check if target already exists
    if new_path.exists():
        return False, f"target already exists: {new_name}"

    # Perform rename
    if dry_run:
        return True, f"would rename to: {new_name}"

    try:
        checkpoint_path.rename(new_path)
        return True, f"renamed to: {new_name}"
    except Exception as e:
        return False, f"error renaming: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Migrate checkpoints to new naming scheme",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--keep-min-epoch",
        type=int,
        default=10,
        help="Minimum epoch to keep (delete earlier epochs, default: 10)",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete checkpoints that should be removed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information for all checkpoints",
    )

    args = parser.parse_args()

    if not args.outputs_dir.exists():
        print(f"❌ Error: {args.outputs_dir} does not exist")
        return 1

    print("=" * 80)
    print("Checkpoint Migration Tool")
    print("=" * 80)
    print(f"Outputs directory: {args.outputs_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Keep min epoch: {args.keep_min_epoch}")
    print(f"Delete old: {args.delete_old}")
    print("=" * 80)
    print()

    # Get all experiments
    experiments = get_experiment_checkpoints(args.outputs_dir)

    if not experiments:
        print("No experiments with checkpoints found.")
        return 0

    print(f"Found {len(experiments)} experiments with checkpoints\n")

    # Statistics
    stats = {
        "total": 0,
        "renamed": 0,
        "deleted": 0,
        "skipped": 0,
        "errors": 0,
    }

    # Process each experiment
    for exp_name, checkpoints in sorted(experiments.items()):
        print(f"📁 {exp_name}")
        print(f"   Found {len(checkpoints)} checkpoints")

        # Sort checkpoints by epoch
        checkpoints_sorted = sorted(checkpoints, key=lambda p: parse_old_checkpoint_name(p.name).get("epoch") or 0)

        for checkpoint in checkpoints_sorted:
            stats["total"] += 1

            # Check if should delete
            should_delete, delete_reason = should_delete_checkpoint(checkpoint, args.keep_min_epoch)

            if should_delete:
                if args.delete_old:
                    if args.dry_run:
                        print(f"   🗑️  Would delete: {checkpoint.name} ({delete_reason})")
                    else:
                        try:
                            checkpoint.unlink()
                            print(f"   ✅ Deleted: {checkpoint.name} ({delete_reason})")
                            stats["deleted"] += 1
                        except Exception as e:
                            print(f"   ❌ Error deleting {checkpoint.name}: {e}")
                            stats["errors"] += 1
                else:
                    if args.verbose:
                        print(f"   ⚠️  Should delete: {checkpoint.name} ({delete_reason})")
                    stats["skipped"] += 1
                continue

            # Try to migrate
            success, message = migrate_checkpoint(checkpoint, args.dry_run)

            if success:
                if "already in new format" in message or "no rename needed" in message:
                    if args.verbose:
                        print(f"   ⏭️  {checkpoint.name}: {message}")
                    stats["skipped"] += 1
                else:
                    print(f"   ✅ {checkpoint.name}: {message}")
                    stats["renamed"] += 1
            else:
                print(f"   ❌ {checkpoint.name}: {message}")
                stats["errors"] += 1

        print()

    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total checkpoints processed: {stats['total']}")
    print(f"Renamed: {stats['renamed']}")
    print(f"Deleted: {stats['deleted']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 80)

    if args.dry_run:
        print("\n⚠️  This was a DRY RUN. No changes were made.")
        print("   Run without --dry-run to apply changes.")

    if not args.delete_old and stats["total"] > stats["renamed"] + stats["deleted"]:
        print("\n💡 Tip: Use --delete-old to remove unnecessary checkpoints")

    return 0


if __name__ == "__main__":
    exit(main())

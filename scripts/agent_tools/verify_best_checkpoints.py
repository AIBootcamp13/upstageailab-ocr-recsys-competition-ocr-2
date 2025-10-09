#!/usr/bin/env python3
"""
Script to verify that best checkpoints are being saved correctly.
"""

import argparse
from pathlib import Path


def find_best_checkpoints(outputs_dir: Path) -> list[tuple[Path, str]]:
    """
    Find all checkpoint files that appear to be best checkpoints.

    Returns list of (path, type) tuples where type is 'best', 'last', or 'epoch'.
    """
    checkpoints = []

    for ckpt_path in outputs_dir.rglob("*.ckpt"):
        filename = ckpt_path.name.lower()

        if "last" in filename:
            checkpoints.append((ckpt_path, "last"))
        elif "best" in filename:
            checkpoints.append((ckpt_path, "best"))
        else:
            checkpoints.append((ckpt_path, "epoch"))

    return checkpoints


def analyze_checkpoint_naming(checkpoints: list[tuple[Path, str]]) -> None:
    """
    Analyze the checkpoint naming and provide recommendations.
    """
    best_count = sum(1 for _, ctype in checkpoints if ctype == "best")
    last_count = sum(1 for _, ctype in checkpoints if ctype == "last")
    epoch_count = sum(1 for _, ctype in checkpoints if ctype == "epoch")

    print("=== Checkpoint Analysis ===")
    print(f"Total checkpoints found: {len(checkpoints)}")
    print(f"Best checkpoints: {best_count}")
    print(f"Last checkpoints: {last_count}")
    print(f"Epoch checkpoints: {epoch_count}")
    print()

    if best_count == 0:
        print("❌ WARNING: No best checkpoints found!")
        print("   This means the ModelCheckpoint is not saving the best model.")
        print("   Check that:")
        print("   - 'val/hmean' metric is being logged during validation")
        print("   - ModelCheckpoint monitor is set to 'val/hmean'")
        print("   - save_top_k > 0")
        print("   - The metric is actually improving during training")
    else:
        print("✅ Best checkpoints found!")

    if last_count == 0:
        print("❌ WARNING: No last checkpoints found!")
        print("   Check that save_last=True in ModelCheckpoint config")
    else:
        print("✅ Last checkpoints found!")

    print()
    print("=== Detailed Checkpoint List ===")
    for path, ctype in sorted(checkpoints, key=lambda x: (x[1], x[0])):
        marker = "🏆" if ctype == "best" else "📁" if ctype == "last" else "📄"
        print(f"{marker} {ctype.upper()}: {path}")


def main():
    parser = argparse.ArgumentParser(description="Verify best checkpoint saving")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Outputs directory to search")

    args = parser.parse_args()

    if not args.outputs_dir.exists():
        print(f"Error: {args.outputs_dir} does not exist")
        return

    print(f"Analyzing checkpoints in {args.outputs_dir}")
    print("=" * 60)

    checkpoints = find_best_checkpoints(args.outputs_dir)

    if not checkpoints:
        print("No checkpoint files found.")
        return

    analyze_checkpoint_naming(checkpoints)


if __name__ == "__main__":
    main()

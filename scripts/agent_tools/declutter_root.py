#!/usr/bin/env python3
"""
Script to relocate lightning_logs and other clutter from root directory.
Moves TensorBoard logs and checkpoints to proper output directories.
"""

import shutil
from pathlib import Path


def relocate_lightning_logs():
    """Move lightning_logs contents to outputs directory."""
    lightning_logs_dir = Path("lightning_logs")
    outputs_dir = Path("outputs")

    if not lightning_logs_dir.exists():
        print("No lightning_logs directory found.")
        return

    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True)

    print(f"Found lightning_logs directory with {len(list(lightning_logs_dir.rglob('*')))} items")

    # Move each version to outputs/lightning_logs_backup/
    backup_dir = outputs_dir / "lightning_logs_backup"
    backup_dir.mkdir(exist_ok=True)

    moved_count = 0
    for version_dir in lightning_logs_dir.iterdir():
        if version_dir.is_dir() and version_dir.name.startswith("version_"):
            dst_dir = backup_dir / version_dir.name
            print(f"Moving {version_dir} -> {dst_dir}")
            try:
                shutil.move(str(version_dir), str(dst_dir))
                moved_count += 1
            except Exception as e:
                print(f"Error moving {version_dir}: {e}")

    # Remove the now-empty lightning_logs directory
    try:
        lightning_logs_dir.rmdir()
        print("Removed empty lightning_logs directory")
    except Exception as e:
        print(f"Could not remove lightning_logs directory: {e}")

    print(f"Successfully relocated {moved_count} version directories to {backup_dir}")


def relocate_wandb_offline_data():
    """Move W&B offline data directories to outputs."""
    # Common W&B offline directory names
    wandb_patterns = [
        "receipt-text-recognition-ocr-project",  # From config
        "offline-*",  # Generic offline pattern
    ]

    outputs_dir = Path("outputs")
    backup_dir = outputs_dir / "wandb_offline_backup"
    backup_dir.mkdir(exist_ok=True)

    moved_items = []

    for pattern in wandb_patterns:
        for item in Path(".").glob(pattern):
            if item.is_dir():
                print(f"Found W&B offline directory: {item}")
                dst_dir = backup_dir / item.name
                try:
                    shutil.move(str(item), str(dst_dir))
                    moved_items.append(str(item))
                    print(f"Moved {item} -> {dst_dir}")
                except Exception as e:
                    print(f"Error moving {item}: {e}")

    if moved_items:
        print(f"Moved {len(moved_items)} W&B offline directories to {backup_dir}")
    else:
        print("No W&B offline directories found")


def clean_other_clutter():
    """Clean up other common clutter files in root directory."""
    root_clutter_patterns = [
        "*.log",  # Log files
        "*.tmp",  # Temporary files
        "*.bak",  # Backup files
        "*.swp",  # Vim swap files
        "*.pyc",  # Python bytecode
        "__pycache__",  # Python cache directories
        ".DS_Store",  # macOS files
        "Thumbs.db",  # Windows files
    ]

    cleaned_items = []

    for pattern in root_clutter_patterns:
        for item in Path(".").glob(pattern):
            if item.is_file():
                try:
                    item.unlink()
                    cleaned_items.append(str(item))
                    print(f"Removed file: {item}")
                except Exception as e:
                    print(f"Could not remove {item}: {e}")
            elif item.is_dir() and item.name == "__pycache__":
                try:
                    shutil.rmtree(item)
                    cleaned_items.append(str(item))
                    print(f"Removed directory: {item}")
                except Exception as e:
                    print(f"Could not remove {item}: {e}")

    if cleaned_items:
        print(f"Cleaned up {len(cleaned_items)} clutter items")
    else:
        print("No clutter files found to clean")


def update_gitignore():
    """Ensure .gitignore includes lightning_logs and other clutter."""
    gitignore_path = Path(".gitignore")
    entries_to_add = [
        "\n# Lightning logs (should be in outputs/)",
        "lightning_logs/",
        "\n# Common clutter",
        "*.log",
        "*.tmp",
        "*.bak",
        "*.swp",
        "*.pyc",
        "__pycache__/",
        ".DS_Store",
        "Thumbs.db",
    ]

    if gitignore_path.exists():
        content = gitignore_path.read_text()
    else:
        content = ""

    added_entries = []
    for entry in entries_to_add:
        if entry.strip() and entry.strip() not in content:
            content += entry + "\n"
            added_entries.append(entry.strip())

    if added_entries:
        gitignore_path.write_text(content)
        print(f"Updated .gitignore with {len(added_entries)} new entries")
    else:
        print(".gitignore already up to date")


def main():
    print("=== Root Directory Declutter Script ===")
    print()

    print("1. Relocating lightning_logs...")
    relocate_lightning_logs()

    print("\n2. Relocating W&B offline data...")
    relocate_wandb_offline_data()

    print("\n3. Cleaning other clutter...")
    clean_other_clutter()

    print("\n4. Updating .gitignore...")
    update_gitignore()

    print("\n=== Declutter Complete ===")
    print("Root directory should now be cleaner!")
    print("Future logs will be saved to outputs/{exp_name}/logs/")


if __name__ == "__main__":
    main()

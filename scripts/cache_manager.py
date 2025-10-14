#!/usr/bin/env python3
"""
Cache Management Utility for OCR Dataset System

This script provides CLI commands for managing dataset caches including:
- Viewing cache status and health
- Clearing stale or invalid caches
- Analyzing cache performance
- Validating cache consistency

Usage:
    python scripts/cache_manager.py status
    python scripts/cache_manager.py clear --all
    python scripts/cache_manager.py health
    python scripts/cache_manager.py validate --config configs/data/base.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocr.utils.path_utils import setup_project_paths

setup_project_paths()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_cache_directory() -> Path:
    """Get the default cache directory."""
    return Path("/tmp/ocr_cache")


def clear_cache(cache_type: str = "all", dry_run: bool = False):
    """Clear cache files.

    Args:
        cache_type: Type of cache to clear ("all", "tensor", "image", "maps")
        dry_run: If True, only show what would be deleted without actually deleting
    """
    cache_dir = get_cache_directory()

    if not cache_dir.exists():
        print(f"✓ No cache directory found at {cache_dir}")
        return

    if cache_type == "all":
        print(f"{'[DRY RUN] ' if dry_run else ''}Clearing all caches in {cache_dir}...")
        if not dry_run:
            import shutil

            shutil.rmtree(cache_dir)
            print(f"✓ All caches cleared from {cache_dir}")
        else:
            print(f"[DRY RUN] Would remove directory: {cache_dir}")
    else:
        print(f"{'[DRY RUN] ' if dry_run else ''}Clearing {cache_type} cache...")
        # This is a simplified version - actual implementation would need to
        # distinguish between different cache types based on file patterns
        print("⚠️  Specific cache type clearing not yet implemented. Use --all instead.")


def show_cache_status():
    """Display current cache status."""
    cache_dir = get_cache_directory()

    if not cache_dir.exists():
        print(f"✓ No cache directory at {cache_dir}")
        print("Cache Status: CLEAN")
        return

    print("=" * 60)
    print("CACHE STATUS")
    print("=" * 60)
    print(f"Cache Directory: {cache_dir}")
    print(f"Exists: {cache_dir.exists()}")

    if cache_dir.exists():
        # Count files and estimate size
        total_files = 0
        total_size = 0

        for item in cache_dir.rglob("*"):
            if item.is_file():
                total_files += 1
                total_size += item.stat().st_size

        print(f"Total Files: {total_files}")
        print(f"Total Size: {total_size / (1024**3):.2f} GB")

        # Show recent cache activity
        if total_files > 0:
            recent_files = sorted(cache_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            print("\nRecent Cache Files (last 5):")
            for f in recent_files:
                if f.is_file():
                    mtime = f.stat().st_mtime
                    size = f.stat().st_size
                    from datetime import datetime

                    mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  - {f.name} ({size / 1024:.1f} KB, modified: {mod_time})")
    print("=" * 60)


def show_cache_health():
    """Analyze and display cache health information."""
    print("=" * 60)
    print("CACHE HEALTH ANALYSIS")
    print("=" * 60)

    cache_dir = get_cache_directory()

    if not cache_dir.exists():
        print("Status: ✓ HEALTHY (No cache)")
        print("Recommendation: Cache will be created on first training run")
        print("=" * 60)
        return

    # Basic health checks
    warnings = []
    errors = []

    # Check cache size
    total_size = 0
    for item in cache_dir.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size

    size_gb = total_size / (1024**3)

    if size_gb > 10:
        warnings.append(f"⚠️  Large cache size: {size_gb:.1f} GB (> 10 GB threshold)")
    elif size_gb > 20:
        errors.append(f"🚨 Extremely large cache: {size_gb:.1f} GB (> 20 GB threshold)")

    # Check for stale caches (older than 7 days)
    from datetime import datetime, timedelta

    cutoff_date = datetime.now() - timedelta(days=7)
    stale_files = []

    for item in cache_dir.rglob("*"):
        if item.is_file():
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            if mtime < cutoff_date:
                stale_files.append(item)

    if len(stale_files) > 0:
        warnings.append(f"⚠️  Found {len(stale_files)} stale cache files (> 7 days old)")

    # Display results
    if errors:
        print("Status: 🚨 CRITICAL ISSUES DETECTED")
        for error in errors:
            print(f"  {error}")
    elif warnings:
        print("Status: ⚠️  WARNINGS DETECTED")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("Status: ✓ HEALTHY")

    print(f"\nCache Size: {size_gb:.2f} GB")
    print(f"Stale Files: {len(stale_files)}")

    if warnings or errors:
        print("\nRecommendations:")
        if size_gb > 10:
            print("  • Consider clearing cache with: python scripts/cache_manager.py clear --all")
        if len(stale_files) > 0:
            print("  • Stale cache may be invalid - clear before training")

    print("=" * 60)


def validate_cache_config(config_path: Path):
    """Validate cache configuration against current cache state.

    Args:
        config_path: Path to dataset configuration YAML file
    """
    print("=" * 60)
    print("CACHE CONFIGURATION VALIDATION")
    print("=" * 60)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    # This would require parsing the YAML and checking cache version
    # For now, provide a placeholder
    print(f"Config file: {config_path}")
    print("⚠️  Configuration validation not yet fully implemented")
    print("\nManual validation checklist:")
    print("  1. Check cache_transformed_tensors setting matches intended use")
    print("  2. Verify load_maps setting if using pre-computed maps")
    print("  3. Ensure cache is cleared after config changes")
    print("  4. Validate cache version in logs during training")
    print("=" * 60)


def export_cache_stats(output_path: Path):
    """Export cache statistics to JSON file.

    Args:
        output_path: Path to output JSON file
    """
    cache_dir = get_cache_directory()

    stats = {
        "cache_directory": str(cache_dir),
        "exists": cache_dir.exists(),
        "total_files": 0,
        "total_size_bytes": 0,
        "total_size_gb": 0,
        "recent_files": [],
    }

    if cache_dir.exists():
        for item in cache_dir.rglob("*"):
            if item.is_file():
                stats["total_files"] += 1
                stats["total_size_bytes"] += item.stat().st_size

        stats["total_size_gb"] = stats["total_size_bytes"] / (1024**3)

        # Get 10 most recent files
        recent_files = sorted(cache_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        for f in recent_files:
            if f.is_file():
                from datetime import datetime

                stats["recent_files"].append(
                    {
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    }
                )

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Cache statistics exported to {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage OCR dataset caches", formatter_class=argparse.RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    subparsers.add_parser("status", help="Show current cache status")

    # Health command
    subparsers.add_parser("health", help="Analyze cache health and provide recommendations")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache files")
    clear_parser.add_argument("--all", action="store_true", help="Clear all caches")
    clear_parser.add_argument("--type", choices=["tensor", "image", "maps"], help="Specific cache type to clear")
    clear_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate cache configuration")
    validate_parser.add_argument("--config", type=Path, help="Path to configuration file")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export cache statistics to JSON")
    export_parser.add_argument("--output", type=Path, default=Path("cache_stats.json"), help="Output JSON file path")

    # Global options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == "status":
        show_cache_status()
    elif args.command == "health":
        show_cache_health()
    elif args.command == "clear":
        if args.all:
            clear_cache(cache_type="all", dry_run=args.dry_run)
        elif args.type:
            clear_cache(cache_type=args.type, dry_run=args.dry_run)
        else:
            print("Error: Must specify --all or --type")
            sys.exit(1)
    elif args.command == "validate":
        if args.config:
            validate_cache_config(args.config)
        else:
            print("Error: Must specify --config")
            sys.exit(1)
    elif args.command == "export":
        export_cache_stats(args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

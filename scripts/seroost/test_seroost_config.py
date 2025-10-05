# Test the Seroost configuration

import json
from pathlib import Path


def test_seroost_config():
    """Test that the seroost configuration is valid."""

    # Load the configuration
    config_path = Path(__file__).parent.parent.parent / "configs" / "tools" / "seroost_config.json"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return

    with open(config_path) as f:
        config = json.load(f)

    print("Seroost configuration loaded successfully.")
    print(f"Include patterns: {len(config.get('index', {}).get('include', []))}")
    print(f"Exclude patterns: {len(config.get('index', {}).get('exclude', []))}")

    # Test a few key patterns
    include_patterns = config.get("index", {}).get("include", [])
    exclude_patterns = config.get("index", {}).get("exclude", [])

    # Check if key patterns are present
    expected_includes = ["**/*.py", "**/*.yaml", "**/*.md", "**/docs/**/*", "**/configs/**/*"]

    expected_excludes = ["**/__pycache__/**", "**/logs/**", "**/venv/**", "**/*.log", "**/data/datasets/**"]

    print("\nVerifying expected include patterns:")
    for pattern in expected_includes:
        if pattern in include_patterns:
            print(f"  ✓ {pattern}")
        else:
            print(f"  ✗ {pattern} (missing)")

    print("\nVerifying expected exclude patterns:")
    for pattern in expected_excludes:
        if pattern in exclude_patterns:
            print(f"  ✓ {pattern}")
        else:
            print(f"  ✗ {pattern} (missing)")


if __name__ == "__main__":
    test_seroost_config()

#!/usr/bin/env python3
"""Validate UI inference compatibility schema."""

import sys
from pathlib import Path

import yaml


def validate_schema(schema_path: Path) -> bool:
    """
    Validate the UI inference compatibility schema.

    Args:
        schema_path: Path to the schema YAML file

    Returns:
        True if validation passes, False otherwise
    """
    try:
        with open(schema_path, encoding="utf-8") as f:
            schema = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load schema: {e}")
        return False

    families = schema.get("model_families", [])
    if not families:
        print("❌ No model families found in schema")
        return False

    print(f"Found {len(families)} model families\n")

    errors = []
    warnings = []

    # Check for required fields
    for i, family in enumerate(families):
        family_id = family.get("id", f"<unnamed-{i}>")

        # Required fields
        if "id" not in family:
            errors.append(f"Family {i}: Missing 'id'")
            continue

        if "encoder" not in family:
            errors.append(f"{family_id}: Missing 'encoder'")
        if "decoder" not in family:
            errors.append(f"{family_id}: Missing 'decoder'")
        if "head" not in family:
            errors.append(f"{family_id}: Missing 'head'")

        # Check encoder
        encoder = family.get("encoder", {})
        if "model_names" not in encoder:
            errors.append(f"{family_id}: Missing encoder 'model_names'")
        elif not isinstance(encoder["model_names"], list):
            errors.append(f"{family_id}: encoder 'model_names' must be a list")
        elif not encoder["model_names"]:
            warnings.append(f"{family_id}: encoder 'model_names' is empty")
        else:
            encoder_names = encoder["model_names"]
            print(f"✓ {family_id}: {len(encoder_names)} encoder(s) - {', '.join(encoder_names)}")

        # Check decoder
        decoder = family.get("decoder", {})
        if "class" not in decoder:
            errors.append(f"{family_id}: Missing decoder 'class'")
        if "in_channels" not in decoder:
            errors.append(f"{family_id}: Missing decoder 'in_channels'")
        elif not isinstance(decoder["in_channels"], list):
            errors.append(f"{family_id}: decoder 'in_channels' must be a list")

        if "output_channels" not in decoder:
            warnings.append(f"{family_id}: Missing decoder 'output_channels' (optional but recommended)")
        if "inner_channels" not in decoder:
            warnings.append(f"{family_id}: Missing decoder 'inner_channels' (optional but recommended)")

        # Check head
        head = family.get("head", {})
        if "class" not in head:
            errors.append(f"{family_id}: Missing head 'class'")

        # Optional description
        if "description" not in family:
            warnings.append(f"{family_id}: Missing 'description' (recommended for documentation)")

    # Print warnings
    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")

    # Print errors
    if errors:
        print(f"\n❌ {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return False

    print(f"\n✅ Schema validation passed! All {len(families)} families are valid.")
    return True


def main():
    schema_path = Path("configs/schemas/ui_inference_compat.yaml")

    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        print("   Make sure you're running this from the project root directory.")
        return 1

    if validate_schema(schema_path):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

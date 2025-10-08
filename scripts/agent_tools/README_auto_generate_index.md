# AI Handbook Index Auto-Generation Script

This script automatically generates the `index.json` file for the AI handbook by scanning the directory structure and extracting metadata from Markdown files.

## Usage

```bash
python scripts/agent_tools/auto_generate_index.py [options]
```

## Options

- `--handbook-dir PATH`: Path to the ai_handbook directory (default: `docs/ai_handbook`)
- `--output PATH`: Output path for the generated index.json (default: `docs/ai_handbook/index.json`)
- `--validate`: Run validation after generation using the existing validation script

## What it does

1. **Scans directories**: Recursively scans all subdirectories in the handbook directory
2. **Extracts metadata**: Reads titles and other metadata from Markdown files
3. **Generates entries**: Creates entry objects with proper IDs, paths, sections, tags, and priorities
4. **Creates bundles**: Organizes entries into logical bundles based on content tags
5. **Validates output**: Optionally runs the validation script to ensure the generated index is correct

## Directory Structure

The script expects the handbook to be organized with numbered directories (01_onboarding, 02_protocols, etc.) containing Markdown files.

## Validation

The generated index.json is validated against the existing validation rules to ensure:
- All required fields are present
- Paths exist and are within the handbook directory
- Bundle references are valid
- Schema compliance

## Integration

This script replaces manual maintenance of the index.json file. After reorganizing directories or adding new content, simply run this script to update the index automatically.

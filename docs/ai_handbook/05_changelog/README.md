# Changelog Directory Organization

This directory contains changelog entries documenting changes, fixes, and developments in the project.

## Folder Structure
- `2025-XX/` - Monthly folders containing changelog entries for that month
- `debugging/` - Debugging sessions and logs organized by date
- `_deprecated/` - Deprecated changelog entries

## Naming Convention
All changelog files must follow this format:
```
DD_descriptive-name.md
```

Where:
- `DD` is the two-digit day of the month (01-31)
- `descriptive-name` is a lowercase kebab-case description of the change
- Use hyphens (-) instead of underscores (_) for word separation
- Avoid special characters, keep it simple and descriptive

### Examples
- `01_cleval-config-preset.md`
- `03_command-builder-refactor-progress.md`
- `07_summary-hydra-config-issues-fixes.md`

## For AI Agents
When creating new changelog entries:
1. Place files in the appropriate monthly folder (e.g., `2025-10/` for October 2025)
2. Use the exact naming format: `DD_descriptive-name.md`
3. Ensure the descriptive name is concise but clear
4. If it's a debugging session, place it in `debugging/YYYY-MM-DD_description/`

This convention ensures chronological ordering and easy navigation.

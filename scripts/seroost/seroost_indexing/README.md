# Seroost Indexing Configuration

This project includes a comprehensive Seroost indexing configuration that's optimized for the OCR project. The configuration ensures efficient indexing of relevant files while excluding unnecessary artifacts.

## Files

- `seroost_config.json`: The main indexing configuration file
- `setup_seroost_indexing.py`: A Python script to set up and run the indexing
- `run_seroost_indexing.sh`: A shell script to execute the indexing process

## Included File Types

The configuration includes:

- **Source Code**: Python files, configuration files, shell scripts
- **Documentation**: Markdown files, documentation files in the docs/ directory
- **Configuration**: YAML, TOML, JSON, configuration files in the configs/ directory
- **Tests**: Test files in the tests/ directory
- **UI Components**: Files in the ui/ directory
- **Scripts**: Shell and Python scripts in the scripts/ directory

## Excluded File Types

The configuration excludes:

- **Virtual Environments**: `.venv/`, `venv/`, `env/` directories
- **Cache Directories**: `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, etc.
- **Build Artifacts**: `build/`, `dist/`, `*.egg-info/`, compiled files
- **Logs**: `.log` files and `logs/` directories
- **Data Files**: Large data files, images, video, audio files
- **Model Checkpoints**: `*.pth`, `*.ckpt`, and output directories
- **Dependencies**: `node_modules/`, and other dependency directories
- **Deprecated/Archive**: `DEPRECATED/`, `_deprecated/`, `_archive/` directories

## Usage

To set up and run the indexing:

```bash
python setup_seroost_indexing.py
```

Or using the shell script:

```bash
chmod +x run_seroost_indexing.sh
./run_seroost_indexing.sh
```

## Notes

- The indexing process may take several minutes for large codebases
- The configuration is designed to optimize search performance by excluding irrelevant files
- Regular expressions and glob patterns follow standard conventions
- For large projects, consider running indexing during off-peak hours

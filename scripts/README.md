# Scripts Directory

This directory contains various utility scripts and tools for the OCR project.

## Quick Reference

### Checkpoint Migration
Migrate existing checkpoints to the new naming scheme:
```bash
# Dry run (preview changes)
python scripts/migrate_checkpoints.py --dry-run --verbose

# Migrate and cleanup (delete early epochs)
python scripts/migrate_checkpoints.py --delete-old
```

See `docs/CHECKPOINT_MIGRATION_GUIDE.md` for details.

### UI Schema Validation
Validate the UI inference compatibility schema:
```bash
# Validate schema correctness
python scripts/validate_ui_schema.py
```

See `docs/UI_INFERENCE_COMPATIBILITY_SCHEMA.md` for details.

---

## Organization

### Root Scripts

#### `migrate_checkpoints.py` ⭐ NEW
Migrate checkpoints to the new hierarchical naming scheme.

**Features:**
- Rename checkpoints: `epoch_epoch_XX_...` → `epoch-XX_step-XXXXXX.ckpt`
- Delete early-epoch checkpoints (configurable threshold)
- Dry-run mode for safe preview
- Detailed reporting and statistics

**Usage:**
```bash
# Preview changes
python migrate_checkpoints.py --dry-run

# Apply changes
python migrate_checkpoints.py --delete-old --keep-min-epoch 10
```

See `docs/CHECKPOINT_MIGRATION_GUIDE.md` for full documentation.

#### `validate_ui_schema.py` ⭐ NEW
Validate the UI inference compatibility schema.

**Features:**
- Validate schema YAML syntax
- Check for required fields in each model family
- Verify encoder, decoder, and head configurations
- Report errors and warnings

**Usage:**
```bash
# Validate schema
python validate_ui_schema.py
```

See `docs/UI_INFERENCE_COMPATIBILITY_SCHEMA.md` for full documentation.

### `agent_tools/`
Scripts and tools for AI agent integration and automation.

### `analysis_validation/`
Scripts for analyzing experiments, validating pipeline contracts, and profiling performance.
- `analyze_experiment.py` - Analyze training runs and detect anomalies
- `validate_pipeline_contracts.py` - Validate data contracts across OCR pipeline
- `profile_data_loading.py` - Profile data loading performance
- `profile_transforms.py` - Profile data transformation performance

### `data_processing/`
Scripts for preprocessing and transforming data.
- `preprocess_maps.py` - Generate probability/threshold maps for training
- `fix_canonical_orientation_images.py` - Fix image orientation issues
- `report_orientation_mismatches.py` - Report orientation mismatch issues

### `migration_refactoring/`
Scripts for migrating data formats and refactoring code.
- `migrate_checkpoint_names.py` - Migrate checkpoint filenames to new format
- `refactor_ocr_pl.py` - Automated refactoring of OCR Lightning module

### `monitoring/`
System monitoring and resource management tools.
- `monitor.sh` - AI-powered system monitoring via Qwen MCP server
- `monitor_resources.sh` - Resource monitoring script
- `process_monitor.py` - Process monitoring utilities

### `performance/`
Performance analysis and baseline reporting tools.
- `generate_baseline_report.py` - Generate performance baseline reports

### `performance_benchmarking/`
Scripts for benchmarking and measuring performance.
- `benchmark_optimizations.py` - Benchmark data loading optimizations
- `benchmark_validation.py` - Validate benchmarking results
- `decoder_benchmark.py` - Benchmark decoder performance
- `performance_measurement.py` - General performance measurements
- `performance_test.py` - Performance testing utilities
- `quick_performance_validation.py` - Quick performance validation

### `seroost/`
Seroost semantic search indexing tools and configuration.
- `SEROOST_INDEXING_SETUP.md` - Complete setup documentation
- `setup_seroost_indexing.py` - Python script for indexing setup
- `test_seroost_config.py` - Configuration validation script
- `run_seroost_indexing.sh` - Shell script for running indexing
- `install_and_run_seroost.sh` - Complete installation and setup script
- `seroost_indexing/` - Additional indexing documentation

### `setup/`
Project setup and configuration scripts.
- `00_setup-environment.sh` - Environment setup
- `01_setup-professional-linting.sh` - Linting setup
- `02_setup-bash-aliases.sh` - Bash aliases setup
- `code-quality.sh` - Automated code quality maintenance
- `qwen-advanced.sh` - Advanced Qwen memory optimization
- `qwen-memfix.sh` - Qwen memory fixes
- `qwen-version.sh` - Qwen version management
- `setup_mcp_github_only.sh` - MCP GitHub setup

## Usage

### System Monitoring
```bash
# AI-powered system monitoring
./scripts/monitoring/monitor.sh "Show system health status"

# Resource monitoring
./scripts/monitoring/monitor_resources.sh
```

### Seroost Indexing
```bash
# Complete setup (install + index)
./scripts/seroost/install_and_run_seroost.sh

# Just run indexing (if seroost is already installed)
./scripts/seroost/run_seroost_indexing.sh

# Manual setup
python ./scripts/seroost/setup_seroost_indexing.py
```

## Contributing

When adding new scripts:
1. Place them in the appropriate subdirectory
2. Update this README with documentation
3. Ensure scripts are executable (`chmod +x script.sh`)
4. Include usage examples and descriptions

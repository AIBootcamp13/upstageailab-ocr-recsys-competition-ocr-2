# Debug Scripts Directory

This directory contains debugging, verification, and performance testing scripts that are useful for development but are not part of the main test suite.

## Directory Structure

### `bug_reproduction/`
Scripts for reproducing and diagnosing bugs during development.

- `test_albumentations_contract.py` - Tests Albumentations transform contract to understand transform bugs
- `test_bug_reproduction.py` - Reproduces PIL Image vs numpy array bug in transforms
- `test_collate_bug.py` - Diagnoses collate function polygon handling issues

### `verification/`
Scripts for verifying that bug fixes and features work correctly.

- `test_bug_fix_verification.py` - Verifies fix for PIL Image handling in DBTransforms
- `test_load_maps_disabled.py` - Verifies load_maps=False parameter functionality
- `verify_cache_implementation.py` - Verifies tensor cache statistics implementation

### `performance/`
Scripts for performance testing and optimization.

- `test_transform_optimizations.py` - Tests different transform configurations for performance

## Usage

These scripts are typically run manually during development and debugging:

```bash
# Run a verification script
python debug/verification/test_bug_fix_verification.py

# Run a bug reproduction script
python debug/bug_reproduction/test_albumentations_contract.py

# Run a performance test
python debug/performance/test_transform_optimizations.py
```

## Note

These scripts are not part of the automated test suite and may require specific test data or conditions to run successfully. They are primarily for development and debugging purposes.

# # **filename: docs/ai_handbook/05_changelog/2025-10/15_checkpoint_naming_implementation.md**

**Date**: 2025-10-15
**Type**: Feature Enhancement
**Component**: Training Infrastructure, PyTorch Lightning Callbacks
**Impact**: High - Affects all training runs and checkpoint management

## **Summary**

Implemented an enhanced checkpoint naming scheme for better organization, clarity, and management of model checkpoints. The new hierarchical structure provides clear experiment identification, consistent formatting, and improved searchability.

## **Changes Made**

### **1. Enhanced Checkpoint Callback**

**File**: `ocr/lightning_modules/callbacks/unique_checkpoint.py`

- Complete rewrite of `UniqueModelCheckpoint` class (~200 lines changed)
- Added experiment tag and training phase parameters
- Implemented hierarchical directory structure
- Simplified checkpoint filenames removing redundancy
- Added model architecture extraction from checkpoint metadata

**Key Methods**:
- `format_checkpoint_name()`: Creates clean names (epoch-XX_step-XXXXXX.ckpt)
- `_setup_dirpath()`: Builds hierarchical directory structure
- `setup()`: Configures directory with model info extraction

### **2. Configuration Updates**

**File**: `configs/callbacks/model_checkpoint.yaml`

Added new parameters:
- `experiment_tag`: Unique experiment identifier (supports env var override)
- `training_phase`: Training phase/stage (training, validation, finetuning, etc.)
- Comprehensive documentation for all parameters

### **3. Migration Script**

**File**: `scripts/migrate_checkpoints.py` (~300 lines)

Created comprehensive migration tool with features:
- Dry-run mode for safe preview
- Parses 3 different old checkpoint formats
- Configurable deletion threshold for early epochs
- Detailed logging and error handling
- Preserves special checkpoints (best, last)

**Results**:
- 39 checkpoints processed
- 8 checkpoints renamed to new format
- 31 unnecessary early-epoch checkpoints deleted
- 0 errors during migration

### **4. Cleanup Operations**

- Removed `lightning_logs_backup` directory (obsolete backup)
- Cleaned up old checkpoint files following new retention policy
- All remaining checkpoints follow new naming convention

## **New Checkpoint Structure**

### **Directory Format**

```
outputs/<experiment_tag>-<model>_<phase>_<timestamp>/
├── checkpoints/
│   ├── epoch-XX_step-XXXXXX.ckpt
│   ├── last.ckpt
│   └── best-metricname-X.XXXX.ckpt
├── logs/
└── submissions/
```

### **Naming Examples**

**Before**:
```
epoch_epoch_22_step_step_001932_20251009_015037.ckpt
epoch_epoch_02_step_step_000176_20251009_011245.ckpt
```

**After**:
```
ocr_baseline_v1-dbnet-resnet18_training_20251015_120000/checkpoints/
├── epoch-22_step-001932.ckpt
├── last.ckpt
└── best-hmean-0.8920.ckpt
```

## **Benefits**

1. **Clarity**: Each component provides actionable information
2. **Consistency**: Standardized separators and formatting
3. **Searchability**: Easy filtering with glob patterns and find commands
4. **Automation-Friendly**: Scripts can parse and manage checkpoints easily
5. **Version Control**: Track experiment iterations over time

## **Usage**

### **Training with New Naming**

```bash
export EXPERIMENT_TAG="ocr_improved_baseline_v1"
python runners/train.py preset=example
```

### **Searching Checkpoints**

```bash
# Find all phase1 experiments
ls outputs/ocr_pl_refactor_phase1-*

# Find all ResNet18 checkpoints
ls outputs/*-resnet18_*/checkpoints/

# Find all best checkpoints
find outputs -name "best-*.ckpt"
```

## **Documentation Created**

1. **Checkpoint Naming Scheme Reference**
   - Location: `docs/ai_handbook/03_references/architecture/07_checkpoint_naming_scheme.md`
   - Content: Complete specification, examples, configuration options
   - Lines: 500+

2. **Checkpoint Migration Protocol**
   - Location: `docs/ai_handbook/02_protocols/components/18_checkpoint_migration_protocol.md`
   - Content: Step-by-step migration procedure, troubleshooting
   - Lines: 400+

3. **Scripts Documentation**
   - Updated: `scripts/README.md`
   - Added migration script documentation and usage examples

## **Breaking Changes**

**None** - The new system is backward compatible:
- Old checkpoints can still be loaded
- Migration is optional (recommended)
- Existing training runs unaffected

## **Migration Path**

For users with existing checkpoints:

```bash
# 1. Preview migration
python scripts/migrate_checkpoints.py --dry-run --verbose

# 2. Execute migration
python scripts/migrate_checkpoints.py --delete-old

# 3. Verify results
find outputs -name "*.ckpt" -type f | sort
```

## **Testing**

- ✅ Migration script tested on 39 existing checkpoints
- ✅ Dry-run mode verified
- ✅ Checkpoint loading tested (PyTorch Lightning)
- ✅ No linting errors
- ✅ Type annotations verified

## **Related Changes**

- See also: [UI Schema Updates](./15_ui_schema_updates.md) for related UI compatibility work

## **Future Work**

- Implement automated cleanup cron job
- Add checkpoint compression for archived experiments
- Create checkpoint catalog service for UI
- Add checkpoint versioning metadata

## **References**

- [Checkpoint Naming Scheme](../../03_references/architecture/07_checkpoint_naming_scheme.md)
- [Migration Protocol](../../02_protocols/components/18_checkpoint_migration_protocol.md)
- [Training Protocol](../../02_protocols/components/13_training_protocol.md)

---

**Author**: AI Agent
**Reviewers**: Core Team
**Related PRs**: N/A (direct commit to branch)

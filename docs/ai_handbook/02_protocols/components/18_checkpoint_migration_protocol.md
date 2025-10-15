# # **filename: docs/ai_handbook/02_protocols/components/18_checkpoint_migration_protocol.md**

<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=when_migrating_or_cleaning_up_old_checkpoints -->

## **Overview**

This protocol guides you through migrating existing checkpoints from old naming schemes to the new hierarchical format, and managing checkpoint cleanup to save disk space.

## **Prerequisites**

- Python environment set up with project dependencies
- Access to the `outputs/` directory containing checkpoints
- Understanding of [Checkpoint Naming Scheme](../../03_references/architecture/07_checkpoint_naming_scheme.md)
- **(Optional)** Backup of important checkpoints created

## **Procedure**

### **Step 1: Backup Important Checkpoints** (Recommended)

Before migration, create a backup:

```bash
# Create backup of entire outputs directory
cp -r outputs outputs_backup_$(date +%Y%m%d)

# Or backup specific experiments
cp -r outputs/important_experiment outputs_backup/
```

### **Step 2: Run Dry-Run Migration**

Preview changes without modifying files:

```bash
# See what would be changed
python scripts/migrate_checkpoints.py --dry-run --verbose

# Expected output format:
# Processing experiment: ocr_training
#   ✓ Renamed: epoch_epoch_22_step_step_001932_20251009_015037.ckpt
#            → epoch-22_step-001932.ckpt
#   ⊗ Deleted: epoch_epoch_02_step_step_000176_20251009_011245.ckpt (epoch < 10)
```

### **Step 3: Review Changes**

Check the dry-run output for:
- Checkpoints that will be renamed
- Checkpoints that will be deleted
- Any errors or warnings

### **Step 4: Execute Migration**

Once satisfied with the preview:

#### Option A: Rename Only (No Deletion)
```bash
python scripts/migrate_checkpoints.py
```

#### Option B: Rename and Delete Old Checkpoints
```bash
# Delete early epoch checkpoints (< epoch 10)
python scripts/migrate_checkpoints.py --delete-old

# Or use custom threshold
python scripts/migrate_checkpoints.py --delete-old --keep-min-epoch 15
```

### **Step 5: Verify Migration Results**

```bash
# List all remaining checkpoints
find outputs -name "*.ckpt" -type f | sort

# Count checkpoints per experiment
for dir in outputs/*/checkpoints; do
    echo "$(basename $(dirname $dir)): $(ls $dir/*.ckpt 2>/dev/null | wc -l) checkpoints"
done

# Check specific experiment
ls -lh outputs/your_experiment/checkpoints/
```

### **Step 6: Test Checkpoint Loading**

Verify migrated checkpoints work correctly:

```bash
# Start inference UI
python run_ui.py --app inference

# Load a migrated checkpoint and test inference
# OR

# Test programmatically
python -c "
import torch
ckpt = torch.load('outputs/experiment/checkpoints/epoch-18_step-003895.ckpt')
print(f'Loaded checkpoint with epoch: {ckpt[\"epoch\"]}')
"
```

### **Step 7: Clean Up**

Once verified, remove backups:

```bash
# Remove backup if everything works
rm -rf outputs_backup_*

# Or keep for longer if unsure
mv outputs_backup_* /archive/
```

## **Validation**

✅ **Success Criteria:**
- All checkpoints renamed to new format
- Early epoch checkpoints deleted (if requested)
- Migrated checkpoints load successfully
- No errors in checkpoint discovery UI
- Training can resume from migrated checkpoints

❌ **Failure Indicators:**
- Parse errors during migration
- Checkpoints fail to load
- UI cannot discover checkpoints
- Training crashes on checkpoint loading

## **Troubleshooting**

### **Issue: "Unable to parse epoch"**

**Cause**: Checkpoint filename doesn't match any known format.

**Solution**:
1. Check if it's a special checkpoint (best, last) - these are skipped automatically
2. Manually inspect filename: `ls outputs/experiment/checkpoints/`
3. If needed, manually rename following the pattern: `epoch-XX_step-XXXXXX.ckpt`
4. Skip if checkpoint is not important

### **Issue: "Target already exists"**

**Cause**: Multiple checkpoints would map to the same new name.

**Solution**:
1. This indicates duplicate checkpoints from different runs
2. Compare file sizes and timestamps: `ls -lh outputs/experiment/checkpoints/`
3. Keep the better checkpoint (usually larger/newer)
4. Delete duplicates before re-running migration

### **Issue: Wrong Checkpoints Deleted**

**Cause**: `keep-min-epoch` threshold too aggressive.

**Solution**:
1. Restore from backup: `cp -r outputs_backup/* outputs/`
2. Adjust parameter: `--keep-min-epoch 5` (lower threshold)
3. Run dry-run again: `--dry-run --verbose`
4. Review carefully before executing

### **Issue: Migration Script Fails**

**Cause**: Permission errors, disk space, or corrupted checkpoints.

**Solution**:
1. Check disk space: `df -h outputs/`
2. Check permissions: `ls -la outputs/`
3. Check for corrupted files: `find outputs -name "*.ckpt" -size 0`
4. Remove or move corrupted files before retrying

## **Related Documents**

- [Checkpoint Naming Scheme](../../03_references/architecture/07_checkpoint_naming_scheme.md) - Complete naming convention reference
- [Training Protocol](./13_training_protocol.md) - How to use checkpoints in training
- [UI Inference Compatibility Schema](../../03_references/guides/ui_inference_compatibility_schema.md) - Checkpoint discovery in UI

## **Configuration**

### **Migration Script Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--outputs-dir` | `outputs` | Path to outputs directory |
| `--dry-run` | `False` | Preview changes without modifying files |
| `--keep-min-epoch` | `10` | Minimum epoch to keep (delete earlier) |
| `--delete-old` | `False` | Delete unnecessary checkpoints |
| `--verbose` | `False` | Show detailed info for all checkpoints |

### **Format Migration Rules**

The script recognizes and converts three old formats:

| Old Format | Pattern | New Format |
|------------|---------|------------|
| Format 1 | `epoch_epoch_NN_step_step_SSSSSS_TIMESTAMP.ckpt` | `epoch-NN_step-SSSSSS.ckpt` |
| Format 2 | `epoch_NN_step_SSSSSS.ckpt` | `epoch-NN_step-SSSSSS.ckpt` |
| Format 3 | `epoch=NN-step=SSSSSS.ckpt` | `epoch-NN_step-SSSSSS.ckpt` |

**Special checkpoints never modified**:
- `last.ckpt`
- `best-*.ckpt`
- Checkpoints already in new format

### **Deletion Rules**

When `--delete-old` is enabled, checkpoints are deleted if:
- Epoch number < `keep-min-epoch` threshold
- NOT a special checkpoint (last/best)
- NOT the only checkpoint in the experiment

## **Best Practices**

### **1. Always Run Dry-Run First**

```bash
python scripts/migrate_checkpoints.py --dry-run --verbose
```

Review all changes before applying them.

### **2. Backup Before Major Migrations**

Create timestamped backups:

```bash
backup_name="outputs_backup_$(date +%Y%m%d_%H%M%S)"
cp -r outputs "$backup_name"
echo "Backup created: $backup_name"
```

### **3. Migrate in Stages**

For large numbers of checkpoints:

```bash
# Stage 1: Rename only
python scripts/migrate_checkpoints.py

# Stage 2: Verify loading works
# Test checkpoints in UI or scripts

# Stage 3: Clean up old epochs
python scripts/migrate_checkpoints.py --delete-old
```

### **4. Keep Experiment Logs**

Document migrations in your experiment log:

```markdown
## 2025-10-15: Checkpoint Migration
- Migrated 39 checkpoints to new naming scheme
- Deleted 31 early-epoch checkpoints (< epoch 10)
- Kept 8 mature checkpoints for continued use
```

### **5. Update References**

After migration, update any hardcoded checkpoint paths in:
- Training scripts
- Inference scripts
- Jupyter notebooks
- Documentation

## **Cleanup Strategies**

### **Strategy 1: Remove Early Epoch Checkpoints**

Keep only mature epochs:

```bash
python scripts/migrate_checkpoints.py --delete-old --keep-min-epoch 10
```

### **Strategy 2: Keep Only Best and Last**

For completed experiments:

```bash
# After migration, manually remove epoch checkpoints
find outputs/experiment/checkpoints -name "epoch-*.ckpt" -delete

# Keeps: best-*.ckpt and last.ckpt
```

### **Strategy 3: Archive Old Experiments**

Move experiments to archive before cleanup:

```bash
# Create archive directory
mkdir -p /archive/ocr_experiments

# Move old experiments
mv outputs/old_experiment_* /archive/ocr_experiments/

# Or compress before archiving
tar -czf /archive/old_experiments_$(date +%Y%m%d).tar.gz outputs/old_*
rm -rf outputs/old_*
```

### **Strategy 4: Automated Periodic Cleanup**

Set up regular cleanup with cron:

```bash
# Add to crontab (weekly cleanup on Sundays at 2 AM)
0 2 * * 0 cd /path/to/project && python scripts/migrate_checkpoints.py --delete-old --keep-min-epoch 15
```

---

**Last Updated**: 2025-10-15
**Owner**: ml-platform
**Related Files**:
- `scripts/migrate_checkpoints.py` - Migration script
- `ocr/lightning_modules/callbacks/unique_checkpoint.py` - Checkpoint callback

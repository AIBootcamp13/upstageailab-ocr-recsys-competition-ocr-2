# 06. Wandb Integration Reference

## Overview

This document serves as a comprehensive reference for working with Weights & Biases (Wandb) in the OCR project. It covers common issues, best practices, troubleshooting, and lessons learned from integrating Wandb with Hydra-based ML pipelines.

## Table of Contents

- [Configuration and Setup](#configuration-and-setup)
- [Per Batch Image Logging](#per-batch-image-logging)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Config Serialization Issues](#config-serialization-issues)
- [Best Practices](#best-practices)
- [Troubleshooting Guide](#troubleshooting-guide)
- [API Reference](#api-reference)

## Configuration and Setup

### Environment Variables

Required environment variables for Wandb integration:

```bash
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=receipt-text-recognition-ocr-project
WANDB_ENTITY=ocr-team2
WANDB_USER=your_username
```

### Config Structure

Wandb configuration is defined in `configs/logger/wandb.yaml`:

```yaml
# Logger config
wandb:
  enabled: True
project_name: "receipt-text-recognition-ocr-project"
exp_version: "v1.0"

# Per batch image logging for error analysis
per_batch_image_logging:
  enabled: true
  recall_threshold: 0.8
```

The logger is composed via `configs/logger/default.yaml`:

```yaml
# Logger config
defaults:
  - wandb
  - _self_
```

### Enabling Wandb in Training

Wandb can be enabled in several ways:

1. **Via config override:**
   ```bash
   python runners/train.py logger.wandb.enabled=true
   ```

2. **Via environment variable:**
   ```bash
   export WANDB_ENABLED=true
   python runners/train.py
   ```

3. **Via UI parameter** (when using the web interface)

## Per Batch Image Logging

The per batch image logging feature automatically logs images from validation batches that perform poorly, helping with error analysis and debugging.

### How It Works

During validation, the system computes per-batch metrics (recall, precision, hmean) for each batch. When a batch's recall falls below the configured threshold, the system:

1. Loads all images from that problematic batch
2. Creates WandB Image objects with captions showing the batch index and recall score
3. Logs the images, batch metrics, and metadata to WandB

### Configuration

Configure this feature in `configs/logger/wandb.yaml`:

```yaml
per_batch_image_logging:
  enabled: true          # Enable/disable the feature
  recall_threshold: 0.8  # Threshold below which batches are logged (0.0-1.0)
```

### Usage Examples

**Enable with default threshold:**
```bash
python runners/train.py
```

**Disable the feature:**
```bash
python runners/train.py logger.per_batch_image_logging.enabled=false
```

**Adjust threshold:**
```bash
python runners/train.py logger.per_batch_image_logging.recall_threshold=0.7
```

### What Gets Logged

For each problematic batch, WandB receives:
- `problematic_batch_{idx}_images`: Array of WandB Image objects with captions
- `problematic_batch_{idx}_count`: Number of images in the batch
- `problematic_batch_{idx}_recall`: The batch's recall score
- `problematic_batch_{idx}_precision`: The batch's precision score
- `problematic_batch_{idx}_hmean`: The batch's harmonic mean score

### Use Cases

- **Error Analysis**: Identify which types of images cause model failures
- **Data Quality**: Detect batches with consistently poor performance
- **Debugging**: Visualize problematic inputs during development
- **Model Comparison**: Compare which images different models struggle with

### Performance Considerations

- Only triggers when batches perform poorly (below threshold)
- Images are loaded from disk and converted to RGB format
- Memory usage scales with batch size and number of problematic batches
- Consider disabling in production or for very large datasets

## Common Issues and Solutions

### Config.json Not Appearing in Overview Tab

**Symptoms:**
- Charts and metrics display correctly
- Config.json and summary metrics missing from overview tab
- Run appears incomplete in Wandb UI

**Root Cause:**
Config serialization fails due to unresolved Hydra interpolations like `${hydra:runtime.cwd}`.

**Solution:**
Use `OmegaConf.to_container()` with proper error handling:

```python
from omegaconf import OmegaConf

# Properly serialize config for wandb, handling hydra interpolations
try:
    # Try to resolve interpolations for cleaner config
    wandb_config = OmegaConf.to_container(config, resolve=True)
except Exception:
    # Fall back to unresolved config if resolution fails
    wandb_config = OmegaConf.to_container(config, resolve=False)

logger = WandbLogger(
    run_name,
    project=config.logger.project_name,
    config=wandb_config,
)
```

**Files affected:**
- `runners/train.py` ✅ (already fixed)
- `runners/test.py` ✅ (fixed in this update)

### Summary Metrics Not Appearing

**Symptoms:**
- Config.json appears but summary metrics are missing
- `wandb.summary` not populated

**Possible causes:**
1. `finalize_run()` not called
2. Metrics not properly extracted from `trainer.callback_metrics`
3. Non-finite values in metrics

**Solution:**
Ensure `finalize_run()` is called after training with proper metrics:

```python
# Finalize wandb run if wandb was used
if config.logger.wandb:
    from ocr.utils.wandb_utils import finalize_run

    metrics = {}
    for key, value in trainer.callback_metrics.items():
        cast_value = _to_float(value)
        if cast_value is not None and math.isfinite(cast_value):
            metrics[key] = cast_value

    finalize_run(metrics)
```

### Run Names Not Updating

**Symptoms:**
- Run names don't include final metrics/scores
- Names remain as placeholders

**Solution:**
Ensure `finalize_run()` properly updates the run name:

```python
# In finalize_run()
formatted_score = f"{metric_label}{metric_value:.{precision}f}"
current_name = wandb.run.name or "run_SCORE_PLACEHOLDER"
if "_SCORE_PLACEHOLDER" in current_name:
    final_name = current_name.replace("_SCORE_PLACEHOLDER", f"_{formatted_score}")
else:
    final_name = f"{current_name}_{formatted_score}"

wandb.run.name = final_name
wandb.summary["final_run_name"] = final_name
```

### Authentication Issues

**Symptoms:**
- "Login required" errors
- API key not found

**Solutions:**
1. **Check API key:**
   ```bash
   wandb login --relogin
   ```

2. **Environment variables:**
   ```bash
   export WANDB_API_KEY=your_key_here
   ```

3. **Check .env files:**
   - `.env.local`
   - `.env`

4. **Verify key format:**
   - Should be 40 characters
   - No extra whitespace

### Offline Mode Issues

**Symptoms:**
- Runs not syncing to cloud
- Local storage filling up

**Solutions:**
1. **Check offline setting:**
   ```python
   wandb_logger_kwargs["offline"] = bool(logger_cfg.get("offline", False))
   ```

2. **Sync manually:**
   ```bash
   wandb sync wandb/latest-run
   ```

3. **Check storage location:**
   ```bash
   export WANDB_DIR=/path/to/storage
   ```

## Config Serialization Issues

### Hydra Interpolation Resolution

**Problem:**
`dict(config)` fails when config contains `${hydra:runtime.cwd}` or similar interpolations.

**Why it happens:**
- Hydra interpolations require runtime context to resolve
- `dict()` conversion doesn't have access to Hydra's resolution engine

**Solutions:**

1. **Use OmegaConf.to_container with resolve=True:**
   ```python
   config_dict = OmegaConf.to_container(config, resolve=True)
   ```

2. **Handle resolution failures gracefully:**
   ```python
   try:
       config_dict = OmegaConf.to_container(config, resolve=True)
   except Exception:
       config_dict = OmegaConf.to_container(config, resolve=False)
   ```

3. **Manual interpolation replacement:**
   ```python
   # For known interpolations
   config_str = str(config)
   config_str = config_str.replace("${hydra:runtime.cwd}", project_root)
   ```

### Complex Object Serialization

**Problem:**
Config contains non-serializable objects (functions, classes, custom objects).

**Solutions:**
1. **Filter out complex objects:**
   ```python
   def _serialize_for_wandb(obj):
       if isinstance(obj, (str, int, float, bool, list, dict)):
           return obj
       elif hasattr(obj, '__dict__'):
           return str(obj)
       else:
           return str(type(obj).__name__)
   ```

2. **Use custom serialization:**
   ```python
   import json

   class WandbSafeEncoder(json.JSONEncoder):
       def default(self, obj):
           if isinstance(obj, (str, int, float, bool, list, dict)):
               return obj
           return str(type(obj).__name__)
   ```

## Best Practices

### Run Management

1. **Use descriptive run names:**
   ```python
   run_name = generate_run_name(config)  # Includes model arch, params, etc.
   ```

2. **Tag runs appropriately:**
   ```python
   wandb_logger_kwargs["tags"] = ["experiment", "baseline", " ablation"]
   ```

3. **Set proper job types:**
   ```python
   wandb_logger_kwargs["job_type"] = "training"
   ```

### Config Logging

1. **Log resolved config:**
   - Use `resolve=True` when possible
   - Fall back to `resolve=False` if needed

2. **Include relevant metadata:**
   ```python
   config_dict.update({
       "git_commit": get_git_commit(),
       "hostname": socket.gethostname(),
       "timestamp": datetime.now().isoformat(),
   })
   ```

3. **Exclude sensitive data:**
   - API keys
   - Private paths
   - Personal information

### Metrics and Artifacts

1. **Log key metrics:**
   ```python
   wandb.log({
       "train/loss": train_loss,
       "val/accuracy": val_acc,
       "epoch": epoch
   })
   ```

2. **Use summary for final metrics:**
   ```python
   wandb.summary["final_accuracy"] = best_accuracy
   wandb.summary["total_epochs"] = epoch
   ```

3. **Log artifacts:**
   ```python
   artifact = wandb.Artifact("model", type="model")
   artifact.add_file("checkpoints/best.ckpt")
   wandb.log_artifact(artifact)
   ```

### Resource Management

1. **Clean up runs:**
   ```python
   wandb.finish()  # Always call at end
   ```

2. **Handle interruptions:**
   ```python
   try:
       # training code
   except KeyboardInterrupt:
       wandb.finish()
       raise
   ```

3. **Monitor storage:**
   - Regular cleanup of old runs
   - Use offline mode for large experiments

## Troubleshooting Guide

### Run Not Appearing in Wandb

**Check:**
1. API key validity
2. Network connectivity
3. Offline mode setting
4. Project permissions

**Commands:**
```bash
wandb login --relogin
wandb status
ping api.wandb.ai
```

### Config Appears Incomplete

**Check:**
1. Serialization method
2. Interpolation resolution
3. Object complexity
4. Size limits

**Debug:**
```python
# Test config serialization
try:
    test_config = OmegaConf.to_container(config, resolve=True)
    print("Serialization successful")
except Exception as e:
    print(f"Serialization failed: {e}")
```

### Metrics Not Logging

**Check:**
1. Metric names (should not contain special characters)
2. Value types (should be numeric)
3. Logging frequency
4. Step counters

**Debug:**
```python
# Check metric values
for key, value in metrics.items():
    print(f"{key}: {value} (type: {type(value)})")
```

### Performance Issues

**Symptoms:**
- Slow startup
- High memory usage
- Network timeouts

**Solutions:**
1. **Reduce logging frequency:**
   ```python
   # Log every N steps instead of every step
   if step % log_every_n_steps == 0:
       wandb.log(metrics)
   ```

2. **Use offline mode:**
   ```python
   wandb_logger_kwargs["offline"] = True
   ```

3. **Batch log calls:**
   ```python
   # Collect metrics, log once per epoch
   epoch_metrics = {}
   # ... accumulate metrics ...
   wandb.log(epoch_metrics)
   ```

## API Reference

### Key Functions

#### `generate_run_name(cfg: DictConfig) -> str`
Generates descriptive run names from config parameters.

**Parameters:**
- `cfg`: Hydra config object

**Returns:**
- Formatted run name string

#### `finalize_run(metrics: Mapping[str, float])`
Finalizes wandb run with summary metrics and updated name.

**Parameters:**
- `metrics`: Dictionary of metric names to values

#### `load_env_variables()`
Loads wandb credentials from `.env` files.

### WandbLogger Parameters

```python
WandbLogger(
    name="run_name",           # Run name
    project="project_name",    # Project name
    entity="entity_name",      # Team/entity name
    config=config_dict,        # Config to log
    tags=["tag1", "tag2"],    # Run tags
    job_type="training",       # Job type
    offline=False,             # Offline mode
    save_dir="/path/to/dir",   # Local save directory
)
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_API_KEY` | API key for authentication | Yes |
| `WANDB_PROJECT` | Default project name | No |
| `WANDB_ENTITY` | Default entity/team | No |
| `WANDB_USER` | Username for run attribution | No |
| `WANDB_DIR` | Local storage directory | No |
| `WANDB_DISABLED` | Disable wandb globally | No |

## Lessons Learned

### Config Serialization (October 2025)
- **Issue:** `dict(config)` fails with Hydra interpolations
- **Impact:** Config.json missing from wandb overview
- **Solution:** Use `OmegaConf.to_container()` with error handling
- **Files:** `runners/train.py`, `runners/test.py`

### Future Considerations

1. **Config validation:** Add schema validation before wandb logging
2. **Metric standardization:** Define consistent metric naming conventions
3. **Artifact management:** Implement automatic model checkpoint uploading
4. **Run grouping:** Better experiment organization and comparison
5. **Custom dashboards:** Create project-specific wandb dashboards

## Related Documentation

- [Hydra Configuration Guide](02_hydra_and_registry.md)
- [Experiment Logging](04_experiments/)
- [PyTorch Lightning Integration](../external/lightning-hydra-template/)
- [Wandb Official Documentation](https://docs.wandb.ai/)

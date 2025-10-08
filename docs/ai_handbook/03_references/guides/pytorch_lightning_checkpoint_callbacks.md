# PyTorch Lightning Callbacks and Checkpoint Loading

## Overview
When using PyTorch Lightning's `ckpt_path` parameter to load a checkpoint during testing or prediction, the `Trainer` must include the same callbacks that were active when the checkpoint was saved. Failure to do so results in warnings about missing callbacks.

## Why This Matters
- **Reproducibility**: Ensures the testing/prediction environment matches the training setup
- **Warning Avoidance**: Prevents PyTorch Lightning warnings about mismatched callbacks
- **Proper State Restoration**: Callbacks may store state that needs to be restored for correct behavior

## How PyTorch Lightning Checks Callbacks
PyTorch Lightning generates a string representation for each callback based on:
- The callback's class name
- Key initialization parameters (e.g., `monitor`, `mode`, `every_n_train_steps`, etc.)

When loading a checkpoint, it compares these strings against the current trainer's callbacks.

## Example Issue
```
Be aware that when using `ckpt_path`, callbacks used to create the checkpoint need to be provided during `Trainer` instantiation. Please add the following callbacks: ["EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}", "UniqueModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"].
```

## Solution
1. **Inspect the checkpoint** to see what callbacks were saved:
   ```python
   import torch
   ckpt = torch.load('path/to/checkpoint.ckpt', map_location='cpu', weights_only=False)
   print('Saved callbacks:', list(ckpt['callbacks'].keys()))
   ```

2. **Configure matching callbacks** in your config files (e.g., `test.yaml`, `predict.yaml`, or callback-specific configs)

3. **Key parameters to match**:
   - Class type (e.g., `EarlyStopping`, `ModelCheckpoint`)
   - `monitor` parameter (exact string match, including slashes if present)
   - `mode` parameter
   - Other relevant parameters like `every_n_train_steps`, `every_n_epochs`, `train_time_interval`

## Configuration Examples

### For Testing with Custom Callbacks
```yaml
# configs/callbacks/performance_profiler.yaml
performance_profiler:
  _target_: ocr.lightning_modules.callbacks.PerformanceProfilerCallback
  # ... other params

# Add required callbacks to match checkpoint
early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: "val_loss"
  patience: 5
  mode: "min"

model_checkpoint:
  _target_: ocr.lightning_modules.callbacks.unique_checkpoint.UniqueModelCheckpoint
  monitor: "val_loss"
  mode: "min"
  every_n_train_steps: 0
  every_n_epochs: 1
  train_time_interval: null
```

### For Prediction
```yaml
# configs/predict.yaml
callbacks:
  model_checkpoint:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: "val/loss"  # Note: exact match including slashes
    mode: "min"
    every_n_train_steps: 0
    every_n_epochs: 1
    train_time_interval: null
```

## Best Practices
- Always inspect checkpoint contents before configuring callbacks
- Use the exact parameter values from the checkpoint's callback strings
- Test with `--cfg job` to verify the resolved configuration
- Document callback configurations for different checkpoints used in the project

## Related Files
- `configs/callbacks/performance_profiler.yaml`
- `configs/test.yaml`
- `configs/predict.yaml`
- `runners/test.py`
- `runners/predict.py`

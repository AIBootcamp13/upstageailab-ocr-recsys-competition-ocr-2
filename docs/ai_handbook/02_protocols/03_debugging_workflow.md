# **filename: docs/ai_handbook/02_protocols/03_debugging_workflow.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=debugging,triage -->

# **Protocol: Debugging Workflow**

This protocol outlines a systematic approach to debugging common issues within the project, from data pipeline problems to model training errors.

## **1. Initial Triage: Isolate the Problem**

Before diving deep, quickly determine the nature of the issue.

* **Is it a configuration error?** Errors often occur at startup and mention Hydra or instantiate. Check your config files first.
* **Is it a data error?** Problems happening during the first few training steps, like shape mismatches or TypeError, often point to the data pipeline.
* **Is it a model error?** CUDA out of memory, NaN loss, or errors deep within a model's forward pass suggest a problem with the model architecture or hyperparameters.

## **2. Lightweight Tooling for Safe Inspection**

Use the project's built-in tools to inspect behavior without running a full training session. Refer to the **Command Registry** for a full list of commands.

### **Key Inspection Commands**

* **Validate Configuration:** Check for syntax errors or missing values in your experiment's configuration.
  `uv run python scripts/agent_tools/validate_config.py --config-name your_config_name`

* **Run a Smoke Test:** Run one or two batches of training and validation to quickly surface errors in the data or model pipeline. Most scripts support a fast_dev_run flag.
  `uv run python runners/train.py --config-name your_config_name trainer.fast_dev_run=True`

* **Visualize a Data Batch:** If you suspect a data issue, generate and inspect a sample batch to check augmentations and shapes.
  `uv run python scripts/agent_tools/visualize_batch.py`

* **Profile Dataset Health:** Use the diagnostics CLI to check EXIF orientation distribution or polygon retention before refactors.
  `uv run python tests/debug/data_analyzer.py --mode both`

## **3. Recommended Debugging Tools**

* **icecream for "Print" Debugging:** Use ic() instead of print() for more informative output that includes the variable name and location. It is configured project-wide.
```python
  from icecream import ic

  def forward(self, x):
      features = self.encoder(x)
      ic(features.shape) # Example: features.shape: torch.Size([8, 256, 64, 64])
      return features
```
* **Rich for Logging:** The project uses the rich library for clear, color-coded logging. Pay close attention to WARNING and ERROR messages in the console.

## **4. Common Scenarios & Solutions**

### **Scenario: CUDA out of memory**

1. **Action:** Reduce the data.batch_size in your configuration.
2. **Action:** Enable mixed-precision training by setting trainer.precision=16-mixed.
3. **Action:** If available, enable gradient accumulation (trainer.accumulate_grad_batches=2).

### **Scenario: NaN Loss**

1. **Cause:** Often caused by an exploding gradient.
2. **Action:** Lower the model.optimizer.lr (learning rate) significantly (e.g., from 1e-3 to 1e-5).
3. **Action:** Enable gradient clipping in the trainer config (trainer.gradient_clip_val=1.0).
4. **Action:** Inspect the data batch for corrupted images or annotations.

### **Scenario: Shape Mismatch Error**

1. **Cause:** The output channels of one layer (e.g., encoder) do not match the input channels of the next (e.g., decoder).
2. **Action:** Check the out_channels and in_channels parameters for the connected components in your model's configuration file.
3. **Action:** Use ic(tensor.shape) at each step of the model's forward pass to trace where the shape becomes incorrect.
4. **Action:** Run `uv run python tests/debug/data_analyzer.py --mode polygons` to confirm that annotations and transforms still emit valid polygons.

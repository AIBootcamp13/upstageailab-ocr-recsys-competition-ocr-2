# **filename: docs/ai_handbook/02_protocols/development/03_debugging_workflow.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=debugging,triage,issue_resolution -->

# **Protocol: Debugging Workflow**

## **Overview**
This protocol establishes a systematic approach to debugging common issues within the OCR project, from data pipeline problems to model training errors. The workflow emphasizes quick triage, lightweight inspection tools, and targeted fixes to minimize development downtime.

## **Prerequisites**
- Access to project repository and development environment
- Basic understanding of PyTorch, Hydra configuration, and the project's architecture
- Familiarity with the Command Registry for available debugging tools
- Active development environment with uv package manager configured

## **Procedure**

### **Step 1: Initial Triage - Isolate the Problem**
Before diving deep, quickly determine the nature of the issue:

- **Configuration Error:** Errors at startup mentioning Hydra or instantiate - check config files first
- **Data Error:** Problems during first training steps like shape mismatches or TypeError - inspect data pipeline
- **Model Error:** CUDA out of memory, NaN loss, or errors in model's forward pass - examine model architecture/hyperparameters

### **Step 2: Lightweight Inspection - Use Built-in Tools**
Employ the project's safe inspection tools without running full training sessions:

**Validate Configuration:**
```bash
uv run python scripts/agent_tools/validate_config.py --config-name your_config_name
```

**Run Smoke Test:**
```bash
uv run python runners/train.py --config-name your_config_name trainer.fast_dev_run=True
```

**Visualize Data Batch:**
```bash
uv run python scripts/agent_tools/visualize_batch.py
```

**Profile Dataset Health:**
```bash
uv run python tests/debug/data_analyzer.py --mode both
```

### **Step 3: Apply Debugging Techniques**
Use recommended tools for effective debugging:

**Icecream for Print Debugging:**
```python
from icecream import ic

def forward(self, x):
    features = self.encoder(x)
    ic(features.shape)  # Example: features.shape: torch.Size([8, 256, 64, 64])
    return features
```

**Rich Logging:** Monitor WARNING and ERROR messages in console output for diagnostic information.

### **Step 4: Address Common Scenarios**
Apply targeted fixes for frequent issues:

**CUDA Out of Memory:**
- Reduce `data.batch_size` in configuration
- Enable mixed-precision: `trainer.precision=16-mixed`
- Enable gradient accumulation: `trainer.accumulate_grad_batches=2`

**NaN Loss (Exploding Gradients):**
- Lower learning rate: `model.optimizer.lr` (e.g., 1e-3 → 1e-5)
- Enable gradient clipping: `trainer.gradient_clip_val=1.0`
- Inspect data batch for corruption

**Shape Mismatch Error:**
- Verify `out_channels`/`in_channels` parameters in model config
- Use `ic(tensor.shape)` throughout forward pass
- Run polygon validation: `uv run python tests/debug/data_analyzer.py --mode polygons`

## **Validation**
- Issue is resolved without introducing new errors
- Training/validation loops complete successfully
- Model outputs are within expected ranges (no NaN/inf values)
- Resource usage remains within system limits

## **Troubleshooting**
- If smoke tests fail, isolate to specific component (data vs model vs config)
- For persistent CUDA OOM, consider model architecture changes or smaller batch sizes
- When NaN loss persists, check data preprocessing and augmentation pipelines
- Shape mismatches often indicate config/model architecture misalignment

## **Related Documents**
- [Command Registry](02_command_registry.md) - Available debugging and inspection tools
- [Coding Standards](01_coding_standards.md) - Development best practices
- [Configuration Management](../../configuration/01_hydra_config_system.md) - Config validation and debugging
- [Data Pipeline](../../data/01_data_processing_pipeline.md) - Data-related debugging guidance

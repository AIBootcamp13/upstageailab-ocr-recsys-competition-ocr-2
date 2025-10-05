# Experiment Results

This directory contains results and artifacts from model training experiments, evaluations, and analyses.

## Directory Structure

### `experiments/`
Contains experiment-specific results and artifacts.

#### Experiment Naming Convention
Experiments are named using the pattern: `{user}_{model}_{backbone}_{head}_{loss}_{batch_size}_{learning_rate}_{metric}_{value}`

**Example:** `wchoi189_dbnetpp-resnet18-unet-db-head-db-loss-bs8-lr1e-3_hmean0.898`

**Components:**
- `user`: Researcher identifier (e.g., `wchoi189`)
- `model`: Model architecture (e.g., `dbnetpp`)
- `backbone`: Feature extractor (e.g., `resnet18`)
- `head`: Detection head (e.g., `db-head`)
- `loss`: Loss function (e.g., `db-loss`)
- `bs`: Batch size (e.g., `bs8`)
- `lr`: Learning rate (e.g., `lr1e-3`)
- `metric`: Primary evaluation metric (e.g., `hmean`)
- `value`: Metric value (e.g., `0.898`)

## File Types

### `experiment_summary.json`
Contains experiment metadata including:
- Model configuration
- Training parameters
- Performance metrics
- Hardware information
- Timestamps

### `wandb_export_*.csv`
WandB (Weights & Biases) logging exports containing:
- Training metrics over time
- Validation results
- System resource usage
- Model checkpoints information

## Usage

### Viewing Results
```bash
# View experiment summary
cat results/experiments/{experiment_name}/experiment_summary.json | jq

# Analyze training curves
python -c "import pandas as pd; df = pd.read_csv('results/experiments/{experiment_name}/wandb_export_*.csv'); print(df.head())"
```

### Comparing Experiments
```bash
# List all experiments
ls results/experiments/

# Compare metrics across experiments
find results/experiments/ -name "experiment_summary.json" -exec jq -r '.metrics.hmean' {} \;
```

## Organization

Results are organized by experiment for:
- **Reproducibility:** Complete experiment artifacts
- **Analysis:** Easy comparison across experiments
- **Backup:** Persistent storage of training outputs
- **Collaboration:** Shareable experiment results

## Cleanup

Old or unsuccessful experiments can be archived:
```bash
# Archive experiment
mv results/experiments/{experiment_name} results/archive/

# Compress for storage
tar -czf results/archive/{experiment_name}.tar.gz results/archive/{experiment_name}
```

## Related Directories

- `outputs/` - Active training outputs (auto-generated)
- `lightning_logs/` - PyTorch Lightning logs
- `wandb/` - Weights & Biases local storage

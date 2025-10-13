# **filename: docs/ai_handbook/02_protocols/components/17_advanced_training_techniques.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=training,experimentation,parallel-processing -->

# **Protocol: Advanced Training Techniques**

## **Overview**

This protocol covers advanced training methodologies including parallel experimentation, model architecture selection, and parameter optimization for specific use cases like curved text detection. It provides systematic approaches for efficient model development and optimization.

## **Prerequisites**

- Basic understanding of Hydra configuration system
- Familiarity with PyTorch Lightning training workflows
- Access to GPU resources for parallel training
- Knowledge of OCR model architectures (DBNet, DBNet++, CRAFT)

## **Component Architecture**

### **Core Components**
- **Parallel Training Engine**: Hydra multirun system for simultaneous experiments
- **Architecture Selection Framework**: Decision trees for model architecture choice
- **Parameter Optimization System**: Systematic hyperparameter tuning workflows
- **Result Analysis Pipeline**: Automated comparison and visualization tools

### **Integration Points**
- `runners/train.py`: Main training entry point
- `ablation_study/`: Result collection and analysis tools
- `configs/model/`: Architecture-specific configurations
- `outputs/`: Experiment output directory structure

## **Procedure**

### **Step 1: Environment Setup and Validation**
```bash
# Verify GPU availability and Hydra installation
nvidia-smi
python -c "import hydra; print('Hydra version:', hydra.__version__)"
```

### **Step 2: Architecture Selection and Configuration**
Choose appropriate architecture based on use case:

**For High Accuracy (DBNet++)**:
```bash
python runners/train.py \
  +model=architectures/dbnetpp \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=20 \
  data.batch_size=6
```

**For Curved Text (CRAFT)**:
```bash
python runners/train.py \
  +model=architectures/craft \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=15 \
  data.batch_size=8
```

**For Fast Iteration (DBNet)**:
```bash
python runners/train.py \
  exp_name="fast_iteration" \
  model.architecture_name=dbnet \
  model.encoder.model_name=resnet18 \
  trainer.max_epochs=8 \
  data.batch_size=16 \
  training.learning_rate=1e-3
```

### **Step 3: Parallel Experimentation Setup**
Configure and execute parallel training runs:

```bash
# Architecture comparison for curved text
python runners/train.py \
  exp_name="architecture_comparison_curved_text" \
  model.architecture_name=dbnetpp,craft \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=15 \
  data.batch_size=8 \
  -m
```

```bash
# Parameter sweep example
python runners/train.py \
  exp_name="parameter_sweep" \
  data.batch_size=4,8,16 \
  training.learning_rate=1e-3,5e-4,1e-4 \
  trainer.max_epochs=10 \
  -m
```

### **Step 4: Results Analysis and Optimization**
Analyze parallel run results and optimize configurations:

```bash
# Collect and compare results
python ablation_study/collect_results.py \
  --project "OCR_Ablation" \
  --tag "architecture_comparison" \
  --output results.csv

# Generate comparison table
python ablation_study/generate_ablation_table.py \
  --input results.csv \
  --metric val/hmean \
  --output-md comparison.md
```

## **API Reference**

### **Training Commands**
- `python runners/train.py [params] -m`: Execute parallel training runs
- `python runners/train.py resume=[checkpoint_path]`: Resume interrupted training

### **Analysis Tools**
- `python ablation_study/collect_results.py`: Collect experiment results
- `python ablation_study/generate_ablation_table.py`: Generate comparison tables

### **Key Parameters**
- `exp_name`: Experiment identifier (auto-generates output directory)
- `model.architecture_name`: Model architecture selection
- `model.encoder.model_name`: Backbone encoder selection
- `data.batch_size`: Training batch size
- `trainer.max_epochs`: Maximum training epochs
- `training.learning_rate`: Learning rate for optimization

## **Configuration Structure**

```
outputs/${exp_name}/
├── 0_model.architecture_name=dbnetpp/
│   ├── checkpoints/
│   ├── logs/
│   └── .hydra/
├── 1_model.architecture_name=craft/
│   ├── checkpoints/
│   ├── logs/
│   └── .hydra/
└── multirun.yaml  # Summary of all runs
```

## **Validation**

### **Training Validation**
- Monitor GPU utilization: `nvidia-smi`
- Check process status: `ps aux | grep train.py`
- Verify output directories are created
- Confirm parallel processes are running

### **Result Validation**
- Check convergence in training logs
- Validate metric improvements across epochs
- Compare results between parallel runs
- Ensure checkpoint files are generated

## **Troubleshooting**

### **Common Issues**

**"Could not find 'model/architecture'"**
- Use `model.architecture_name=` instead of `+model=`
- Verify architecture name spelling

**Memory Errors**
- Reduce `data.batch_size`
- Check available GPU memory with `nvidia-smi`
- Consider using smaller backbone models

**Slow Training**
- Increase `data.batch_size` if GPU memory allows
- Check `num_workers` in dataloader configuration
- Verify data preprocessing pipeline efficiency

**No Parallel Execution**
- Ensure `-m` flag is at the end of command
- Check Hydra installation: `python -c "import hydra"`

**Experiment Naming Conflicts**
- Use timestamps: `exp_name="experiment_$(date +%Y%m%d_%H%M%S)"`
- Include key parameters in experiment name

## **Related Documents**

- `22_command_builder_hydra_configuration_fixes.md`: Hydra configuration troubleshooting
- `21_experiment_analysis_framework_handbook.md`: Experiment analysis and monitoring
- `23_hydra_configuration_testing_implementation_plan.md`: Configuration testing frameworks
- `13_training_protocol.md`: Basic training workflows
- `01_model_architecture_selection.md`: Architecture decision guidance

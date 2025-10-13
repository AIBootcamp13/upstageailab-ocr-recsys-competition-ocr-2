# **filename: docs/ai_handbook/02_protocols/components/13_training_protocol.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=training,experimentation -->

# **Protocol: Training & Experimentation**

## **Overview**

This protocol provides systematic guidance for planning, executing, analyzing, and iterating on model training experiments. It ensures all runs are reproducible, learnings are captured, and context is efficiently passed between collaborators through a structured hypothesis-testing cycle.

## **Prerequisites**

- Access to experiment logging templates in `docs/ai_handbook/04_experiments/`
- Weights & Biases (W&B) account and project setup
- Understanding of Hydra configuration system
- Familiarity with ablation study tools in `ablation_study/`

## **Component Architecture**

### **Core Components**
- **Experiment Planning Framework**: Template-based hypothesis formulation
- **Execution Engine**: Structured training run orchestration
- **Analysis Pipeline**: Automated result collection and visualization
- **Iteration System**: Hypothesis refinement and next-step planning

### **Integration Points**
- `docs/ai_handbook/04_experiments/`: Experiment documentation and logs
- `ablation_study/`: Result analysis and comparison tools
- `runners/train.py`: Training execution entry point
- W&B: Experiment tracking and monitoring

## **Procedure**

### **Step 1: Experiment Planning and Documentation**
Create structured experiment plan:

```bash
# Copy experiment template
cp docs/ai_handbook/04_experiments/TEMPLATE.md docs/ai_handbook/04_experiments/$(date +%Y-%m-%d)_experiment_name.md
```

Define clear hypothesis and configuration:
- **Objective**: State testable hypothesis
- **Configuration**: Full command-line invocation with all overrides

### **Step 2: Context Logging Setup**
Initialize structured logging:

```bash
# Start context log
make context-log-start LABEL="experiment_name"

# Log experiment plan reference
# (First logged action should reference the plan file)
```

### **Step 3: Training Execution and Monitoring**
Execute and monitor training run:

```bash
# Run the exact command from experiment plan
python runners/train.py [experiment_config] [overrides]

# Monitor via W&B dashboard
# Check real-time metrics and system resources
```

### **Step 4: Results Analysis and Documentation**
Transform raw results into knowledge:

```bash
# Link W&B run in experiment log
# Record key metrics (val/hmean, test/recall, test/precision)

# Analyze results against hypothesis
# Document surprising findings and visual insights
```

Generate automated summaries:
```bash
# Summarize session actions
make context-log-summarize LOG=path/to/log.jsonl
```

### **Step 5: Ablation Study Analysis (Optional)**
For parameter sweeps and comparisons:

```bash
# Collect W&B results
uv run python ablation_study/collect_results.py \
  --project "receipt-text-recognition-ocr-project" \
  --tag experiment_tag \
  --output outputs/ablation/results.csv

# Generate comparison tables
uv run python ablation_study/generate_ablation_table.py \
  --input outputs/ablation/results.csv \
  --ablation-type parameter_name \
  --metric val/hmean \
  --output-md docs/ablation/comparison.md
```

## **API Reference**

### **Key Commands**
- `make context-log-start LABEL="name"`: Initialize experiment logging
- `make context-log-summarize LOG=path`: Generate session summary
- `python runners/train.py [config]`: Execute training run
- `python ablation_study/collect_results.py`: Gather experiment results

### **Configuration Parameters**
- `exp_name`: Experiment identifier for output organization
- `trainer.max_epochs`: Training duration
- `data.batch_size`: Batch size for training
- `training.learning_rate`: Optimization learning rate
- `model.*`: Architecture-specific parameters

### **Output Structure**
```
outputs/${exp_name}/
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── .hydra/              # Configuration snapshots
└── wandb/               # W&B artifacts
```

## **Configuration Structure**

```
docs/ai_handbook/04_experiments/
├── TEMPLATE.md              # Experiment planning template
├── YYYY-MM-DD_experiment.md # Individual experiment logs
└── summaries/               # Session summaries

ablation_study/
├── collect_results.py       # Result aggregation
├── generate_ablation_table.py # Comparison tables
└── demo_ablation.py         # Analysis examples
```

## **Validation**

### **Pre-Experiment Validation**
- [ ] Experiment plan created with clear hypothesis
- [ ] Configuration command is complete and runnable
- [ ] Context logging initialized
- [ ] W&B project and entity configured

### **Execution Validation**
- [ ] Training command runs without immediate errors
- [ ] W&B logging active and receiving metrics
- [ ] System resources monitored (GPU, memory)
- [ ] Checkpoint saving functional

### **Analysis Validation**
- [ ] W&B run URL captured in experiment log
- [ ] Key metrics recorded and compared to baseline
- [ ] Analysis section completed with insights
- [ ] Next steps clearly defined

### **Ablation Validation**
```bash
# Verify result collection
wc -l outputs/ablation/results.csv

# Check table generation
head -10 docs/ablation/comparison.md

# Validate metrics
python -c "import pandas as pd; df = pd.read_csv('outputs/ablation/results.csv'); print(df.describe())"
```

## **Troubleshooting**

### **Common Issues**

**Experiment Planning Problems**
- Use TEMPLATE.md as starting point
- Ensure hypothesis is testable and specific
- Include all necessary configuration overrides

**Training Execution Failures**
- Verify configuration syntax
- Check GPU/memory availability
- Validate data paths and permissions

**W&B Integration Issues**
- Confirm API key and project settings
- Check network connectivity
- Verify entity/project permissions

**Result Analysis Challenges**
- Ensure consistent metric naming
- Check for missing or corrupted runs
- Validate sweep parameter ranges

**Ablation Study Errors**
- Verify W&B run tagging
- Check metric availability across runs
- Ensure consistent configuration naming

## **Related Documents**

- `17_advanced_training_techniques.md`: Advanced training methodologies
- `21_experiment_analysis_framework_handbook.md`: Experiment analysis tools
- `23_hydra_configuration_testing_implementation_plan.md`: Configuration testing
- `22_command_builder_hydra_configuration_fixes.md`: Command construction

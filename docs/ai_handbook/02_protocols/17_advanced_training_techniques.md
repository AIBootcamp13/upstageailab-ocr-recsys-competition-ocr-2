# **filename: docs/ai_handbook/02_protocols/17_advanced_training_techniques.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=training,experimentation,parallel-processing -->

# **Protocol: Advanced Training Techniques**

This protocol covers advanced training methodologies including parallel experimentation, model architecture selection, and parameter optimization for specific use cases like curved text detection.

## **1. Parallel Training with Hydra Multirun**

### **Why Parallel Training?**

Parallel training allows you to:
- Compare multiple architectures simultaneously
- Test parameter combinations efficiently
- Maximize GPU utilization
- Reduce total experimentation time

### **Basic Multirun Syntax**

```bash
# Run multiple configurations in parallel
python runners/train.py [parameters] -m
```

**Key Points:**
- `-m` flag enables multirun mode
- Each parameter combination runs as a separate process
- Output directories are automatically created under `outputs/${exp_name}/`
- GPU resources are automatically distributed

### **Multirun Examples**

#### **Architecture Comparison**
```bash
# Compare DBNet++ vs CRAFT for curved text detection
python runners/train.py \
  exp_name="architecture_comparison_curved_text" \
  model.architecture_name=dbnetpp,craft \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=15 \
  data.batch_size=8 \
  -m
```

#### **Parameter Sweep**
```bash
# Test different batch sizes and learning rates
python runners/train.py \
  exp_name="parameter_sweep" \
  data.batch_size=4,8,16 \
  training.learning_rate=1e-3,5e-4,1e-4 \
  trainer.max_epochs=10 \
  -m
```

#### **Backbone Comparison**
```bash
# Compare different encoder backbones
python runners/train.py \
  exp_name="backbone_comparison" \
  model.encoder.model_name=resnet18,resnet34,resnet50,efficientnet_b0 \
  trainer.max_epochs=12 \
  -m
```

### **Multirun Output Structure**

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

### **Monitoring Parallel Runs**

```bash
# Check running processes
ps aux | grep "python runners/train.py"

# Monitor GPU usage
nvidia-smi

# View individual run logs
tail -f outputs/architecture_comparison_curved_text/0_*/logs/train.log
```

## **2. Model Architecture Selection Guide**

### **Architecture Overview**

| Architecture | Best For | Strengths | Weaknesses | Use Case |
|-------------|----------|-----------|------------|----------|
| **DBNet** | General OCR | Fast, good baseline | Limited curved text | Straight text, speed priority |
| **DBNet++** | High Accuracy | Best overall accuracy | Slower training | Competition submissions, general OCR |
| **CRAFT** | Curved/Slanted Text | Character-level detection | Complex post-processing | Receipts, curved text, irregular layouts |

### **When to Choose Each Architecture**

#### **Choose DBNet++ When:**
- Maximum accuracy is needed
- Training time is not a major constraint
- Dataset has varied text orientations
- Competition submission optimization

#### **Choose CRAFT When:**
- Dataset contains significant curved/slanted text
- Character-level precision is important
- Text appears in irregular layouts
- Post-processing can be customized

#### **Choose DBNet When:**
- Fast iteration is needed
- Baseline performance testing
- Limited computational resources
- Straight, horizontal text dominates

### **Architecture-Specific Configuration**

#### **DBNet++ for High Accuracy**
```bash
python runners/train.py \
  +model=architectures/dbnetpp \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=20 \
  data.batch_size=6
```

#### **CRAFT for Curved Text**
```bash
python runners/train.py \
  +model=architectures/craft \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=15 \
  data.batch_size=8
```

## **3. Parameter Optimization for Specific Use Cases**

### **Curved and Slanted Text Detection**

#### **Key Parameters**
- `model.head.postprocess.use_polygon=true`: Enables polygon detection instead of rectangles
- `model.head.postprocess.unclip_ratio`: Controls text region expansion (higher = more context)
- `data.batch_size`: Smaller batches for complex geometries

#### **Recommended Configuration**
```bash
python runners/train.py \
  exp_name="curved_text_optimized" \
  model.architecture_name=craft \
  model.head.postprocess.use_polygon=true \
  model.head.postprocess.unclip_ratio=2.0 \
  data.batch_size=8 \
  trainer.max_epochs=15
```

### **High-Accuracy Training**
```bash
python runners/train.py \
  exp_name="high_accuracy_training" \
  model.architecture_name=dbnetpp \
  model.encoder.model_name=resnet50 \
  trainer.max_epochs=25 \
  data.batch_size=4 \
  training.learning_rate=3e-4 \
  training.weight_decay=1e-4
```

### **Fast Iteration**
```bash
python runners/train.py \
  exp_name="fast_iteration" \
  model.architecture_name=dbnet \
  model.encoder.model_name=resnet18 \
  trainer.max_epochs=8 \
  data.batch_size=16 \
  training.learning_rate=1e-3
```

## **4. Advanced Command Patterns**

### **Complex Parameter Sweeps**
```bash
# Architecture + backbone + batch size sweep
python runners/train.py \
  exp_name="comprehensive_sweep" \
  model.architecture_name=dbnetpp,craft \
  model.encoder.model_name=resnet34,resnet50 \
  data.batch_size=8,12 \
  trainer.max_epochs=10 \
  -m
```

### **Conditional Overrides**
```bash
# Different settings based on architecture
python runners/train.py \
  exp_name="conditional_training" \
  model.architecture_name=dbnetpp \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=20 \
  data.batch_size=6
```

### **Resume Parallel Runs**
```bash
# Resume specific runs from a multirun
python runners/train.py \
  exp_name="architecture_comparison_curved_text" \
  resume=outputs/architecture_comparison_curved_text/0_model.architecture_name=dbnetpp/checkpoints/last.ckpt \
  trainer.max_epochs=25
```

## **5. Best Practices**

### **Experiment Naming**
- Use descriptive `exp_name` values
- Include key parameters in the name
- Use timestamps for uniqueness: `exp_name="dbnetpp_resnet50_$(date +%Y%m%d_%H%M%S)"`

### **Resource Management**
- Monitor GPU memory usage with `nvidia-smi`
- Adjust `data.batch_size` based on available VRAM
- Use smaller batches for complex architectures

### **Result Analysis**
```bash
# Compare results across runs
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

### **Troubleshooting**
- **"Could not find 'model/architecture'"**: Use `model.architecture_name=`
- **Memory errors**: Reduce `data.batch_size`
- **Slow training**: Check `num_workers` in dataloader config
- **No parallel execution**: Ensure `-m` flag is at the end

## **6. Common Patterns**

### **Architecture A/B Testing**
```bash
python runners/train.py \
  exp_name="ab_test_$(date +%Y%m%d_%H%M%S)" \
  model.architecture_name=dbnetpp,craft \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=15 \
  data.batch_size=8 \
  -m
```

### **Hyperparameter Optimization**
```bash
python runners/train.py \
  exp_name="hpo_run_$(date +%Y%m%d_%H%M%S)" \
  training.learning_rate=1e-3,5e-4,1e-4 \
  training.weight_decay=1e-4,5e-5 \
  data.batch_size=8,12 \
  trainer.max_epochs=12 \
  -m
```

### **Production Training**
```bash
python runners/train.py \
  exp_name="production_training_$(date +%Y%m%d_%H%M%S)" \
  model.architecture_name=dbnetpp \
  model.encoder.model_name=resnet50 \
  model.head.postprocess.use_polygon=true \
  trainer.max_epochs=30 \
  data.batch_size=4 \
  training.learning_rate=3e-4 \
  wandb=true \
  project_name="production_runs"
```

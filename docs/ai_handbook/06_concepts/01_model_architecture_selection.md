# **Concept: Model Architecture Selection for OCR**

**Objective:** Guide the selection of appropriate OCR model architectures based on dataset characteristics, performance requirements, and computational constraints.

### **1. What is Model Architecture Selection?**

Model architecture selection in OCR involves choosing the right neural network design (DBNet, DBNet++, CRAFT, etc.) and its components (backbones, decoders, heads) to optimize performance for specific text detection challenges. This decision impacts accuracy, speed, and the ability to handle different text layouts.

### **2. Why Do We Use Different Architectures? (The Problem It Solves)**

* **Problem 1:** Different OCR datasets have varying text characteristics - some have mostly straight horizontal text, others have curved, slanted, or irregular layouts that require different detection approaches.
* **Problem 2:** Training time and computational resources vary significantly between architectures, requiring trade-offs between accuracy and efficiency.
* **Problem 3:** Competition requirements may prioritize different metrics (precision vs recall) or have specific constraints on inference time.
* **Our Solution:** A systematic approach to architecture selection based on dataset analysis, performance requirements, and resource constraints, documented in this guide.

### **3. How Does Architecture Selection Work in Our Codebase?**

* **Key File(s):**
  * `configs/preset/models/dbnetpp.yaml` - DBNet++ configuration
  * `configs/preset/models/craft.yaml` - CRAFT configuration
  * `configs/preset/models/model_example.yaml` - Base model configuration
  * `ocr/models/architectures/` - Architecture implementations
  * `ablation_study/` - Performance comparison tools

* **Example Configuration (DBNet++ for High Accuracy):**
  ```yaml
  # configs/preset/models/dbnetpp.yaml
  defaults:
    - /model/architectures: dbnetpp
    - /model/optimizers: adamw
    - /preset/models/encoder/timm_backbone
    - /preset/models/decoder/dbpp_decoder
    - /preset/models/head/db_head
    - /model/loss/db_loss
    - _self_

  component_overrides:
    encoder:
      model_name: "resnet50"
      select_features: [1, 2, 3, 4]
      pretrained: true
  ```

* **Example Usage:**
  ```bash
  # Select architecture via command line
  python runners/train.py model.architecture_name=dbnetpp

  # Or use preset configuration
  python runners/train.py +model=dbnetpp
  ```

### **4. Architecture Decision Framework**

#### **Step 1: Analyze Dataset Characteristics**
```bash
# Use data analyzer to understand text layout distribution
uv run python tests/debug/data_analyzer.py --mode both --limit 1000
```

**Key Questions:**
- What percentage of text is curved/slanted?
- Are there irregular text layouts?
- What's the typical text size distribution?

#### **Step 2: Define Performance Requirements**
- **Speed Priority:** DBNet with lightweight backbone
- **Accuracy Priority:** DBNet++ with ResNet50
- **Curved Text Priority:** CRAFT with polygon post-processing

#### **Step 3: Consider Resource Constraints**
- **GPU Memory:** Affects maximum batch size
- **Training Time:** DBNet++ takes ~2x longer than DBNet
- **Inference Speed:** Important for production deployment

### **5. Architecture Comparison Matrix**

| Criteria | DBNet | DBNet++ | CRAFT |
|----------|-------|---------|-------|
| **Accuracy** | Good | Excellent | Very Good |
| **Speed** | Fastest | Slow | Medium |
| **Curved Text** | Limited | Good | Excellent |
| **Memory Usage** | Low | High | Medium |
| **Training Time** | 1x | 2x | 1.5x |
| **Post-processing** | Simple | Moderate | Complex |

### **6. When to Choose Each Architecture**

#### **Choose DBNet When:**
- Dataset has mostly straight, horizontal text
- Fast iteration and experimentation needed
- Limited computational resources
- Baseline performance testing
- Real-time applications with speed constraints

#### **Choose DBNet++ When:**
- Maximum accuracy required
- Competition submissions
- Mixed text orientations
- Sufficient training time available
- High-quality labeled data available

#### **Choose CRAFT When:**
- Significant curved or slanted text (>20% of samples)
- Character-level precision needed
- Irregular text layouts common
- Post-processing customization possible
- Research or specialized applications

### **7. Trade-offs and Considerations**

* **Pro (DBNet++):** Highest accuracy, robust to various text layouts
* **Con (DBNet++):** High computational cost, longer training time
* **Pro (CRAFT):** Excellent curved text detection, character-level precision
* **Con (CRAFT):** Complex post-processing, higher memory usage
* **Pro (DBNet):** Fast training/inference, simple pipeline
* **Con (DBNet):** Limited curved text capability, lower accuracy ceiling

* **Consideration:** Always enable `use_polygon=true` for curved text datasets, regardless of architecture choice.

### **8. Performance Benchmarks**

Based on our ablation studies:

- **DBNet + ResNet18:** H-Mean ~0.85-0.88, ~20-25 min/epoch
- **DBNet++ + ResNet50:** H-Mean ~0.90-0.92, ~35-45 min/epoch
- **CRAFT + ResNet50:** H-Mean ~0.88-0.91, ~30-40 min/epoch

*Note: Performance varies by dataset characteristics and hyperparameter tuning.*

### **9. Migration Between Architectures**

#### **From DBNet to DBNet++**
```bash
# Incremental upgrade
python runners/train.py \
  model.architecture_name=dbnetpp \
  model.encoder.model_name=resnet34 \
  resume=outputs/dbnet_training/checkpoints/last.ckpt
```

#### **From DBNet++ to CRAFT**
```bash
# Architecture switch with fine-tuning
python runners/train.py \
  model.architecture_name=craft \
  model.encoder.model_name=resnet50 \
  training.learning_rate=1e-4  # Lower LR for fine-tuning
```

### **10. Future Architecture Considerations**

- **Hybrid Approaches:** Combining strengths of multiple architectures
- **Lightweight Variants:** MobileNet/EfficientNet backbones for edge deployment
- **Multi-stage Pipelines:** Separate detection and recognition stages
- **Domain Adaptation:** Architecture selection based on target domain characteristics</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/02_protocols/17_advanced_training_techniques.md

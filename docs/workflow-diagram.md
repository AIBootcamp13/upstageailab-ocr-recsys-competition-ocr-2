# OCR Pipeline Workflow

```mermaid
graph TB
    %% Input Data
    subgraph "Input Data"
        A[Raw Images<br/>JPG/PNG]
        B[Annotation JSON<br/>Polygon Labels]
    end

    %% Data Loading
    subgraph "Data Pipeline"
        C[OCRDataset<br/>Class]
        D[Data Transforms<br/>Augmentation]
        E[DataLoader<br/>Batch Processing]
    end

    %% Model Architecture
    subgraph "OCR Model"
        F[Encoder<br/>TIMM Backbone]
        G[Decoder<br/>UNet Architecture]
        H[DB Head<br/>Text Detection]
        I[Loss Function<br/>DB Loss]
    end

    %% Training
    subgraph "Training Pipeline"
        J[Lightning Module<br/>OCR_PL]
        K[PyTorch Lightning<br/>Trainer]
        L[Optimizers<br/>Schedulers]
    end

    %% Evaluation
    subgraph "Evaluation"
        M[CLEval Metric<br/>Character-Level<br/>Evaluation]
        N[Polygon Extraction<br/>Post-processing]
        O[Performance Metrics<br/>Precision/Recall/F1]
    end

    %% Outputs
    subgraph "Outputs"
        P[Model Checkpoints]
        Q[Prediction Maps]
        R[Detected Polygons]
        S[WandB Logs<br/>TensorBoard]
    end

    %% Flow connections
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    K --> P
    K --> S

    %% Evaluation flow
    H --> N
    N --> M
    M --> O
    H --> Q
    N --> R

    %% Styling
    classDef inputClass fill:#e1f5fe
    classDef processClass fill:#f3e5f5
    classDef modelClass fill:#e8f5e8
    classDef trainingClass fill:#fff3e0
    classDef evalClass fill:#ffebee
    classDef outputClass fill:#f3e5f5

    class A,B inputClass
    class C,D,E processClass
    class F,G,H,I modelClass
    class J,K,L trainingClass
    class M,N,O evalClass
    class P,Q,R,S outputClass
```

## Component Details

### Data Pipeline Flow
1. **Raw Images** → OCRDataset loads and validates image files
2. **Annotations** → JSON files with polygon coordinates for text regions
3. **Transforms** → Albumentations for data augmentation and preprocessing
4. **DataLoader** → PyTorch DataLoader for batch processing

### Model Architecture Flow
1. **Encoder** → TIMM backbone (ResNet, EfficientNet, etc.) extracts features
2. **Decoder** → UNet architecture upsamples features to original resolution
3. **DB Head** → Differentiable Binarization head for text region detection
4. **Loss** → DB Loss combines binary and threshold maps for training

### Training Flow
1. **Lightning Module** → Wraps model with training/validation logic
2. **Trainer** → PyTorch Lightning handles training loop, logging, checkpointing
3. **Optimizers** → Configurable optimizers with learning rate schedulers

### Evaluation Flow
1. **Prediction Maps** → Raw model outputs (probability maps)
2. **Polygon Extraction** → Post-processing to extract text polygons
3. **CLEval Metric** → Character-level evaluation against ground truth
4. **Metrics** → Precision, Recall, F1-score computation

## Configuration Management

```mermaid
---
config:
  theme: "forest"
---
graph LR
    A[Base Config<br/>base.yaml] --> B[Dataset Config<br/>datasets/db.yaml]
    A --> C[Model Config<br/>models/model_example.yaml]
    A --> D[Lightning Config<br/>lightning_modules/base.yaml]

    B --> E[Runner Config<br/>train.yaml]
    C --> E
    D --> E

    E --> F[Hydra<br/>Configuration]
    F --> G[Experiment<br/>Execution]

    classDef configClass fill:#e3f2fd
    classDef runnerClass fill:#fff9c4
    classDef systemClass fill:#c8e6c9

    class A,B,C,D configClass
    class E runnerClass
    class F,G systemClass
```

## Key Integration Points

- **Hydra**: Configuration management for experiments
- **PyTorch Lightning**: Training framework with built-in features
- **TIMM**: Model zoo for backbone selection
- **Albumentations**: Image augmentation library
- **WandB/TensorBoard**: Experiment tracking and visualization
- **CLEval**: Industry-standard OCR evaluation metric

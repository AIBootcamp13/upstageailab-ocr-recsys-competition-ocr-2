# Component Diagrams

## Current Architecture Diagram

```mermaid
---
config:
  theme: "forest"
---
graph TB
    subgraph "Data Layer"
        A[Input Images] --> B[OCRTransforms]
        B --> C[OCRDataset]
        C --> D[DataLoader]
        D --> E[DBCollateFn]
    end

    subgraph "Model Layer"
        E --> F[OCRLightningModule]
        F --> G[OCRModel]

        subgraph "OCRModel Components"
            G --> H[TimmBackbone<br/>Encoder]
            G --> I[UNet<br/>Decoder]
            G --> J[DBNetHead]
            G --> K[DBLoss]
        end
    end

    subgraph "Training Layer"
        F --> L[PyTorch Lightning<br/>Trainer]
        L --> M[Training Loop]
        M --> N[Validation]
        N --> O[Checkpointing]
    end

    subgraph "Evaluation Layer"
        P[CLEvalMetric] --> Q[Precision/Recall/F1]
        R[Visualization] --> S[Prediction Overlay]
    end

    J --> P
    K --> M
    O --> T[Model Artifacts]
```

## Plug-and-Play Architecture Diagram

```mermaid
graph TB
    subgraph "Configuration Layer"
        A[Hydra Config] --> B[Architecture Registry]
        B --> C[Model Factory]
    end

    subgraph "Component Registry"
        D[Encoder Registry<br/>• TimmBackbone<br/>• EfficientNet<br/>• Custom CNN]
        E[Decoder Registry<br/>• UNet<br/>• FPN<br/>• Custom Decoder]
        F[Head Registry<br/>• DBNet Head<br/>• EAST Head<br/>• CRAFT Head]
        G[Loss Registry<br/>• DB Loss<br/>• Dice Loss<br/>• Focal Loss]
    end

    subgraph "Model Assembly"
        C --> H[CompositeModel]
        D --> H
        E --> H
        F --> H
        G --> H
    end

    subgraph "Abstract Interfaces"
        I[BaseEncoder] --> D
        J[BaseDecoder] --> E
        K[BaseHead] --> F
        L[BaseLoss] --> G
    end

    subgraph "Data Pipeline"
        M[BaseDataset] --> N[DataLoader]
        O[Transform Pipeline] --> M
        P[Collate Functions] --> N
    end

    subgraph "Training & Evaluation"
        H --> Q[Lightning Module]
        Q --> R[Trainer]
        S[BaseMetric] --> T[Evaluation Pipeline]
        U[Callbacks] --> R
    end
```

## Component Interface Diagrams

### BaseEncoder Interface

```mermaid
classDiagram
    class BaseEncoder {
        <<abstract>>
        +config: Dict[str, Any]
        +forward(x: Tensor): Tensor
        +output_channels*: int
        +output_stride*: int
        +get_feature_maps()*: List[Tensor]
    }

    class TimmBackbone {
        +backbone: str
        +pretrained: bool
        +freeze_backbone: bool
        +forward(x: Tensor): Tensor
        +output_channels: int
        +output_stride: int
    }

    class EfficientNetBackbone {
        +model_name: str
        +pretrained: bool
        +forward(x: Tensor): Tensor
        +output_channels: int
        +output_stride: int
    }

    BaseEncoder <|-- TimmBackbone
    BaseEncoder <|-- EfficientNetBackbone
```

### BaseDecoder Interface

```mermaid
classDiagram
    class BaseDecoder {
        <<abstract>>
        +config: Dict[str, Any]
        +forward(features: Tensor): Tensor
        +get_feature_maps()*: List[Tensor]
        +in_channels*: int
        +out_channels*: int
    }

    class UNetDecoder {
        +num_layers: int
        +bilinear: bool
        +forward(features: Tensor): Tensor
        +get_feature_maps(): List[Tensor]
    }

    class FPNDecoder {
        +num_levels: int
        +forward(features: Tensor): Tensor
        +get_feature_maps(): List[Tensor]
    }

    BaseDecoder <|-- UNetDecoder
    BaseDecoder <|-- FPNDecoder
```

### BaseHead Interface

```mermaid
classDiagram
    class BaseHead {
        <<abstract>>
        +config: Dict[str, Any]
        +forward(features: Tensor): Any
        +postprocess(predictions: Any): Dict[str, Any]
        +task_type*: str
    }

    class DBNetHead {
        +forward(features: Tensor): Tuple[Tensor, Tensor]
        +postprocess(predictions: Tuple): Dict
        +task_type: "text_detection"
    }

    class EASTHead {
        +forward(features: Tensor): Tuple[Tensor, Tensor, Tensor]
        +postprocess(predictions: Tuple): Dict
        +task_type: "text_detection"
    }

    BaseHead <|-- DBNetHead
    BaseHead <|-- EASTHead
```

### BaseLoss Interface

```mermaid
classDiagram
    class BaseLoss {
        <<abstract>>
        +config: Dict[str, Any]
        +forward(predictions: Any, targets: Any): Tensor
        +get_loss_components(): Dict[str, Tensor]
        +loss_type*: str
    }

    class DBLoss {
        +weight_bce: float
        +weight_dice: float
        +weight_l1: float
        +forward(predictions, targets): Tensor
        +get_loss_components(): Dict
    }

    class DiceLoss {
        +smooth: float
        +forward(predictions, targets): Tensor
        +get_loss_components(): Dict
    }

    BaseLoss <|-- DBLoss
    BaseLoss <|-- DiceLoss
```

## Data Flow Diagrams

### Training Data Flow

```mermaid
sequenceDiagram
    participant DataLoader
    participant Dataset
    participant Transforms
    participant LightningModule
    participant Model
    participant Loss

    DataLoader->>Dataset: __getitem__(idx)
    Dataset->>Transforms: __call__(image, annotations)
    Transforms-->>Dataset: transformed_image, targets
    Dataset-->>DataLoader: batch
    DataLoader->>LightningModule: training_step(batch, batch_idx)
    LightningModule->>Model: forward(batch['image'])
    Model-->>LightningModule: predictions
    LightningModule->>Loss: forward(predictions, batch['targets'])
    Loss-->>LightningModule: loss_value
    LightningModule-->>DataLoader: loss_dict
```

### Inference Data Flow

```mermaid
sequenceDiagram
    participant Image
    participant Transforms
    participant Model
    participant Head
    participant PostProcess

    Image->>Transforms: preprocess
    Transforms-->>Model: normalized_image
    Model->>Head: forward
    Head-->>Model: raw_predictions
    Model-->>PostProcess: raw_predictions
    PostProcess-->>Image: polygons, scores, text_regions
```

## Configuration Flow

```mermaid
graph TD
    A[config.yaml] --> B[OmegaConf.load]
    B --> C[ConfigValidator]
    C --> D{Valid?}
    D -->|Yes| E[ModelFactory.create_model]
    D -->|No| F[ValidationError]

    E --> G[Registry.get_components]
    G --> H[Component Instantiation]
    H --> I[CompositeModel]

    I --> J[LightningModule]
    J --> K[Trainer.fit]

    K --> L[Training Loop]
    L --> M[Validation]
    M --> N[Model Checkpoint]
```

## Registry System Architecture

```mermaid
graph TD
    subgraph "Registry Components"
        A[ArchitectureRegistry] --> B[Encoder Registry]
        A --> C[Decoder Registry]
        A --> D[Head Registry]
        A --> E[Loss Registry]
        A --> F[Architecture Presets]
    end

    subgraph "Registration Process"
        G[DBNet Architecture] --> H[register_architecture]
        I[TimmBackbone] --> J[register_encoder]
        K[UNetDecoder] --> L[register_decoder]
        M[DBNetHead] --> N[register_head]
        O[DBLoss] --> P[register_loss]
    end

    subgraph "Usage"
        Q[ModelFactory] --> R[get_architecture]
        R --> S[Component Classes]
        S --> T[Model Instantiation]
    end

    H --> A
    J --> B
    L --> C
    N --> D
    P --> E
```

## Experiment Management Flow

```mermaid
graph TD
    subgraph "Experiment Setup"
        A[experiment.yaml] --> B[ExperimentConfig]
        B --> C[Seed Setting]
        C --> D[Reproducibility Config]
    end

    subgraph "Model & Data"
        E[Architecture Selection] --> F[ModelFactory]
        G[Dataset Selection] --> H[DataModule]
        F --> I[Model]
        H --> J[DataLoader]
    end

    subgraph "Training"
        I --> K[LightningModule]
        J --> K
        K --> L[Trainer]
        L --> M[Training Loop]
        M --> N[Validation Loop]
    end

    subgraph "Logging & Monitoring"
        O[WandbLogger] --> P[Metrics Logging]
        Q[CheckpointCallback] --> R[Model Saving]
        S[EarlyStopping] --> T[Training Control]
    end

    N --> O
    M --> Q
    N --> S
```

## Component Dependencies

```mermaid
graph TD
    subgraph "Core Dependencies"
        A[torch] --> B[torchvision]
        A --> C[timm]
        A --> D[albumentations]
    end

    subgraph "ML Framework"
        E[pytorch_lightning] --> F[torch]
        G[hydra-core] --> H[omegaconf]
    end

    subgraph "Project Components"
        I[ocr_framework.core] --> A
        J[ocr_framework.architectures] --> I
        K[ocr_framework.models] --> J
        L[ocr_framework.training] --> K
        M[ocr_framework.datasets] --> A
        N[ocr_framework.evaluation] --> A
    end

    subgraph "Development Tools"
        O[pytest] --> P[Project]
        Q[black] --> P
        R[isort] --> P
        S[flake8] --> P
    end

    subgraph "Optional Dependencies"
        T[wandb] --> U[Logging]
        V[rich] --> W[CLI Output]
        X[icecream] --> Y[Debugging]
    end
```

## Error Handling Flow

```mermaid
graph TD
    A[Component Error] --> B{Error Type}
    B -->|Config Error| C[ConfigValidationError]
    B -->|Runtime Error| D[ModelRuntimeError]
    B -->|Data Error| E[DataProcessingError]

    C --> F[Validation Hook]
    D --> G[Exception Handler]
    E --> H[Data Recovery]

    F --> I[User Feedback]
    G --> J[Graceful Degradation]
    H --> K[Skip Batch]

    I --> L[Config Fix]
    J --> M[Continue Training]
    K --> N[Log Warning]
```

## Testing Architecture

```mermaid
graph TD
    subgraph "Test Categories"
        A[Unit Tests] --> B[Component Tests]
        C[Integration Tests] --> D[Pipeline Tests]
        E[Manual Tests] --> F[Visualization Tests]
        G[Debug Tests] --> H[Reproducibility Tests]
    end

    subgraph "Test Fixtures"
        I[Mock Data] --> B
        J[Sample Configs] --> B
        K[Test Images] --> D
        L[Mock Models] --> D
    end

    subgraph "Test Infrastructure"
        M[pytest] --> N[Test Discovery]
        O[Coverage] --> P[Report Generation]
        Q[CI/CD] --> R[Automated Testing]
    end

    B --> N
    D --> N
    F --> N
    H --> N
```

These diagrams provide a comprehensive visual representation of the current architecture and the planned plug-and-play system, showing component relationships, data flows, and system interactions.</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/component-diagrams.md

# Naming Conventions

## Python Naming Standards

### 1. General Rules
- **Case**: Follow PEP 8 naming conventions
- **Consistency**: Use same patterns throughout codebase
- **Clarity**: Names should be descriptive and unambiguous
- **Length**: Balance between clarity and brevity

### 2. Files and Modules
```python
# Module names (snake_case)
data_loader.py
model_utils.py
config_validator.py

# Package names (snake_case, no underscores for top-level)
ocr_framework/
├── core/
├── architectures/
├── datasets/
└── utils/

# Test files
test_ocr_model.py
test_data_loader.py
test_config_validator.py
```

### 3. Classes and Types
```python
# Class names (PascalCase)
class OCRModel(nn.Module):
class TimmBackbone(nn.Module):
class DBNetHead(nn.Module):
class CLEvalMetric(Metric):

# Abstract base classes
class BaseEncoder(nn.Module, ABC):
class BaseDecoder(nn.Module, ABC):

# Exception classes
class ConfigurationError(ValueError):
class DatasetError(RuntimeError):

# Data classes
@dataclass
class ModelConfig:
@dataclass
class TrainingConfig:
```

### 4. Functions and Methods
```python
# Function names (snake_case)
def load_config():
def process_image():
def validate_polygons():
def calculate_metrics():

# Method names (snake_case)
def forward(self, x):
def training_step(self, batch, batch_idx):
def configure_optimizers(self):

# Private methods (single underscore prefix)
def _validate_config(self):
def _preprocess_image(self):

# Special methods (double underscore)
def __init__(self):
def __call__(self):
def __repr__(self):
```

### 5. Variables and Attributes
```python
# Local variables (snake_case)
image_path = "data/train/image1.jpg"
batch_size = 8
learning_rate = 0.001

# Instance attributes (snake_case)
self.model = model
self.config = config
self.logger = logger

# Constants (UPPER_CASE)
MAX_IMAGE_SIZE = 4096
DEFAULT_THRESHOLD = 0.5
SUPPORTED_BACKBONES = ["resnet50", "resnet101"]

# Loop variables
for image, polygons in dataset:
for batch_idx, batch in enumerate(dataloader):
```

### 6. Configuration Keys
```yaml
# Top-level sections (snake_case)
model:
  encoder:
  decoder:
  head:
  loss:

data:
  train_path: "data/train"
  val_path: "data/val"
  batch_size: 8

training:
  max_epochs: 100
  learning_rate: 0.001

# Component configurations
backbone: "resnet50"
pretrained: true
freeze_backbone: false

# Target specifications
_target_: "ocr.models.encoder.TimmBackbone"
```

## Domain-Specific Naming

### 1. OCR-Specific Terms
```python
# Model components
probability_map
text_region
polygon_coordinates
ground_truth_polygons
predicted_polygons

# Dataset terms
image_annotation
word_bbox
text_line
character_region

# Evaluation terms
precision_score
recall_score
f1_score
character_level_evaluation
```

### 2. Architecture Names
```python
# Architecture identifiers
dbnet_architecture
east_architecture
unet_backbone
resnet_encoder

# Component names
db_head
db_loss
cleval_metric
timm_backbone
```

### 3. File and Directory Names
```python
# Data directories
images/
annotations/
pseudo_labels/
checkpoints/
logs/
submissions/

# Configuration files
train.yaml
val.yaml
test.yaml
model_config.yaml

# Output files
predictions.json
metrics.json
visualizations.png
```

## Hydra Configuration Naming

### 1. Config Groups
```
configs/
├── architectures/     # Architecture-specific configs
│   ├── dbnet.yaml
│   ├── east.yaml
│   └── custom.yaml
├── datasets/          # Dataset-specific configs
│   ├── icdar.yaml
│   ├── synthtext.yaml
│   └── custom.yaml
└── experiments/       # Experiment configurations
    ├── baseline.yaml
    ├── ablation.yaml
    └── hyperopt.yaml
```

### 2. Config File Structure
```yaml
# configs/architectures/dbnet.yaml
defaults:
  - encoder: timm_resnet50
  - decoder: unet
  - head: db_head
  - loss: db_loss
  - _self_

architecture_name: dbnet
model:
  encoder:
    _target_: ocr_framework.architectures.dbnet.encoder.DBNetEncoder
    backbone: resnet50
  decoder:
    _target_: ocr_framework.architectures.dbnet.decoder.DBNetDecoder
  head:
    _target_: ocr_framework.architectures.dbnet.head.DBNetHead
  loss:
    _target_: ocr_framework.architectures.dbnet.loss.DBNetLoss
```

## Testing Naming Conventions

### 1. Test Files
```python
# Unit tests
test_ocr_model.py
test_data_loader.py
test_config_validator.py

# Integration tests
test_training_pipeline.py
test_evaluation_pipeline.py

# Manual tests
test_visualization.py
test_data_validation.py
```

### 2. Test Functions
```python
def test_model_initialization():
def test_forward_pass():
def test_loss_calculation():
def test_config_validation():

# Parameterized tests
@pytest.mark.parametrize("backbone", ["resnet50", "resnet101"])
def test_backbone_compatibility(backbone):

# Fixtures
@pytest.fixture
def sample_config():
@pytest.fixture
def mock_dataset():
```

### 3. Test Classes
```python
class TestOCRModel:
class TestDataLoader:
class TestConfigValidator:

# Test methods
def test_initialization(self):
def test_forward_pass(self):
def test_error_handling(self):
```

## Documentation Naming

### 1. Documentation Files
```markdown
# Main documentation
README.md
CONTRIBUTING.md
CHANGELOG.md

# Development docs
docs/coding-standards.md
docs/naming-conventions.md
docs/testing-guide.md

# Architecture docs
docs/architecture-overview.md
docs/component-diagrams.md
docs/api-reference.md
```

### 2. Section Headers
```markdown
# Main Title (H1)
## Section (H2)
### Subsection (H3)
#### Sub-subsection (H4)

## Implementation Details
### Class Architecture
### Method Signatures
### Configuration Options
```

## Version and Release Naming

### 1. Version Numbers
```
v1.0.0          # Major release
v1.1.0          # Minor release
v1.1.1          # Patch release
v2.0.0-alpha    # Pre-release
```

### 2. Branch Names
```bash
# Feature branches
feature/dbnet-architecture
feature/multi-gpu-support
feature/advanced-augmentations

# Bug fix branches
fix/memory-leak
fix/config-validation

# Experiment branches
experiment/alternative-loss
experiment/efficientnet-backbone
```

### 3. Experiment Names
```python
# Experiment identifiers
exp_001_baseline_dbnet
exp_002_east_comparison
exp_003_data_augmentation
exp_004_hyperparameter_tuning
```

## API and Interface Naming

### 1. Function Parameters
```python
def load_dataset(
    data_path: str,
    annotation_file: str,
    image_size: Tuple[int, int],
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True
):
```

### 2. Return Values
```python
def get_model_metrics() -> Dict[str, float]:
    """Get comprehensive model metrics.

    Returns:
        Dictionary containing:
        - 'precision': Precision score
        - 'recall': Recall score
        - 'f1': F1 score
        - 'accuracy': Accuracy score
    """
```

### 3. Callback Functions
```python
def on_epoch_end(epoch: int, logs: Dict[str, float]) -> None:
def on_batch_start(batch_idx: int) -> None:
def on_validation_end(metrics: Dict[str, float]) -> None:
```

## Database and Storage Naming

### 1. Database Tables/Collections
```python
# Experiment tracking
experiments_table
model_configs_table
training_runs_table

# Results storage
predictions_table
metrics_table
artifacts_table
```

### 2. File Storage
```python
# Model files
model_weights.pth
model_config.json
optimizer_state.pth

# Data files
train_annotations.json
val_annotations.json
test_predictions.json

# Log files
training.log
validation.log
debug.log
```

## Environment and Deployment Naming

### 1. Environment Variables
```bash
# Project paths
OCR_PROJECT_ROOT=/path/to/project
OCR_DATA_DIR=/path/to/data
OCR_OUTPUT_DIR=/path/to/outputs

# Configuration
OCR_CONFIG_FILE=config.yaml
OCR_LOG_LEVEL=INFO

# GPU settings
CUDA_VISIBLE_DEVICES=0,1
OMP_NUM_THREADS=4
```

### 2. Docker Naming
```dockerfile
# Image names
ocr-training:latest
ocr-inference:v1.0.0
ocr-dataset-prep:dev

# Container names
ocr_training_container
ocr_inference_service
ocr_data_processor
```

### 3. Kubernetes Resources
```yaml
# Deployment names
ocr-model-deployment
ocr-api-deployment
ocr-worker-deployment

# Service names
ocr-api-service
ocr-model-service

# ConfigMap names
ocr-config
ocr-secrets
```

This naming convention ensures consistency across the entire codebase, making it easier for team members to understand and navigate the project structure.</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/development/naming-conventions.md

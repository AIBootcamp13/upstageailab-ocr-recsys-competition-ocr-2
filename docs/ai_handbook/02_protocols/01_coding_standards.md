# **filename: docs/ai_handbook/02_protocols/01_coding_standards.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=code_changes,style -->

# **Protocol: Coding Standards & Naming Conventions**

This document defines the coding standards and naming conventions for the project to ensure consistency, readability, and maintainability.

## **1. Code Formatting & Linting**

* **Formatter:** Ruff Formatter (or Black with line length 88).
* **Linter:** Ruff.
* **Execution:** All formatting and linting is handled automatically via pre-commit hooks and the CI pipeline. Run uv run ruff check . --fix && uv run ruff format . to apply manually.
* **Import Sorting:** Imports are organized by Ruff into three groups: standard library, third-party, and local application imports.

## **2. Type Hinting**

* **Requirement:** All public functions, methods, and class attributes **must** include type hints.
* **Clarity:** Use specific types from the typing module (Dict, List, Optional, Tuple, Union) where appropriate.

```python
from typing import Dict, List, Optional, Tuple
import torch

def process_batch(
    images: torch.Tensor,
    targets: List[Dict[str, any]],
    device: Optional[str] = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Processes a batch of images and targets."""
    # ... function implementation
    pass
```

## **3. Naming Conventions**

Adherence to PEP 8 is required. The key conventions are summarized below.

* **Modules & Packages:** snake_case (e.g., ocr_framework, data_loader.py).
* **Classes:** PascalCase (e.g., OCRLightningModule, TimmBackbone).
  * Abstract Base Classes should start with Base (e.g., BaseEncoder).
  * Exceptions should end with Error (e.g., ConfigurationError).
* **Functions & Methods:** snake_case (e.g., validate_polygons, training_step).
  * Private methods should be prefixed with a single underscore (e.g., _preprocess_image).
* **Variables & Attributes:** snake_case (e.g., learning_rate, self.batch_size).
* **Constants:** UPPER_SNAKE_CASE (e.g., MAX_IMAGE_SIZE, DEFAULT_THRESHOLD).

## **4. Architecture & Design Patterns**

* **Configuration-Driven:** All components (models, datasets, optimizers) are instantiated from Hydra configurations. Avoid hard-coding component choices in Python code.
* **Abstract Base Classes:** New core components (encoders, decoders, heads) should inherit from their respective Base class in ocr_framework/core/ to ensure a consistent interface.
* **Registry Pattern:** Architectures and components are registered in and retrieved from the ArchitectureRegistry to enable plug-and-play experimentation.
* **Factory Functions:** Use factory functions (e.g., get_encoder_by_cfg) that take a config object as input and return an instantiated component.

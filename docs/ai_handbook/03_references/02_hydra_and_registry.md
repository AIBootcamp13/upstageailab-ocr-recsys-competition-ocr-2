# **filename: docs/ai_handbook/03_references/02_hydra_and_registry.md**

# **Reference: Hydra & Component Registry**

This document provides a technical reference for the project's configuration system (Hydra) and the custom component registry.

## **1. Hydra Configuration System**

Hydra is the single source of truth for all experiment parameters. It allows for modular and overridable configurations.

### **1.1. Configuration Structure**

The configuration is organized into a hierarchy under the configs/ directory.

configs/
├── data/
├── model/
│   ├── encoder/
│   ├── decoder/
│   ├── head/
│   └── loss/
├── trainer/
├── logger/
│   ├── default.yaml
│   ├── wandb.yaml
│   └── csv.yaml
├── hydra/
│   └── default.yaml
├── extras/
│   └── default.yaml
└── train.yaml  # Main config file

### **1.2. Instantiation with _target_**

Hydra instantiates Python objects directly from the configuration using the special _target_ key. This is the primary mechanism for building models, datasets, and other components.

**Example (configs/model/encoder/timm_backbone.yaml):**

_target_: ocr_framework.architectures.dbnet.encoder.TimmBackbone
backbone: resnet50
pretrained: true

When this config is passed to Hydra's instantiate function, it creates an instance of the TimmBackbone class with backbone='resnet50' and pretrained=True.

### **1.3. Overriding Parameters**

You can override any parameter from the command line, which is the standard way to run experiments.

# Override the learning rate and batch size
uv run python runners/train.py model.optimizer.lr=0.0005 data.batch_size=16

### **1.4. Logging Configuration**

The project supports multiple logging backends for comprehensive experiment tracking:

* **WandB Logger**: Primary logging for experiment visualization and comparison
* **CSV Logger**: Structured logging for programmatic analysis and backup
* **Hydra Logging**: Rich-formatted console output with color coding

Logging configurations are managed through:
- `configs/logger/default.yaml` - Combines WandB and CSV loggers
- `configs/hydra/default.yaml` - Console logging with Rich formatting
- `configs/extras/default.yaml` - Miscellaneous logging settings

## **2. Component Registry**

The Component Registry is a custom system that works with Hydra to enable plug-and-play architecture experimentation. It acts as a catalog of available components.

### **2.1. Purpose**

The registry allows you to define entire architectures (like DBNet or CRAFT) and their components (encoders, decoders, etc.) in a central location. This makes it easy to switch between them using a single configuration parameter.

### **2.2. How It Works**

1. **Registration:** Components are "registered" with a unique name (e.g., timm_resnet50, unet_decoder). This happens in ocr_framework/architectures/registry.py.
2. **Architecture Presets:** A complete model is defined as a "preset" that references the registered component names.
3. **Factory Instantiation:** The ModelFactory reads the desired architecture from the config, looks up the component names in the registry, and instantiates them.

### **2.3. Registry Compatibility**

When adding or modifying components, you must ensure they are compatible.

* **Shape Compatibility:** The out_channels of an encoder must match the in_channels of a decoder. The registry system does not automatically validate this; it must be checked manually or via smoke tests.
* **Data Compatibility:** Some models may require specific data formats (e.g., CRAFT requires character-level annotations). Ensure the chosen dataset is compatible with the selected architecture.

## **3. Pydantic Policy**

* **Hydra is Authoritative:** Hydra manages all configuration for model training, validation, and testing.
* **Pydantic for UI Only:** The use of Pydantic models is strictly limited to validating user inputs within the Streamlit UI. It **must not** be used for managing or composing model configuration files. The UI's role is to generate valid Hydra overrides, not to manage its own configuration state.

## **4. Best Practices & Template Adoption**

For guidance on adopting best practices from external templates like the lightning-hydra-template, refer to the [**Template Adoption Protocol**](../02_protocols/16_template_adoption_protocol.md). This protocol provides a systematic approach to analyzing template structures, comparing with our current setup, and proposing incremental improvements to align with community standards.

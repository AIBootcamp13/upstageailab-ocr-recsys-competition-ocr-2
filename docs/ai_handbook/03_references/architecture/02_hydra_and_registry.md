# **filename: docs/ai_handbook/03_references/architecture/02_hydra_and_registry.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=configuration,hydra,registry,setup -->

# **Reference: Hydra & Component Registry**

This reference document provides comprehensive information about the OCR system's configuration system using Hydra and the custom component registry for quick lookup and detailed understanding.

## **Overview**

The OCR system uses Hydra as the authoritative configuration management system combined with a custom component registry to enable plug-and-play architecture experimentation. This setup allows declarative configuration of all experiment parameters and modular component assembly.

## **Key Concepts**

### **Hydra Configuration System**
Hydra serves as the single source of truth for all experiment parameters, enabling modular and overridable configurations through hierarchical YAML files and command-line overrides.

### **Component Registry**
A custom catalog system that works with Hydra to enable plug-and-play architecture experimentation by registering components with unique names and assembling them into complete models.

### **Instantiation with _target_**
Hydra's mechanism for directly instantiating Python objects from configuration using the special _target_ key, allowing declarative object construction.

### **Architecture Presets**
Complete model definitions that reference registered component names, enabling easy switching between different architectures through configuration.

## **Detailed Information**

### **Configuration Structure**
The configuration is organized into a hierarchical structure under the configs/ directory:

```
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
```

### **Logging Configuration**
The project supports multiple logging backends for comprehensive experiment tracking:

- **WandB Logger**: Primary logging for experiment visualization and comparison
- **CSV Logger**: Structured logging for programmatic analysis and backup
- **Hydra Logging**: Rich-formatted console output with color coding

### **Registry Compatibility**
When adding or modifying components, ensure compatibility:

- **Shape Compatibility**: Output channels of encoders must match input channels of decoders
- **Data Compatibility**: Some models may require specific data formats (e.g., CRAFT requires character-level annotations)

## **Examples**

### **Basic Usage**
```python
# Example Hydra instantiation
from hydra.utils import instantiate

config = {
    '_target_': 'ocr_framework.architectures.dbnet.encoder.TimmBackbone',
    'backbone': 'resnet50',
    'pretrained': True
}

encoder = instantiate(config)
```

### **Advanced Usage**
```python
# Command line overrides
# Override learning rate and batch size
uv run python runners/train.py model.optimizer.lr=0.0005 data.batch_size=16

# Switch architectures
uv run python runners/train.py model.architecture=east
```

## **Configuration Options**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| _target_ | str | - | Python class path for instantiation |
| backbone | str | resnet50 | Encoder backbone name |
| pretrained | bool | true | Use pretrained weights |
| logger | str | default | Logging backend (wandb, csv, default) |
| architecture | str | dbnet | Model architecture preset |

## **Best Practices**

- **Hydra Authoritative**: Use Hydra for all model training, validation, and testing configuration
- **Pydantic for UI Only**: Limit Pydantic models to Streamlit UI input validation only
- **Registry Registration**: Register all components with unique names for discoverability
- **Shape Validation**: Manually validate component compatibility (encoder output → decoder input)
- **Configuration Overrides**: Use command-line overrides for experiment variations

## **Troubleshooting**

### **Common Issues**
- **Instantiation Errors**: Verify _target_ paths are correct and classes are importable
- **Configuration Not Found**: Check config file paths and Hydra search paths
- **Component Not Registered**: Ensure components are registered in the appropriate registry
- **Shape Mismatches**: Validate tensor dimensions between connected components

### **Debug Information**
- Enable debug logging: `export HYDRA_LOGGING=DEBUG`
- List registered components: `python -c "from ocr_framework.architectures.registry import list_components; print(list_components())"`
- Validate config: `python -c "from omegaconf import OmegaConf; OmegaConf.select(config, 'model')"`

## **Related References**

- `docs/ai_handbook/03_references/architecture/01_architecture.md` - System architecture overview
- `docs/ai_handbook/02_protocols/configuration/20_hydra_config_resolution_troubleshooting.md` - Hydra troubleshooting guide
- `docs/ai_handbook/02_protocols/16_template_adoption_protocol.md` - Template adoption best practices

---

*This document follows the references template. Last updated: 2025-01-15*

# Streamlit UI Development Guide

## Overview
This document provides essential context for Agentic AI systems working on the OCR Training Command Builder Streamlit UI. The UI enables users to build and execute training, testing, and prediction commands through an intuitive interface.

## Architecture

### Core Components
- **command_builder.py**: Main Streamlit application entry point
- **ui/schemas/**: YAML schema files defining UI elements and validation
- **ui/utils/**: Utility modules for UI generation, validation, and configuration parsing

### Key Files
```
ui/
├── command_builder.py          # Main Streamlit app
├── schemas/
│   ├── command_builder_train.yaml
│   ├── command_builder_test.yaml
│   └── command_builder_predict.yaml
└── utils/
    ├── ui_generator.py         # Dynamic UI generation from schemas
    ├── config_parser.py        # Configuration and model discovery
    ├── command_builder.py      # Command generation logic
    ├── ui_validator.py         # Input validation
    └── config_parser.py        # Configuration parsing
```

## UI Schema System

### Schema Structure
Each YAML schema defines:
- **title**: UI section title
- **constant_overrides**: Fixed Hydra overrides
- **ui_elements**: List of form elements

### UI Element Types
- **text_input**: Single-line text input
- **selectbox**: Dropdown selection
- **checkbox**: Boolean toggle
- **slider**: Numeric range selection
- **number_input**: Numeric input field

### Dynamic Options
Options can be sourced dynamically:
- `models.architectures`: Available OCR architectures (dbnet, craft, dbnetpp)
- `models.backbones`: Available encoder backbones (resnet18, resnet50, etc.)
- `models.decoders`: Available decoder types (for future use)
- `checkpoints`: Available model checkpoints
- `datasets`: Available datasets

### Validation
- **required**: Field must be filled
- **required_if**: Conditional requirement based on other fields
- **range**: Numeric min/max validation
- **min_length**: String minimum length

### Visibility Control
- **visible_if**: Show/hide elements based on conditions
- Uses simple boolean expressions: `field_name == "value"`

## Configuration System

### Hydra Integration
- UI elements map to Hydra configuration overrides
- `hydra_override`: Path in config hierarchy
- Supports multiple overrides per element

### Model Registry
Components are registered dynamically:
- **Encoders**: Feature extractors (timm_backbone, craft_vgg)
- **Decoders**: Upsampling networks (unet, craft_decoder, dbpp_decoder)
- **Heads**: Prediction heads (db_head, craft_head)
- **Losses**: Training objectives (db_loss, craft_loss)
- **Architectures**: Complete presets combining components

## Development Workflow

### Adding New UI Elements
1. Update appropriate YAML schema in `ui/schemas/`
2. Add validation rules if needed
3. Test UI generation and command building
4. Update documentation

### Adding New Options Sources
1. Update `_get_options_from_source()` in `ui_generator.py`
2. Add corresponding method in `ConfigParser` if needed
3. Test option loading

### Testing UI Changes
```bash
# Validate YAML syntax
uv run python -c "import yaml; yaml.safe_load(open('ui/schemas/command_builder_train.yaml'))"

# Test options loading
uv run python -c "from ui.utils.ui_generator import _get_options_from_source; print(_get_options_from_source('models.architectures'))"
```

## Command Building

### Command Generation Flow
1. User fills form based on schema
2. `generate_ui_from_schema()` creates Streamlit elements
3. `validate_inputs()` checks requirements
4. `CommandBuilder.build_command_from_overrides()` generates command
5. Command execution via `execute_command()`

### Override Processing
- **Constant overrides**: Always applied
- **Dynamic overrides**: Generated from user selections
- **Conditional logic**: Experiment name suffixing, resume handling

## Best Practices

### UI Design
- Use clear, descriptive labels and help text
- Group related options logically
- Provide sensible defaults
- Use validation to prevent invalid configurations

### Schema Organization
- Keep schemas modular and focused
- Use consistent naming conventions
- Document complex validation logic
- Test schema changes thoroughly

### Error Handling
- Validate inputs before command generation
- Provide clear error messages
- Handle edge cases gracefully
- Log issues for debugging

## Common Patterns

### Conditional Visibility
```yaml
- key: checkpoint_path
  visible_if: "resume_training == true"
  required_if: "resume_training == true"
```

### Multiple Overrides
```yaml
- key: batch_size
  hydra_override:
    - "dataloaders.train_dataloader.batch_size"
    - "dataloaders.val_dataloader.batch_size"
```

### Dynamic Options
```yaml
- key: architecture
  options_source: "models.architectures"
  default: "dbnet"
```

## Troubleshooting

### Common Issues
- **Empty options lists**: Check ConfigParser methods and registry imports
- **Schema validation errors**: Verify YAML syntax and required fields
- **Command generation failures**: Check hydra_override paths and value types
- **UI not updating**: Clear Streamlit cache and restart app

### Debugging
- Use `st.write()` for debugging UI state
- Check browser console for JavaScript errors
- Validate commands manually before execution
- Test individual components in isolation

## Architecture Context

### Available Components
- **Architectures**: dbnet, craft, dbnetpp (currently selectable in UI)
- **Encoders**: timm_backbone, craft_vgg, dbnetpp_backbone
- **Decoders**: unet, craft_decoder, dbpp_decoder, dbnetpp_decoder (not currently selectable in UI)
- **Heads**: db_head, craft_head, dbnetpp_head
- **Losses**: db_loss, craft_loss, dbnetpp_loss

### Configuration Hierarchy
```
model/
├── architecture/
│   └── name: "dbnet"
├── encoder/
│   ├── _target_: "ocr.models.encoders..."
│   └── model_name: "resnet18"
├── decoder/
│   └── _target_: "ocr.models.decoders..."
└── head/
    └── _target_: "ocr.models.heads..."
```

This context enables Agentic AI to understand the UI system, make informed modifications, and maintain consistency with the existing architecture.</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/copilot/streamlit-instructions.md

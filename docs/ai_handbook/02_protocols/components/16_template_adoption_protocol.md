# **filename: docs/ai_handbook/02_protocols/components/16_template_adoption_protocol.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=refactor,architecture,onboarding -->

# **Protocol: Template Adoption & Best Practices**

## **Overview**

This protocol governs systematic adoption of patterns from external template repositories, specifically the lightning-hydra-template. It provides a controlled workflow for aligning project structure with community best practices while maintaining incremental, safe refactoring.

## **Prerequisites**

- Access to external template documentation in `docs/external/lightning-hydra-template/`
- Understanding of current project structure in `ocr/` and `configs/` directories
- Familiarity with the Modular Refactoring Protocol
- Knowledge of PyTorch Lightning and Hydra configuration patterns

## **Component Architecture**

### **Core Components**
- **Template Analysis Engine**: Systematic comparison between external templates and current structure
- **Incremental Refactor Framework**: Safe, step-by-step adoption of template patterns
- **Validation Pipeline**: Ensures adopted patterns maintain functionality
- **Documentation Sync**: Keeps local template documentation current

### **Integration Points**
- `docs/external/lightning-hydra-template/`: Local copy of template documentation
- `ocr/`: Current project source code structure
- `configs/`: Current configuration organization
- Modular Refactoring Protocol: Governs implementation approach

## **Procedure**

### **Step 1: Template Analysis and Documentation Review**
Load and analyze template documentation:

```bash
# Review template structure
cat docs/external/lightning-hydra-template/README.md

# Examine configuration examples
ls docs/external/lightning-hydra-template/configs/

# Study code organization patterns
ls docs/external/lightning-hydra-template/src/
```

Identify key structural patterns:
- `src/` layout for Python code organization
- `configs/` directory structure (callbacks/, datamodule/, model/, etc.)
- Root-level entry points (train.py, eval.py)
- PyTorch Lightning module structure

### **Step 2: Current Project Structure Assessment**
Analyze existing project organization:

```bash
# Examine current source structure
find ocr/ -type f -name "*.py" | head -20

# Review configuration organization
find configs/ -name "*.yaml" | sort

# Document current patterns
ls configs/train.yaml  # Check current callback definitions
```

### **Step 3: Comparative Analysis and Delta Identification**
Perform systematic comparison:

```bash
# Compare directory structures
diff -r docs/external/lightning-hydra-template/src/ ocr/

# Analyze configuration differences
# Template: configs/callbacks/default.yaml
# Current: configs/train.yaml (inline definitions)
```

Document key differences and improvement opportunities.

### **Step 4: Incremental Change Proposal and Implementation**
Formulate single, safe improvement:

**Example Hypothesis**: "Refactoring callback configurations to match lightning-hydra-template pattern will improve modularity and reusability."

**Implementation Plan**:
```yaml
# Create: configs/callbacks/default.yaml
defaults:
  - _self_
  - model: ???
  - datamodule: ???
  - callbacks: default

# Create: configs/callbacks/early_stopping.yaml
early_stopping:
  patience: 10
  monitor: val_loss

# Modify: configs/train.yaml
# Remove inline callback definitions
# Add to defaults list: - callbacks: early_stopping
```

## **API Reference**

### **Key Template Patterns**
- **src/ Layout**: Centralized Python package structure
- **Config Organization**: Modular configuration files by component type
- **Entry Points**: Root-level scripts for training/evaluation
- **LitModule Structure**: Standardized PyTorch Lightning module organization

### **Configuration Structure**
```
configs/
├── callbacks/          # Callback configurations
├── datamodule/         # Data module settings
├── model/             # Model architecture configs
├── trainer/           # Training configurations
└── train.yaml         # Main training config with defaults
```

### **Source Structure**
```
src/
├── models/            # Model implementations
├── datamodules/        # Data loading modules
├── callbacks/          # Training callbacks
├── utils/             # Utility functions
└── __init__.py
```

## **Configuration Structure**

```
docs/external/lightning-hydra-template/
├── README.md              # Template documentation
├── configs/               # Configuration examples
│   ├── callbacks/
│   ├── datamodule/
│   └── model/
└── src/                   # Code structure examples
    ├── models/
    ├── datamodules/
    └── utils/
```

## **Validation**

### **Pre-Adoption Validation**
- [ ] Template documentation is current and accessible
- [ ] Current project structure is well understood
- [ ] Modular Refactoring Protocol is available
- [ ] Backup/snapshot of current state exists

### **Post-Adoption Validation**
- [ ] Adopted pattern maintains existing functionality
- [ ] Configuration files load without errors
- [ ] Training/evaluation scripts still work
- [ ] No breaking changes to existing workflows

### **Incremental Safety Checks**
```bash
# Test configuration loading
python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/train.yaml')"

# Verify training still works
python runners/train.py --config-name=train ++dry_run=true

# Check for import errors
python -c "import ocr; print('Import successful')"
```

## **Troubleshooting**

### **Common Issues**

**Template Documentation Outdated**
- Update local copy from external repository
- Verify version compatibility
- Document version differences

**Structural Conflicts**
- Identify incompatible existing patterns
- Plan transitional approach
- Consider hybrid solutions

**Configuration Loading Failures**
- Check YAML syntax
- Verify path references
- Test individual config files

**Import Path Issues**
- Update Python path configurations
- Modify `__init__.py` files
- Check package structure

**Breaking Changes**
- Implement feature flags for gradual rollout
- Maintain backward compatibility
- Provide migration scripts

## **Related Documents**

- `12_streamlit_refactoring_protocol.md`: UI refactoring patterns
- `17_advanced_training_techniques.md`: Training workflow improvements
- `22_command_builder_hydra_configuration_fixes.md`: Configuration management
- `23_hydra_configuration_testing_implementation_plan.md`: Testing frameworks

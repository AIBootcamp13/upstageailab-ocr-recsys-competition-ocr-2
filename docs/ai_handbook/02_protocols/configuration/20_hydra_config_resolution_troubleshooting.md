# **filename: docs/ai_handbook/02_protocols/configuration/20_hydra_config_resolution_troubleshooting.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=hydra_config_troubleshooting,config_resolution_issues,runtime_interpolation_errors -->

# **Protocol: Hydra Configuration Resolution Troubleshooting**

## **Overview**
This protocol provides systematic troubleshooting for Hydra configuration resolution issues, particularly `InterpolationResolutionError` exceptions that occur when loading configs from different working directories or contexts. It covers safe initialization patterns, runtime interpolation handling, and best practices for robust config loading.

## **Prerequisites**
- Basic understanding of Hydra configuration system
- Access to `ocr.utils.path_utils.get_path_resolver()` utility
- Familiarity with OmegaConf interpolation syntax
- Understanding of project directory structure and config locations

## **Procedure**

### **Step 1: Identify the Resolution Error**
**Action:** Examine the specific error message and context:
```python
# Common error patterns:
# omegaconf.errors.InterpolationResolutionError: ValueError raised while resolving interpolation: HydraConfig was not set
# Key indicators: ${hydra:runtime.cwd}, missing Hydra context, working directory dependencies
```

**Expected Outcome:** Clear identification of whether the issue is runtime interpolation, missing Hydra context, or path resolution.

### **Step 2: Use Safe Initialization Pattern**
**Action:** Replace problematic initialization with robust approach:
```python
from hydra import initialize_config_dir
from ocr.utils.path_utils import get_path_resolver

# Get config directory using the standardized path resolver
config_dir = str(get_path_resolver().config.config_dir)

with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
    config = hydra.compose(config_name="predict")
```

**Expected Outcome:** Config loads successfully regardless of working directory.

### **Step 3: Handle Runtime Interpolations**
**Action:** Process configs with runtime-dependent values safely:
```python
from omegaconf import OmegaConf

# Convert config to container without resolving to avoid immediate errors
config_dict = OmegaConf.to_container(config, resolve=False)

# Get project root for manual interpolation replacement
project_root = str(get_path_resolver().config.project_root)

# Manually handle problematic interpolations
if 'dataset_base_path' in config_dict and "${hydra:runtime.cwd}" in str(config_dict['dataset_base_path']):
    config_dict['dataset_base_path'] = config_dict['dataset_base_path'].replace("${hydra:runtime.cwd}", project_root)

# Recreate config from processed dictionary
config = OmegaConf.create(config_dict)
```

**Expected Outcome:** Runtime interpolations resolved without Hydra context errors.

### **Step 4: Add Missing Configuration Values**
**Action:** Provide fallbacks for required config values using open_dict:
```python
from omegaconf import open_dict

with open_dict(config):
    config.dataset_path = getattr(config, 'dataset_path', 'ocr.datasets')
    config.model_path = getattr(config, 'model_path', 'ocr.models')
    # Add other required paths as needed
```

**Expected Outcome:** Config is complete and ready for use in downstream processes.

## **Configuration Structure**
```
config/
├── base.yaml              # Base configuration with common settings
├── predict.yaml           # Prediction-specific configuration
└── [other configs]        # Domain-specific configurations

Key interpolation patterns:
- ${hydra:runtime.cwd}     # Runtime working directory
- ${dataset_path}          # Dataset module path
- ${model_path}            # Model module path
- ${oc.env:VAR_NAME}       # Environment variable access
```

## **Validation**
Run these checks after implementing the troubleshooting steps:

```python
# Test 1: Verify config loads from different directories
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)
    # Attempt to load config - should work regardless of cwd
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        config = hydra.compose(config_name="predict")
        print("✓ Config loads successfully from arbitrary directory")

# Test 2: Verify interpolations resolve
try:
    OmegaConf.resolve(config)
    print("✓ All interpolations resolved successfully")
except Exception as e:
    print(f"✗ Resolution error: {e}")

# Test 3: Verify required paths exist
required_paths = ['dataset_path', 'model_path']
for path_key in required_paths:
    if hasattr(config, path_key):
        print(f"✓ {path_key} is configured: {getattr(config, path_key)}")
    else:
        print(f"✗ Missing required path: {path_key}")
```

## **Troubleshooting**

### **Issue: Still getting InterpolationResolutionError**
**Solution:** Check if config contains unhandled runtime interpolations:
```python
# Debug: Find all interpolations in config
def find_interpolations(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            find_interpolations(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_interpolations(item, f"{path}[{i}]")
    elif isinstance(obj, str) and "${" in obj:
        print(f"Found interpolation at {path}: {obj}")

config_dict = OmegaConf.to_container(config, resolve=False)
find_interpolations(config_dict)
```

### **Issue: Path resolver not available**
**Solution:** Fall back to manual path construction:
```python
import os
# Get project root (adjust dirname calls based on script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Adjust as needed
config_dir = os.path.join(project_root, "configs")
```

### **Issue: Config missing required values**
**Solution:** Implement comprehensive fallback strategy:
```python
def ensure_config_complete(config):
    defaults = {
        'dataset_path': 'ocr.datasets',
        'model_path': 'ocr.models',
        'trainer': {'max_epochs': 100},
        # Add other common defaults
    }

    with open_dict(config):
        for key, default_value in defaults.items():
            if not OmegaConf.select(config, key):
                OmegaConf.set(config, key, default_value)

    return config
```

## **Related Documents**
- `02_protocols/configuration/20_command_builder_testing_guide.md` - Testing strategies for config-related components
- `02_protocols/configuration/22_command_builder_hydra_configuration_fixes.md` - Specific Hydra configuration fixes
- `02_protocols/configuration/23_hydra_configuration_testing_implementation_plan.md` - Comprehensive testing for Hydra configs
- `03_references/architecture/02_hydra_and_registry.md` - Hydra architecture reference
- `01_coding_standards.md` - Configuration-related coding standards

---

*This document follows the configuration protocol template. Last updated: October 13, 2025*

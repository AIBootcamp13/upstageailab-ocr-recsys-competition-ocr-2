# **Protocol: Hydra Configuration Resolution Troubleshooting**

This protocol documents lessons learned about Hydra configuration resolution, particularly issues that arise when initializing Hydra from different working directories.

## **1. Problem: Hydra Interpolation Resolution Failure**

### **1.1 Symptom**
```
omegaconf.errors.InterpolationResolutionError: ValueError raised while resolving interpolation: HydraConfig was not set
```

### **1.2 Common Scenarios**
- When loading models from scripts outside the main training pipeline (e.g., visualization scripts)
- When initializing Hydra from subdirectories rather than the project root
- When accessing configs that contain `${hydra:runtime.cwd}` or other runtime-dependent interpolations

### **1.3 Root Cause**
Hydra configurations may contain interpolations that require a full Hydra context to resolve, such as:
- `${hydra:runtime.cwd}` - Requires Hydra runtime context
- `${dataset_path}`, `${model_path}` - May refer to global config paths
- Other runtime-specific values that depend on Hydra's internal state

## **2. Solution Approaches**

### **2.1 Use initialize_config_dir Instead of initialize**

**PREFERRED APPROACH:**
```python
from hydra import initialize_config_dir
from ocr.utils.path_utils import get_path_resolver

# Get config directory using the standardized path resolver
config_dir = str(get_path_resolver().config.config_dir)

with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
    config = hydra.compose(config_name="predict")
```

**LEGACY APPROACH (AVOID):**
```python
from hydra import initialize_config_dir
import os

# Get absolute path to configs directory from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = os.path.join(project_root, "configs")

with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
    config = hydra.compose(config_name="predict")
```

**AVOID:**
```python
# This can fail if working directory is not as expected
with hydra.initialize(config_path="configs", version_base="1.2"):
    config = hydra.compose(config_name="predict")
```

### **2.2 Handle Runtime Interpolations**

For configs containing `${hydra:runtime.cwd}` or similar runtime values:

```python
# First, get the project root directory using the path resolver
from ocr.utils.path_utils import get_path_resolver
project_root = str(get_path_resolver().config.project_root)

# Convert config to container without resolving to avoid immediate errors
config_dict = OmegaConf.to_container(config, resolve=False)

# Manually handle problematic interpolations
if 'dataset_base_path' in config_dict and "${hydra:runtime.cwd}" in str(config_dict['dataset_base_path']):
    config_dict['dataset_base_path'] = config_dict['dataset_base_path'].replace("${hydra:runtime.cwd}", project_root)

# Recreate config from processed dictionary
config = OmegaConf.create(config_dict)
```

**LEGACY APPROACH (AVOID):**

```python
# First, get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Convert config to container without resolving to avoid immediate errors
config_dict = OmegaConf.to_container(config, resolve=False)

# Manually handle problematic interpolations
if 'dataset_base_path' in config_dict and "${hydra:runtime.cwd}" in str(config_dict['dataset_base_path']):
    config_dict['dataset_base_path'] = config_dict['dataset_base_path'].replace("${hydra:runtime.cwd}", project_root)

# Recreate config from processed dictionary
config = OmegaConf.create(config_dict)
```

### **2.3 Use open_dict for Safe Configuration Updates**

```python
from omegaconf import open_dict

with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
    config = hydra.compose(config_name="predict")

    # Safely add missing keys without structured config restrictions
    with open_dict(config):
        if 'dataset_path' not in config:
            config.dataset_path = 'ocr.datasets'
        # Add other required paths...
```

## **3. Best Practices**

### **3.1 Always Use Absolute Paths**
When initializing Hydra from scripts that might be run from different working directories:

**PREFERRED APPROACH:**
```python
from ocr.utils.path_utils import get_path_resolver

# Get config directory using the standardized path resolver
config_dir = str(get_path_resolver().config.config_dir)
```

**LEGACY APPROACH (AVOID):**
```python
# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root (adjust number of dirname calls as needed)
project_root = os.path.dirname(script_dir)  # or os.path.dirname(os.path.dirname(script_dir)) for deeper nesting
config_dir = os.path.join(project_root, "configs")
```

### **3.2 Handle Missing Configuration Values**
Always provide fallbacks for required config values:

```python
with open_dict(config):
    config.dataset_path = getattr(config, 'dataset_path', 'ocr.datasets')
    config.model_path = getattr(config, 'model_path', 'ocr.models')
    # etc.
```

### **3.3 Test Configuration Resolution**
Before using a config in downstream processes, ensure it resolves properly:

```python
try:
    OmegaConf.resolve(config)
except Exception as e:
    print(f"Warning: Error resolving full config: {e}")
    # Handle the error appropriately
```

## **4. Maintenance Checklist for Hydra Config Scripts**

When creating or updating scripts that load Hydra configurations:

- [ ] Use `initialize_config_dir` with absolute paths
- [ ] Handle runtime-dependent interpolations before accessing config values
- [ ] Provide fallback values for common paths (dataset_path, model_path, etc.)
- [ ] Test the script from different working directories
- [ ] Verify the script works with both new and existing checkpoint formats

## **5. Anti-Patterns to Avoid**

- Using relative paths that depend on working directory
- Accessing config values containing interpolations without proper Hydra context
- Assuming default values exist in loaded configs
- Forgetting to handle `${hydra:runtime.cwd}` and similar runtime interpolations

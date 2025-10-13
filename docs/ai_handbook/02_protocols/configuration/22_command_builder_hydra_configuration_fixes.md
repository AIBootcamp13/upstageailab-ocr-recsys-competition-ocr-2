# **filename: docs/ai_handbook/02_protocols/configuration/22_command_builder_hydra_configuration_fixes.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=command_builder_fixes,hydra_config_fixes,preprocessing_profile_issues -->

# **Protocol: Command Builder Hydra Configuration Fixes**

## **Overview**
This protocol documents the systematic fixes applied to resolve Hydra configuration issues in the Streamlit UI Command Builder, specifically addressing invalid transform override paths and duplicate preprocessing profile handling that caused command generation failures.

## **Prerequisites**
- Access to Streamlit UI Command Builder
- Understanding of Hydra configuration structure and override syntax
- Familiarity with preprocessing profiles and transform configurations
- Access to `configs/ui_meta/preprocessing_profiles.yaml` and UI generator code

## **Procedure**

### **Step 1: Identify Configuration Path Issues**
**Action:** Analyze the error messages and configuration structure:
```bash
# Error pattern:
# Could not override 'data.transforms.train_transform'.
# Key 'transforms' is not in struct

# Check Hydra config structure
uv run python runners/train.py --cfg job | grep -A 10 "transforms:"
```

**Expected Outcome:** Clear identification that `data.transforms.*` paths don't exist in the Hydra config structure.

### **Step 2: Fix Transform Override Paths**
**Action:** Update preprocessing profile configurations to use correct paths:
```yaml
# In configs/ui_meta/preprocessing_profiles.yaml
overrides:
  - "+preset/datasets=preprocessing_docTR_demo"
  - "transforms.train_transform=${enhanced_transforms.train_transform}"  # ✅ Correct
  # Instead of: "data.transforms.train_transform=${enhanced_transforms.train_transform}"  # ❌ Wrong
```

**Expected Outcome:** Transform overrides target the correct `transforms.*` paths at the global level.

### **Step 3: Remove Duplicate Preprocessing Handling**
**Action:** Eliminate duplicate preprocessing profile logic in UI generator:
```python
# Remove from ui/utils/ui_generator.py:
# - Hardcoded __preprocessing_profile__ handling
# - _get_preprocessing_profile_overrides() function
# - preprocessing_overrides_applied variable

# Rely on build_additional_overrides() in overrides.py which uses ConfigParser
```

**Expected Outcome:** Single source of truth for preprocessing profile handling through ConfigParser.

### **Step 4: Validate Configuration Changes**
**Action:** Test the fixed command generation:
```bash
# Test configuration structure
uv run python runners/train.py --cfg job

# Test command generation for preprocessing profiles
python3 -c "
from ui.utils.config_parser import ConfigParser
cp = ConfigParser()
profiles = cp.get_preprocessing_profiles()
print('Available profiles:', list(profiles.keys()))
"
```

**Expected Outcome:** Commands generate successfully without path errors or duplicates.

## **Configuration Structure**
```
# Hydra Configuration Layout
transforms:                    # Global level transforms
  train_transform: ...
  val_transform: ...
  test_transform: ...
  predict_transform: ...

data:                         # Data section references transforms
  datasets:
    train_dataset:
      transform: \${transforms.train_transform}

# Preprocessing Profile Integration
enhanced_transforms:          # Defined by +preset/datasets=preprocessing_docTR_demo
  train_transform: ...        # Preprocessing-enabled transforms

# Profile overrides change references:
transforms.train_transform: \${enhanced_transforms.train_transform}
```

## **Validation**
Run these checks after implementing the fixes:

```python
# Test 1: Verify config structure
import subprocess
result = subprocess.run([
    "uv", "run", "python", "runners/train.py", "--cfg", "job"
], capture_output=True, text=True)
assert "transforms:" in result.stdout, "transforms section missing"

# Test 2: Validate preprocessing profiles
from ui.utils.config_parser import ConfigParser
cp = ConfigParser()
profiles = cp.get_preprocessing_profiles()
assert 'doctr_demo' in profiles, "doctr_demo profile missing"

# Test 3: Test command generation
from ui.apps.command_builder.services.overrides import build_additional_overrides
overrides = build_additional_overrides({"preprocessing_profile": "doctr_demo"})
# Should contain single +preset/datasets=preprocessing_docTR_demo
preset_count = sum(1 for o in overrides if "preset/datasets=preprocessing_docTR_demo" in o)
assert preset_count == 1, f"Expected 1 preset override, got {preset_count}"

print("✅ All validation checks passed")
```

## **Troubleshooting**

### **Issue: Still getting 'Key transforms is not in struct'**
**Solution:** Verify the preprocessing profile YAML uses correct paths:
```yaml
# Check configs/ui_meta/preprocessing_profiles.yaml
overrides:
  - "transforms.train_transform=..."  # ✅ Correct
  # NOT: "data.transforms.train_transform=..."  # ❌ Wrong
```

### **Issue: Duplicate preset overrides in generated commands**
**Solution:** Ensure UI generator doesn't have hardcoded preprocessing logic:
```python
# Search for hardcoded preprocessing in ui/utils/ui_generator.py
grep -n "__preprocessing_profile__" ui/utils/ui_generator.py
# Should return no results after cleanup
```

### **Issue: Preprocessing profiles not loading**
**Solution:** Verify ConfigParser integration:
```python
from ui.utils.config_parser import ConfigParser
cp = ConfigParser()
try:
    profiles = cp.get_preprocessing_profiles()
    print(f"Loaded {len(profiles)} profiles")
except Exception as e:
    print(f"ConfigParser error: {e}")
```

## **Related Documents**
- `02_protocols/configuration/20_command_builder_testing_guide.md` - Testing strategies for command builder
- `02_protocols/configuration/20_hydra_config_resolution_troubleshooting.md` - General Hydra config troubleshooting
- `02_protocols/configuration/23_hydra_configuration_testing_implementation_plan.md` - Comprehensive testing for Hydra configs
- `03_references/architecture/02_hydra_and_registry.md` - Hydra architecture reference
- `configs/ui_meta/preprocessing_profiles.yaml` - Preprocessing profile definitions

---

*This document follows the configuration protocol template. Last updated: October 13, 2025*

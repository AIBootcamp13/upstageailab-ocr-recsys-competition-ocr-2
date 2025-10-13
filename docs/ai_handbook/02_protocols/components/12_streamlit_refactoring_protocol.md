# **filename: docs/ai_handbook/02_protocols/components/12_streamlit_refactoring_protocol.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=streamlit,refactor,ui-development -->

# **Protocol: Streamlit Refactoring**

## **Overview**

This protocol provides systematic guidance for executing structural changes to Streamlit UI modules, including new features, component rewrites, and state management overhauls. It ensures disciplined, reproducible refactoring while maintaining compatibility with existing configurations and schemas.

## **Prerequisites**

- Baseline behavior captured (screenshots, logs, or video documentation)
- Streamlit app confirmed working: `uv run streamlit run ui/<app_entry>.py --server.port=8504`
- Compatibility schemas in `configs/schemas/` are clean and validated
- Tests and linting pass: `uv run ruff check ui/apps/<app_name> ui/utils`
- Pydantic models in `ui/apps/<app_name>/models/` updated for any schema changes

## **Component Architecture**

### **Core Components**
- **App Entry Point**: Thin entrypoint in `ui/<app_entry>.py`
- **App Module**: Page lifecycle orchestrator in `ui/apps/<app_name>/app.py`
- **Components**: UI fragments in `ui/apps/<app_name>/components/`
- **Models**: Pydantic/dataclass state validation in `ui/apps/<app_name>/models/`
- **Services**: Pure-Python business logic in `ui/apps/<app_name>/services/`
- **State Management**: Session helpers in `ui/apps/<app_name>/state.py`

### **Integration Points**
- `configs/ui/<app_name>.yaml`: UI configuration
- `configs/schemas/`: Domain schema definitions
- `ui_meta/<app_name>/`: UI copy and demo payloads
- `ui/inference_ui.py`: Legacy entrypoint (maintained for compatibility)

## **Procedure**

### **Step 1: Planning and Analysis**
Define the refactoring scope and document current architecture:

```bash
# Verify current state
uv run streamlit run ui/<app_entry>.py --server.port=8504

# Check existing structure
find ui/apps/<app_name> -type f -name "*.py" | head -20
```

Document contracts and external touchpoints:
- Service method signatures and return types
- Streamlit widget keys for session state
- Configuration keys in `configs/ui/<app_name>.yaml`
- Schema dependencies in `configs/schemas/`

### **Step 2: Isolate Changes and Create New Modules**
Fork new modules within appropriate packages:

```python
# Create new component
# ui/apps/<app_name>/components/new_feature.py
import streamlit as st

def render_new_feature():
    """New UI component with proper separation of concerns"""
    # Pure UI logic here
    pass
```

```python
# Create new service
# ui/apps/<app_name>/services/new_service.py

class NewService:
    """Pure business logic, no Streamlit imports"""

    def process_data(self, data):
        # Processing logic here
        return processed_data
```

### **Step 3: Update Configuration and Schemas**
Surface new toggles in configuration:

```yaml
# configs/ui/<app_name>.yaml
new_feature:
  enabled: true
  option1: "default_value"
  option2: 42
```

Update schema families for new capabilities:

```yaml
# configs/schemas/ui_inference_compat.yaml
new_model_family:
  encoder: "resnet50"
  head: "new_head_type"
  compatible: true
```

### **Step 4: Update Entry Points and Integration**
Modify app orchestration while preserving caching:

```python
# ui/apps/<app_name>/app.py
import streamlit as st
from components.new_feature import render_new_feature
from services.new_service import NewService

@st.cache_resource
def get_new_service():
    return NewService()

def main():
    service = get_new_service()

    # New feature integration
    if st.session_state.get('new_feature_enabled', False):
        render_new_feature()
```

### **Step 5: Testing and Validation**
Run comprehensive testing:

```bash
# Unit testing
uv run python -c "
from ui.apps.<app_name>.services.new_service import NewService
service = NewService()
result = service.process_data(test_data)
print('Service test passed')
"

# Streamlit smoke test
uv run streamlit run ui/<app_entry>.py --server.port=8504
```

## **API Reference**

### **Key Classes and Methods**
- `InferenceService.run()`: Main inference execution
- `InferenceState`: Session state management
- `build_catalog()`: Model checkpoint discovery
- `InferenceEngine.load_model()`: Model loading and initialization

### **Configuration Parameters**
- `enabled`: Feature toggle flags
- `ttl`: Cache time-to-live settings
- `batch_size`: Processing batch sizes
- `model_families`: Compatible model configurations

### **Session State Keys**
- `selected_checkpoint`: Current model selection
- `inference_results`: Cached results
- `ui_config`: UI-specific settings

## **Configuration Structure**

```
ui/apps/<app_name>/
├── app.py                 # Main orchestrator
├── components/            # UI fragments
│   ├── sidebar.py
│   ├── results.py
│   └── ...
├── models/                # State validation
│   ├── inference.py
│   └── ui_state.py
├── services/              # Business logic
│   ├── inference.py
│   ├── catalog.py
│   └── ...
└── state.py               # Session helpers
```

## **Validation**

### **Pre-Refactor Validation**
- [ ] UI renders without warnings
- [ ] All existing features work
- [ ] Session state persists correctly
- [ ] Caching boundaries respected

### **Post-Refactor Validation**
- [ ] New features render correctly
- [ ] Configuration toggles work
- [ ] Schema validation passes
- [ ] Performance metrics unchanged
- [ ] Fallback paths functional

### **Integration Testing**
```bash
# Test multiple checkpoints
# Test batch inference
# Test configuration overrides
# Test error handling
```

## **Troubleshooting**

### **Common Issues**

**Monolithic Logic Regression**
- Keep `ui/<app_entry>.py` as thin proxy
- Move all logic to `ui/apps/<app_name>/` modules

**Broken Cache Keys**
- Include config versions in cache key inputs
- Set explicit `ttl` values for time-sensitive caches

**Schema Compatibility Issues**
- Update `configs/schemas/*.yaml` for new model families
- Run catalogue validation after schema changes

**Hard-coded UI Strings**
- Use configuration values from `configs/ui/<app_name>.yaml`
- Keep copy in `ui_meta/<app_name>/`

**Pydantic vs Hydra Confusion**
- Use Pydantic only for UI/session payload validation
- Keep Hydra/OmegaConf for runtime configuration

**Mock Fallbacks Hiding Issues**
- Check logs for actual error conditions
- Ensure fallbacks only trigger on explicit exceptions

## **Related Documents**

- `16_template_adoption_protocol.md`: Template adoption workflows
- `11_docTR_preprocessing_workflow.md`: UI preprocessing integration
- `17_advanced_training_techniques.md`: Training workflow integration
- `22_command_builder_hydra_configuration_fixes.md`: Configuration management

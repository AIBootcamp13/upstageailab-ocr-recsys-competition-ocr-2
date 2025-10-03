# **Actionable Refactor Plan: Standardize Path Management**

**Objective:** Eliminate all hardcoded and inconsistent path resolution logic, and establish the OCRPathResolver from ocr/utils/path_utils.py as the single source of truth for all file paths.

### **Phase 1: Audit and Identify Problem Areas**

The first step is to locate all instances of inconsistent path management.

Action 1: Perform a codebase search.
Search the entire project for the following patterns:

* ../configs
* Path(__file__).parent
* os.environ.get("OP_CONFIG_DIR")
* Any direct usage of the legacy PathUtils class.

Action 2: Review documentation.
Check project-overview.md and other documentation for any instructions asking the user to manually edit paths in configuration files.

### **Phase 2: Deprecate Legacy PathUtils**

Clearly mark the old utility as deprecated and guide developers to the new one.

**Action 1: Modify ocr/utils/path_utils.py.**

* Add a prominent DeprecationWarning to the PathUtils class docstring and its methods.
  ``` class PathUtils:
      """
      DEPRECATED: This class is for backward compatibility only.
      Please use the `get_path_resolver()` instance for all new code.
      """
      # ... add warnings to methods as well
  ```

* Perform a search-and-replace across the project to replace any calls to PathUtils.get_...() with the modern equivalent, get_path_resolver().config....

### **Phase 3: Refactor Scripts and Runners**

Update all scripts to use the centralized resolver.

**Action 1: Modify runners/train.py, runners/test.py, runners/predict.py.**

* Remove the hardcoded CONFIG_DIR logic:
  ``` # REMOVE THIS LINE
  CONFIG_DIR = os.environ.get("OP_CONFIG_DIR") or "../configs"
  ```

* Instead of hardcoding, get the path from the resolver. The setup_paths() call already ensures the project root is set up correctly.
  ``` # In runners/train.py
  from ocr.utils.path_utils import get_path_resolver

  # ... inside the train function, or globally
  config_dir = get_path_resolver().config.config_dir

  @hydra.main(config_path=str(config_dir), config_name="train", version_base="1.2")
  def train(config: DictConfig):
      # ...
  ```

  *Note: The @hydra.main decorator requires a relative path from the script's location or an absolute path. Ensure the path passed is compatible.* A better approach might be to let Hydra discover the default configs directory by running the script from the project root.

Action 2: Update run_ui.py.
This script uses Path(__file__).parent. Refactor it to use the path resolver for consistency.
``` # In run_ui.py
from ocr.utils.path_utils import get_path_resolver

ui_path = get_path_resolver().config.project_root / "ui" / "command_builder.py"
```

### **Phase 4: Update Documentation**

Ensure user-facing documentation no longer contains incorrect instructions.

Action 1: Modify project-overview.md.
Remove any sentences that instruct the user to manually edit .yaml files to update paths. The new system should handle this automatically.

### **Prompt for Agentic AI (Next Session)**

Objective: Execute the path management standardization plan. Follow the four phases in `refactor_plan_standardize_paths.md`. Audit the codebase for incorrect path logic, deprecate the legacy `PathUtils` class, refactor the runner scripts to remove hardcoded paths, and update the project documentation.

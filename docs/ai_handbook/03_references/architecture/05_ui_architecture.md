# **Reference: UI Architecture**

**Objective:** This document provides the architectural overview of the Streamlit-based UI applications located in the ui/ directory.

### **1. Core Principles**

* **Modularity:** Each major UI function (e.g., Command Builder, Inference) is a self-contained application within the ui/apps/ directory.
* **Thin Entry Points:** Top-level files like ui/command_builder.py are minimal wrappers that delegate immediately to their respective application modules (e.g., ui/apps/command_builder/app.py). This prevents logic from spilling into the root ui/ directory.
* **Configuration-Driven:** The UI's appearance, options, and behavior are controlled by YAML files in configs/ui/ and configs/schemas/, not hardcoded in Python.
* **Stateful Services:** Pure Python logic (e.g., parsing configs, running inference) is encapsulated in services that are separate from the Streamlit rendering code.

### **2. Directory Structure**

The ui/ directory is organized as follows:

``` ui/
├── apps/                     # Contains individual Streamlit applications
│   ├── command_builder/      # The Command Builder application
│   │   ├── components/       # Reusable UI parts (e.g., sidebar, command preview)
│   │   ├── services/         # Business logic (e.g., formatting, overrides)
│   │   ├── models/           # Pydantic/dataclass models for UI state
│   │   ├── schemas/          # YAML schemas that define the UI layout and widgets
│   │   └── app.py            # Main application logic for the Command Builder
│   └── inference/            # The Real-time Inference application
│       └── ...               # (Similar structure to command_builder)
├── utils/                    # Shared utilities used by multiple UI apps
│   ├── command_builder.py    # Service for building and running CLI commands
│   ├── config_parser.py      # Service for parsing model and UI configurations
│   ├── ui_generator.py       # Dynamically generates Streamlit widgets from a schema
│   └── ...
└── command_builder.py        # Thin entry point for the Command Builder
└── inference_ui.py           # Thin entry point for the Inference UI
```

### **3. State Management**

* **Mechanism:** We use Streamlit's st.session_state.
* **Best Practice:** To avoid polluting the global session state, each application encapsulates its state within a single dataclass (e.g., ui/apps/command_builder/state.py).
* **Lifecycle:**
  1. At the start of the app run, a state object is loaded from st.session_state. If it doesn't exist, a new default one is created.
  2. Widgets and services read from and write to this state object.
  3. At the end of the app run, the state object is persisted back into st.session_state.

### **4. Configuration and Data Flow**

The UI is dynamically generated from configuration to keep it decoupled from the core application logic.

1. **Schema (ui/apps/.../schemas/*.yaml):** A YAML file defines every widget to be displayed: its type (selectbox, slider), label, default value, and the corresponding Hydra override key.
2. **UI Generator (ui/utils/ui_generator.py):** This service reads the schema and programmatically calls the appropriate Streamlit functions (st.selectbox, st.slider) to render the UI. It collects the user's input.
3. **Config Parser (ui/utils/config_parser.py):** This service reads the main project configurations (configs/model/, configs/ui_meta/) to populate dynamic options, like the list of available encoders or pre-defined use cases.
4. **Command Builder (ui/utils/command_builder.py):** The collected user inputs are converted into a list of Hydra overrides (e.g., model.optimizer.lr=0.005). This service then assembles the final uv run python ... command string.
5. **Execution:** The final command is executed in a subprocess, with its output streamed back to the UI.

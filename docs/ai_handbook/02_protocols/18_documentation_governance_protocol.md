# **filename: docs/ai_handbook/02_protocols/18_documentation_governance_protocol.md**

# **Protocol: Documentation Governance**

**Objective:** This protocol defines the rules for managing the docs/ai_handbook/ directory. You MUST adhere to this protocol for all file creation, modification, and deletion tasks. The master schema is defined in docs/ai_handbook/index.json.

### **Rule 1: Adhere to the Central Schema**

Before adding or moving a file, you MUST consult the "schema" section within docs/ai_handbook/index.json. This schema defines the required naming convention, purpose, and allowed content for each top-level directory.

### **Rule 2: Do Not Mix Artifact Types**

Documentation directories (01_onboarding, 02_protocols, etc.) MUST ONLY contain Markdown (.md) documents.

* **PROHIBITED:** You MUST NOT save generated outputs (images, logs, checkpoints, JSON files) inside the docs/ai_handbook/ directories. All generated outputs belong in the project's top-level outputs/ directory.
* **PROHIBITED:** You MUST NOT store planning documents, work-in-progress notes, or temporary files within the handbook. These belong in a separate project management directory.

### **Rule 3: Follow Naming and Numbering Conventions**

All new documents within a numbered directory (e.g., 01_onboarding, 02_protocols) MUST follow the NN_descriptive_name.md format, where NN is the next sequential number in that directory.

### **Rule 4: Update the Manifest**

After creating a new document, you MUST add a corresponding entry to docs/ai_handbook/index.json. Ensure you provide a unique id, title, path, and other relevant metadata.

### **Rule 5: Validate Your Changes**

After modifying the handbook, run the validation script to ensure integrity.

```bash
uv run python scripts/agent_tools/validate_manifest.py
```

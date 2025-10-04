# **filename: refactor_plan_cleanup_handbook.md**

# **Actionable Refactor Plan: Clean Up AI Handbook**

**Objective:** Reorganize the docs/ai_handbook/ directory to conform to the new schema defined in index.json and the Documentation Governance Protocol.

### **Phase 1: Relocate Mis-categorized Artifacts**

Move non-documentation files out of the handbook directories.

Action 1: Relocate protocol outputs.
The outputs directory inside 02_protocols contains generated data. Move its contents to the appropriate top-level project directory.
# Assuming execution from project root
```bash
mkdir -p outputs/protocols/
mv docs/ai_handbook/02_protocols/outputs/* outputs/protocols/
rm -rf docs/ai_handbook/02_protocols/outputs/
```

Action 2: Relocate changelog planning documents.
The 05_changelog/01_planning directory contains refactor plans, which are project management artifacts, not changelogs. Move them to a more appropriate location. For now, we will place them in a new top-level directory for clarity.
# Assuming execution from project root
```bash
mkdir -p docs/ai_handbook/07_project_management/
mv docs/ai_handbook/05_changelog/01_planning/* docs/ai_handbook/07_project_management/
rm -rf docs/ai_handbook/05_changelog/01_planning/
```

Action 3: Relocate changelog work-in-progress notes.
Similarly, move the 02_work_in_progress directory.
# Assuming execution from project root
```bash
mv docs/ai_handbook/05_changelog/02_work_in_progress/* docs/ai_handbook/07_project_management/
rm -rf docs/ai_handbook/05_changelog/02_work_in_progress/
```

Action 4: Relocate nested protocol.
The 04_experiments directory contains a misplaced protocol.
```bash
mv docs/ai_handbook/04_experiments/protocols/experiment_analysis_framework_handbook.md docs/ai_handbook/02_protocols/
rm -rf docs/ai_handbook/04_experiments/protocols/
```

### **Phase 2: Fix Naming and Numbering**

Correct the filenames in 02_protocols to be consistent with the schema.

Action 1: Re-number duplicated file.
There are two files named 11_.... Let's re-number them sequentially. First, find the highest number currently in use. Let's assume it's 17.
# Rename the second "11_..." file
```bash
mv docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md docs/ai_handbook/02_protocols/19_streamlit_maintenance_protocol.md
```

**Action 2: Number the un-numbered file.**

```bash
mv docs/ai_handbook/02_protocols/command_builder_testing_guide.md docs/ai_handbook/02_protocols/20_command_builder_testing_guide.md
```

### **Phase 3: Update Manifest**

Update index.json to reflect all the file moves and renames.

**Action 1: Manually edit docs/ai_handbook/index.json.**

* Update the path for all moved files.
* Add entries for the newly numbered protocols.
* Add a schema entry for the new 07_project_management directory if desired.

Action 2: Validate the manifest.
Run the validation script to ensure all paths are correct and the structure is sound.
```bash
uv run python scripts/agent_tools/validate_manifest.py
```

### **Prompt for Agentic AI (Next Session)**

```markdown
Objective: Execute the documentation cleanup plan. Follow the three phases outlined in `refactor_plan_cleanup_handbook.md`. Relocate all mis-categorized artifacts, correct the file naming and numbering in the protocols directory, and update the `index.json` manifest to reflect all changes. Finally, run the validation script to confirm the handbook's integrity.
```

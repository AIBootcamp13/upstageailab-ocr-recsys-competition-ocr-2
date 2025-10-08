# 04_experiments/ Directory

This directory contains documentation for experiments, debugging sessions, and related operational artifacts in the OCR project.

## Directory Structure

```
04_experiments/
├── experiment_logs/           # Core experiment documentation and results
│   ├── [YYYY-MM-DD]_experiment_name.md
│   └── templates/
│       └── experiment_log_template.md
├── debugging/                 # Debug sessions and troubleshooting documentation
│   ├── [YYYY-MM-DD]_debug_topic.md
│   └── summaries/
│       └── [YYYY-MM-DD]_debug_summary.md
├── sessions/                  # Session handovers and knowledge transfer
│   └── [YYYY-MM-DD]_session_topic_handover.md
└── agent_runs/                # Automated agent activity logs
    └── [YYYY-MM-DD]_agent_activity_summary.md
```

## Subdirectory Descriptions

### experiment_logs/
Contains formal experiment documentation including:
- Hypothesis testing results
- Configuration comparisons
- Performance analyses
- Model architecture evaluations

### debugging/
Technical troubleshooting documentation:
- Root cause analyses
- Debug session logs
- Issue resolution summaries
- Performance regression investigations

### sessions/
Knowledge transfer documents:
- Session handovers between developers/agents
- Work continuation instructions
- Context summaries for ongoing tasks

### agent_runs/
Operational logs from automated agents:
- Task completion summaries
- Code modification logs
- Testing and validation results

## Filenaming Convention

All files must follow the format: `YYYY-MM-DD_descriptive_name.md`

- **YYYY-MM-DD**: Date in ISO 8601 format (e.g., 2025-10-08)
- **descriptive_name**: Brief, descriptive name using underscores instead of spaces
- **.md**: Markdown file extension

### Examples
- `2025-10-08_performance_regression_debug.md`
- `2025-10-04_dbnetpp_vs_craft_analysis.md`
- `2025-09-30_detection_pipeline_enhancements.md`

## Guidelines

1. **Placement**: Place files in the most appropriate subdirectory based on content type.
2. **Naming**: Use descriptive names that clearly indicate the file's purpose.
3. **Consistency**: Follow the naming convention strictly to maintain organization.
4. **Updates**: When updating existing files, preserve the original date unless creating a new version.
5. **References**: Use relative paths when linking between files in this directory.

## Maintenance

- Regularly review and move misplaced files to appropriate subdirectories.
- Archive outdated files to prevent clutter.
- Update this README if the structure evolves.</content>
<filePath>/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/04_experiments/README.md

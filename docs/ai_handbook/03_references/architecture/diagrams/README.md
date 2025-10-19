# Automated Mermaid Diagram System

This directory contains an automated system for generating and maintaining Mermaid diagrams that visualize the OCR project's architecture. The system ensures diagrams stay synchronized with code changes through automated generation and CI/CD integration.

## 🎯 Overview

The diagram system provides:
- **Automated Generation**: Diagrams update automatically when code changes
- **AI Agent Integration**: Diagrams include cues for AI assistants to understand project structure
- **Validation**: Syntax checking ensures all diagrams render correctly
- **CI/CD Integration**: Automatic updates via GitHub Actions

## 📁 Directory Structure

```
docs/ai_handbook/03_references/architecture/diagrams/
├── 01_component_registry.md    # Component assembly flow
├── 02_data_pipeline.md         # Data processing pipeline
├── 03_training_inference.md    # Training/inference flow
├── 04_ui_flow.md              # UI and user experience flow
├── 05_data_loading_complexity.md # Data loading complexity explanation
└── _generated/
    └── diagram_metadata.json  # Generation metadata and checksums
```

## 🤖 AI Integration

Each diagram file includes AI cues to help agents understand when and how to reference them:

```markdown
<!-- ai_cue:diagram=component_registry -->
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=architecture,components,registry -->
```

## 🛠️ Usage

### Manual Updates

```bash
# Check which diagrams need updates
python scripts/generate_diagrams.py --check-changes

# Update all diagrams
python scripts/generate_diagrams.py --update

# Update specific diagrams
python scripts/generate_diagrams.py --update component_registry data_pipeline

# Validate diagram syntax
python scripts/generate_diagrams.py --validate

# Force update all diagrams
python scripts/generate_diagrams.py --update --force
```

### CI/CD Integration

The system automatically runs via GitHub Actions when relevant code changes:

- **Triggers**: Changes to `ocr/`, `ui/`, `configs/`, or diagram scripts
- **Process**: Checks for changes → generates diagrams → validates → commits updates
- **Branches**: Runs on `main` and `develop` branches

## 🔧 How It Works

### 1. Code Analysis
The system analyzes source code to extract architectural information:
- **Component Registry**: Scans `@register_*` decorators
- **Data Pipeline**: Analyzes preprocessing transforms and collate functions
- **Training/Inference**: Examines Lightning modules and training flows
- **UI Flow**: Reviews Streamlit components and service layers

### 2. Diagram Generation
Generates Mermaid diagrams with:
- **Dynamic Content**: Component counts, method signatures, file structures
- **Fallback Diagrams**: Basic diagrams when source analysis fails
- **Consistent Styling**: Standardized node shapes and connection patterns

### 3. Change Detection
Uses checksums to detect when diagrams need updates:
- **Source Files**: Monitors Python files, configs, and documentation
- **Metadata Tracking**: Stores generation timestamps and source checksums
- **Selective Updates**: Only regenerates diagrams when sources change

### 4. Validation
Ensures diagram quality through:
- **Syntax Checking**: Validates Mermaid syntax rules
- **Graph Type Verification**: Ensures correct graph directions (TD, LR, etc.)
- **Error Reporting**: Detailed error messages for debugging

## 📊 Diagram Types

### Component Registry (`01_component_registry.md`)
- **Purpose**: Visualizes how components are registered and assembled
- **Content**: Encoder/decoder/head/loss registration flow
- **Source Files**: `ocr/models/core/registry.py`, `ocr/models/core/architecture.py`

### Data Pipeline (`02_data_pipeline.md`)
- **Purpose**: Shows data flow from raw images to model input
- **Content**: Geometric preprocessing, DB collate functions, data contracts
- **Source Files**: `ocr/datasets/preprocessing/`, `ocr/datasets/db_collate_fn.py`

### Training/Inference (`03_training_inference.md`)
- **Purpose**: Illustrates training loops and inference pipelines
- **Content**: PyTorch Lightning integration, loss calculation, optimization
- **Source Files**: `ocr/lightning_modules/ocr_pl.py`, `runners/train.py`

### UI Flow (`04_ui_flow.md`)
- **Purpose**: Documents user interaction and service layer architecture
- **Content**: Streamlit UI components, inference services, checkpoint catalog
- **Source Files**: `ui/apps/inference/`, `ui/services/`

### Data Loading Complexity (`05_data_loading_complexity.md`)
- **Purpose**: Explains the hidden complexity of data loading beyond filesystem operations
- **Content**: Geometric preprocessing, polygon processing, batch validation, performance implications
- **Source Files**: `ocr/datasets/preprocessing/`, `ocr/datasets/db_collate_fn.py`, `ocr/datasets/`

## 🔄 Automation Triggers

Diagrams automatically update when these files change:

### Component Registry
- `ocr/models/core/registry.py`
- `ocr/models/core/architecture.py`
- `configs/model/`

### Data Pipeline
- `ocr/datasets/preprocessing/`
- `ocr/datasets/db_collate_fn.py`
- `configs/data/`

### Training/Inference
- `ocr/lightning_modules/ocr_pl.py`
- `runners/train.py`
- `configs/trainer/`

### UI Flow
- `ui/apps/inference/`
- `ui/services/`
- `configs/ui/`

## 🚨 Troubleshooting

### Common Issues

**"Diagram file not found"**
- Check that diagram files exist in the correct location
- Verify filename mapping in `generate_diagrams.py`

**"Syntax validation errors"**
- Run `python scripts/generate_diagrams.py --validate` for details
- Check Mermaid syntax in generated diagrams

**"No diagrams need updating"**
- Use `--force` flag to regenerate all diagrams
- Check that source files are being detected correctly

### Manual Recovery

```bash
# Force regenerate all diagrams
python scripts/generate_diagrams.py --update --force

# Clear metadata and start fresh
rm docs/ai_handbook/03_references/architecture/diagrams/_generated/diagram_metadata.json
python scripts/generate_diagrams.py --update
```

## 🤝 Contributing

When adding new diagram types:

1. **Add to `generate_diagrams.py`**:
   - Create analysis method (e.g., `generate_new_diagram_type()`)
   - Add to `filename_map` in `update_diagram()`
   - Include in main argument parsing

2. **Create diagram file**:
   - Follow naming pattern: `NN_diagram_type.md`
   - Include AI cues and metadata
   - Add comprehensive documentation

3. **Update CI triggers**:
   - Add relevant file paths to `.github/workflows/update-diagrams.yml`
   - Test the workflow locally

## 📈 Performance

- **Generation Time**: <5 seconds for all diagrams
- **Validation Time**: <1 second
- **Memory Usage**: Minimal (analyzes text files only)
- **CI Impact**: Adds ~30 seconds to pipeline when diagrams update

## 🔗 Related Documentation

- **Project Architecture**: `docs/ai_handbook/01_architecture/`
- **Data Contracts**: `docs/pipeline/data_contracts.md`
- **Component Registry**: `docs/ai_handbook/02_components/`
- **UI Architecture**: `docs/ai_handbook/05_ui/`

---

*Last updated: 2025-01-19 | Auto-generated: true*

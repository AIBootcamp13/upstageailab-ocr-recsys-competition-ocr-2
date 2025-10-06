# Generating Submission Files (submission.csv)

## Overview

The OCR project generates predictions in **JSON format** first, which then needs to be converted to **CSV format** for competition submission. This is a two-step process.

## Current Workflow

### Step 1: Generate Predictions (JSON)

Use the **Predict** page in Command Builder or run the predict script:

#### Option A: Using Command Builder UI
1. Open Streamlit: `streamlit run ui/command_builder.py`
2. Navigate to **Predict** page
3. Configure:
   - Select checkpoint
   - Match architecture/encoder/decoder/head/loss to checkpoint
   - Set experiment name (e.g., `final_submission`)
4. Click "Generate command" and "Run command"

#### Option B: Using Terminal
```bash
uv run python runners/predict.py \
  exp_name=final_submission \
  'checkpoint_path="outputs/your-model/checkpoints/best.ckpt"' \
  model.architecture_name=dbnet \
  model/architectures=dbnet \
  model.encoder.model_name=resnet34 \
  model.component_overrides.decoder.name=pan_decoder \
  model.component_overrides.head.name=db_head \
  model.component_overrides.loss.name=db_loss \
  minified_json=false
```

**Output Location:**
```
outputs/{exp_name}/submissions/{timestamp}.json
```

Example:
```
outputs/final_submission/submissions/20251003_143022.json
```

### Step 2: Convert JSON to CSV

Use the conversion utility:

```bash
uv run python ocr/utils/convert_submission.py \
  --json_path outputs/final_submission/submissions/20251003_143022.json \
  --output_path submission.csv
```

**Arguments:**
- `-J` or `--json_path`: Path to the JSON prediction file
- `-O` or `--output_path`: Desired path for the output CSV file

**Output Format (submission.csv):**
```csv
filename,polygons
image001.jpg,123 456 789 012|234 567 890 123
image002.jpg,111 222 333 444|555 666 777 888
```

Each row contains:
- `filename`: Image filename
- `polygons`: Pipe-separated (`|`) list of polygons, where each polygon is space-separated coordinates

## File Locations

### Generated Files
```
outputs/
└── {exp_name}/
    ├── submissions/
    │   └── {timestamp}.json    # Predictions in JSON format
    └── checkpoints/
        └── *.ckpt
```

### Submission File
```
project_root/
└── submission.csv              # Final submission file for competition
```

## Inference UI vs Command Builder

### Inference UI (`ui/inference_ui.py`)
- **Purpose**: Interactive single-image testing and visualization
- **Output**: Visual results, bounding boxes overlaid on images
- **Use case**: Debugging, model inspection, quick tests
- **Does NOT generate**: Batch predictions or submission files

### Command Builder Predict Page
- **Purpose**: Batch prediction on test dataset
- **Output**: JSON prediction file → converted to CSV
- **Use case**: Competition submissions, full test set evaluation
- **Generates**: submission.json → submission.csv

## Complete Example Workflow

### 1. Train Your Model
```bash
# Using Command Builder Train page or terminal
uv run python runners/train.py \
  exp_name=dbnet_pan_resnet34 \
  model.architecture_name=dbnet \
  model.encoder.model_name=resnet34 \
  model.component_overrides.decoder.name=pan_decoder \
  trainer.max_epochs=30
```

### 2. Find Best Checkpoint
```bash
# List checkpoints
ls outputs/dbnet_pan_resnet34/checkpoints/

# Example output:
# epoch=28_step=23722.ckpt  <- Use this
# last.ckpt
```

### 3. Generate Predictions
```bash
uv run python runners/predict.py \
  exp_name=final_submission \
  'checkpoint_path="outputs/dbnet_pan_resnet34/checkpoints/epoch=28_step=23722.ckpt"' \
  model.architecture_name=dbnet \
  model/architectures=dbnet \
  model.encoder.model_name=resnet34 \
  model.component_overrides.decoder.name=pan_decoder \
  model.component_overrides.head.name=db_head \
  model.component_overrides.loss.name=db_loss \
  minified_json=true
```

**Output:**
```
outputs/final_submission/submissions/20251003_143022.json
```

### 4. Convert to CSV
```bash
uv run python ocr/utils/convert_submission.py \
  --json_path outputs/final_submission/submissions/20251003_143022.json \
  --output_path submission.csv
```

**Output:**
```
Successfully converted 1234 rows to 'submission.csv'
```

### 5. Verify and Submit
```bash
# Check the file
head -n 5 submission.csv

# Upload to competition platform
# (Kaggle, AIStages, etc.)
```

## Tips & Best Practices

### Organizing Submissions
```bash
# Create a submissions directory
mkdir -p competition_submissions

# Name files descriptively
uv run python ocr/utils/convert_submission.py \
  --json_path outputs/final_submission/submissions/20251003_143022.json \
  --output_path competition_submissions/dbnet_pan_resnet34_epoch28.csv
```

### Finding Latest JSON
```bash
# Find most recent submission JSON
ls -lt outputs/final_submission/submissions/ | head -n 2

# Or use wildcard
JSON_FILE=$(ls -t outputs/final_submission/submissions/*.json | head -n 1)
echo "Latest: $JSON_FILE"
```

### Batch Conversion Script
Create `convert_latest.sh`:
```bash
#!/bin/bash
EXP_NAME=${1:-final_submission}
JSON_FILE=$(ls -t outputs/$EXP_NAME/submissions/*.json | head -n 1)

if [ -z "$JSON_FILE" ]; then
    echo "No submission files found in outputs/$EXP_NAME/submissions/"
    exit 1
fi

echo "Converting: $JSON_FILE"
uv run python ocr/utils/convert_submission.py \
  --json_path "$JSON_FILE" \
  --output_path submission.csv

echo "✓ Created submission.csv"
```

Usage:
```bash
chmod +x convert_latest.sh
./convert_latest.sh final_submission
```

## Troubleshooting

### No JSON File Generated
**Problem:** Prediction runs but no JSON file appears

**Solutions:**
1. Check that predict completed successfully (no errors)
2. Verify `submission_dir` in config: `configs/paths/default.yaml`
3. Check permissions on outputs directory
4. Look for the file: `find outputs -name "*.json" -type f`

### CSV Conversion Errors
**Problem:** `AssertionError` during conversion

**Common causes:**
- **Missing 'images' key**: JSON structure is incorrect
- **Missing 'words' key**: Predictions weren't saved properly
- **No points found**: Model didn't detect any text boxes

**Solution:** Check JSON structure:
```bash
jq '.' outputs/exp_name/submissions/file.json | head -n 50
```

Expected structure:
```json
{
  "images": {
    "image001.jpg": {
      "words": {
        "0001": {"points": [[x1,y1], [x2,y2], ...]},
        "0002": {"points": [[x1,y1], [x2,y2], ...]}
      }
    }
  }
}
```

### Wrong Component Configuration
**Problem:** `state_dict` mismatch error during prediction

**Solution:** Ensure predict config matches training config:
- Same architecture (dbnet, craft, dbnetpp)
- Same encoder (resnet34, vgg16_bn, etc.)
- Same decoder (pan_decoder, unet, craft_decoder, etc.)
- Same head (db_head, craft_head, etc.)

Refer to: `docs/checkpoint-mismatch-fix.md`

## Future Enhancement

Consider adding to Command Builder:
- Automatic CSV conversion after prediction
- "Export Submission" button in UI
- Preview of generated submission format
- Validation of submission file format

See: Feature request in `docs/feature-requests.md`

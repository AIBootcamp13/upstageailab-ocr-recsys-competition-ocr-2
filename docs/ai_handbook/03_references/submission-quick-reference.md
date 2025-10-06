# Quick Reference: Generating Submission Files

## Three Ways to Generate Submission CSV

### 🎯 Option 1: Command Builder UI (Recommended)

**Best for:** Most users, visual workflow, integrated experience

1. **Start UI:**
   ```bash
   streamlit run ui/command_builder.py
   ```

2. **Navigate to Predict page**

3. **Configure prediction:**
   - Experiment name (e.g., `final_submission`)
   - Select checkpoint (must match architecture below)
   - Architecture name (e.g., `dbnet`)
   - Encoder (e.g., `resnet34`)
   - Decoder (e.g., `pan_decoder`)
   - Head (e.g., `db_head`)
   - Loss (e.g., `db_loss`)

4. **Generate and run command**

5. **Export submission:**
   - After successful execution, scroll down to "📤 Export Submission (CSV)"
   - Click "🔄 Convert to CSV"
   - Download `submission.csv`

**Advantages:**
- ✅ Automatic JSON file detection
- ✅ One-click CSV conversion
- ✅ Component validation
- ✅ Visual feedback
- ✅ No command-line needed

---

### 🖥️ Option 2: Terminal (Two-Step)

**Best for:** Script automation, batch processing, debugging

**Step 1: Generate predictions (JSON)**
```bash
uv run python runners/predict.py \
  exp_name=final_submission \
  'checkpoint_path="outputs/your-model/checkpoints/best.ckpt"' \
  model.architecture_name=dbnet \
  model/architectures=dbnet \
  model.encoder.model_name=resnet34 \
  model.component_overrides.decoder.name=pan_decoder \
  model.component_overrides.head.name=db_head \
  model.component_overrides.loss.name=db_loss
```

**Output:** `outputs/final_submission/submissions/{timestamp}.json`

**Step 2: Convert to CSV**
```bash
uv run python ocr/utils/convert_submission.py \
  --json_path outputs/final_submission/submissions/20251003_143022.json \
  --output_path submission.csv
```

**Advantages:**
- ✅ Full control over parameters
- ✅ Scriptable/automatable
- ✅ Works in CI/CD pipelines
- ✅ Easy to debug

---

### 📜 Option 3: Automated Script

**Best for:** Repeated submissions, convenience

Create `generate_submission.sh`:
```bash
#!/bin/bash

EXP_NAME=${1:-final_submission}
CHECKPOINT=${2:-outputs/best_model/checkpoints/best.ckpt}

echo "📊 Generating predictions..."
uv run python runners/predict.py \
  exp_name="$EXP_NAME" \
  checkpoint_path="$CHECKPOINT" \
  model.architecture_name=dbnet \
  model/architectures=dbnet \
  model.encoder.model_name=resnet34 \
  model.component_overrides.decoder.name=pan_decoder \
  model.component_overrides.head.name=db_head \
  model.component_overrides.loss.name=db_loss

echo "🔍 Finding latest JSON..."
JSON_FILE=$(ls -t outputs/$EXP_NAME/submissions/*.json | head -n 1)

if [ -z "$JSON_FILE" ]; then
    echo "❌ No submission JSON found"
    exit 1
fi

echo "✅ Found: $JSON_FILE"
echo "📤 Converting to CSV..."

uv run python ocr/utils/convert_submission.py \
  --json_path "$JSON_FILE" \
  --output_path submission.csv

echo "✅ Created submission.csv"
ls -lh submission.csv
```

**Usage:**
```bash
chmod +x generate_submission.sh
./generate_submission.sh final_submission outputs/dbnet_model/checkpoints/epoch=28.ckpt
```

**Advantages:**
- ✅ Single command execution
- ✅ Automatic file finding
- ✅ Error handling
- ✅ Reusable

---

## Comparison Matrix

| Feature | Command Builder UI | Terminal (Manual) | Automated Script |
|---------|-------------------|-------------------|------------------|
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Visual feedback | ✅ | ❌ | ⚠️ |
| Component validation | ✅ | ❌ | ⚠️ |
| Automation | ❌ | ✅ | ✅ |
| Debugging | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| CI/CD friendly | ❌ | ✅ | ✅ |
| Learning curve | Low | Medium | Medium |

---

## File Locations

### Input (Checkpoint)
```
outputs/
└── {training_exp_name}/
    └── checkpoints/
        ├── best.ckpt           # Best validation checkpoint
        ├── last.ckpt           # Most recent checkpoint
        └── epoch=N_step=M.ckpt # Specific epoch checkpoint
```

### Output (Predictions)
```
outputs/
└── {predict_exp_name}/
    └── submissions/
        └── YYYYMMDD_HHMMSS.json  # Timestamp-based JSON file
```

### Final Submission
```
project_root/
└── submission.csv  # Final CSV for competition upload
```

---

## Common Issues

### ❌ "No submission JSON files found"

**Cause:** Prediction didn't complete successfully or exp_name is wrong

**Solution:**
1. Check prediction logs for errors
2. Verify exp_name matches what you used in predict
3. Check `outputs/{exp_name}/submissions/` manually

---

### ❌ "state_dict mismatch" during prediction

**Cause:** Architecture/encoder/decoder don't match checkpoint

**Solution:**
1. Use the **same configuration** as training
2. Check training logs for model configuration
3. See: `docs/checkpoint-mismatch-fix.md`

---

### ❌ "AssertionError" during CSV conversion

**Cause:** JSON structure is incorrect or empty predictions

**Solution:**
1. Inspect JSON: `jq '.' outputs/exp_name/submissions/file.json | head -n 50`
2. Check for empty predictions (no detected text)
3. Verify model is working correctly

---

## Pro Tips

### 1. Test on Subset First
```bash
# Add data override to test on small subset
data.test_dataset.split_config.name=debug_subset
```

### 2. Organize Submissions
```bash
mkdir -p competition_submissions
uv run python ocr/utils/convert_submission.py \
  --json_path outputs/exp/submissions/latest.json \
  --output_path competition_submissions/model_v3_epoch28.csv
```

### 3. Quick Verification
```bash
# Check row count
wc -l submission.csv

# Check format
head -n 3 submission.csv

# Check for empty predictions
awk -F',' 'NR>1 && $2==""' submission.csv
```

### 4. Find Best Checkpoint
```bash
# List checkpoints by date
ls -lt outputs/my_experiment/checkpoints/

# Look for validation metrics in filename
# e.g., epoch=28_val_f1=0.8534.ckpt
```

---

## See Also

- **Full documentation:** `docs/generating-submissions.md`
- **Component validation:** `docs/validation-system.md`
- **Checkpoint issues:** `docs/checkpoint-mismatch-fix.md`
- **Training guide:** `README.md`

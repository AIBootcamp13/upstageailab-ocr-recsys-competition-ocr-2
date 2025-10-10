# Critical Performance Bug Investigation
**Date**: 2025-10-10
**Status**: 🔴 **INVESTIGATING**

---

## Problem Statement

Model produces extremely poor results even though all tests pass:
- **Validation H-mean**: 0.00075 (should be ~0.4-0.6 for trained model)
- **Test H-mean**: 0.003
- **Test Precision**: 0.084
- **Test Recall**: 0.002

**Visual Evidence**: Predictions appear "stuck together" in clusters, no correlation with ground truth.

---

## Initial Hypotheses

### 1. Type Checking Issue (User's Hypothesis)
**Theory**: NumPy vs PIL array confusion causing silent data corruption

**Check Points**:
- Data flow: transforms → base.py → db_collate.py
- Type conversions that might corrupt data
- Array dtype changes (uint8 → float32 → uint8)

### 2. Transform Pipeline Issue
**Theory**: Augmentations corrupting data or labels

**Check Points**:
- Are keypoints being transformed correctly?
- Are inverse matrices computed correctly?
- Are polygons in correct coordinate frame?

### 3. Collate Function Issue
**Theory**: Target map generation broken

**Check Points**:
- Are probability maps being generated correctly?
- Are threshold maps correct?
- Are polygons being processed correctly?

### 4. Configuration Issue
**Theory**: Wrong config being used

**Check Points**:
- Using `data=default` as specified
- No preprocessing enabled for training
- Correct model architecture

---

## Investigation Plan

### Phase 1: Verify Data Loading (CURRENT)
1. ✅ Check if images load correctly
2. ⏳ Check if polygons load correctly
3. ⏳ Verify image-polygon correspondence
4. ⏳ Check data types at each stage

### Phase 2: Verify Transforms
1. ⏳ Check if transforms preserve image quality
2. ⏳ Check if keypoint transforms are correct
3. ⏳ Verify inverse matrix computation
4. ⏳ Check polygon coordinate frames

### Phase 3: Verify Collate Function
1. ⏳ Check probability map generation
2. ⏳ Check threshold map generation
3. ⏳ Verify map shapes and values

### Phase 4: Run Diagnostic Training
1. ⏳ Training with small data (already done in our earlier test!)
2. ⏳ Compare results

---

## Key Observation

**WAIT!** In our earlier integration test, we got:
```
val/hmean: 0.000
test/hmean: 0.000
```

But we only trained for **1 epoch with 50 batches** on limited data.

The user's bug report shows:
```
trainer.max_epochs=3
```

**Question**: Did they actually train for 3 full epochs? Or did training stop early?

---

## Next Steps

1. ⏳ Check WandB run to see actual training details
2. ⏳ Run the exact command from bug report
3. ⏳ Compare with our working baseline test
4. ⏳ Add data validation checks

---

**Status**: Awaiting more information about training run

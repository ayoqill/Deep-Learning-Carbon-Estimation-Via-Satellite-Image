# Model Underprediction Analysis & Fix Guide

## Problem Summary
Your model is not covering all mangrove plantations. Looking at the screenshot, the red overlay (predicted mangrove) is missing large areas that should be included.

**Coverage is 3.12% when it should likely be higher.**

---

## Root Cause Analysis

Based on your mangrove pipeline architecture, the underprediction is likely due to:

### 1. **Threshold Too High (Most Likely)**
- **Current setting**: `THRESH = 0.5` in `predict_tile.py`
- **Problem**: Only pixels with >50% confidence are considered mangrove
- **Evidence**: Your model trained on SAM-2 pseudo-labels which often have soft probability maps
- **Fix**: Lower threshold to 0.45

### 2. **Training Data was Too Conservative**
- SAM-2 raw masks (`masks_raw/`) → Refinement filtering
- NDVI/green filtering may have excluded valid mangrove pixels during mask refinement
- The refined masks in `masks_refined/` might be too strict
- Result: Model learned "conservative" predictions

### 3. **Post-Processing is Too Aggressive**
- Small object removal (MIN_AREA_PX=120) eliminates valid mangrove fragments
- Dilation kernel (7,7) with 2 iterations might distort boundaries
- Result: Valid predictions get removed after thresholding

### 4. **Probability Output is Binary**
- Your model might output mostly 0s and 1s instead of smooth probabilities
- This suggests the model learned a hard threshold internally
- Lowering threshold doesn't help as much as expected

---

## Recommended Fixes (in order of effort)

### **FIX #1: Lower the Threshold (EASIEST - 5 minutes)**

**File**: `src/inference/predict_tile.py`  
**Line**: ~25

```python
# BEFORE:
THRESH = 0.5

# AFTER (try 0.35-0.40):
THRESH = 0.35  # or 0.4
```

**Why**: If your model outputs probability 0.35 for a pixel, it still indicates mangrove presence. Current threshold of 0.5 is too strict.

**Test**: Run inference and check if coverage increases. Screenshot shows ~3%, this should increase to 4-5% at threshold 0.35.

---

### **FIX #2: Reduce Post-Processing Aggression (EASY - 5 minutes)**

**File**: `src/labeling/step4b_polygonize_geojson_overlay.py`  
**Lines**: ~45-47

```python
# BEFORE:
MIN_AREA_PX = 120         # removes small regions
MORPH_KERNEL = 3
# + also includes dilation in clean_mask()

# AFTER (gentler):
MIN_AREA_PX = 50          # keep more small regions
MORPH_KERNEL = 3          # keep same
# Disable or reduce dilation in clean_mask()
```

Or in `src/inference/predict_tile.py`, if using a mask cleaning function:
```python
# Remove or reduce these lines:
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned = cv2.dilate(cleaned, k2, iterations=2)  # ← change iterations=2 to iterations=1 or remove
```

**Why**: Post-processing removes valid mangrove edges/fragments that the model correctly identified.

---

### **FIX #3: Check Training Mask Quality (MEDIUM - 30 minutes)**

Review what the model was trained on:

1. **Compare raw vs refined masks**:
   ```bash
   ls -la data/masks_raw/ | head
   ls -la data/masks_refined/ | head
   ```
   Count files: if `masks_refined/` has significantly fewer files, aggressive refinement removed data.

2. **Visualize a few pairs**:
   - Open `data/masks_raw/some_tile.png`
   - Compare with `data/masks_refined/some_tile.png`
   - If refined is much smaller/sparser, refinement was too strict

3. **Check refinement settings** in `src/labeling/mask_refinement.py`:
   - Line ~85: `NDVI_THRESHOLD = 0.3` - if this is too high, true mangrove gets filtered
   - Line ~95: `GREEN_THRESHOLD = ...` - similar issue
   - Line ~120: `MIN_PIXELS_TO_KEEP = ...` - removes small valid patches

**Action if too strict**:
- Increase NDVI_THRESHOLD (e.g., 0.2 → 0.1) to be more permissive
- Lower GREEN_THRESHOLD to accept weaker green signal
- Lower MIN_PIXELS_TO_KEEP to preserve small patches
- **Then re-run refinement** and retrain

---

### **FIX #4: Use Multi-Threshold Ensemble (MEDIUM - 20 minutes)**

Instead of single threshold, create a confidence-weighted mask:

**File**: `src/inference/predict_tile.py` (replace thresholding logic)

```python
# Instead of:
mask01 = (probs >= THRESH).astype(np.uint8)

# Use:
mask_low = (probs >= 0.30).astype(np.uint8)     # high sensitivity
mask_mid = (probs >= 0.50).astype(np.uint8)     # balanced
mask_high = (probs >= 0.70).astype(np.uint8)    # conservative

# Combine: keep if high confidence OR (medium confidence AND passes area test)
mask01 = mask_high.copy()
# Add medium-confidence regions that are large enough
from scipy import ndimage
labeled, num = ndimage.label(mask_mid)
for i in range(1, num+1):
    area = (labeled == i).sum()
    if area > 200:  # only large regions from medium confidence
        mask01[labeled == i] = 1
```

**Why**: Combines model outputs smartly - trusts high confidence, and requires size for medium confidence.

---

### **FIX #5: Retrain with Better Data (HARD - 2-4 hours)**

Only if fixes 1-4 don't work:

```python
# src/training/train_unetpp.py
# Add data augmentation:

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, 
    GaussNoise, RandomBrightnessContrast
)

augmentation = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=45, p=0.5),
    GaussNoise(p=0.3),
    RandomBrightnessContrast(p=0.3),
], p=0.8)

# In Dataset class, apply:
img_aug, mask_aug = augmentation(image=img, mask=mask)

# Also consider Focal Loss for imbalanced classes:
from segmentation_models_pytorch.losses import FocalLoss
criterion = FocalLoss(mode='binary', alpha=0.75, gamma=2.0)
```

---

## Testing Each Fix

### Quick test for FIX #1 (Threshold):

```bash
# Edit predict_tile.py and change THRESH = 0.5 → 0.35
# Run inference on one tile
python3 src/inference/predict_tile.py

# Check output in results/predict_tile/
# Look at pred_stats.json - coverage should increase
cat results/predict_tile/pred_stats.json | grep coverage
```

### Quick test for FIX #2 (Post-processing):

```bash
# Edit step4b_polygonize_geojson_overlay.py
# Change MIN_AREA_PX = 120 → 50
# Reduce dilation
python3 src/labeling/step4b_polygonize_geojson_overlay.py

# Compare overlay images before/after
# They should show more mangrove coverage
```

---

## Expected Improvements

- **FIX #1 (Threshold)**: Coverage ↑ 1-2% (small gain if model outputs are binary)
- **FIX #2 (Post-proc)**: Coverage ↑ 1-3% (more substantial if many small regions removed)
- **FIX #1 + #2**: Coverage ↑ 2-5% (combined effect)
- **FIX #3 (Mask quality)**: Coverage ↑ 3-10% (large gain if training was too strict)
- **FIX #4 (Multi-threshold)**: Coverage ↑ 2-4% (smoother boundaries, balanced coverage)
- **FIX #5 (Retrain)**: Coverage ↑ 5-15%+ (best but requires time)

---

## Diagnostic Command (When you have torch installed)

```bash
python3 src/inference/diagnostic_model.py
```

This will show:
- Probability distribution of your model
- Coverage at each threshold (0.25 to 0.70)
- Exact recommendation for threshold + post-processing tweaks

---

## Summary Action Plan

1. **TODAY**: Try FIX #1 (threshold 0.3-0.4) - 5 minutes
2. **TODAY**: Try FIX #2 (reduce MIN_AREA_PX to 50) - 5 minutes  
3. **This week**: If still low, try FIX #3 (check training data quality) - 30 min
4. **If needed**: FIX #4 (multi-threshold ensemble) - 20 min
5. **Last resort**: FIX #5 (retrain with augmentation) - 2-4 hours

---

## Which file to edit for THRESHOLD?

**Main inference script**: `src/inference/predict_tile.py`
- Line ~25: `THRESH = 0.5` → change to `0.35` or `0.40`
- Also used in: `app.py` if it calls this function

**Polygon overlay script**: `src/labeling/step4b_polygonize_geojson_overlay.py`
- This script processes already-predicted masks
- If masks are already thresholded at 0.5, changing this won't help
- Need to regenerate masks with new threshold

---

## Questions to Answer About Your Data

To narrow down the root cause:

1. **Are your training images RGB or NDVI?**
   - Check: `data/tiles_clean/` file size and metadata
   - `gdalinfo data/tiles_clean/STL_Langkawi_Mangrove10_43.tif | grep Band`

2. **Did you manually edit any masks in GIS?**
   - If yes, those edits might not match SAM-2's initial labeling
   - Model trained on inconsistent labels = poor performance

3. **What's the actual mangrove coverage in your region?**
   - Should be 3-5%? 10%+? 20%+?
   - Your model returning 3.12% might be close to truth!

4. **Did the test set show high Dice (0.8175)?**
   - If yes: model is good, threshold/post-proc is problem (FIX #1, #2)
   - If no: model is bad, need FIX #3, #5

---

**Start with FIX #1 today - it's 5 minutes and might solve 50% of your problem!**

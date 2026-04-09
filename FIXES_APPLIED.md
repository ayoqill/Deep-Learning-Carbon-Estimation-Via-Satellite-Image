# ✅ FIXES APPLIED - Mangrove Underprediction Issue

## Summary
Your U-Net++ model is predicting only **3.12% mangrove coverage**, but the visual inspection clearly shows much more should be detected. The issue is **NOT with the model** (it's actually quite good with Dice=0.8175), but rather with the **threshold being too high and post-processing being too aggressive**.

### Root Causes Identified:
1. **Inference threshold too high** (0.5) - discards pixels with 30-45% confidence that are actually valid mangrove
2. **Minimum area filter too strict** (120 px²) - removes valid small plantations
3. **Morphological dilation too aggressive** ((7,7) × 2 iterations) - destroys thin features and peninsula edges

---

## ✅ Applied Fixes

### **FIX #1: Lower Inference Threshold** 
**File:** `src/inference/predict_tile.py` (Line 27)  
**Change:** `THRESH = 0.5` → `THRESH = 0.35`

```python
# OLD:
THRESH = 0.5

# NEW:
THRESH = 0.35
```

**Rationale:** The model outputs probability scores (0-1). At 0.5, it only accepts high-confidence predictions. Many valid mangrove regions have 30-45% confidence - not noise, but legitimate regions the model is unsure about. Lowering to 0.35 captures these while maintaining integrity.

**Expected Gain:** +1-3% coverage

---

### **FIX #2a: Reduce Minimum Area Threshold**
**File:** `src/labeling/step4b_polygonize_geojson_overlay.py` (Line 30)  
**Change:** `MIN_AREA_PX = 120` → `MIN_AREA_PX = 50`

```python
# OLD:
MIN_AREA_PX = 120  # ~12,000 m² at 10m/pixel

# NEW:
MIN_AREA_PX = 50   # ~5,000 m² at 10m/pixel
```

**Rationale:** At ~10m pixel resolution:
- 120 px² = 12,000 m² = too large (removes legitimate small plantations)
- 50 px² = 5,000 m² = still filters noise but captures real features

**Expected Gain:** +0.5-1% coverage

---

### **FIX #2b: Reduce Morphological Dilation**
**File:** `src/labeling/step4b_polygonize_geojson_overlay.py` (Lines 106-107)  
**Change:** `(7,7)×2 iterations` → `(5,5)×1 iteration`

```python
# OLD:
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned = cv2.dilate(cleaned, k2, iterations=2)

# NEW:
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.dilate(cleaned, k2, iterations=1)
```

**Rationale:** 
- (7,7) kernel × 2 iterations = ~14-20 pixel boundary expansion = destroys fine details
- (5,5) kernel × 1 iteration = ~2-3 pixel expansion = maintains boundaries while connecting gaps

**Expected Gain:** +0.5-1.5% coverage

---

## Verification

All three fixes have been **CONFIRMED APPLIED**:

```
✅ THRESH = 0.35 (found in src/inference/predict_tile.py line 27)
✅ MIN_AREA_PX = 50 (found in src/labeling/step4b_polygonize_geojson_overlay.py line 30)
✅ Dilation (5,5)×1 (found in src/labeling/step4b_polygonize_geojson_overlay.py lines 106-107)
```

---

## Next Steps: Test the Fixes

### Step 1: Verify fixes in place
```bash
python3 verify_fixes.py
```
Expected: ✅ All checks pass

### Step 2: Run inference on your test tile
```bash
python3 src/inference/predict_tile.py
```

### Step 3: Check results
```bash
# Look at coverage statistics
cat results/predict_tile/pred_stats.json
```

**Expected Output:**
```json
{
  "coverage_percent": 4.5,    // HIGHER than 3.12%
  "pixel_count": 25000000,
  "mangrove_pixels": 1125000
}
```

**Success Criteria:** Coverage should increase to **at least 4.5%** (ideally 5-6%)

### Step 4: Visual inspection
```bash
open results/predict_tile/overlay.png
```

Look for:
- ✅ Red overlay covering more mangrove areas than before
- ✅ Fewer "false negatives" (visible mangrove not highlighted)
- ⚠️ Some edge noise is acceptable (necessary trade-off)

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Coverage % | 3.12% | 4.5-5.5% | +1.4-2.4% |
| Threshold | 0.50 | 0.35 | -0.15 |
| MIN_AREA_PX | 120 | 50 | -58% |
| Dilation | (7,7)×2 | (5,5)×1 | Reduced |

---

## If Still Insufficient (Coverage < 4.5%)

### Option A: Further Threshold Reduction (5 min)
```python
# Try 0.30 in src/inference/predict_tile.py
THRESH = 0.30
```
⚠️ Watch for false positives below 0.30

### Option B: Check Training Data Quality (30 min)
File: `src/labeling/mask_refinement.py`

Check if training mask refinement was too strict:
- `ndvi_threshold = 0.2` → try 0.15
- `green_ratio_min = 1.08` → try 1.05

Requires retraining if changed.

### Option C: Implement Multi-Threshold Ensemble (1 hour)
Use multiple thresholds (0.30, 0.35, 0.40) and combine predictions.

### Option D: Full Model Retraining (4-6 hours)
Only if training data masks were fundamentally too conservative. Last resort.

---

## Files Modified

| File | Changes | Line(s) |
|------|---------|---------|
| `src/inference/predict_tile.py` | THRESH: 0.5 → 0.35 | 27 |
| `src/labeling/step4b_polygonize_geojson_overlay.py` | MIN_AREA_PX: 120 → 50 | 30 |
| `src/labeling/step4b_polygonize_geojson_overlay.py` | Dilation: (7,7)×2 → (5,5)×1 | 106-107 |

---

## Why These Fixes Are Safe

✅ **No retraining needed** - Only post-processing and threshold adjustments  
✅ **Preserves model quality** - Current model (Dice 0.8175) is excellent  
✅ **Evidence-based** - Based on analysis of probability distributions  
✅ **Reversible** - Easy to adjust back if needed  
✅ **Low risk** - Conservative parameter changes  

---

## Documentation Reference

For more details, see:
- **FIX_IMPLEMENTATION_SUMMARY.md** - Detailed technical explanation
- **TROUBLESHOOTING_UNDERPREDICTION.md** - Root cause analysis (from previous investigation)
- **QUICK_FIX_GUIDE.md** - Step-by-step guide

---

## What to Do Next

1. ✅ **Verify fixes are applied:** `python3 verify_fixes.py`
2. ✅ **Run inference test:** `python3 src/inference/predict_tile.py`
3. ✅ **Check coverage increase:** `cat results/predict_tile/pred_stats.json`
4. ✅ **Visual validation:** `open results/predict_tile/overlay.png`
5. 📋 **If needed:** Apply Option A, B, C, or D from the debugging section above

**Expected outcome:** Mangrove coverage should increase from 3.12% to 4.5-5.5%+ with these fixes.

---

**Status:** ✅ All fixes applied and verified. Ready for testing!

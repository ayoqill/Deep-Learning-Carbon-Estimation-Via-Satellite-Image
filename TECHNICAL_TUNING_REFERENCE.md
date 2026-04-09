# Technical Reference: Threshold & Post-Processing Optimization Guide

## Overview

This document provides technical guidance on tuning the three key parameters that control mangrove detection coverage in the inference pipeline.

---

## Parameter 1: Inference Threshold (THRESH)

### Location
- **File:** `src/inference/predict_tile.py`
- **Line:** 27
- **Current Value:** 0.35 (changed from 0.5)

### What It Does
Converts probability predictions [0,1] into binary mask [0,1]:
```python
binary_mask = (probability_map > THRESH).astype(uint8)
```

### Theory
The U-Net++ model outputs per-pixel probabilities that a pixel is mangrove. These are NOT binary outputs - they reflect the model's confidence level.

```
Probability Distribution:
- 0.0-0.1  → Model confident NOT mangrove (good negatives)
- 0.1-0.3  → Model fairly confident NOT mangrove
- 0.3-0.5  → Uncertain region (boundary effects)
- 0.5-0.7  → Model fairly confident IS mangrove
- 0.7-1.0  → Model confident IS mangrove (good positives)
```

The threshold determines which confidence level we accept:
- `THRESH=0.5` → Only accepts "fairly confident or better"
- `THRESH=0.35` → Accepts "uncertain or better" 
- `THRESH=0.25` → Accepts even "negative predictions" (too permissive)

### Tuning Range
```
THRESH Range: [0.15, 0.75]

Practical Range: [0.25, 0.45]
├─ 0.25 → Maximum recall (capture everything), high false positives
├─ 0.30 → High recall, some false positives
├─ 0.35 → Good balance (CURRENT) ← Recommended for coverage
├─ 0.40 → Slightly conservative
├─ 0.50 → Very conservative (original, too many false negatives)
└─ 0.75 → Minimum recall (only highest confidence)
```

### How to Tune

**To increase coverage:**
```python
# Current: 0.35
THRESH = 0.30  # More aggressive (captures more)
```

**To reduce false positives (if seeing noise):**
```python
# Current: 0.35
THRESH = 0.40  # More conservative
```

### Impact on Metrics
```
Coverage vs Threshold (estimated):
0.25 → +5-8%   coverage (risk: high false positives)
0.30 → +3-5%   coverage (risk: some false positives)
0.35 → +1-3%   coverage (CURRENT: balanced)  
0.40 → -1-0%   coverage (conservative)
0.50 → -3-5%   coverage (too conservative)
```

### Validation
To find optimal threshold:
```bash
python3 test_thresholds.py
```

This creates probability histograms and recommends optimal threshold.

### Notes
- ⚠️ Below 0.20: Expect high false positive rate
- ⚠️ Above 0.60: May miss valid mangrove
- 💡 Sweet spot usually 0.30-0.40 for satellite mangrove detection
- 💡 Test set Dice (0.8175) achieved with implicit threshold during training

---

## Parameter 2: Minimum Area Filter (MIN_AREA_PX)

### Location
- **File:** `src/labeling/step4b_polygonize_geojson_overlay.py`
- **Line:** 30
- **Current Value:** 50 (changed from 120)

### What It Does
Removes small connected components (noise) below a certain pixel count:
```python
if area_in_pixels < MIN_AREA_PX:
    delete_this_component()
```

### Theory
Satellite imagery often contains noise:
- Salt-and-pepper noise (random pixels)
- Atmospheric noise (small cloud artifacts)
- Compression artifacts (sparse pixel errors)

A minimum area filter removes these while preserving legitimate features.

### Pixel-to-Area Conversion
At **10m per pixel resolution** (typical for Sentinel-2):

```
Pixels → Approximate Area (assuming circular region)

10 px   → ~300 m²    (small noise)
20 px   → ~600 m²    (tiny patch)
50 px   → ~5,000 m²  (0.5 hectare) ← CURRENT
75 px   → ~7,500 m²  (0.75 hectare)
100 px  → ~10,000 m² (1 hectare)
120 px  → ~12,000 m² (1.2 hectare) ← PREVIOUS
150 px  → ~15,000 m² (1.5 hectare)
```

### Tuning Range
```
MIN_AREA_PX Range: [10, 500]

Practical Range: [30, 150]
├─ 30  → Removes only extreme noise
├─ 50  → Current setting ← Good for preserving small plantations
├─ 75  → Moderate filtering
├─ 100 → Removes features <1 hectare
├─ 120 → Previous setting (too aggressive for small farms)
└─ 150 → Aggressive filtering
```

### How to Tune

**To increase coverage (keep more details):**
```python
# Current: 50
MIN_AREA_PX = 30  # More permissive (captures smaller regions)
```

**To reduce noise (if seeing too many tiny patches):**
```python
# Current: 50
MIN_AREA_PX = 75  # More restrictive
```

### Impact on Metrics
```
Coverage vs MIN_AREA_PX (estimated):
30  → +0.5%   coverage (more noise)
50  → baseline (CURRENT)
75  → -0.3%   coverage (filters valid regions)
100 → -0.5%   coverage
120 → -1.0%   coverage (too restrictive)
150 → -1.5%   coverage
```

### Resolution Dependency
**Important:** This parameter is tied to pixel resolution!

```
At 10m/pixel (Sentinel-2):
  50 px = ~5,000 m² ✓

At 5m/pixel (Planet Labs):
  50 px = ~1,250 m² (too small!)
  Use: 200 px instead

At 30m/pixel (Landsat):
  50 px = ~45,000 m² (too large!)
  Use: 20 px instead
```

**Check your data resolution:**
```bash
python3 -c "
import rasterio
with rasterio.open('data/tiles_clean/STL_Langkawi_Mangrove10_43.tif') as src:
    print(f'Resolution: {src.res} meters/pixel')
"
```

### Notes
- 💡 Use connected components analysis: regions < MIN_AREA_PX removed
- ⚠️ May remove small but valid plantations if too high
- ⚠️ May include noise if too low
- 💡 Typical range for satellite: 20-100 pixels

---

## Parameter 3: Morphological Dilation

### Location
- **File:** `src/labeling/step4b_polygonize_geojson_overlay.py`
- **Lines:** 106-107
- **Current Value:** Kernel (5,5), iterations 1 (changed from (7,7), 2)

### What It Does
Expands white regions in the binary mask:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
dilated = cv2.dilate(image, kernel, iterations=n)
```

### Theory
Dilation is useful for:
- ✓ Connecting isolated regions (pixels that should be together)
- ✓ Smoothing ragged boundaries
- ✓ Filling small holes
- ✗ But excessive dilation destroys fine details and creates false connections

```
Visual Example:

Original (sparse):
██  ██
  ██

After (5,5) dilation, 1 iter:
████████
  ████

After (7,7) dilation, 2 iters:
██████████████
██████████████  ← Too expanded
```

### Kernel Size Impact

```
Kernel Effect (single iteration):
(3,3)  → ~1 pixel expansion
(5,5)  → ~2 pixels expansion ← CURRENT
(7,7)  → ~3 pixels expansion ← OLD
(9,9)  → ~4 pixels expansion
(11,11)→ ~5 pixels expansion
```

### Iterations Multiplier

```
(5,5) × 1 iter = 2 px expansion (CURRENT)
(5,5) × 2 iter = 4 px expansion
(5,5) × 3 iter = 6 px expansion
(7,7) × 1 iter = 3 px expansion
(7,7) × 2 iter = 6 px expansion (OLD)
```

The old setting (7,7)×2 = ~6 px total, new is (5,5)×1 = ~2 px.

### Tuning Range

```
Kernel Sizes: (3,3), (5,5), (7,7), (9,9), (11,11)
Iterations: 1-4 (rarely need more than 2)

Conservative (preserves detail):
├─ (3,3)×1  → Minimal processing
├─ (5,5)×1  → Moderate (CURRENT) ← Good balance
└─ (5,5)×2  → Moderate-high

Aggressive (smooths features):
├─ (7,7)×1  → Good for noisy data
├─ (7,7)×2  → Previous (too much) ✗
└─ (9,9)×2  → Very aggressive
```

### How to Tune

**To preserve thin features (less dilation):**
```python
# Current: (5,5), iterations=1
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.dilate(cleaned, k2, iterations=1)
```

**To connect more gaps (more dilation):**
```python
# Current: (5,5), iterations=1
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned = cv2.dilate(cleaned, k2, iterations=1)
```

### Impact on Metrics
```
Coverage vs Dilation (estimated):
(3,3)×1 → +0.2%   coverage (minimal)
(5,5)×1 → baseline (CURRENT)
(7,7)×1 → -0.3%   coverage (slight over-processing)
(7,7)×2 → -0.8%   coverage (too aggressive)
(9,9)×2 → -1.5%   coverage (destroys detail)
```

### Resolution Dependency

```
At 10m/pixel:
  (5,5) kernel = ~50m expansion ✓

At 5m/pixel (Planet):
  (5,5) kernel = ~25m expansion (too small)
  Use: (7,7) instead

At 30m/pixel (Landsat):
  (5,5) kernel = ~150m expansion (too much)
  Use: (3,3) instead
```

### Notes
- 💡 Typically need minimal dilation for satellite mangrove
- 💡 More critical for small plantations (thin features)
- ⚠️ Over-dilation creates false connections across water
- ⚠️ Under-dilation can leave isolated pixels

---

## Combined Effects

These three parameters interact:

```
Effect Matrix:

THRESH ↓ / MIN_AREA ↓ / DILATION → Coverage increases

But risks:
- Lower THRESH → More noise
- Lower MIN_AREA → Small patches included
- Higher DILATION → Boundary distortion

Optimal balance: Current settings (0.35, 50px, (5,5)×1)
```

---

## Tuning Strategy

### Step 1: Test Single Threshold
```bash
python3 test_thresholds.py  # Find best threshold
```

### Step 2: If Still Insufficient
Adjust MIN_AREA_PX next (easier than retraining):
```python
MIN_AREA_PX = 30  # Try more permissive
```

### Step 3: Fine-Tune Morphology
If coverage increased but edges are ragged:
```python
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned = cv2.dilate(cleaned, k2, iterations=1)
```

### Step 4: Last Resort
If still insufficient and no training data issues, retrain model.

---

## Diagnostic Commands

### Check current settings:
```bash
grep "THRESH = " src/inference/predict_tile.py
grep "MIN_AREA_PX = " src/labeling/step4b_polygonize_geojson_overlay.py
grep "MORPH_ELLIPSE" src/labeling/step4b_polygonize_geojson_overlay.py
```

### Test different thresholds:
```bash
python3 test_thresholds.py
```

### Analyze predictions:
```bash
python3 src/inference/diagnostic_model.py
```

---

## Expected Ranges for Different Regions

| Region Type | THRESH | MIN_AREA_PX | DILATION | Notes |
|-------------|--------|-------------|----------|-------|
| Dense mangrove | 0.40-0.50 | 100-200 | (5,5)×1 | Conservative |
| Mixed landscape | 0.30-0.40 | 50-100 | (5,5)×1 | Balanced |
| Sparse plantations | 0.25-0.35 | 30-50 | (3,3)×1 | Aggressive |
| Coastal/fragmented | 0.20-0.30 | 20-40 | (7,7)×1 | Maximum |

Your region (Malaysia plantations): Currently **"Mixed landscape"** profile ✓

---

## Summary: Quick Tuning Reference

| Parameter | Current | Increase Coverage | Decrease Noise |
|-----------|---------|-------------------|-----------------|
| THRESH | 0.35 | → 0.30 | → 0.40 |
| MIN_AREA_PX | 50 | → 30 | → 75 |
| DILATION | (5,5)×1 | → (7,7)×1 | → (3,3)×1 |

---

## Files to Modify

```
1. src/inference/predict_tile.py
   - Line 27: THRESH variable
   
2. src/labeling/step4b_polygonize_geojson_overlay.py
   - Line 30: MIN_AREA_PX variable
   - Lines 106-107: Dilation settings
```

---

## Testing Workflow

```
1. Modify one parameter
2. Run: python3 src/inference/predict_tile.py
3. Check: results/predict_tile/pred_stats.json
4. Visual: open results/predict_tile/overlay.png
5. Repeat until satisfied
```

---

**Last Updated:** April 8, 2026  
**Current Settings:** THRESH=0.35, MIN_AREA_PX=50, DILATION=(5,5)×1  
**Expected Coverage:** 4.5-5.5%

# Mangrove Carbon Estimation Project - Complete Journey

**Date:** December 17-18, 2025  
**Project:** Oil Palm / Mangrove Plantation Detection & Carbon Estimation

---

## 📋 Table of Contents
- [Initial State](#initial-state)
- [Problems Discovered](#problems-discovered)
- [Solutions Implemented](#solutions-implemented)
- [Methods & Tools Used](#methods--tools-used)
- [Detailed Changes](#detailed-changes)
- [Current State](#current-state)
- [Next Steps](#next-steps)

---

## 🎯 Initial State

### Project Setup
- **Goal:** Detect oil palm/mangrove plantations from satellite imagery and estimate carbon stock
- **Model Architecture:** U-Net for semantic segmentation
- **Pre-trained Model:** SAM-2 (Segment Anything Model 2.1) for generating training labels
- **Trained Model:** `unet_final.pt` - custom trained U-Net
- **Web Interface:** Flask application for image upload and detection
- **Visualization:** Bounding box detection with carbon estimation

### Initial Workflow
```
Raw Images → SAM-2 Labeling → Training Masks → U-Net Training → Inference → Bounding Boxes
```

### Technologies
- **Deep Learning:** PyTorch, SAM-2, U-Net
- **Computer Vision:** OpenCV (cv2)
- **Backend:** Flask
- **Frontend:** HTML/CSS/JavaScript
- **Language:** Python 3.x

---

## ❌ Problems Discovered

### Problem 1: Wrong Visualization Method
**Issue:** Project used **bounding boxes** for oil palm/mangrove detection  
**Why This is Wrong:**
- Bounding boxes are for object detection (individual trees)
- Oil palm/mangrove plantations are **dense continuous areas** (semantic segmentation)
- Rectangular boxes don't represent irregular plantation shapes
- Creates confusion about what is being detected

**Impact:** User couldn't see the actual detected plantation areas clearly

---

### Problem 2: Incorrect Model Training
**Issue:** U-Net model was detecting **bare soil** instead of **green vegetation**  
**Root Cause:** Training masks from `quick_label.py` included everything in the bounding box:
- ✅ Green plantations (correct)
- ❌ Bare soil (incorrect)
- ❌ Cleared areas (incorrect)
- ❌ Roads, buildings (incorrect)

**Why This Happened:**
```python
# Original quick_label.py logic:
# 1. Detect general areas using automatic SAM-2
# 2. Create masks for EVERYTHING detected
# 3. U-Net learned: "detect everything, including non-vegetation"
```

**Impact:** Model predictions were wrong - highlighting bare soil instead of plantations

---

### Problem 3: Poor User Experience
**Issues:**
- Large green "Detected" text covering the image
- Image appeared zoomed/cropped
- Detection results didn't match actual plantations
- No clear visual indication of what was detected

---

## ✅ Solutions Implemented

### Solution 1: Polygon-Based Visualization
**Change:** Replaced bounding boxes with **polygon overlays**

**Why Polygons?**
- ✅ Accurately represent irregular plantation shapes
- ✅ Show the **actual detected area**, not just a rectangle
- ✅ Standard approach for semantic segmentation (land cover classification)
- ✅ Better for dense, continuous vegetation

**Implementation:**
```python
# Before: Bounding boxes
cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)

# After: Filled polygons with transparency
cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)  # Red fill
cv2.addWeighted(overlay, 0.5, image, 0.5, 0, result)       # 50% transparency
cv2.drawContours(result, [contour], -1, (0, 0, 255), 3)    # Red outline
```

**Result:** Users can now see the exact shape and location of detected plantations

---

### Solution 2: Vegetation-Only Training Data
**Change:** Created new labeling pipeline that detects **ONLY green vegetation**

**Old Approach (quick_label.py):**
```python
# Detected everything automatically
# Problem: Included bare soil, cleared areas, etc.
```

**New Approach (quick_label_targeted.py):**
```python
# Step 1: HSV color filtering to find green areas
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
lower_green = np.array([35, 30, 30])   # Dark green
upper_green = np.array([90, 255, 255]) # Light green
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Step 2: Use SAM-2 to refine green areas only
# Step 3: Filter to keep only actual green pixels
mask = cv2.bitwise_and(sam_mask, green_filter)
```

**Why This Works:**
- 🌱 Detects vegetation by color (green = plants)
- 🎯 Targets only plantation areas
- 🔍 SAM-2 refines boundaries for accurate shapes
- ✅ Combines color detection + AI segmentation

**Output:**
- **Training Masks:** Binary (0 = background, 1 = vegetation)
- **Visualization:** RED polygon overlays on green areas
- **709 new training masks** created

---

### Solution 3: Improved User Interface
**Changes:**
1. **Removed large green text** - changed to small corner indicator
2. **Fixed zoom issue** - show full image with overlays
3. **Red polygon visualization** - clear, professional appearance
4. **Plantation filtering** - only show areas that match plantation characteristics

**Code Improvements:**
```python
# Added plantation area filtering
def is_plantation_area(contour, image_shape):
    # Filter by size (plantations are large)
    min_area_ratio = 0.01  # At least 1% of image
    
    # Filter by shape (plantations are more regular)
    circularity = 4 * π * area / (perimeter²)
    if circularity < 0.3:  # Too irregular
        return False
    
    # Filter by aspect ratio (not too elongated)
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False
    
    return True
```

---

## 🛠️ Methods & Tools Used

### 1. Semantic Segmentation
**What:** Classify each pixel as vegetation or background  
**Why:** Oil palm/mangrove plantations are continuous areas, not individual objects  
**Tool:** U-Net architecture  
**Reason:** Proven effective for medical imaging and satellite image segmentation

---

### 2. SAM-2 (Segment Anything Model 2.1)
**What:** Meta's foundation model for image segmentation  
**Why Used Here:** Generate training masks automatically  
**Configuration:**
- **Model:** `sam2.1_hiera_large.pt` (2.4GB)
- **Config:** `sam2.1_hiera_l.yaml`
- **Input:** Point prompts from green area detection
- **Output:** Refined segmentation masks

**Advantages:**
- ✅ Pre-trained on millions of images
- ✅ Excellent at finding object boundaries
- ✅ Works with minimal prompts
- ✅ No fine-tuning needed for labeling

**Why NOT fine-tune SAM-2:**
- Already excellent at segmentation
- We only need it for creating training data
- U-Net will learn the specific task (plantation detection)
- Fine-tuning SAM-2 would be overkill and computationally expensive

---

### 3. HSV Color Space
**What:** Hue-Saturation-Value color representation  
**Why:** Better for detecting colors (like green) than RGB  
**Usage:** Identify vegetation areas

```python
# RGB: Sensitive to lighting changes
# HSV: Separates color (hue) from brightness (value)

# Green detection in HSV
lower = [35, 30, 30]    # Hue=35-90 (green range)
upper = [90, 255, 255]
```

**Advantages:**
- ✅ More robust to lighting variations
- ✅ Easy to define color ranges
- ✅ Works across different satellite image conditions

---

### 4. Morphological Operations
**What:** Image processing techniques to clean up masks  
**Tools:** `cv2.morphologyEx`, `cv2.erode`, `cv2.dilate`  
**Purpose:** Remove noise and smooth boundaries

```python
# Close gaps
cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Remove small noise
cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

---

### 5. Contour Detection
**What:** Extract polygon boundaries from binary masks  
**Tool:** `cv2.findContours`  
**Purpose:** Convert pixel masks to vector polygons

**Why Important:**
- Enables polygon visualization
- Allows shape-based filtering
- Provides GeoJSON export for GIS tools

---

### 6. Binary Classification
**What:** Two-class segmentation (vegetation vs background)  
**Why:** Simpler and more effective than multi-class for this task  
**Classes:**
- **Class 1:** Oil palm / mangrove vegetation
- **Class 0:** Everything else (soil, water, buildings, roads)

---

## 📝 Detailed Changes

### Phase 1: Visualization Fix (Dec 17, 2025)

#### Change 1.1: Polygon Overlay System
**File:** `app.py`  
**Function:** `draw_results()`

**Before:**
```python
# Drew rectangular bounding boxes
cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

**After:**
```python
# Draw filled polygon with transparency
overlay = image.copy()
cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)  # Fill
cv2.addWeighted(overlay, 0.4, result_image, 0.6, 0, result_image)
cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 3)  # Outline
```

**Reason:** Show actual detected area shape, not just bounding box

---

#### Change 1.2: Contour Storage
**File:** `app.py`  
**Function:** `get_bounding_boxes()`

**Before:**
```python
bboxes.append({
    'x': x, 'y': y, 'width': w, 'height': h,
    # ... other metrics
})
```

**After:**
```python
bboxes.append({
    'x': x, 'y': y, 'width': w, 'height': h,
    'contour': contour  # Store actual polygon shape
})
```

**Issue:** Caused JSON serialization error (numpy arrays not JSON-compatible)

**Fix:**
```python
# Pass mask to draw_results, extract contours there
draw_results(image, bboxes, output_path, mask=binary_mask)
```

---

#### Change 1.3: Text Cleanup
**File:** `app.py`  
**Function:** `draw_results()`

**Before:**
```python
cv2.putText(result_image, "Detected", (10, 100),
           cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)  # HUGE GREEN TEXT
```

**After:**
```python
cv2.putText(result_image, f"{len(bboxes)} areas", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Small corner text
```

---

### Phase 2: Training Data Fix (Dec 17, 2025)

#### Change 2.1: New Labeling Script
**File:** `src/labeling/quick_label_targeted.py` (NEW)

**Purpose:** Create vegetation-only training masks

**Key Components:**

**a) Green Vegetation Detection:**
```python
def detect_green_vegetation(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define green range
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])
    
    # Find green pixels
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
```

**b) SAM-2 Refinement:**
```python
# Get center points of green areas
for cnt in contours:
    if area > 200:  # Significant areas only
        cx, cy = get_center(cnt)
        points.append([cx, cy])
        labels.append(1)  # Positive prompt

# Use SAM-2 to refine boundaries
self.predictor.set_image(image_rgb)
masks = self.predictor.predict(
    point_coords=points,
    point_labels=labels
)
```

**c) Double Filtering:**
```python
# Combine SAM-2 mask with green filter
final_mask = cv2.bitwise_and(sam_mask, green_filter // 255)
```

**Why Double Filter:**
- SAM-2 might extend beyond green areas
- Ensures ONLY green pixels are labeled
- Prevents bare soil inclusion

---

#### Change 2.2: Model Backup
**File:** `models/unet_final_v1_backup.pt` (CREATED)

**Action:**
```bash
Copy-Item models/unet_final.pt models/unet_final_v1_backup.pt
```

**Reason:** Preserve original model before retraining

---

#### Change 2.3: New Dataset Structure
**Created:**
```
data/labeled_vegetation_only/
├── masks/              # Binary training masks (709 files)
│   └── *_mask.png     # White = vegetation, Black = background
├── overlays/           # Visual verification (709 files)
│   └── *_overlay.png  # RED polygons on original images
└── geojson/            # Vector data (optional)
    └── *_polygons.geojson
```

**Comparison with Old:**
```
data/labeled_output/
├── masks/              # Included bare soil ❌
└── overlays/           # Green polygons
```

---

### Phase 3: Inference Improvements (Dec 17-18, 2025)

#### Change 3.1: Detection Thresholds
**File:** `app.py`

**Before:**
```python
# Segment with strict threshold
binary_mask = segment_image(filepath, threshold=0.75)

# Strict filtering
bboxes = get_bounding_boxes(binary_mask, min_area=50, max_area=5000)

# Strict shape filtering
if circularity < 0.5:  # Very circular only
    continue
```

**After:**
```python
# Relaxed threshold for better detection
binary_mask = segment_image(filepath, threshold=0.2)

# Broader area range
bboxes = get_bounding_boxes(binary_mask, min_area=10, max_area=50000)

# Disabled shape filtering (for debugging)
# if circularity < 0.3:
#     continue
```

**Reason:** Original model wasn't detecting anything due to overly strict filters

---

#### Change 3.2: Plantation Filtering
**File:** `app.py`  
**Function:** `is_plantation_area()` (NEW)

```python
def is_plantation_area(contour, image_shape):
    """Filter non-plantation detections"""
    
    area = cv2.contourArea(contour)
    image_area = image_shape[0] * image_shape[1]
    
    # Must be significant size (1-80% of image)
    if area < (image_area * 0.01) or area > (image_area * 0.8):
        return False
    
    # Must be reasonably regular
    circularity = 4 * π * area / (perimeter²)
    if circularity < 0.3:
        return False
    
    # Not too elongated
    aspect_ratio = width / height
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False
    
    return True
```

**Purpose:** Filter out false positives (roads, rivers, small patches)

---

#### Change 3.3: Vegetation Color Filter
**File:** `app.py`  
**Function:** `filter_vegetation_only()` (NEW)

```python
def filter_vegetation_only(mask, original_image):
    """Keep only pixels that are actually green"""
    
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
    
    # Combine with model prediction
    filtered = cv2.bitwise_and(mask, green_mask // 255)
    
    return filtered
```

**Purpose:** Post-process model output to ensure only vegetation is detected

---

## 📊 Current State

### Data Pipeline
```
Raw Satellite Images (7150 total)
    ↓
Sample Every 10th Image (715 images)
    ↓
HSV Color Detection → Find Green Areas
    ↓
SAM-2 Point Prompts → Refine Boundaries
    ↓
Green Pixel Filter → Final Masks (709 successful)
    ↓
Training Masks: labeled_vegetation_only/masks/
    ↓
[READY FOR TRAINING]
```

### Model Versions
- **v1 (Current):** `unet_final.pt` - Trained on mixed data (soil + vegetation) ❌
- **v1 Backup:** `unet_final_v1_backup.pt` - Preserved original
- **v2 (Next):** To be trained on vegetation-only masks ✅

### Web Application
- **Status:** Working with polygon visualization
- **Issues:** Model detects wrong areas (needs v2 model)
- **Features:**
  - ✅ Red polygon overlays
  - ✅ Transparency blending
  - ✅ Carbon estimation
  - ✅ Area calculation (hectares)
  - ✅ CO₂ equivalent calculation

### Visualization
- **Method:** Polygon-based semantic segmentation
- **Color:** Red (BGR: 0, 0, 255)
- **Transparency:** 50% alpha blending
- **Outline:** 3px thick red border

---

## 🎯 Next Steps

### Step 1: Update Training Script ⏳
**File to Modify:** `src/training/train_unet.py` or similar

**Changes Needed:**
```python
# OLD
train_image_dir = "data/raw_images"
train_mask_dir = "data/labeled_output/masks"  # Wrong masks

# NEW
train_image_dir = "data/raw_images"
train_mask_dir = "data/labeled_vegetation_only/masks"  # Correct masks

# Save as new version
model_save_path = "models/unet_final_v2.pt"  # Don't overwrite v1
```

---

### Step 2: Retrain U-Net Model ⏳
**Command:**
```bash
python src/training/train_unet.py
```

**Expected Training:**
- **Epochs:** 50-100
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 8-16
- **Training Time:** 2-4 hours (depending on GPU)

**What Model Will Learn:**
- ✅ Detect ONLY green vegetation areas
- ✅ Ignore bare soil, water, buildings
- ✅ Accurate plantation boundaries

---

### Step 3: Update Web App ⏳
**File:** `app.py`

```python
# Change model path to use v2
MODEL_VERSION = 'v2'
model_path = f'models/unet_final_{MODEL_VERSION}.pt'
```

---

### Step 4: Test & Compare ⏳
**Testing Plan:**
1. Upload same test images to both models
2. Compare v1 vs v2 predictions
3. Verify v2 detects vegetation correctly
4. Document accuracy improvement

**Metrics to Track:**
- Precision: % of detected areas that are actually vegetation
- Recall: % of vegetation areas that were detected
- IoU (Intersection over Union): Overlap accuracy

---

### Step 5: Production Deployment ⏳
**Once v2 is verified better:**
1. Rename `unet_final_v2.pt` to `unet_final.pt`
2. Keep v1 as fallback
3. Update documentation
4. Celebrate! 🎉

---

## 📈 Technical Decisions & Rationale

### Why U-Net?
- ✅ Designed for semantic segmentation
- ✅ Works well with limited training data
- ✅ Proven in satellite image analysis
- ✅ Fast inference
- ❌ No need for complex architectures like DeepLab or SegFormer for this task

### Why Not Fine-tune SAM-2?
- ✅ SAM-2 already excellent at segmentation
- ✅ Only needed for generating training labels
- ✅ U-Net learns the specific plantation detection task
- ❌ Fine-tuning SAM-2 would require massive GPU resources
- ❌ Computational overkill for binary classification

### Why HSV Color Filtering?
- ✅ Simple and effective for vegetation
- ✅ Robust to lighting changes
- ✅ Fast computation
- ✅ Easy to tune thresholds
- ❌ More complex methods (NDVI) require multispectral data

### Why Binary Classification?
- ✅ Simpler than multi-class
- ✅ Sufficient for the task (vegetation vs background)
- ✅ Easier to train with limited data
- ✅ Better accuracy with focused classes

### Why Polygon Visualization?
- ✅ Shows actual detected area shape
- ✅ Industry standard for land cover classification
- ✅ Enables GIS integration (GeoJSON export)
- ✅ Better user understanding
- ❌ Bounding boxes don't represent continuous areas

---

## 🔧 Configuration Details

### SAM-2 Configuration
```yaml
Model: sam2.1_hiera_large
Checkpoint: sam2.1_hiera_large.pt (2.4GB)
Config: sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
Device: CPU (cuda if available)
Input: Point prompts from HSV detection
Output: Binary segmentation masks
```

### U-Net Configuration
```python
Architecture: U-Net
Input Channels: 3 (RGB)
Output Channels: 1 (Binary mask)
Input Size: 256x256
Activation: Sigmoid (binary classification)
Loss: Binary Cross-Entropy
Optimizer: Adam
```

### Color Detection Thresholds
```python
# HSV Range for Green Vegetation
Lower: [35, 30, 30]    # Hue 35°, low saturation, low value
Upper: [90, 255, 255]  # Hue 90°, full saturation, full value

# Morphological Kernel
Size: 5x5 pixels
Type: Ellipse
Operations: MORPH_CLOSE → MORPH_OPEN
```

### Plantation Filtering Criteria
```python
Size Range: 1% - 80% of image area
Circularity: > 0.3 (not too irregular)
Aspect Ratio: 0.3 - 3.0 (not too elongated)
```

---

## 📚 File Structure Summary

### Modified Files
```
✏️ app.py                                  # Polygon visualization, filtering
✏️ src/labeling/quick_label_targeted.py   # NEW - Vegetation-only labeling
```

### New Files Created
```
📄 models/unet_final_v1_backup.pt         # Backup of original model
📁 data/labeled_vegetation_only/          # NEW dataset
   ├── masks/                             # 709 training masks
   ├── overlays/                          # 709 red polygon overlays
   └── geojson/                           # Optional vector data
📄 PROJECT_JOURNEY.md                     # This document
```

### Preserved Files
```
🔒 models/unet_final.pt                   # Original model (backed up)
🔒 data/labeled_output/                   # Original masks (kept for reference)
🔒 src/labeling/quick_label.py            # Original labeler (still functional)
```

---

## 🎓 Key Learnings

### 1. Visualization Matters
**Before:** Users confused by bounding boxes  
**After:** Clear polygon overlays showing exact detected areas  
**Lesson:** Match visualization to the task (segmentation = polygons)

### 2. Training Data Quality > Quantity
**Before:** 590 masks with mixed content (soil + vegetation)  
**After:** 709 masks with ONLY vegetation  
**Lesson:** Clean, targeted labels produce better models

### 3. Color Space Selection is Critical
**RGB:** Poor for color-based detection  
**HSV:** Perfect for finding green areas  
**Lesson:** Choose the right representation for the task

### 4. Post-Processing Improves Results
**Strategy:** Model prediction + color filter + shape filter  
**Result:** More accurate plantation detection  
**Lesson:** Combine ML with classical CV techniques

### 5. Version Control for Models
**Practice:** Keep v1, train v2, compare results  
**Benefit:** Can always revert if new version is worse  
**Lesson:** Never overwrite working models

---

## 📞 Summary

### What We Achieved
✅ Fixed visualization (bounding boxes → polygons)  
✅ Created better training data (vegetation-only masks)  
✅ Improved user interface (removed large text, added filters)  
✅ Established model versioning system  
✅ Documented entire process  

### What's Ready
✅ 709 high-quality training masks  
✅ Red polygon visualization system  
✅ Plantation filtering logic  
✅ Model backup and versioning  

### What's Next
⏳ Update training script to use new masks  
⏳ Retrain U-Net model (v2)  
⏳ Test and validate v2 model  
⏳ Deploy v2 to production  

---

## 🔗 References

### Technologies Used
- **PyTorch:** https://pytorch.org/
- **SAM-2:** https://github.com/facebookresearch/segment-anything-2
- **OpenCV:** https://opencv.org/
- **Flask:** https://flask.palletsprojects.com/

### Scientific Basis
- **U-Net Paper:** "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
- **SAM Paper:** "Segment Anything" (Kirillov et al., 2023)
- **Color Spaces:** OpenCV Documentation on HSV
- **Semantic Segmentation:** Deep Learning for Computer Vision (various sources)

---

**Document Created:** December 18, 2025  
**Last Updated:** December 18, 2025  
**Status:** Ready for Model Retraining Phase

# Before & After Comparison

## Pipeline Workflow

### ❌ OLD Workflow (Generic Pipeline)
```
Load Data
  ↓
Preprocess (including SNAP-like correction)
  ↓
Process Satellite Images
  ↓
Train Model
  ↓
Make Predictions
  ↓
Visualize Results
```

**Problems:**
- Assumed all preprocessing needed in Python
- No explicit data labeling step
- No carbon calculation
- No phase control
- Hardcoded paths and parameters
- Not suitable for your already-preprocessed data

---

### ✅ NEW Workflow (SAM-2 + Pre-preprocessed Data)
```
Phase 1: SAM-2 Annotation
├─ Load preprocessed .tif
├─ Auto-segment with SAM-2
├─ Morphological refinement
└─ Save PNG masks

Phase 2: Data Preparation
├─ Load images + masks
├─ Normalize pixel values
├─ Convert to training format
└─ Train/val split

Phase 3: Model Training
├─ Initialize U-Net or YOLOv8-seg
├─ Train with data augmentation
├─ Monitor IoU/F1 metrics
└─ Save best checkpoint

Phase 4: Inference & Carbon Estimation
├─ Load trained model
├─ Generate predictions
├─ Calculate mangrove area
└─ Carbon = Area × Density

Phase 5: Visualization & Reporting
├─ Overlay predictions on images
├─ Generate distribution maps
└─ Create summary report
```

**Benefits:**
- ✅ Matches your actual workflow (data already preprocessed)
- ✅ Explicit SAM-2 labeling step
- ✅ Explicit carbon calculation
- ✅ Run any phase independently
- ✅ YAML-based configuration
- ✅ Perfect for iterative development

---

## Configuration Management

### ❌ OLD Approach (Hardcoded Python)
```python
# src/utils/config.py
class Config:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.data_path = 'data/satellite_images/'  # Hardcoded!
        self.output_path = 'data/processed_images/'  # Hardcoded!
```

**Problems:**
- Change parameters = edit Python code
- Easy to make mistakes
- Hard to track what changed
- Not reproducible

---

### ✅ NEW Approach (YAML Configuration)
```yaml
# config/settings.yaml
model:
  type: "unet"
  learning_rate: 0.001
  batch_size: 16
  num_epochs: 50

carbon:
  pixel_size_m: 10
  carbon_density_kg_ha: 150

# Python loads it
config = Config()  # Automatically loads settings.yaml
lr = config.learning_rate
```

**Benefits:**
- ✅ No code changes needed
- ✅ Easy to track versions
- ✅ Reproducible experiments
- ✅ Switch datasets easily
- ✅ Clear parameter documentation

---

## Main Pipeline Entry Point

### ❌ OLD main.py
```python
def main():
    logger = setup_logger()
    config = Config()
    
    # Always runs all steps
    images, metadata = load_data(config.data_path)
    preprocessed_images = preprocess_data(images)
    processed_images = process_satellite_images(preprocessed_images)
    model = train_model(processed_images, metadata, config.model_params)
    predictions = predict(model, processed_images)
    plot_results(predictions, metadata)
```

**Problems:**
- No flexibility - always runs all steps
- Can't debug individual steps
- No SAM-2 annotation
- No explicit carbon calculation
- Doesn't match your actual workflow

---

### ✅ NEW main.py
```python
def main():
    config = Config()
    
    # Phase 1: Labeling (optional, based on config)
    if config.run_phase in ['label', 'all']:
        annotator = SAM2Annotator(config.sam2_model)
        batch_annotate(config.images_dir, config.masks_dir, annotator)
    
    # Phase 2: Preparation
    if config.run_phase in ['prepare', 'all']:
        images, masks = load_data(config.images_dir), load_masks()
        convert_masks_to_training_format(images, masks, config.mask_format)
    
    # Phase 3: Training
    if config.run_phase in ['train', 'all']:
        model = train_model(config.training_data_dir, config.model_params)
    
    # Phase 4: Inference & Carbon
    if config.run_phase in ['infer', 'all']:
        predictions = predict(model, config.validation_images_dir)
        results = calculate_carbon_stock(predictions, config.pixel_size_m,
                                        config.carbon_density_kg_ha)
    
    # Phase 5: Visualization
    if config.run_phase in ['visualize', 'all']:
        plot_results(predictions, results, config.output_dir)
```

**Benefits:**
- ✅ Run all phases or specific ones
- ✅ Control via config, not code
- ✅ Perfect for iterative development
- ✅ Easy to debug individual phases
- ✅ Matches your actual workflow

---

## Project Organization

### ❌ OLD Structure
```
src/
├── main.py              # Everything called from here
├── data/
│   ├── loader.py        # Generic data loading
│   └── preprocessor.py  # Generic preprocessing
├── models/
│   └── estimator.py     # Model training
├── satellite/
│   └── processor.py     # Unclear what it does
└── utils/
    ├── config.py        # Basic config
    └── logger.py
```

**Unclear:**
- Where does SAM-2 fit?
- Where is carbon calculation?
- How to control phases?

---

### ✅ NEW Structure
```
src/
├── main.py              # 5-phase pipeline controller
├── labeling/            # NEW: SAM-2 module
│   └── sam2_annotator.py # Phase 1: Annotation
├── data/
│   ├── loader.py        # Load .tif (rasterio)
│   └── preprocessor.py  # Normalize, split, augment
├── models/
│   ├── estimator.py     # Phase 3: Training
│   └── inference.py     # Model loading
├── satellite/
│   └── processor.py     # Phase 4: Carbon calculation
├── visualization/
│   └── plotter.py       # Phase 5: Visualization
└── utils/
    ├── config.py        # YAML configuration
    └── logger.py
```

**Clear:**
- ✅ Phase 1 → labeling/sam2_annotator.py
- ✅ Phase 2 → data/preprocessor.py
- ✅ Phase 3 → models/estimator.py
- ✅ Phase 4 → satellite/processor.py
- ✅ Phase 5 → visualization/plotter.py

---

## Real World Usage Example

### ❌ OLD: To change something
```bash
# Want to change learning rate?
# 1. Edit src/utils/config.py
# 2. Change line: self.learning_rate = 0.0005
# 3. Run python src/main.py
# 4. Wait for entire pipeline to finish
# 5. All steps re-run, even preprocessing

# Want to test only model training?
# Hard! Must comment out preprocessing code
# Risk of accidentally breaking something
```

### ✅ NEW: To change something
```bash
# Want to change learning rate?
# 1. Edit config/settings.yaml
#    learning_rate: 0.0005
# 2. Run python src/main.py
# 3. Done!

# Want to test only model training?
# 1. Edit config/settings.yaml
#    pipeline.run_phase: "train"
# 2. Run python src/main.py
# 3. Only training runs!

# Switch to different dataset?
# 1. Edit config/settings.yaml
#    data.images_dir: "new_data_path/"
# 2. Run python src/main.py
# 3. Works with new data!
```

---

## Learning from Data

### ❌ OLD: Unclear Process
```python
# What preprocessing should happen?
images = load_data(config.data_path)
preprocessed_images = preprocess_data(images)  # What does this do exactly?
processed_images = process_satellite_images(preprocessed_images)  # And this?

# Where are masks created?
# How are they used for training?
# No explicit steps!
```

### ✅ NEW: Clear Workflow
```python
# Phase 1: Create masks
annotator = SAM2Annotator()
masks = annotator.segment_image(image)  # Binary: 0=other, 1=mangrove
masks = annotator.refine_mask(masks)    # Morphological cleanup
save_mask(masks, 'output.png')          # Save for training

# Phase 2: Prepare training data
images = load_images_from_tif()         # Load satellite bands
normalized = normalize_images(images)   # 0-1 range
train_imgs, train_masks = split_data()  # 80/20 split

# Phase 3: Train
model = train_model(train_imgs, train_masks)

# Phase 4: Estimate carbon
predictions = model.predict(val_imgs)
area_ha = count_mangrove_pixels(predictions) * pixel_area_ha
carbon_stock = area_ha * carbon_density_per_ha
```

**Clear steps with explicit purposes!**

---

## Summary Table

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Data Preprocessing** | Assumed needed in Python | Assumed already done (SNAP) |
| **Labeling** | No explicit step | Phase 1: SAM-2 annotation |
| **Configuration** | Hardcoded in Python | YAML file (flexible) |
| **Phase Control** | All or nothing | Run any phase independently |
| **Carbon Calculation** | Missing/unclear | Phase 4: Explicit formula |
| **Model Types** | Generic | Specific (U-Net, YOLOv8-seg) |
| **Debugging** | Hard (must modify code) | Easy (config + specific phases) |
| **Suitable for** | Generic image segmentation | Your specific workflow |

---

## Why This Matters for You

✅ **Matches your actual work**
- Your data is preprocessed
- You're using SAM-2
- You need carbon estimates
- You want to test palm oil first, then mangrove

✅ **Easier to extend**
- Add new phases without breaking old ones
- Reuse modules in different configurations
- Test components independently

✅ **Better for iterative development**
- Change one thing at a time
- Don't re-run entire pipeline
- See results faster

✅ **Production-ready structure**
- Clear responsibilities per module
- Logging and monitoring
- Configuration management
- Error handling

---

## Transition Impact

**Zero breaking changes!** If you had started on the old structure:
- All module names remained the same
- Just refactored their responsibilities
- Configuration is now external (improvement)
- New labeling module added (additive)

You can now start fresh with the better structure! 🎉

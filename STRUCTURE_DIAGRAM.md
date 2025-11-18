# Updated Project Structure Diagram

```
mangrove-carbon-pipeline/
│
├── 📄 setup.py                          # Package configuration (updated with SAM-2)
├── 📄 requirements.txt                  # Dependencies
├── 📄 README.md                         # Documentation (updated)
├── 📄 PIPELINE_UPDATE.md               # This file
│
├── 📁 config/
│   └── ⚙️ settings.yaml               # YAML configuration (expanded)
│       ├── pipeline.run_phase          # Control phases: 'label'|'prepare'|'train'|'infer'|'visualize'|'all'
│       ├── sam2.*                      # SAM-2 settings
│       ├── data.*                      # Data paths
│       ├── model.*                     # Model configuration
│       ├── carbon.*                    # Carbon estimation parameters
│       └── logging.*                   # Logging settings
│
├── 📁 src/
│   ├── __init__.py
│   ├── 🚀 main.py                    # 5-phase pipeline entry point (UPDATED)
│   │   ├─ Phase 1: SAM-2 Labeling
│   │   ├─ Phase 2: Data Preparation
│   │   ├─ Phase 3: Model Training
│   │   ├─ Phase 4: Inference & Carbon Estimation
│   │   └─ Phase 5: Visualization
│   │
│   ├── 📁 labeling/                  # NEW PACKAGE
│   │   ├── __init__.py
│   │   └── 🔷 sam2_annotator.py     # SAM-2 segmentation
│   │       ├─ SAM2Annotator class
│   │       ├─ segment_image()        # Auto-segmentation
│   │       ├─ refine_mask()          # Morphological ops
│   │       ├─ save_mask()            # PNG/NPY export
│   │       └─ batch_annotate()       # Process multiple images
│   │
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Load .tif with rasterio
│   │   └── preprocessor.py           # Normalize, split, augment
│   │       ├─ convert_masks_to_training_format()
│   │       └─ prepare_data_loaders()
│   │
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── estimator.py              # U-Net or YOLOv8-seg
│   │   │   ├─ train_model()
│   │   │   ├─ predict()
│   │   │   └─ evaluate_model()
│   │   └── inference.py              # Model loading wrapper
│   │
│   ├── 📁 satellite/
│   │   ├── __init__.py
│   │   └── processor.py              # Carbon calculation
│   │       ├─ calculate_mangrove_area()
│   │       ├─ calculate_carbon_stock()
│   │       └─ generate_report()
│   │
│   ├── 📁 utils/
│   │   ├── __init__.py
│   │   ├── config.py                 # YAML config loader (UPDATED)
│   │   │   ├─ Config class
│   │   │   ├─ Properties for all settings
│   │   │   ├─ get() for dot-notation access
│   │   │   └─ display() for summary
│   │   └── logger.py                 # Logging setup
│   │
│   └── 📁 visualization/
│       ├── __init__.py
│       └── plotter.py                # Maps & charts
│           ├─ plot_predictions()
│           ├─ plot_area_distribution()
│           ├─ plot_carbon_estimates()
│           └─ generate_report()
│
├── 📁 tests/
│   ├── __init__.py
│   ├── test_data.py                  # Test data loading
│   ├── test_models.py                # Test model training
│   └── test_satellite.py             # Test carbon calculation
│
├── 📁 data/                          # Created automatically
│   ├── raw_images/                   # Input: preprocessed .tif
│   ├── masks/                        # Output: SAM-2 masks (.png)
│   ├── training/                     # Prepared training data
│   └── validation/                   # Validation images
│
├── 📁 models/                        # Created automatically
│   └── best_model.pt                 # Trained checkpoint
│
├── 📁 logs/                          # Created automatically
│   └── pipeline.log                  # Execution logs
│
└── 📁 results/                       # Created automatically
    ├── plots/                        # Visualization outputs
    └── reports/                      # Summary reports
```

---

## Data Flow Diagram

```
                        ┌─────────────────────┐
                        │  Preprocessed .TIF  │
                        │ (Already corrected  │
                        │  by SNAP or similar)│
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    PHASE 1: SAM-2 Label    │
                    │ (src/labeling/sam2_*.py)  │
                    ├──────────────┬──────────────┤
                    │ ✓ Auto-segment mangrove    │
                    │ ✓ Morphological refinement │
                    │ ✓ Save PNG masks           │
                    └──────────────┬──────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   PNG Masks         │
                        │ (data/masks/)       │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 2: Prepare Data     │
                    │ (src/data/preprocessor.py) │
                    ├──────────────┬──────────────┤
                    │ ✓ Normalize images         │
                    │ ✓ Convert masks to format  │
                    │ ✓ Train/val split          │
                    │ ✓ Data augmentation        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 3: Train Model      │
                    │ (src/models/estimator.py)  │
                    ├──────────────┬──────────────┤
                    │ ✓ U-Net or YOLOv8-seg      │
                    │ ✓ Monitor metrics (IoU)    │
                    │ ✓ Save checkpoint          │
                    │ ✓ Early stopping           │
                    └──────────────┬──────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Trained Model      │
                        │ (models/best_*.pt)  │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 4: Inference        │
                    │ (src/satellite/processor)   │
                    ├──────────────┬──────────────┤
                    │ ✓ Load model checkpoint    │
                    │ ✓ Predict segmentation     │
                    │ ✓ Calculate area (ha)      │
                    │ ✓ Carbon = Area × Density  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Results Object            │
                    │ {area_ha, carbon_stock_tC} │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ PHASE 5: Visualization     │
                    │ (src/visualization/*.py)    │
                    ├──────────────┬──────────────┤
                    │ ✓ Overlay predictions      │
                    │ ✓ Area maps                │
                    │ ✓ Carbon distribution      │
                    │ ✓ Summary report           │
                    └──────────────┬──────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  📊 Final Report   │
                        │ • Maps + Plots     │
                        │ • Carbon estimate  │
                        │ • Confidence       │
                        └────────────────────┘
```

---

## Configuration Hierarchy

```
settings.yaml
├── pipeline
│   └── run_phase        → config.run_phase
├── sam2
│   ├── model_name       → config.sam2_model
│   ├── device           → config.sam2_device
│   └── confidence_*     → config.get('sam2.confidence_threshold')
├── data
│   ├── images_dir       → config.images_dir
│   ├── masks_dir        → config.masks_dir
│   ├── training_*       → config.training_data_dir
│   └── ...
├── model
│   ├── type             → config.model_type
│   ├── learning_rate    → config.learning_rate
│   └── batch_size       → config.batch_size
├── carbon
│   ├── pixel_size_m     → config.pixel_size_m
│   └── carbon_density   → config.carbon_density_kg_ha
└── logging
    ├── log_file         → config.log_file
    └── log_level        → config.log_level
```

---

## Phase Execution Control

Control pipeline execution via `config/settings.yaml`:

```yaml
pipeline:
  run_phase: "all"          # Run all phases

# OR

pipeline:
  run_phase: "label"        # Only Phase 1: SAM-2

# OR

pipeline:
  run_phase: "prepare"      # Only Phase 2: Prepare data

# OR

pipeline:
  run_phase: "train"        # Only Phase 3: Training

# OR

pipeline:
  run_phase: "infer"        # Only Phase 4: Inference

# OR

pipeline:
  run_phase: "visualize"    # Only Phase 5: Viz
```

---

## Key Advantages

✅ **Flexible** - Run any phase independently  
✅ **Reproducible** - All settings in YAML  
✅ **Scalable** - Handle multiple tiles/scenes  
✅ **Debuggable** - Detailed logging per phase  
✅ **Extensible** - Easy to add new models/phases  
✅ **SAM-2 Ready** - Interactive annotation built-in  
✅ **UzmaSat Compatible** - Switch datasets seamlessly  

---

## Running Your First Test

```bash
# 1. Place test images in data/raw_images/
cp your_test.tif data/raw_images/

# 2. Edit config/settings.yaml
#    - Set pipeline.run_phase = "label"
#    - Verify data.images_dir points to raw_images/

# 3. Run pipeline
python src/main.py

# 4. Check results
# - Masks in data/masks/ (PNG files)
# - Logs in logs/pipeline.log
```

Done! Your pipeline is now **SAM-2 ready** and **production-grade**. 🚀

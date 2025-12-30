# Quick Reference Card

## 🚀 Quick Start (30 seconds)

```bash
# 1. Navigate to project
cd "c:\Users\aniq\OneDrive\Desktop\Mangrove Carbon Estimation\mangrove-carbon-pipeline"

# 2. Check structure
ls src/
# Should show: labeling/, data/, models/, satellite/, utils/, visualization/

# 3. Test config loading
python -c "from src.utils.config import Config; c = Config(); c.display()"

# 4. Run pipeline (when ready)
python src/main.py
```

---

## 📋 5 Phases at a Glance

| Phase | What It Does | Input | Output | Module |
|-------|-----------|-------|--------|--------|
| **1. Label** | SAM-2 segmentation | `.tif` images | PNG masks | `src/labeling/sam2_annotator.py` |
| **2. Prepare** | Normalize & split data | Images + masks | Training data | `src/data/preprocessor.py` |
| **3. Train** | Train deep learning model | Training data | Model checkpoint | `src/models/estimator.py` |
| **4. Infer** | Segment & calculate carbon | Validation images | Carbon estimates | `src/satellite/processor.py` |
| **5. Visualize** | Generate maps & plots | Predictions | PNG/PDF reports | `src/visualization/plotter.py` |

---

## ⚙️ Configuration Quick Reference

**File:** `config/settings.yaml`

```yaml
# Which phase to run
pipeline.run_phase: "all"  # 'label'|'prepare'|'train'|'infer'|'visualize'|'all'

# SAM-2 settings
sam2.device: "cuda"  # 'cuda' or 'cpu'

# Data paths
data.images_dir: "data/raw_images/"      # Your .tif files
data.masks_dir: "data/masks/"            # SAM-2 output
data.training_data_dir: "data/training/" # Training data

# Model choice
model.type: "unet"  # 'unet' or 'yolov8-seg'

# Carbon estimation
carbon.carbon_density_kg_ha: 150  # Mangroves: 100-200
carbon.pixel_size_m: 10           # Sentinel-2: 10m
```

---

## 📁 File Structure Map

```
mangrove-carbon-pipeline/
│
├── 🔧 Configuration
│   └── config/settings.yaml         ← Edit this to control pipeline
│
├── 📜 Documentation (Read These!)
│   ├── README.md                    ← Start here
│   ├── UPDATE_SUMMARY.md            ← What changed
│   ├── BEFORE_AFTER.md             ← Old vs new
│   ├── STRUCTURE_DIAGRAM.md        ← Visual structure
│   ├── IMPLEMENTATION_CHECKLIST.md ← What to code next
│   └── QUICK_REFERENCE.md          ← This file
│
├── 🐍 Main Pipeline
│   └── src/main.py                  ← Run: python src/main.py
│
├── Phase 1: Labeling
│   └── src/labeling/sam2_annotator.py
│
├── Phase 2-3: Data & Models
│   ├── src/data/loader.py          ← Load .tif files
│   ├── src/data/preprocessor.py    ← Normalize data
│   └── src/models/estimator.py     ← Train models
│
├── Phase 4-5: Carbon & Viz
│   ├── src/satellite/processor.py   ← Carbon calculation
│   └── src/visualization/plotter.py ← Make plots
│
└── Utilities
    └── src/utils/config.py          ← Config loader
```

---

## 🎯 Command Cheat Sheet

```bash
# Test configuration loads
python -c "from src.utils.config import Config; c = Config(); print(c.model_type)"

# Run all phases
python src/main.py

# Run only labeling
# Edit: config/settings.yaml → pipeline.run_phase: "label"
python src/main.py

# Run only training
# Edit: config/settings.yaml → pipeline.run_phase: "train"
python src/main.py

# Check logs
tail logs/pipeline.log

# View results
ls results/
```

---

## 🔍 Debugging Guide

| Problem | Check |
|---------|-------|
| "Config file not found" | Path to `config/settings.yaml` correct? |
| "No module named 'labeling'" | Did you create `src/labeling/__init__.py`? |
| Imports fail | Packages installed? `pip install -r requirements.txt` |
| SAM-2 not running | `config.sam2_device` set to 'cuda' with GPU? |
| Data not found | `config.images_dir` points to correct folder? |
| Phase not running | `config.run_phase` set correctly? |

---

## 📊 Data Format Reference

**Input:** `.tif` files (multi-band satellite images)
```
Shape: (Height, Width, Bands)
Example: (512, 512, 4)  # 4-band satellite image
Bands: RGB + NDVI or similar
```

**Masks from SAM-2:** PNG binary
```
Values: 0 (non-mangrove) or 1 (mangrove)
Shape: (Height, Width)
Format: PNG uint8
```

**Training Data:** Normalized images + masks
```
Images: normalized to [0, 1]
Masks: 0-255 or as tensor
Split: 80% train, 20% validation
```

**Carbon Output:** Dictionary
```python
{
    'area_ha': 1234.56,        # Mangrove area in hectares
    'carbon_stock_tC': 185000,  # Carbon in tonnes
    'confidence': 0.92          # Optional: confidence interval
}
```

---

## 🎓 Understanding Carbon Calculation

```python
# Formula used in Phase 4
area_m2 = mangrove_pixel_count × (pixel_size_m²)
area_ha = area_m2 / 10000

carbon_kg = area_ha × carbon_density_kg_ha
carbon_tC = carbon_kg / 1000

# Example with real numbers
# pixel_size_m = 10 (Sentinel-2)
# mangrove_pixels = 1,000,000
# carbon_density_kg_ha = 150,000 kg/ha = 150 tC/ha

area_m2 = 1,000,000 × (10²) = 100,000,000 m²
area_ha = 100,000,000 / 10,000 = 10,000 ha
carbon_tC = 10,000 × 150 = 1,500,000 tC
```

---

## ✅ Pre-Run Checklist

Before running `python src/main.py`:

- [ ] Data files placed in `data/raw_images/` as `.tif`
- [ ] Config file edited if needed (`config/settings.yaml`)
- [ ] Required Python packages installed
- [ ] Output directories exist or will be created
- [ ] Sufficient disk space for results
- [ ] GPU available if using CUDA

```bash
# Quick check
python -c "
from src.utils.config import Config
c = Config()
print(f'Images dir: {c.images_dir}')
print(f'Model type: {c.model_type}')
print(f'Phase to run: {c.run_phase}')
"
```

---

## 📚 Documentation Reading Order

1. **First time?** → Start with `README.md`
2. **What changed?** → Read `UPDATE_SUMMARY.md`
3. **How to implement?** → Check `IMPLEMENTATION_CHECKLIST.md`
4. **Understand structure?** → See `STRUCTURE_DIAGRAM.md`
5. **How different from before?** → Compare `BEFORE_AFTER.md`
6. **Need quick reference?** → You're reading it! (this file)

---

## 🚦 Status & Next Steps

### ✅ Complete
- Project structure
- Configuration system
- Pipeline skeleton
- SAM-2 module scaffold
- Documentation

### ⏳ To Implement
- Data loading (Rasterio)
- Preprocessing functions
- Model training logic
- Carbon calculation
- Visualization functions

### 🎯 Priority
1. `src/data/loader.py` - Cannot proceed without this
2. `src/data/preprocessor.py` - Prepare training data
3. `src/models/estimator.py` - Train model
4. Rest of modules

---

## 💾 Saving Your Configuration

```bash
# Save current config as backup
cp config/settings.yaml config/settings_backup.yaml

# Save after successful experiments
cp config/settings.yaml config/settings_palm_oil.yaml
cp config/settings.yaml config/settings_mangrove.yaml

# Restore from backup if needed
cp config/settings_backup.yaml config/settings.yaml
```

---

## 🔗 Key Concepts

| Term | Meaning | Where |
|------|---------|-------|
| **SAM-2** | Segment Anything Model 2 | Phase 1: Labeling |
| **Segmentation Mask** | Binary image (0 or 1 per pixel) | Phase 1 output |
| **U-Net** | Deep learning segmentation model | Phase 3 choice |
| **YOLOv8-seg** | Instance segmentation model | Phase 3 choice |
| **IoU** | Intersection over Union (metric) | Model training |
| **Carbon Density** | tC/ha from literature | Phase 4 input |
| **Georeferencing** | Map coordinates to pixels | Rasterio metadata |

---

## 🆘 Getting Help

**Within the project:**
- `README.md` - Usage & overview
- `IMPLEMENTATION_CHECKLIST.md` - Code examples
- `STRUCTURE_DIAGRAM.md` - Visual guides
- Docstrings in `.py` files - Function descriptions

**When stuck:**
1. Check `logs/pipeline.log`
2. Review relevant documentation file
3. Test with smaller dataset first
4. Add print statements to debug
5. Verify configuration is correct

---

## 📝 Important Notes

⚠️ **Phase Execution:**
- Phase 1 output (masks) → Phase 2 input
- Phase 2 output (training data) → Phase 3 input
- Phase 3 output (model) → Phase 4 input
- Phase 4 output (carbon) → Phase 5 input

⚠️ **Data Formats:**
- Always use `.tif` for input images
- Configuration expects specific directory structure
- Masks should be single-channel PNG (0/1 values)

⚠️ **Carbon Density:**
- Default: 150 tC/ha (mangroves)
- Adjust based on your literature review
- Document your source!

---

## 🎉 You're Ready!

This pipeline is now:
- ✅ **Structured** - Clear phases with specific purposes
- ✅ **Flexible** - Control via configuration
- ✅ **Complete** - Skeleton ready for implementation
- ✅ **Documented** - Multiple guides and reference docs
- ✅ **Ready for UzmaSat** - Just swap data directory

**Next: Implement data loading and start with palm oil testing!** 🚀

# Mangrove Carbon Estimation Pipeline

Estimates carbon stock in mangrove ecosystems using satellite imagery, SAM-2 segmentation, and deep learning models.

## Updated Workflow (SAM-2 + Pre-Preprocessed Data)

Since your satellite data is **already preprocessed** (atmospheric correction, georeferencing, etc.), the pipeline focuses on:

1. **Labeling** - Use SAM-2 for interactive segmentation of mangrove areas
2. **Data Preparation** - Convert masks to training format (segmentation or YOLO)
3. **Model Training** - Train U-Net or YOLOv8-seg for pixel-level classification
4. **Inference** - Generate mangrove segmentation masks
5. **Carbon Estimation** - Calculate area and estimate total carbon stock

## Project Structure

```
mangrove-carbon-pipeline/
├── src/
│   ├── main.py                     # Entry point (5-phase pipeline)
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── sam2_annotator.py       # SAM-2 interactive segmentation
│   ├── data/
│   │   ├── loader.py               # Load .tif images (rasterio)
│   │   └── preprocessor.py         # Normalize, augment, split data
│   ├── models/
│   │   ├── estimator.py            # U-Net or YOLOv8-seg models
│   │   └── inference.py            # Model prediction wrapper
│   ├── satellite/
│   │   └── processor.py            # Carbon stock calculation
│   ├── utils/
│   │   ├── config.py               # YAML configuration loader
│   │   └── logger.py               # Logging setup
│   └── visualization/
│       └── plotter.py              # Maps and visualizations
├── config/
│   └── settings.yaml               # Pipeline configuration
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_satellite.py
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use setup.py
pip install -e .
```

### 2. Configuration

Edit `config/settings.yaml` to set:
- Data directories (input images, output masks)
- SAM-2 model and device
- Model type (unet or yolov8-seg)
- Carbon density parameters

### 3. Run Pipeline

```bash
# Run all phases
python src/main.py

# Or run specific phase (edit config: pipeline.run_phase)
# - label: Use SAM-2 to segment mangrove areas
# - prepare: Prepare masks for training
# - train: Train the model
# - infer: Make predictions
# - visualize: Generate plots
```

## Pipeline Phases

### Phase 1: SAM-2 Annotation
- Loads preprocessed .tif files
- Uses SAM-2 for automatic segmentation with optional prompts
- Applies morphological refinement
- Saves PNG masks

### Phase 2: Data Preparation  
- Loads images and masks
- Normalizes pixel values
- Converts masks to training format (PNG or YOLO polygons)
- Splits into train/val sets

### Phase 3: Model Training
- Trains U-Net or YOLOv8-seg
- Monitors IoU, F1-score, accuracy
- Saves best checkpoint
- Supports early stopping

### Phase 4: Inference & Carbon Estimation
- Loads trained model
- Generates segmentation masks for validation set
- Calculates total mangrove area from pixel count
- Multiplies by carbon density to estimate total carbon stock
  ```
  Area (ha) = pixel_count × (pixel_size_m²) / 10000
  Carbon (tC) = Area (ha) × carbon_density (tC/ha)
  ```

### Phase 5: Visualization
- Overlays predictions on original images
- Generates area/carbon distribution maps
- Creates summary report

## Configuration Parameters

Key settings in `config/settings.yaml`:

```yaml
sam2:
  model_name: "facebook/sam2-hiera-large"  # SAM-2 from Hugging Face
  device: "cuda"  # or "cpu"

data:
  images_dir: "data/raw_images/"          # Input .tif files
  masks_dir: "data/masks/"                # SAM-2 output

model:
  type: "unet"                             # or "yolov8-seg"
  input_channels: 4                        # Satellite bands
  output_channels: 2                       # Mangrove vs non-mangrove
  learning_rate: 0.001
  batch_size: 16
  num_epochs: 50

carbon:
  pixel_size_m: 10                         # Resolution (m)
  carbon_density_kg_ha: 150                # Literature value
```

## Dependencies

- **PyTorch** - Deep learning framework
- **Rasterio** - Geospatial raster I/O
- **GeoPandas** - Vector data handling
- **Transformers** - SAM-2 model loading
- **OpenCV** - Image processing
- **NumPy, Pandas, Matplotlib** - Data & visualization

See `requirements.txt` for exact versions.

## Workflow for UzmaSat

When your mangrove UzmaSat dataset arrives:
1. Place .tif files in `data/raw_images/`
2. Update `config/settings.yaml` with correct parameters
3. Run `python src/main.py` - same pipeline works seamlessly!

## References

- [SAM-2](https://github.com/facebookresearch/segment-anything-2) - Segment Anything Model 2
- [U-Net](https://arxiv.org/abs/1505.04597) - Semantic Segmentation
- [YOLOv8-seg](https://github.com/ultralytics/ultralytics) - Instance Segmentation
- Mangrove carbon density: ~100-200 tC/ha (varies by region)

## Contributing

Contributions welcome! Submit issues or pull requests.

## License

MIT License - See LICENSE file
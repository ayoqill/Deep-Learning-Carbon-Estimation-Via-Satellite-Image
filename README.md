# Mangrove Carbon Mapper

**Deep Learning-Based Carbon Stock Estimation from Multispectral Satellite Imagery**

Mangrove Carbon Mapper is a Flask-based web application that detects mangrove areas from satellite imagery using a deep learning segmentation model, then estimates mangrove area, carbon stock, and CO₂ equivalent from the predicted mask.

This project was developed as a Final Year Project (FYP) prototype to support faster mangrove monitoring and carbon estimation using remote sensing and artificial intelligence.

---
<img width="1671" height="945" alt="Screenshot 2026-05-13 at 5 08 54 PM" src="https://github.com/user-attachments/assets/0d45fe8e-897f-4be6-a905-17030e9808ae" />

<img width="1671" height="945" alt="Screenshot 2026-05-13 at 5 09 25 PM" src="https://github.com/user-attachments/assets/b789c1dd-1842-44e2-b5b5-609b0a46c5d0" />

<img width="1640" height="938" alt="Screenshot 2026-05-13 at 5 09 44 PM" src="https://github.com/user-attachments/assets/35c95bec-0543-44d7-befa-b515fc3d708c" />

<img width="1644" height="950" alt="Screenshot 2026-05-13 at 5 10 16 PM" src="https://github.com/user-attachments/assets/9dc2c8f7-5a9a-49d0-94f2-f4129166e8e1" />


## Project Overview

Traditional mangrove carbon estimation often depends on manual field surveys and GIS-based analysis. These methods are useful, but they can be time-consuming when repeated across large coastal regions.

This system provides a web-based workflow where users can upload satellite images, run mangrove segmentation, and receive carbon-related estimates automatically.

The main flow is:

```text
Satellite Image Upload
        ↓
Image Preprocessing
        ↓
DeepLabV3+ Mangrove Segmentation
        ↓
Binary Mangrove Mask
        ↓
Pixel-to-Area Calculation
        ↓
Carbon Stock and CO₂ Equivalent Estimation
        ↓
Visualization and Result Dashboard
```

> Important: This system provides carbon stock estimation only. It does not perform official carbon credit certification, trading, or verification.

---

## Main Features

- Upload satellite imagery for mangrove analysis
- Supports GeoTIFF images, with PNG/JPG support for demo use
- Detects mangrove regions using a trained DeepLabV3+ model
- Generates segmentation mask and overlay visualization
- Calculates mangrove coverage, area, carbon stock, and CO₂ equivalent
- Provides a web interface for viewing results
- Includes analysis history and result dashboard
- Includes precomputed study area showcase for Langkawi, Kedah
- Supports deployment using Render
- Supports external model hosting using Hugging Face to avoid storing large model files in GitHub

---

## Current Model

The current deployed model is based on semantic segmentation.

| Item | Details |
|---|---|
| Task | Binary semantic segmentation |
| Target class | Mangrove |
| Model | DeepLabV3+ |
| Encoder | ResNet34 |
| Framework | PyTorch |
| Library | segmentation-models-pytorch |
| Output | Binary mangrove mask |
| Input | Satellite image / multispectral image |

SAM-2 was used earlier as a support tool for mask generation and annotation assistance. It is not the main runtime model in the current deployed web application.

---

## Model Performance

Latest reported model performance:

| Metric | Validation | Test |
|---|---:|---:|
| Dice Coefficient | 81.44% | 80.34% |
| IoU | 70.99% | 69.50% |

The Dice Coefficient is the main segmentation score used to evaluate how well the predicted mangrove mask overlaps with the ground truth mask. IoU is stricter because it measures the intersection area divided by the total combined area of prediction and ground truth.

---

## Carbon Estimation Formula

After segmentation, the system counts the number of predicted mangrove pixels and converts them into area.

```text
Mangrove Coverage (%) = (Mangrove Pixels / Total Pixels) × 100

Area (m²) = Mangrove Pixels × Pixel Area (m²)

Area (ha) = Area (m²) / 10,000

Carbon Stock (tC) = Area (ha) × Carbon Density (tC/ha)

CO₂ Equivalent (tCO₂e) = Carbon Stock (tC) × 3.67
```

For Sentinel-2 style 10 m imagery:

```text
1 pixel = 10 m × 10 m = 100 m²
```

The carbon density value should be based on a peer-reviewed mangrove carbon study. Update the value in the application configuration according to the source used in the FYP report.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Flask, Python |
| Deep Learning | PyTorch, segmentation-models-pytorch |
| Image Processing | OpenCV, NumPy, Pillow, Rasterio |
| Frontend | HTML, CSS, JavaScript |
| Database / History | SQLite |
| Deployment | Render |
| Model Hosting | Hugging Face |
| GIS Support | QGIS / GeoTIFF preprocessing |

---

## Project Structure

```text
mangrove-carbon-pipeline/
│
├── app.py                         # Main Flask application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files ignored by Git
│
├── models/
│   └── deeplabv3/
│       └── deeplabv3_best.pth     # Trained model checkpoint (not pushed to GitHub)
│
├── src/
│   ├── models/
│   │   └── inference.py           # Model loading and prediction logic
│   │
│   ├── utils/
│   │   ├── io.py                  # Image loading, RGB rendering, TIFF handling
│   │   ├── carbon.py              # Area and carbon calculation utilities
│   │   └── study_areas.py         # Precomputed Langkawi study area data
│   │
│   └── processing/
│       └── postprocess.py         # Mask thresholding and refinement
│
├── templates/
│   ├── index.html                 # Upload and result page
│   ├── analytics.html             # Analysis dashboard page
│   ├── insight.html               # Blue carbon / project information page
│   └── login.html                 # Login page, if enabled
│
├── static/
│   ├── css/
│   │   └── style.css              # Main styling
│   │
│   ├── js/
│   │   ├── main.js                # Upload and result handling
│   │   ├── analytics.js           # Dashboard and study area logic
│   │   └── auth.js                # Login/logout UI handling, if enabled
│   │
│   ├── uploads/                   # Uploaded images (runtime generated)
│   ├── results/                   # Masks, overlays, and result JSON files
│   └── assets/                    # Logos, maps, and static images
│
├── data/
│   └── study_areas/
│       └── langkawi/              # Precomputed Langkawi showcase data
│
└── instance/
    └── app.db                     # SQLite database, if enabled locally
```

Some file names may differ slightly depending on the latest implementation, but the structure above represents the intended project organization.

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/ayoqill/Deep-Learning-Carbon-Estimation-Via-Satellite-Image.git
cd Deep-Learning-Carbon-Estimation-Via-Satellite-Image
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the project root:

```env
SECRET_KEY=replace_with_your_secret_key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=replace_with_secure_password
MODEL_PATH=models/deeplabv3/deeplabv3_best.pth
MODEL_URL=https://huggingface.co/your-username/your-model-repo/resolve/main/deeplabv3_best.pth
CARBON_DENSITY_TC_PER_HA=replace_with_literature_value
PIXEL_SIZE_M=10
```

Do not commit `.env` to GitHub.

### 5. Run the application

```bash
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

---

## Model Weight Handling

The trained model file is large, so it should not be committed directly to GitHub.

Recommended approach:

1. Upload the model checkpoint to Hugging Face.
2. Store the model URL in an environment variable.
3. Let the Flask app download the model automatically if it does not exist locally.

Example model path:

```text
models/deeplabv3/deeplabv3_best.pth
```

Example Hugging Face direct file URL format:

```text
https://huggingface.co/<username>/<repo-name>/resolve/main/deeplabv3_best.pth
```

---

## Running Inference

When a user uploads an image, the backend performs these steps:

1. Validate uploaded file
2. Save uploaded image
3. Load image using the image utility function
4. Convert image into the model input format
5. Run tiled DeepLabV3+ prediction
6. Generate probability map
7. Apply threshold to create binary mask
8. Refine mask using post-processing
9. Calculate area and carbon estimates
10. Save result images and JSON output
11. Return result to the frontend

Example output files:

```text
static/results/<analysis_id>/pred_mask.png
static/results/<analysis_id>/overlay.png
static/results/<analysis_id>/step5_results.json
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Main upload page |
| `/upload` | POST | Upload image and run mangrove analysis |
| `/analytics` | GET | View analytics dashboard |
| `/insight` | GET | View blue carbon information page |
| `/api/analyses` | GET | Get saved analysis history |
| `/api/analyses/<id>` | DELETE | Delete selected analysis history item |

Endpoint names may be adjusted depending on the final Flask route names.

---

## Example Result Output

The system returns result values similar to:

```json
{
  "coveragePercent": 12.45,
  "areaM2": 52300,
  "areaHectares": 5.23,
  "carbonTons": 784.5,
  "carbonCO2": 2879.12,
  "maskUrl": "/static/results/example/pred_mask.png",
  "overlayUrl": "/static/results/example/overlay.png"
}
```

The actual values depend on the uploaded image, predicted mask, pixel size, and selected carbon density value.

---

## Study Area Showcase

The application includes a precomputed study area showcase for:

```text
Langkawi, Kedah, Malaysia
```

This section is designed for demonstration and presentation purposes. It allows users to view prepared results without running a new model prediction every time.

The study area dashboard may include:

- Before image
- After detection image
- Mangrove coverage
- Estimated area
- Estimated carbon stock
- CO₂ equivalent
- Notes and methodology

---

## Deployment on Render

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180
```

Recommended Render environment variables:

```env
SECRET_KEY=replace_with_your_secret_key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=replace_with_secure_password
MODEL_PATH=models/deeplabv3/deeplabv3_best.pth
MODEL_URL=https://huggingface.co/your-username/your-model-repo/resolve/main/deeplabv3_best.pth
CARBON_DENSITY_TC_PER_HA=replace_with_literature_value
PIXEL_SIZE_M=10
```

### Render Notes

Render free hosting has limited memory, so the app should be kept lightweight.

Recommended deployment settings:

- Use `--workers 1`
- Load the model only when needed
- Avoid committing large `.pth`, `.tif`, and result files
- Store model weights externally on Hugging Face
- Clean temporary upload/result files when necessary
- Use smaller image sizes for demo testing if memory errors occur

---

## Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/

# Environment variables
.env

# Model files
models/**/*.pth
models/**/*.pt
models/**/*.ckpt

# Uploaded and generated files
static/uploads/*
static/results/*

# Large geospatial files
*.tif
*.tiff
*.jp2
*.vrt

# Database / local runtime files
instance/*.db
*.sqlite3

# OS files
.DS_Store
```

Keep folder placeholders if needed:

```text
static/uploads/.gitkeep
static/results/.gitkeep
```

---

## Limitations

- The system estimates carbon stock only; it does not certify carbon credits.
- Accuracy depends on the quality, resolution, and preprocessing of the satellite image.
- PNG/JPG images may lose geospatial metadata, so GeoTIFF is preferred for accurate area calculation.
- Carbon stock is estimated using a fixed literature-based density value, not field-measured biomass.
- Model performance may change when tested on new locations due to domain shift.
- Render free tier may face memory limits when processing large images or loading deep learning models.

---

## Future Improvements

- Add more Malaysian study areas such as Sabah and Sarawak
- Improve model generalization using more diverse satellite datasets
- Compare DeepLabV3+ with U-Net++ and other segmentation models
- Add proper GeoTIFF metadata reading for automatic pixel size extraction
- Export results as GeoJSON, Shapefile, or georeferenced raster
- Add user-based analysis history with persistent cloud storage
- Add confidence map visualization
- Add more carbon density options based on verified regional studies

---

## Academic Scope

This project supports the following FYP objectives:

1. Develop a deep learning model for mangrove segmentation from satellite imagery.
2. Convert predicted mangrove masks into area measurements.
3. Estimate carbon stock and CO₂ equivalent from detected mangrove area.
4. Build a web-based prototype to make the workflow easier to access and demonstrate.

---

## Project Status

Current status:

```text
Model training: Completed
Web application: Completed prototype
Deployment: Render-ready
Model hosting: Hugging Face-supported
Study area showcase: Langkawi included
Carbon estimation: Implemented using literature-based density value
```

---

## Author

**Aqil Aiman**  
Final Year Project  
Bachelor of Computer Science  
International Islamic University Malaysia (IIUM)

---

## License

This project is prepared for academic and research demonstration purposes.

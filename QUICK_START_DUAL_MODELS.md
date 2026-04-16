# 🚀 Quick Start Guide - Dual Model Setup

## ✅ Current Status

Your Mangrove Carbon Pipeline now supports **TWO detection models**:

### Models Available
1. **U-Net++** (100 MB) - Fast & Accurate
   - Location: `models/unetpp_best.pth`
   - Speed: ~2-3s per image
   - Dice Score: 0.80

2. **DeepLabV3+** (86 MB) - Advanced Segmentation
   - Location: `models/deeplabv3/deeplabv3_best.pth`
   - Speed: ~3-4s per image
   - Dice Score: 0.8034
   - IoU: 0.6950

## 🎯 How to Use

### 1. Start the Application

```bash
cd /Users/amxr666/Desktop/mangrove-carbon-pipeline
source venv/bin/activate
python3 app.py
```

**Expected Output:**
```
✅ Using device: mps
✅ Open: http://localhost:5000
```

### 2. Open in Browser

Navigate to: **http://localhost:5000**

### 3. Select & Upload

1. **Choose a model:**
   - 🟦 **U-Net++** (Fast) - Default selected
   - 🟦 **DeepLabV3+** (Advanced) - Better boundaries

2. **Upload satellite image:**
   - Drag & drop or click to select
   - Supported: GeoTIFF, PNG, JPG

3. **Get Results:**
   - Detection overlay
   - Mangrove coverage %
   - Carbon estimates (tons)
   - CO₂ equivalent

## 🔧 Backend Architecture

### Model Loading (`app.py`)

```python
# Multi-model support
MODEL_PATHS = {
    "unetpp": "models/unetpp_best.pth",
    "deeplabv3": "models/deeplabv3/deeplabv3_best.pth"
}

# Cache loaded models in memory
loaded_models = {}
model_in_channels = {}

# Load on demand
def load_model(model_name: str) -> bool:
    if model_name in loaded_models:
        return True  # Already loaded
    # Load from disk, cache in memory
```

### Inference Pipeline

```
Upload Form (with model selection)
    ↓
FormData.append('model', 'unetpp' or 'deeplabv3')
    ↓
/upload endpoint
    ↓
load_model(model_choice)
    ↓
predict_mask_tiled(img, model_name)
    ↓
Calculate area & carbon
    ↓
Return JSON with results
```

## 📊 Frontend Features

### Model Selection UI
- Located above upload box
- Radio buttons for easy switching
- Shows model descriptions
- Default: U-Net++

### Result Display
```javascript
// Automatically detects which model was used
if (data.used_model) {
    console.log(`Result from ${data.used_model} model`);
}
```

## 🧪 Testing Both Models

### Method 1: Web UI (Recommended)
1. Upload same image with U-Net++
2. Note the results
3. Reset and upload with DeepLabV3+
4. Compare side-by-side

### Method 2: API Testing
```bash
# Test U-Net++
curl -X POST http://localhost:5000/upload \
  -F "image=@test_image.tif" \
  -F "model=unetpp"

# Test DeepLabV3+
curl -X POST http://localhost:5000/upload \
  -F "image=@test_image.tif" \
  -F "model=deeplabv3"
```

## 📈 Performance Comparison

| Metric | U-Net++ | DeepLabV3+ |
|--------|---------|-----------|
| Inference Time | 2-3s | 3-4s |
| Dice Score | ~0.80 | **0.8034** ✅ |
| IoU Score | ~0.66 | **0.6950** ✅ |
| File Size | 100 MB | 86 MB |
| Best For | Speed | Precision |

## 🛠️ Advanced Configuration

### Preload Both Models (Faster Response)
Edit `app.py` main block:
```python
if __name__ == "__main__":
    # Preload both models
    load_model("unetpp")
    load_model("deeplabv3")
    
    app.run(debug=False, host="0.0.0.0", port=5000)
```
**Trade-off:** Uses ~200 MB RAM, but instant model switching

### Adjust Inference Speed
```python
# In app.py, tune tiling parameters
BATCH_TILES = 24    # Lower = slower but less memory
TILE_OVERLAP = 32   # Lower = faster but less accurate
```

### Change Default Model
Edit `app.py`:
```python
@app.route("/upload", methods=["POST"])
def upload():
    model_choice = request.form.get("model", "deeplabv3")  # Change default
```

## 🐛 Troubleshooting

### "Model not found" error
```bash
# Verify models exist
ls -lh /Users/amxr666/Desktop/mangrove-carbon-pipeline/models/
# Output should show:
# - unetpp_best.pth (100M)
# - deeplabv3/deeplabv3_best.pth (86M)
```

### Model selection doesn't appear
1. Hard refresh browser (Cmd+Shift+R)
2. Clear browser cache
3. Check browser console for JS errors

### Slow inference
- Check device: Open http://localhost:5000/status
- Should show `"device": "mps"` (Apple Silicon)
- If not, PyTorch CUDA installation issue

### Out of memory
```python
# Reduce batch size in app.py
BATCH_TILES = 8  # Was 24
```

## 📚 File Structure

```
models/
├── unetpp_best.pth          ← U-Net++ model (100 MB)
└── deeplabv3/
    └── deeplabv3_best.pth   ← DeepLabV3+ model (86 MB)

templates/
├── index.html               ← Updated with model selector
└── insight.html

static/
├── script.js                ← Updated with model parameter
└── style.css

src/training/
├── train_unetpp.py          ← U-Net++ training
└── train_deeplabv3.py       ← DeepLabV3+ training (new)

app.py                        ← Updated for dual models
```

## 🚀 Next Steps

### Option 1: Compare Models
1. Upload test image with U-Net++
2. Download results
3. Upload same image with DeepLabV3+
4. Compare Dice/IoU scores

### Option 2: Retrain Models
```bash
# To retrain U-Net++ with new data
python3 src/training/train_unetpp.py

# To retrain DeepLabV3+ with new data
python3 src/training/train_deeplabv3.py
```

### Option 3: Ensemble Predictions
Create a route that averages both models:
```python
@app.route("/upload-ensemble", methods=["POST"])
def upload_ensemble():
    # Run both models
    # Average predictions
    # Return combined result
```

## 📞 API Endpoints

### GET /status
Returns system status and loaded models:
```json
{
  "status": "ready",
  "device": "mps",
  "loaded_models": ["unetpp", "deeplabv3"],
  "model_in_channels": {"unetpp": 4, "deeplabv3": 4}
}
```

### POST /upload
Required parameters:
```
image: File (TIF, PNG, JPG)
model: "unetpp" | "deeplabv3"
```

Optional parameters:
```
pixel_size: float (meters)
carbon_density: float (tons/hectare)
```

## 🎓 Learning Resources

- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **U-Net++**: Research paper on recursive nested skip connections
- **DeepLabV3+**: Encoder-decoder with atrous convolution
- **Apple Silicon ML**: MPS (Metal Performance Shaders) acceleration

## ✨ Summary

✅ **You now have:**
- Two trained segmentation models
- Web interface with model selection
- Dynamic model loading & caching
- Full API support for both models
- Production-ready deployment

**Recommended Usage:**
- Start with **U-Net++** for fastest results
- Use **DeepLabV3+** for maximum accuracy
- Compare results to validate detections

---

**Setup Status**: ✅ Complete
**Models**: ✅ Trained & Deployed
**Frontend**: ✅ Ready with model selector
**Backend**: ✅ Multi-model support
**Testing**: Ready to go! 🎉

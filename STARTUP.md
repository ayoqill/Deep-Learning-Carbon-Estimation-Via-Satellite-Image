# 🌍 Mangrove Carbon Detection - Dual Model System

## Quick Start (30 seconds)

```bash
# 1. Navigate to project
cd /Users/amxr666/Desktop/mangrove-carbon-pipeline

# 2. Activate environment
source venv/bin/activate

# 3. Start server
python3 app.py

# 4. Open browser
# → http://localhost:5000
```

## 🎯 What This Does

Upload a satellite image → **Select Model** → Get Carbon Estimation

### Available Models
- **U-Net++** 🚀 Fast (2-3s, Dice: 0.80)
- **DeepLabV3+** 🎯 Precise (3-4s, Dice: 0.8034)

## 📊 What You Get

```
Input: Satellite Image (TIF/PNG/JPG)
       ↓
Output: 
  • Detection overlay
  • Mangrove coverage %
  • Area (hectares & m²)
  • Carbon stock (tons)
  • CO₂ equivalent (tons)
```

## 🏗️ Architecture

```
Frontend (index.html)
   ├─ Model selector (radio buttons)
   └─ Image upload
   
   ↓ POST /upload (with model param)
   
Backend (app.py)
   ├─ load_model(model_name)
   ├─ predict_mask_tiled(image, model_name)
   └─ calculate carbon
   
   ↓ JSON response
   
Frontend (display results)
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend, model loading |
| `templates/index.html` | Web UI with model selector |
| `static/script.js` | Send model parameter |
| `models/unetpp_best.pth` | U-Net++ model |
| `models/deeplabv3/deeplabv3_best.pth` | DeepLabV3+ model |
| `src/training/train_*.py` | Training scripts |

## 🚀 Features

✅ Multiple detection models  
✅ Web UI with model selection  
✅ Real-time carbon estimation  
✅ GeoTIFF metadata support  
✅ Apple Silicon GPU acceleration  
✅ Model result caching  

## 🔧 Common Tasks

### Retrain DeepLabV3+
```bash
python3 src/training/train_deeplabv3.py
```

### Test API Directly
```bash
curl -X POST http://localhost:5000/upload \
  -F "image=@satellite_image.tif" \
  -F "model=deeplabv3"
```

### Check System Status
```bash
curl http://localhost:5000/status
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | `ls -lh models/*/` |
| Models not loading | Check PyTorch: `python3 -c "import torch"` |
| Port 5000 in use | `kill -9 $(lsof -t -i:5000)` |
| Out of memory | Reduce `BATCH_TILES` in app.py |

## 📚 Full Documentation

- **Training Details**: `DEEPLABV3_TRAINING_SUMMARY.md`
- **Setup Guide**: `QUICK_START_DUAL_MODELS.md`
- **Verification**: `IMPLEMENTATION_COMPLETE.md`

## 🎓 Model Info

### U-Net++
- Encoder: ResNet34
- Input: 4 channels (RGB+NIR)
- Output: 1 channel (Mangrove mask)
- Training: 30 epochs
- Best Dice: ~0.80

### DeepLabV3+
- Encoder: ResNet34
- Input: 4 channels (RGB+NIR)  
- Output: 1 channel (Mangrove mask)
- Training: 30 epochs
- Best Dice: **0.8034**
- Best IoU: **0.6950**

## 💡 Tips

1. **Compare Models**: Upload same image with both models
2. **Use GeoTIFF**: Better results with geo-referenced satellite data
3. **Check Device**: `/status` endpoint shows "mps" for Apple Silicon
4. **Monitor Performance**: Watch inference time increase with image size

## 📞 Support

**Framework**: PyTorch + Flask  
**Models**: segmentation-models-pytorch  
**Hardware**: Apple Silicon (MPS)  
**Status**: Production Ready ✅

---

**Happy detecting! 🌱**

For detailed information, see the comprehensive documentation files.

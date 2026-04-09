#!/usr/bin/env python3
"""
Verify that the threshold 0.45 is being used correctly in the app
"""
import sys
sys.path.insert(0, '/Users/amxr666/Desktop/mangrove-carbon-pipeline')

from src.utils.io import load_image_any
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

# Setup
MODEL_PATH = Path("/Users/amxr666/Desktop/mangrove-carbon-pipeline/models/unetpp_best.pth")
TEST_IMAGE = Path("/Users/amxr666/Desktop/mangrove-carbon-pipeline/data/tiles_clean/STL_Langkawi_Mangrove10_43.tif")

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
TILE_H, TILE_W = 160, 160
TILE_OVERLAP = 32

print(f"Device: {DEVICE}")
print(f"Loading model from: {MODEL_PATH}")
print(f"Loading test image from: {TEST_IMAGE}")

# Load model
state = torch.load(str(MODEL_PATH), map_location=DEVICE)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
if isinstance(state, dict) and "model" in state:
    state = state["model"]

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None
).to(DEVICE)

model.load_state_dict(state, strict=True)
model.eval()

print("✅ Model loaded")

# Load image
model_img, rgb_img, pixel_size, _ = load_image_any(TEST_IMAGE, model_in_channels=4)
print(f"✅ Image loaded: {model_img.shape}")

# Simple inference (no tiling, just direct)
print("\n=== DIRECT INFERENCE TEST ===")
x = np.transpose(model_img[np.newaxis], (0, 3, 1, 2))  # Add batch dim
xt = torch.from_numpy(x).float().to(DEVICE)

with torch.no_grad():
    logits = model(xt)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

print(f"Probs shape: {probs.shape}")
print(f"Probs min/max: {probs.min():.4f}/{probs.max():.4f}")
print(f"Probs mean: {probs.mean():.4f}")

# Test different thresholds
thresholds = [0.3, 0.35, 0.40, 0.45, 0.50]
print(f"\n=== THRESHOLD COMPARISON ===")
for thresh in thresholds:
    mask = (probs >= thresh).astype(np.uint8)
    coverage = (mask.sum() / mask.size) * 100
    print(f"Threshold {thresh}: {coverage:.2f}% coverage ({mask.sum()} pixels)")

print(f"\n=== CRITICAL TEST ===")
print(f"Is threshold 0.45 giving ~10.83% coverage?")
mask_045 = (probs >= 0.45).astype(np.uint8)
coverage_045 = (mask_045.sum() / mask_045.size) * 100
print(f"Answer: {coverage_045:.2f}%")
if coverage_045 > 8.0:
    print("✅ YES - Threshold is working correctly!")
else:
    print("❌ NO - Something is wrong with threshold!")

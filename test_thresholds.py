"""
QUICK FIX #1: Test different thresholds

This script runs inference with multiple thresholds
and saves results for comparison.
"""

import os
import torch
import numpy as np
import rasterio
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp

# --------
# Settings
# --------
MODEL_PATH = "models/unetpp_best.pth"
INPUT_TILE = "data/tiles_clean/STL_Langkawi_Mangrove10_43.tif"
OUT_DIR = Path("results/threshold_tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Thresholds to test
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Load model
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded")

# Read & prep image
with rasterio.open(INPUT_TILE) as src:
    arr = src.read()
    H, W = src.height, src.width

img4 = np.transpose(arr, (1, 2, 0)).astype(np.float32)

# Normalize
out = np.zeros_like(img4)
for c in range(4):
    ch = img4[:, :, c]
    lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
    if hi - lo > 1e-6:
        out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
img4 = out

# Pad
pad_h = max(160 - H, 0)
pad_w = max(160 - W, 0)
if pad_h > 0 or pad_w > 0:
    img4_pad = np.pad(img4, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
else:
    img4_pad = img4

# Infer
x = torch.from_numpy(np.transpose(img4_pad, (2, 0, 1))[None]).float().to(DEVICE)
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

probs = probs[:H, :W]
print(f"✅ Inference done. Image size: {H}×{W}")
print(f"   Prob range: {probs.min():.3f} - {probs.max():.3f}, mean: {probs.mean():.3f}")

# Test each threshold
print("\n" + "="*60)
print("THRESHOLD TEST RESULTS")
print("="*60)

results = {}
for thresh in THRESHOLDS:
    mask = (probs >= thresh).astype(np.uint8)
    coverage = mask.mean() * 100
    pixels = mask.sum()
    
    results[thresh] = {"coverage": coverage, "pixels": pixels}
    
    # Save mask
    mask_path = OUT_DIR / f"mask_thresh_{thresh:.2f}.png"
    cv2.imwrite(str(mask_path), mask * 255)
    
    print(f"Threshold {thresh:.2f}: {coverage:6.2f}% coverage ({pixels:8d} pixels) → saved to {mask_path.name}")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

# Find best
current_cov = results[0.50]["coverage"]
best_thresh = max(results.keys(), key=lambda t: abs(results[t]["coverage"] - current_cov * 1.2))

print(f"\nCurrent setting (0.50): {current_cov:.2f}% coverage")
print(f"\nTo increase coverage:")
for thresh in sorted(THRESHOLDS):
    if thresh < 0.50:
        gain = results[thresh]["coverage"] - current_cov
        if gain > 0.5:
            print(f"  Try {thresh:.2f} → +{gain:.2f}% (to {results[thresh]['coverage']:.2f}%)")

print("\n💡 TIP: Edit predict_tile.py line ~25:")
print("   THRESH = 0.50  → THRESH = 0.35")
print("\n   Then re-run: python3 src/inference/predict_tile.py")

print(f"\n✅ All masks saved to: {OUT_DIR}")

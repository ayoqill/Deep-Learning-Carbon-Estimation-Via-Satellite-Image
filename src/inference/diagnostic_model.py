"""
DIAGNOSTIC: Why your model misses mangrove areas

Key hypothesis based on your screenshot:
1. Model outputs LOW probabilities for true mangrove areas
2. Threshold of 0.5 is too HIGH → misses areas with probs 0.3-0.5
3. Post-processing (morphology + small object removal) removes valid edges

This script tests what works better WITHOUT retraining.
"""

import torch
import numpy as np
import rasterio
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp

MODEL_PATH = "models/unetpp_best.pth"
INPUT_TILE = "data/tiles_clean/STL_Langkawi_Mangrove10_43.tif"
OUT_DIR = Path("results/prediction_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Read image
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
H_orig, W_orig = H, W
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
probs = probs[:H_orig, :W_orig]

# Statistics
print("\n" + "="*60)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*60)
print(f"Min prob:      {probs.min():.4f}")
print(f"Max prob:      {probs.max():.4f}")
print(f"Mean prob:     {probs.mean():.4f}")
print(f"Median prob:   {np.median(probs):.4f}")
print(f"Std dev:       {probs.std():.4f}")

# Percentiles
percs = [10, 25, 50, 75, 90, 95, 99]
print("\nPercentiles:")
for p in percs:
    val = np.percentile(probs, p)
    print(f"  {p}%:  {val:.4f}")

# Coverage at different thresholds
print("\n" + "="*60)
print("COVERAGE AT DIFFERENT THRESHOLDS")
print("="*60)
thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
for thresh in thresholds:
    coverage = (probs >= thresh).mean() * 100
    pixels = (probs >= thresh).sum()
    print(f"Thresh {thresh}: {coverage:6.2f}% ({pixels:8d} pixels)")

# Key insight
print("\n" + "="*60)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*60)

# Find where probabilities are concentrated
under_03 = (probs < 0.3).mean() * 100
under_04 = (probs < 0.4).mean() * 100
under_05 = (probs < 0.5).mean() * 100

print(f"\n1. PROBABILITY DISTRIBUTION:")
print(f"   - {under_03:.1f}% of pixels have prob < 0.3 (likely non-mangrove)")
print(f"   - {under_04:.1f}% of pixels have prob < 0.4")
print(f"   - {under_05:.1f}% of pixels have prob < 0.5")

if under_05 > 85:
    print(f"\n   ⚠️  FINDING: {under_05:.1f}% of image is LOW confidence")
    print("   → Model is very conservative (mostly predicting ~0 or binary)")
else:
    print(f"\n   ℹ️  Model has moderate confidence distribution")

# Look at mangrove probability range
mangrove_like = probs[(probs > 0.3) & (probs < 0.7)]
if len(mangrove_like) > 0:
    print(f"\n2. TRANSITIONAL ZONE (0.3 < prob < 0.7):")
    print(f"   - {len(mangrove_like)} pixels in this zone")
    print(f"   - Mean: {mangrove_like.mean():.3f}")
    print(f"   - THIS is where model is uncertain about mangrove boundaries")
else:
    print(f"\n2. TRANSITIONAL ZONE: Almost no uncertainty (binary output)")

# Recommendations
print(f"\n3. RECOMMENDED ACTIONS (in order of ease):\n")

rec_num = 1

# Recommendation 1: Lower threshold
cov_30 = (probs >= 0.30).mean() * 100
cov_40 = (probs >= 0.40).mean() * 100
cov_50 = (probs >= 0.50).mean() * 100

delta_40_50 = cov_40 - cov_50
if delta_40_50 > 2:
    print(f"   [{rec_num}] LOWER THRESHOLD (easiest fix)")
    print(f"       Current: 0.5 → {cov_50:.2f}% coverage")
    print(f"       Try 0.4 → {cov_40:.2f}% coverage (+{delta_40_50:.2f}%)")
    print(f"       Try 0.3 → {cov_30:.2f}% coverage")
    print(f"       → EDIT: predict_tile.py line ~25: THRESH = 0.4")
    rec_num += 1
else:
    print(f"   [{rec_num}] Threshold adjustment gives small gains")
    print(f"       (0.3-0.5 all ~same coverage) → model is binary")
    rec_num += 1

print(f"\n   [{rec_num}] REDUCE POST-PROCESSING AGGRESSIVENESS")
print(f"       Edit: predict_tile.py or step4b_*.py")
print(f"       - Increase MIN_AREA_PX from 120 → 50 (removes fewer small regions)")
print(f"       - Reduce dilation kernel from (7,7) → (5,5)")
print(f"       - Reduce dilation iterations from 2 → 1")
print(f"       - OR: disable dilation entirely")
rec_num += 1

print(f"\n   [{rec_num}] CHECK TRAINING DATA (Medium effort)")
print(f"       - Were SAM-2 masks too conservative (excluded valid mangrove)?")
print(f"       - Was NDVI/green filter removing true mangrove in refinement?")
print(f"       - Check: masks_refined/ vs masks_raw/")
rec_num += 1

print(f"\n   [{rec_num}] RETRAIN with data augmentation (Hard, last option)")
print(f"       - Add rotation, zoom, brightness augmentation")
print(f"       - Increase training mask diversity")
print(f"       - Use focal loss to focus on mangrove boundaries")
rec_num += 1

print(f"\n" + "="*60)
print("NEXT STEP: Try threshold 0.3-0.4 + reduced post-processing")
print("="*60)

# Save to file
with open(OUT_DIR / 'diagnostic_report.txt', 'w') as f:
    f.write("MANGROVE SEGMENTATION DIAGNOSTIC REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Tile: {INPUT_TILE}\n")
    f.write(f"Size: {H_orig} x {W_orig}\n\n")
    f.write(f"Probability min: {probs.min():.4f}\n")
    f.write(f"Probability max: {probs.max():.4f}\n")
    f.write(f"Probability mean: {probs.mean():.4f}\n")
    f.write(f"Coverage at 0.3: {cov_30:.2f}%\n")
    f.write(f"Coverage at 0.4: {cov_40:.2f}%\n")
    f.write(f"Coverage at 0.5: {cov_50:.2f}%\n")

print(f"\n✅ Report saved to: {OUT_DIR / 'diagnostic_report.txt'}")

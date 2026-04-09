"""
Analyze prediction gaps: why the model misses mangrove areas.

This script:
1. Compares raw model output (logits/probs) vs thresholded output
2. Shows histogram of prediction confidence
3. Tests different thresholds
4. Analyzes post-processing impact
5. Recommends threshold + post-processing tuning
"""

import json
import torch
import numpy as np
import rasterio
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
MODEL_PATH = "models/unetpp_best.pth"
INPUT_TILE = "data/tiles_clean/STL_Langkawi_Mangrove10_43.tif"
OUT_DIR = Path("results/prediction_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAD_TO = 160
DEFAULT_THRESH = 0.5  # Current threshold

# Apple Silicon / CUDA / CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)


# -------------------------
# Utils
# -------------------------
def read_tif_4band(path):
    with rasterio.open(path) as src:
        arr = src.read()  # (C,H,W)
        H, W = src.height, src.width
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # (H,W,C)
    return img, (H, W)


def normalize_per_channel(img):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi - lo < 1e-6:
            out[:, :, c] = 0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def pad_to_size(img, target=PAD_TO):
    H, W = img.shape[:2]
    pad_h = max(target - H, 0)
    pad_w = max(target - W, 0)
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return img_pad, (pad_h, pad_w)


def clean_mask_default(mask_255, min_area=120):
    """Default post-processing from your pipeline."""
    mask_bin = (mask_255 > 127).astype(np.uint8)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    cleaned = np.zeros_like(mask_bin)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    
    # Morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)
    
    # Dilate
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.dilate(cleaned, k2, iterations=2)
    
    # Fill holes
    h, w = cleaned.shape
    flood = cleaned.copy()
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0,0), 1)
    holes = (flood == 0).astype(np.uint8)
    filled = (cleaned | holes)
    
    return filled.astype(np.uint8) * 255


# -------------------------
# Load model
# -------------------------
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")


# -------------------------
# Inference: get RAW probabilities
# -------------------------
img4, (H, W) = read_tif_4band(INPUT_TILE)
img4 = normalize_per_channel(img4)
img4_pad, (pad_h, pad_w) = pad_to_size(img4)

x = torch.from_numpy(np.transpose(img4_pad, (2, 0, 1))[None]).float().to(DEVICE)

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

# Crop back to original size
probs = probs[:H, :W]

print(f"✅ Inference done. Prob shape: {probs.shape}")
print(f"   Min: {probs.min():.3f}, Max: {probs.max():.3f}, Mean: {probs.mean():.3f}")


# -------------------------
# Analysis 1: Histogram of probabilities
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Raw probability distribution
ax = axes[0, 0]
ax.hist(probs.flatten(), bins=100, edgecolor='k', alpha=0.7)
ax.axvline(DEFAULT_THRESH, color='r', linestyle='--', linewidth=2, label=f'Default threshold ({DEFAULT_THRESH})')
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Frequency')
ax.set_title('Raw Model Output Distribution')
ax.legend()

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_stats = {}

for thresh in thresholds:
    mask = (probs >= thresh).astype(np.uint8)
    coverage = mask.sum() / mask.size * 100
    threshold_stats[thresh] = coverage
    print(f"  Threshold {thresh}: {coverage:.2f}% coverage")

ax = axes[0, 1]
ax.plot(list(threshold_stats.keys()), list(threshold_stats.values()), 'o-', linewidth=2, markersize=8)
ax.axvline(DEFAULT_THRESH, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Threshold')
ax.set_ylabel('Mangrove Coverage (%)')
ax.set_title('Coverage vs Threshold')
ax.grid(True, alpha=0.3)

# Analysis 2: Post-processing impact
ax = axes[1, 0]
masks_by_thresh = {}
for thresh in thresholds:
    mask_raw = (probs >= thresh).astype(np.uint8) * 255
    mask_cleaned = clean_mask_default(mask_raw)
    cov_raw = mask_raw.sum() / mask_raw.size * 100
    cov_clean = mask_cleaned.sum() / mask_cleaned.size * 100
    masks_by_thresh[thresh] = (mask_raw, mask_cleaned)
    print(f"  Threshold {thresh}: raw={cov_raw:.2f}%, cleaned={cov_clean:.2f}% (delta={cov_clean-cov_raw:+.2f}%)")

# Show one threshold's impact
thresh_demo = 0.5
mask_raw, mask_cleaned = masks_by_thresh[thresh_demo]
ax.imshow(mask_raw, cmap='gray')
ax.set_title(f'Raw Mask (thresh={thresh_demo})')
ax.axis('off')

ax = axes[1, 1]
ax.imshow(mask_cleaned, cmap='gray')
ax.set_title(f'Cleaned Mask (thresh={thresh_demo})')
ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_thresholds.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Analysis plot saved to {OUT_DIR / 'analysis_thresholds.png'}")


# -------------------------
# Analysis 3: Probability heatmap (sparse sample for visualization)
# -------------------------
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(probs, cmap='RdYlGn', vmin=0, vmax=1)
ax.set_title('Raw Model Probability Map')
plt.colorbar(im, ax=ax, label='Probability')
plt.savefig(OUT_DIR / 'probability_heatmap.png', dpi=150, bbox_inches='tight')
print(f"✅ Heatmap saved to {OUT_DIR / 'probability_heatmap.png'}")
plt.close()


# -------------------------
# Recommendations
# -------------------------
recommendations = {
    "current_coverage_at_0.5": threshold_stats[0.5],
    "coverage_at_0.3": threshold_stats[0.3],
    "coverage_at_0.4": threshold_stats[0.4],
    "coverage_at_0.6": threshold_stats[0.6],
    "coverage_at_0.7": threshold_stats[0.7],
    "probability_min": float(probs.min()),
    "probability_max": float(probs.max()),
    "probability_mean": float(probs.mean()),
    "probability_median": float(np.median(probs)),
    "probability_std": float(probs.std()),
    "recommendations": [
        "1. If current coverage is too LOW (< actual mangrove), try lowering threshold to 0.3-0.4",
        "2. If post-processing removes too much, reduce MIN_AREA_PX or disable dilation",
        "3. If certain areas are consistently missed, check if NDVI/green filter was too strict in training",
        "4. Consider using a probability-weighted mask: mask = (probs > 0.3) OR (probs > 0.5 AND area > 50px)",
    ]
}

print("\n" + "="*60)
print("RECOMMENDATIONS FOR IMPROVING COVERAGE")
print("="*60)
for rec in recommendations["recommendations"]:
    print(rec)

with open(OUT_DIR / 'recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

print(f"\n✅ Full analysis saved to {OUT_DIR / 'recommendations.json'}")

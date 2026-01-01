import json
import torch
import numpy as np
import rasterio
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp

# -------------------------
# Config
# -------------------------
MODEL_PATH   = "models/unetpp_best.pth"
INPUT_TILE   = "data/tiles_clean/STL_Langkawi_Mangrove10_43.tif"

OUT_DIR      = Path("results/predict_tile")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MASK_PNG    = OUT_DIR / "pred_mask.png"
OUT_OVERLAY_PNG = OUT_DIR / "overlay.png"
OUT_JSON        = OUT_DIR / "pred_stats.json"

PAD_TO = 160  # keep if your training padded to 160

# Threshold
THRESH = 0.5

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
        profile = src.profile
        H, W = src.height, src.width
        res = src.res  # (xres, yres)
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # (H,W,C)
    return img, profile, (H, W), res

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
    """Pad only if smaller than target. If bigger, keep as-is."""
    H, W = img.shape[:2]
    pad_h = max(target - H, 0)
    pad_w = max(target - W, 0)
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return img_pad, (pad_h, pad_w)

def save_overlay(img_rgb01, mask01, out_path, alpha=0.45):
    """Filled red overlay."""
    img = (img_rgb01 * 255).astype(np.uint8).copy()
    overlay = img.copy()
    overlay[mask01 == 1] = [255, 0, 0]  # red in RGB
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

def tif_to_rgb_for_display(img_4band_01):
    """
    Create RGB (R,G,B) for overlay display.
    Assumes bands are (B,G,R,NIR) typical.
    """
    if img_4band_01.shape[2] < 3:
        raise ValueError("Need at least 3 bands")
    B = img_4band_01[:, :, 0]
    G = img_4band_01[:, :, 1]
    R = img_4band_01[:, :, 2]
    rgb = np.stack([R, G, B], axis=-1)
    return rgb


# -------------------------
# Load model (must match training)
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


# -------------------------
# Inference
# -------------------------
img4, profile, (H, W), res = read_tif_4band(INPUT_TILE)
img4 = normalize_per_channel(img4)

# pad to training size (if needed)
img4_pad, (pad_h, pad_w) = pad_to_size(img4)

# (H,W,C) → (1,C,H,W)
x = torch.from_numpy(np.transpose(img4_pad, (2, 0, 1))[None]).float().to(DEVICE)

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

# Crop back to original size (remove padding)
probs = probs[:H, :W]

# Threshold
mask01 = (probs >= THRESH).astype(np.uint8)  # 0/1
mask255 = (mask01 * 255).astype(np.uint8)

# Save mask
cv2.imwrite(str(OUT_MASK_PNG), mask255)
print("Saved predicted mask:", OUT_MASK_PNG)

# Save overlay (RGB display from 4-band)
rgb = tif_to_rgb_for_display(img4)  # already 0..1
save_overlay(rgb, mask01, OUT_OVERLAY_PNG)
print("Saved overlay:", OUT_OVERLAY_PNG)

# Simple stats (optional)
pixel_size_m = float(res[0]) if res and res[0] else None
stats = {
    "input_tile": str(INPUT_TILE),
    "shape": [H, W],
    "bands": int(img4.shape[2]),
    "device": DEVICE,
    "threshold": THRESH,
    "mangrove_pixels": int(mask01.sum()),
    "total_pixels": int(mask01.size),
    "coverage_percent": float(mask01.sum() / max(mask01.size, 1) * 100.0),
    "pixel_size_m": pixel_size_m
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print("Saved stats:", OUT_JSON)

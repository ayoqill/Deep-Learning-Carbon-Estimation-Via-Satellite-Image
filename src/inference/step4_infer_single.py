import json
import numpy as np
import torch
import cv2
import rasterio
from pathlib import Path
import segmentation_models_pytorch as smp

# ✅ Import Step 5 (analytics)
from src.analytics.step5_area_carbon import calculate_area_and_carbon, get_pixel_size_m

# ====== CONFIG ======
MODEL_PATH = "models/unetpp_best.pth"
OUT_DIR = "results"

DEFAULT_PIXEL_SIZE_M = 0.7
CARBON_DENSITY_TON_PER_HA = 150.0  # ⚠️ set based on literature

# Apple Silicon GPU (Mac M1/M2/M3)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Training input size
TILE_H, TILE_W = 157, 157

# Must match training
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None

# ✅ IMPORTANT: match your trained model
IN_CHANNELS = 4   # <-- because your predict_tile.py uses 4 bands


# ====== HELPERS ======
def normalize_per_channel(img: np.ndarray) -> np.ndarray:
    """Robust per-channel percentile normalization to [0..1]."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi - lo < 1e-6:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def load_image_any(path: str):
    """
    Loads:
    - PNG/JPG -> returns RGB(3) then adds dummy band to become 4
    - TIF/TIFF -> reads all bands, keeps first 4 bands if available

    Returns:
      img01: (H, W, C) float32 in [0..1]
      meta: dict with pixel_size_m if known
    """
    p = Path(path)
    ext = p.suffix.lower()

    # ---- PNG/JPG ----
    if ext in [".png", ".jpg", ".jpeg"]:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if IN_CHANNELS == 4:
            dummy = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)  # fake NIR
            img = np.concatenate([rgb, dummy], axis=2)
        else:
            img = rgb

        return img, {"source": "opencv", "pixel_size_m": None}

    # ---- TIFF ----
    if ext in [".tif", ".tiff"]:
        with rasterio.open(str(p)) as src:
            arr = src.read()  # (bands, H, W)
            res = getattr(src, "res", None)

        bands, H, W = arr.shape
        if bands < IN_CHANNELS:
            raise ValueError(f"TIFF has only {bands} bands but model expects {IN_CHANNELS} bands.")

        # take first IN_CHANNELS bands
        img = np.transpose(arr[:IN_CHANNELS], (1, 2, 0)).astype(np.float32)
        img = normalize_per_channel(img)

        pixel_size_m = float(res[0]) if res and res[0] else None
        return img, {"source": "rasterio", "pixel_size_m": pixel_size_m}

    raise ValueError(f"Unsupported image extension: {ext}")


def resize_to_training_size(img01: np.ndarray):
    return cv2.resize(img01, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=1,
        activation=None
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def predict_mask(model, img01: np.ndarray) -> np.ndarray:
    """
    img01: (H, W, C) float32 [0..1]
    Returns mask01: (H, W) uint8 0/1
    """
    x = np.transpose(img01, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    return (probs >= 0.5).astype(np.uint8)


def save_mask_png(mask01: np.ndarray, out_path: str):
    out = (mask01 * 255).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, out)


def save_overlay_png(img01: np.ndarray, mask01: np.ndarray, out_path: str, alpha: float = 0.45):
    """
    Red filled overlay.
    If 4-band, uses first 3 channels for display.
    """
    rgb = img01[:, :, :3]
    img = (rgb * 255).astype(np.uint8).copy()
    overlay = img.copy()
    overlay[mask01 == 1] = [255, 0, 0]  # red

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 4 (single inference) + Step 5 (area & carbon).")
    parser.add_argument("--image", required=True, help="User uploaded image (.png/.jpg/.tif)")
    parser.add_argument("--out-name", default="run1", help="Output folder name inside results/")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in meters (needed for PNG/JPG). Example: 0.7")
    parser.add_argument("--carbon-density", type=float, default=CARBON_DENSITY_TON_PER_HA,
                        help="Carbon density (tons/ha) from literature.")
    args = parser.parse_args()

    out_dir = Path(OUT_DIR) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original image
    img_orig, meta = load_image_any(args.image)
    H0, W0 = img_orig.shape[:2]

    # Decide pixel size
    pixel_size_m = meta.get("pixel_size_m", None)
    if pixel_size_m is None:
        pixel_size_m = args.pixel_size if args.pixel_size is not None else DEFAULT_PIXEL_SIZE_M

    # Build model
    model = build_model()

    # Resize to training size → predict → resize mask back
    img_small = resize_to_training_size(img_orig)
    mask_small = predict_mask(model, img_small)
    mask01 = cv2.resize(mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # Save mask + overlay
    mask_path = str(out_dir / "pred_mask.png")
    overlay_path = str(out_dir / "overlay.png")
    save_mask_png(mask01, mask_path)
    save_overlay_png(img_orig, mask01, overlay_path)

    # ✅ Step 5 from analytics module (no duplication here)
    results = calculate_area_and_carbon(
        mask01,
        pixel_size_m=pixel_size_m,
        carbon_density_ton_per_ha=args.carbon_density
    )

    # Save JSON
    json_path = out_dir / "step5_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved outputs:")
    print(f"  Mask:    {mask_path}")
    print(f"  Overlay: {overlay_path}")
    print(f"  JSON:    {json_path}")
    print("\n=== STEP 5 RESULTS ===")
    print(results)


if __name__ == "__main__":
    main()
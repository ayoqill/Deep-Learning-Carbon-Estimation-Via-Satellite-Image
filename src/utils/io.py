# utils/io.py
# I/O helpers for your CURRENT pipeline:
# - Accept PNG/JPG/TIF
# - Keep raw band order for model prediction
# - Use reordered + enhanced RGB only for web display
# - Produce: pred_mask.png, overlay.png, step5_results.json

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import json
import numpy as np
import cv2
import rasterio


# -----------------------------
# Basic helpers
# -----------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()


# -----------------------------
# Normalization for model/display base
# -----------------------------
def normalize_per_channel_percentile(img: np.ndarray) -> np.ndarray:
    """
    img: (H,W,C) float32/uint16/etc
    return: (H,W,C) float32 in [0..1]

    This is safe for model input normalization.
    Do NOT add visual saturation/brightness here because model_img uses this too.
    """
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)

        if hi - lo < 1e-6:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)

    return out


# -----------------------------
# Display-only enhancement
# -----------------------------
def enhance_rgb_for_display(rgb01: np.ndarray) -> np.ndarray:
    """
    Make RGB preview more colourful for the web app.

    Important:
    - Display only.
    - Must NOT be used for model prediction.
    - Helps TOA satellite preview avoid dull/purple look.
    """
    img = np.asarray(rgb01).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)

    # Brighten midtones slightly using gamma correction.
    img = np.power(img, 0.82)

    # Increase contrast around the middle point.
    img = (img - 0.5) * 1.22 + 0.5
    img = np.clip(img, 0.0, 1.0)

    # Reduce purple cast slightly and make water/vegetation more visible.
    # RGB channel order: 0=Red, 1=Green, 2=Blue
    img[:, :, 0] *= 0.77
    img[:, :, 1] *= 1.00
    img[:, :, 2] *= 0.88
    img = np.clip(img, 0.0, 1.0)

    # Boost saturation using HSV for a more colourful preview.
    img_u8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[:, :, 1] *= 1.65  # saturation
    hsv[:, :, 2] *= 1.08  # brightness/value

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out.astype(np.float32) / 255.0


# -----------------------------
# Image loading (PNG/JPG/TIF)
# -----------------------------
def load_image_any(
    path: Path,
    model_in_channels: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], str]:
    """
    Returns:
      model_img: (H,W,Cm) float32 [0..1] -> channels match model_in_channels
      rgb_img:   (H,W,3)  float32 [0..1] -> display-only enhanced RGB preview
      pixel_size_m: float or None from GeoTIFF
      pixel_size_source: "from_tif" | "none"

    Correct TOA GeoTIFF approach:
      - model_img keeps original Band 1, Band 2, Band 3, Band 4 order.
      - rgb_img uses RGB display order: Red=Band 3, Green=Band 2, Blue=Band 1.
    """
    ext = path.suffix.lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if model_in_channels == 3:
            model_img = rgb
        elif model_in_channels == 4:
            zero = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
            model_img = np.concatenate([rgb, zero], axis=2)
        else:
            raise ValueError(f"Unsupported model_in_channels={model_in_channels}")

        rgb_img = enhance_rgb_for_display(rgb)
        return model_img, rgb_img, None, "none"

    if ext in [".tif", ".tiff"]:
        with rasterio.open(str(path)) as src:
            arr = src.read()  # (bands, H, W)
            pixel_size_m = float(abs(src.transform.a)) if src.transform is not None else None

        pixel_source = "from_tif" if pixel_size_m is not None else "none"

        bands, H, W = arr.shape
        arr_hw_c = np.transpose(arr, (1, 2, 0))  # (H,W,bands)

        if model_in_channels == 4:
            if bands < 4:
                pad = np.zeros((H, W, 4 - bands), dtype=arr_hw_c.dtype)
                arr_hw_c = np.concatenate([arr_hw_c, pad], axis=2)

            # MODEL INPUT: keep original TOA order unchanged.
            # Expected: Band 1, Band 2, Band 3, Band 4.
            model_raw = arr_hw_c[:, :, :4]
            model_img = normalize_per_channel_percentile(model_raw)

            # DISPLAY ONLY: convert TOA B,G,R,NIR into true RGB.
            # Red=Band 3, Green=Band 2, Blue=Band 1.
            rgb_raw = np.stack(
                [
                    model_raw[:, :, 2],  # Red
                    model_raw[:, :, 1],  # Green
                    model_raw[:, :, 0],  # Blue
                ],
                axis=-1,
            )

            rgb_img = normalize_per_channel_percentile(rgb_raw)
            rgb_img = enhance_rgb_for_display(rgb_img)

        elif model_in_channels == 3:
            if bands < 3:
                raise ValueError("TIFF has <3 bands; cannot form RGB inference.")

            # For 3-channel model, still display as Band 3,2,1 when TIFF is TOA B,G,R.
            rgb_raw = np.stack(
                [
                    arr_hw_c[:, :, 2],  # Red
                    arr_hw_c[:, :, 1],  # Green
                    arr_hw_c[:, :, 0],  # Blue
                ],
                axis=-1,
            )

            rgb_img = normalize_per_channel_percentile(rgb_raw)
            model_img = rgb_img
            rgb_img = enhance_rgb_for_display(rgb_img)

        else:
            raise ValueError(f"Unsupported model_in_channels={model_in_channels}")

        return model_img, rgb_img, pixel_size_m, pixel_source

    raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# Saving outputs
# -----------------------------
def save_mask_png(mask01: np.ndarray, out_path: Path) -> None:
    """
    mask01: (H,W) uint8 0/1
    """
    ensure_dir(out_path.parent)
    img = mask01.astype(np.uint8) * 255
    cv2.imwrite(str(out_path), img)


def save_overlay_png(
    rgb01: np.ndarray,
    mask01: np.ndarray,
    out_path: Path,
    alpha: float = 0.55,
) -> None:
    """
    rgb01: (H,W,3) float32 [0..1]
    mask01: (H,W) uint8 0/1
    Writes BGR PNG for browser display.

    Lower alpha keeps the image colourful instead of turning everything red/dull.
    """
    ensure_dir(out_path.parent)

    base = np.asarray(rgb01).astype(np.float32)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    base = np.clip(base, 0.0, 1.0)

    img = (base * 255).astype(np.uint8).copy()  # RGB
    overlay = img.copy()

    # Bright red detection mask, but low alpha so original colour stays visible.
    overlay[mask01 == 1] = [255, 30, 30]

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def save_json(data: Dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Run folder helper (results/run_YYYYmmdd_HHMMSS)
# -----------------------------
def create_run_dir(results_dir: Path, timestamp: str) -> Path:
    run_dir = results_dir / f"run_{timestamp}"
    ensure_dir(run_dir)
    return run_dir


def build_run_paths(run_dir: Path) -> Dict[str, Path]:
    """
    Returns standard paths for your pipeline outputs.
    """
    return {
        "mask": run_dir / "pred_mask.png",
        "overlay": run_dir / "overlay.png",
        "json": run_dir / "step5_results.json",
    }

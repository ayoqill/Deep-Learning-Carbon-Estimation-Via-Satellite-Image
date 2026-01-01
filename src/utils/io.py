# utils/io.py
# I/O helpers for your CURRENT pipeline:
# - Accept PNG/JPG/TIF
# - Produce: pred_mask.png, overlay.png, step5_results.json
# - Works with your U-Net++ setup (3ch or 4ch), with TIFF geo pixel size if available

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
# Normalization (same as your app.py)
# -----------------------------
def normalize_per_channel_percentile(img: np.ndarray) -> np.ndarray:
    """
    img: (H,W,C) float32/uint16/etc
    return: (H,W,C) float32 in [0..1]
    """
    img = img.astype(np.float32)
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
# Image loading (PNG/JPG/TIF)
# -----------------------------
def load_image_any(
    path: Path,
    model_in_channels: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], str]:
    """
    Returns:
      model_img: (H,W,Cm) float32 [0..1] -> channels match model_in_channels as best as possible
      rgb_img:   (H,W,3)  float32 [0..1] -> for overlay display
      pixel_size_m: float or None (from GeoTIFF if available)
      pixel_size_source: "from_tif" | "none"

    Notes:
      - For PNG/JPG and model_in_channels==4, pads NIR with zeros (accuracy may drop).
      - For TIFF, assumes common order B,G,R,(NIR) when bands>=4.
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

        return model_img, rgb, None, "none"

    if ext in [".tif", ".tiff"]:
        with rasterio.open(str(path)) as src:
            arr = src.read()  # (bands, H, W)

            # Better pixel size extraction than src.res for some GeoTIFFs
            pixel_size_m = float(abs(src.transform.a)) if src.transform is not None else None

        pixel_source = "from_tif" if pixel_size_m is not None else "none"

        bands, H, W = arr.shape
        arr_hw_c = np.transpose(arr, (1, 2, 0))  # (H,W,bands)

        if model_in_channels == 4:
            if bands < 4:
                pad = np.zeros((H, W, 4 - bands), dtype=arr_hw_c.dtype)
                arr_hw_c = np.concatenate([arr_hw_c, pad], axis=2)

            model_raw = arr_hw_c[:, :, :4]
            model_img = normalize_per_channel_percentile(model_raw)

            # display RGB: assume B,G,R,(NIR) -> RGB = (R,G,B) = (2,1,0)
            rgb_raw = np.stack([model_raw[:, :, 2], model_raw[:, :, 1], model_raw[:, :, 0]], axis=-1)
            rgb_img = normalize_per_channel_percentile(rgb_raw)

        elif model_in_channels == 3:
            if bands < 3:
                raise ValueError("TIFF has <3 bands; cannot form RGB inference.")
            rgb_raw = np.stack([arr_hw_c[:, :, 2], arr_hw_c[:, :, 1], arr_hw_c[:, :, 0]], axis=-1)
            rgb_img = normalize_per_channel_percentile(rgb_raw)
            model_img = rgb_img

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
    img = (mask01.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), img)


def save_overlay_png(
    rgb01: np.ndarray,
    mask01: np.ndarray,
    out_path: Path,
    alpha: float = 0.45,
) -> None:
    """
    rgb01: (H,W,3) float32 [0..1]
    mask01: (H,W) uint8 0/1
    Writes BGR PNG for browser display (OpenCV format)
    """
    ensure_dir(out_path.parent)
    img = (rgb01 * 255).astype(np.uint8).copy()  # RGB
    overlay = img.copy()
    overlay[mask01 == 1] = [255, 0, 0]  # red in RGB
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
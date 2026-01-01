"""
Step 2 — Automatic Mask Refinement (SAM-2 pseudo-labels -> refined labels)

Input:
- tiles_clean/   (GeoTIFF tiles, .tif)
- masks_raw/     (SAM-2 masks, typically .png or .tif)

Output:
- masks_refined/ (refined binary masks, .png)

Refinement (default):
1) Remove masks that are almost empty or almost full
2) Remove small isolated components
3) Morphology open+close
4) Optional: Green filter (RGB)
5) Optional: NDVI filter (needs NIR band)
6) Fill holes
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import cv2
import rasterio


# ----------------------------
# Helpers: reading + normalize
# ----------------------------
def read_tif_as_hwc(tif_path: str) -> np.ndarray:
    """Read GeoTIFF as (H, W, C) float32."""
    with rasterio.open(tif_path) as src:
        arr = src.read()  # (C, H, W)
    arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
    return arr.astype(np.float32)


def normalize_to_255(img: np.ndarray) -> np.ndarray:
    """
    Normalize (H, W, C) or (H, W) to 0..255 float32.
    Works for uint8/uint16/float reflectance.
    """
    x = img.astype(np.float32)

    # Per-channel normalize for stability
    if x.ndim == 3:
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[2]):
            ch = x[:, :, c]
            lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
            if hi - lo < 1e-6:
                out[:, :, c] = 0
            else:
                out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1) * 255.0
        return out
    else:
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        if hi - lo < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0, 1) * 255.0


def read_mask(mask_path: str) -> np.ndarray:
    """Read mask file as binary float32 in {0,1}."""
    # Works for PNG/JPG/TIF mask
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        # try rasterio if cv2 fails (some tif masks)
        with rasterio.open(mask_path) as src:
            m = src.read(1)
    m = m.astype(np.float32)
    if m.max() > 1:
        m = m / 255.0
    return (m > 0.5).astype(np.float32)


# ----------------------------
# Refinement core
# ----------------------------
class MaskRefiner:
    def __init__(
        self,
        min_area: int = 500,
        morph_kernel: int = 5,
        fill_holes: bool = True,
        # filters
        use_green_filter: bool = True,
        green_ratio_min: float = 1.08,
        min_brightness: float = 15.0,
        use_ndvi_filter: bool = False,
        nir_index: Optional[int] = None,
        red_index: int = 2,
        ndvi_threshold: float = 0.2,
        # sanity filters for SAM-2 outputs
        max_full_ratio: float = 0.95,   # if mask coverage > 95%, skip or heavily trim
        min_empty_ratio: float = 0.002  # if mask coverage < 0.2%, skip
    ):
        self.min_area = min_area
        self.fill_holes = fill_holes

        self.use_green_filter = use_green_filter
        self.green_ratio_min = green_ratio_min
        self.min_brightness = min_brightness

        self.use_ndvi_filter = use_ndvi_filter
        self.nir_index = nir_index
        self.red_index = red_index
        self.ndvi_threshold = ndvi_threshold

        self.max_full_ratio = max_full_ratio
        self.min_empty_ratio = min_empty_ratio

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

    def refine(self, image_hwc: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns refined mask (H,W) float32 {0,1}, or None if should be skipped.
        """
        h, w = mask.shape
        coverage = float(mask.mean())

        # 0) sanity: skip useless masks
        if coverage < self.min_empty_ratio:
            return None
        if coverage > self.max_full_ratio:
            # not skipping immediately, but we will try to clean it.
            # If it still stays too full after cleaning, we'll drop it later.
            pass

        # 1) remove small regions
        mask = self._remove_small(mask)

        # 2) morphology open + close
        mask = self._morph(mask)

        # 3) optional green filter (for RGB-like imagery)
        if self.use_green_filter and image_hwc is not None and image_hwc.shape[2] >= 3:
            mask = self._green_filter(image_hwc, mask)

        # 4) optional NDVI filter (needs NIR)
        if self.use_ndvi_filter and (self.nir_index is not None) and image_hwc is not None:
            if image_hwc.shape[2] > max(self.nir_index, self.red_index):
                mask = self._ndvi_filter(image_hwc, mask, self.nir_index, self.red_index)

        # 5) fill holes
        if self.fill_holes:
            mask = self._fill_holes(mask)

        # final cleanup
        mask = self._remove_small(mask)

        # final sanity
        final_cov = float(mask.mean())
        if final_cov < self.min_empty_ratio:
            return None
        if final_cov > self.max_full_ratio:
            return None

        return mask.astype(np.float32)

    def _remove_small(self, mask: np.ndarray) -> np.ndarray:
        mu8 = (mask * 255).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mu8, connectivity=8)
        out = np.zeros_like(mask, dtype=np.float32)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                out[labels == i] = 1.0
        return out

    def _morph(self, mask: np.ndarray) -> np.ndarray:
        mu8 = (mask * 255).astype(np.uint8)
        opened = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, self.kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        return (closed > 0).astype(np.float32)

    def _green_filter(self, image_hwc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Robust green dominance filter:
        - Normalize image to 0..255 first
        - Assumes channels are in any order but uses indices [0,1,2] as RGB-like
        """
        img255 = normalize_to_255(image_hwc[:, :, :3])  # take first 3 channels
        # Treat as RGB (not BGR) — works as long as it's “visual” type imagery.
        r = img255[:, :, 0]
        g = img255[:, :, 1]
        b = img255[:, :, 2]

        r_safe = np.maximum(r, 1.0)
        b_safe = np.maximum(b, 1.0)

        green_dom = (g > r_safe * self.green_ratio_min) | (g > b_safe * self.green_ratio_min)
        bright_ok = (g > self.min_brightness) & (r > self.min_brightness * 0.5) & (b > self.min_brightness * 0.5)

        veg = green_dom & bright_ok
        return (mask * veg.astype(np.float32)).astype(np.float32)

    def _ndvi_filter(self, image_hwc: np.ndarray, mask: np.ndarray, nir_idx: int, red_idx: int) -> np.ndarray:
        """
        NDVI = (NIR - Red) / (NIR + Red)
        Keeps NDVI > threshold.
        """
        nir = image_hwc[:, :, nir_idx].astype(np.float32)
        red = image_hwc[:, :, red_idx].astype(np.float32)

        denom = (nir + red) + 1e-8
        ndvi = (nir - red) / denom
        veg = ndvi > self.ndvi_threshold
        return (mask * veg.astype(np.float32)).astype(np.float32)

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        mu8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mu8)
        cv2.drawContours(filled, contours, -1, 255, -1)
        return (filled > 0).astype(np.float32)


# ----------------------------
# Batch pairing + running
# ----------------------------
def stem_variants(stem: str) -> List[str]:
    """
    Generate possible tile stems from a mask stem.
    e.g. tile_001_mask -> tile_001
    """
    v = {stem}
    v.add(re.sub(r"_mask$", "", stem))
    v.add(re.sub(r"-mask$", "", stem))
    v.add(re.sub(r"_sam2$", "", stem))
    v.add(re.sub(r"_pred$", "", stem))
    return list(v)


def find_matching_tile(tile_dir: Path, mask_file: Path) -> Optional[Path]:
    """Try to find matching .tif tile for a given mask file."""
    for s in stem_variants(mask_file.stem):
        cand = tile_dir / f"{s}.tif"
        if cand.exists():
            return cand
    return None


def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask * 255).astype(np.uint8))


def refine_folder(
    tile_dir: str,
    mask_dir: str,
    out_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    tile_dir = Path(tile_dir)
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect mask files
    mask_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        mask_files.extend(mask_dir.glob(ext))

    stats = {
        "total_masks": len(mask_files),
        "processed": 0,
        "skipped_no_tile": 0,
        "skipped_bad_mask": 0,
        "failed": 0,
    }

    refiner = MaskRefiner(**config)

    for mpath in mask_files:
        try:
            tile_path = find_matching_tile(tile_dir, mpath)
            if tile_path is None:
                stats["skipped_no_tile"] += 1
                continue

            img = read_tif_as_hwc(str(tile_path))
            pmask = read_mask(str(mpath))

            refined = refiner.refine(img, pmask)
            if refined is None:
                stats["skipped_bad_mask"] += 1
                continue

            out_path = out_dir / f"{tile_path.stem}.png"
            save_mask_png(refined, out_path)
            stats["processed"] += 1

        except Exception:
            stats["failed"] += 1

    return stats


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Refine SAM-2 masks (Step 2)")
    parser.add_argument("--tiles", required=True, help="Path to tiles_clean/")
    parser.add_argument("--masks", required=True, help="Path to masks_raw/")
    parser.add_argument("--out", required=True, help="Path to masks_refined/")

    # common knobs
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--no-fill-holes", action="store_true")

    parser.add_argument("--no-green", action="store_true")
    parser.add_argument("--green-ratio", type=float, default=1.08)
    parser.add_argument("--min-bright", type=float, default=15.0)

    parser.add_argument("--use-ndvi", action="store_true")
    parser.add_argument("--nir-index", type=int, default=None)
    parser.add_argument("--red-index", type=int, default=2)
    parser.add_argument("--ndvi-th", type=float, default=0.2)

    parser.add_argument("--max-full", type=float, default=0.95)
    parser.add_argument("--min-empty", type=float, default=0.002)

    args = parser.parse_args()

    cfg = dict(
        min_area=args.min_area,
        morph_kernel=args.kernel,
        fill_holes=not args.no_fill_holes,
        use_green_filter=not args.no_green,
        green_ratio_min=args.green_ratio,
        min_brightness=args.min_bright,
        use_ndvi_filter=args.use_ndvi,
        nir_index=args.nir_index,
        red_index=args.red_index,
        ndvi_threshold=args.ndvi_th,
        max_full_ratio=args.max_full,
        min_empty_ratio=args.min_empty,
    )

    s = refine_folder(args.tiles, args.masks, args.out, cfg)
    print("\nRefinement done:")
    for k, v in s.items():
        print(f"  {k}: {v}")

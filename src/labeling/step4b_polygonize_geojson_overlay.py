# step4b_polygonize_geojson_overlay.py
# -----------------------------------
# STEP 4B: Convert predicted mask -> polygons (GeoJSON) + red outline overlay.
# Input : 1 tile (.tif) + its predicted mask (.png, 0/255)
# Output: overlay_red.png + polygons.geojson
#
# Works for your case: 4-band tile (RGB+NIR), mask is binary (0/255).

import json
from pathlib import Path

import numpy as np
import cv2
import rasterio
from rasterio.transform import xy as pixel_to_map


# -------------------------
# EDIT THESE PATHS
# -------------------------
TILE_PATH = Path("data/tiles_clean/STL_Langkawi_Mangrove10_43.tif")
MASK_PATH = Path("data/pred_mask.png")  # from Step 4A
OUT_OVERLAY = Path("data/overlay_red.png")
OUT_GEOJSON = Path("data/polygons.geojson")


# -------------------------
# TUNING (safe defaults)
# -------------------------
MIN_AREA_PX = 120        # remove tiny polygons/noise (increase if too many small shapes)
MORPH_KERNEL = 3         # 3 or 5 (bigger = smoother, may remove thin details)
APPROX_EPS = 2.0         # polygon simplification in pixels (2-5 good)
OUTLINE_THICKNESS = 2    # red outline thickness


# -------------------------
# Helpers
# -------------------------
def load_tile_rgb_for_display(tile_path: Path):
    """
    Read tile (multi-band tif) and create a display RGB image (uint8).
    Also returns georeferencing info (transform, crs).
    """
    with rasterio.open(tile_path) as src:
        img = src.read()  # (C,H,W)
        transform = src.transform
        crs = src.crs

    # Most common order: Band1=B, Band2=G, Band3=R, Band4=NIR
    if img.shape[0] >= 3:
        b = img[0].astype(np.float32)
        g = img[1].astype(np.float32)
        r = img[2].astype(np.float32)
    else:
        # fallback: single band to RGB
        r = g = b = img[0].astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)  # (H,W,3)

    # Contrast stretch (2-98 percentile) to show nicely
    disp = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi - lo < 1e-6:
            disp[:, :, c] = 0
        else:
            disp[:, :, c] = (np.clip((ch - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    return disp, transform, crs


def load_binary_mask(mask_path: Path):
    """
    Load predicted mask PNG (0/255) -> returns uint8 mask 0/255.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = (m > 127).astype(np.uint8) * 255
    return m


def clean_mask(mask_255: np.ndarray):
    """
    - Remove small connected components
    - Morph close to smooth + connect tiny gaps
    - Additional: dilate with (7,7) kernel, 2 iterations
    """
    mask_bin = (mask_255 > 0).astype(np.uint8)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    cleaned = np.zeros_like(mask_bin)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_AREA_PX:
            cleaned[labels == i] = 1

    # Morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)

    # Additional dilation step
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.dilate(cleaned, k2, iterations=2)

    return cleaned.astype(np.uint8) * 255


def fill_holes(mask_255):
    m = (mask_255 > 0).astype(np.uint8)
    h, w = m.shape
    flood = m.copy()
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0,0), 1)  # fill background
    holes = (flood == 0).astype(np.uint8)
    filled = (m | holes) * 255
    return filled.astype(np.uint8)


def mask_to_contours(mask_255: np.ndarray):
    """
    Extract external contours from mask.
    """
    cnts, _ = cv2.findContours((mask_255 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def draw_red_outline(rgb_uint8: np.ndarray, contours):
    """
    Draw red outline on top of the RGB image.
    """
    # OpenCV drawing uses BGR; convert to BGR then back
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), thickness=OUTLINE_THICKNESS)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def contour_to_geo_polygon(contour, transform, approx_eps=APPROX_EPS):
    """
    Convert a contour (pixel coords) into GeoJSON polygon coordinates using raster transform.
    """
    if contour is None or len(contour) < 3:
        return None

    approx = cv2.approxPolyDP(contour, approx_eps, True)
    if len(approx) < 3:
        return None

    coords = []
    for p in approx:
        x_px, y_px = int(p[0][0]), int(p[0][1])
        # rasterio expects row=y, col=x
        x_geo, y_geo = pixel_to_map(transform, y_px, x_px)
        coords.append([x_geo, y_geo])

    # close polygon ring
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return coords


def export_geojson(contours, transform, crs, out_path: Path):
    features = []

    for cnt in contours:
        poly = contour_to_geo_polygon(cnt, transform, approx_eps=APPROX_EPS)
        if poly is None:
            continue

        features.append({
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly]
            }
        })

    fc = {
        "type": "FeatureCollection",
        "features": features
    }

    # include CRS name (optional, helpful for reporting)
    if crs is not None:
        fc["crs"] = {"type": "name", "properties": {"name": str(crs)}}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, indent=2))
    print("✅ Saved GeoJSON:", out_path)


def draw_red_fill_overlay(rgb_uint8: np.ndarray, mask_255: np.ndarray, alpha: float = 0.7):
    """
    Fill overlay: red color on top of mangrove pixels, semi-transparent.
    alpha: 0.0 (invisible) → 1.0 (solid red)
    """
    overlay = rgb_uint8.copy()

    # Make a red image same size
    red_layer = np.zeros_like(overlay, dtype=np.uint8)
    red_layer[:, :, 0] = 255  # R channel (RGB)

    # Create boolean mask
    m = (mask_255 > 0)

    # Blend only where mask is true
    overlay[m] = (overlay[m] * (1 - alpha) + red_layer[m] * alpha).astype(np.uint8)

    return overlay


# -------------------------
# Main
# -------------------------
def main():
    rgb, transform, crs = load_tile_rgb_for_display(TILE_PATH)
    mask = load_binary_mask(MASK_PATH)

    # safety: if mask size differs from tile, resize mask to match tile
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = clean_mask(mask)
    mask = fill_holes(mask)
    # Fill overlay: red color on top of mangrove pixels, semi-transparent.
    overlay = draw_red_fill_overlay(rgb, mask, alpha=0.7)

    # OPTIONAL: still draw outline on top (nice for clarity)
    contours = mask_to_contours(mask)
    overlay = draw_red_outline(overlay, contours)

    OUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_OVERLAY), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ Saved overlay:", OUT_OVERLAY)

    # geojson
    export_geojson(contours, transform, crs, OUT_GEOJSON)


if __name__ == "__main__":
    main()
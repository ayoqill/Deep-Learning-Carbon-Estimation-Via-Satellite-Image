import json
import numpy as np
import cv2
import rasterio
from pathlib import Path


DEFAULT_PIXEL_SIZE_M = 0.7
DEFAULT_CARBON_DENSITY_TON_PER_HA = 150.0  # ⚠️ replace with your literature-based value


# -------------------------
# Loaders
# -------------------------
def load_binary_mask(mask_path: str) -> np.ndarray:
    """
    Load a predicted mask and return a 2D array of 0/1.
    Supports PNG/JPG (0/255) and TIF/TIFF masks.
    """
    p = Path(mask_path)
    ext = p.suffix.lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        return (m > 127).astype(np.uint8)  # 0/1

    if ext in [".tif", ".tiff"]:
        with rasterio.open(str(p)) as src:
            m = src.read(1)
        return (m > 0).astype(np.uint8)  # treat any >0 as mangrove

    raise ValueError(f"Unsupported mask extension: {ext}")


def get_pixel_size_m(reference_path: str) -> float:
    """
    Extract pixel size (meters per pixel) from a reference GeoTIFF.
    Uses rasterio src.res (xres, yres).
    """
    with rasterio.open(reference_path) as src:
        xres, yres = src.res
    # assume square pixels, return xres
    return float(xres)


# -------------------------
# Core calculation (reusable)
# -------------------------
def calculate_area_and_carbon(
    mask01: np.ndarray,
    pixel_size_m: float,
    carbon_density_ton_per_ha: float = DEFAULT_CARBON_DENSITY_TON_PER_HA
) -> dict:
    """
    Compute coverage (%), area (m²/ha), carbon stock (tons), CO2e (tons)
    from a binary mask (0/1).
    """
    if mask01.ndim != 2:
        raise ValueError("Mask must be 2D (H, W).")

    total_pixels = int(mask01.size)
    mangrove_pixels = int(mask01.sum())

    coverage_percent = (mangrove_pixels / total_pixels) * 100.0 if total_pixels else 0.0

    pixel_area_m2 = float(pixel_size_m * pixel_size_m)
    area_m2 = float(mangrove_pixels * pixel_area_m2)
    area_ha = float(area_m2 / 10000.0)

    carbon_tons = float(area_ha * carbon_density_ton_per_ha)  # tons Carbon
    co2_tons = float(carbon_tons * 3.67)                      # tons CO2 equivalent

    return {
        "pixel_size_m": float(pixel_size_m),
        "pixel_area_m2": float(pixel_area_m2),
        "mangrove_pixels": mangrove_pixels,
        "total_pixels": total_pixels,
        "coverage_percent": float(round(coverage_percent, 2)),
        "area_m2": float(round(area_m2, 2)),
        "area_ha": float(round(area_ha, 4)),
        "carbon_density_ton_per_ha": float(carbon_density_ton_per_ha),
        "carbon_tons": float(round(carbon_tons, 2)),
        "co2_tons": float(round(co2_tons, 2)),
    }


# -------------------------
# CLI runner (optional)
# -------------------------
def main():
    """
    Example usage:
    python -m src.analytics.step5_area_carbon \
      --mask results/run1/pred_mask.png \
      --ref-image uploads/user_image.tif \
      --out results/run1/step5_results.json
    """
    import argparse
    parser = argparse.ArgumentParser(description="Step 5 — Area + Carbon Estimation")
    parser.add_argument("--mask", required=True, help="Predicted mask path (.png or .tif)")
    parser.add_argument("--ref-image", default=None,
                        help="Optional: reference GeoTIFF image to auto-read pixel size")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in meters (used if --ref-image not provided)")
    parser.add_argument("--carbon-density", type=float, default=DEFAULT_CARBON_DENSITY_TON_PER_HA,
                        help="Carbon density (tons/ha) from your literature")
    parser.add_argument("--out", default="results/step5_results.json", help="Output JSON path")
    args = parser.parse_args()

    # 1) load mask
    mask01 = load_binary_mask(args.mask)

    # 2) determine pixel size
    if args.ref_image is not None:
        pixel_size_m = get_pixel_size_m(args.ref_image)
    else:
        pixel_size_m = args.pixel_size if args.pixel_size is not None else DEFAULT_PIXEL_SIZE_M

    # 3) compute results
    res = calculate_area_and_carbon(mask01, pixel_size_m=pixel_size_m, carbon_density_ton_per_ha=args.carbon_density)

    # 4) print + save
    print("\n=== STEP 5 RESULTS ===")
    print(f"Pixel size (m):           {res['pixel_size_m']}")
    print(f"Pixel area (m²):          {res['pixel_area_m2']:.4f}")
    print(f"Mangrove pixels:          {res['mangrove_pixels']}")
    print(f"Total pixels:             {res['total_pixels']}")
    print(f"Mangrove Coverage (%):    {res['coverage_percent']:.2f}")
    print(f"Total Area (m²):          {res['area_m2']:.2f}")
    print(f"Total Area (ha):          {res['area_ha']:.4f}")
    print(f"Carbon Stock (tons):      {res['carbon_tons']:.2f}")
    print(f"CO₂ Equivalent (tons):    {res['co2_tons']:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"\n✅ Saved JSON: {out_path}")
    print("Frontend mapping:")
    print(f"  coveragePercent -> {res['coverage_percent']:.2f}")
    print(f"  areaHectares     -> {res['area_ha']:.4f}")
    print(f"  areaM2           -> {res['area_m2']:.2f}")
    print(f"  carbonTons       -> {res['carbon_tons']:.2f}")
    print(f"  carbonCO2        -> {res['co2_tons']:.2f}")


if __name__ == "__main__":
    main()
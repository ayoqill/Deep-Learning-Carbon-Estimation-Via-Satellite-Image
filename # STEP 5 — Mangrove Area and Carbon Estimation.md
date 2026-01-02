# STEP 5 — Mangrove Area and Carbon Estimation

## Overview
Step 5 converts the binary mangrove segmentation mask produced by the U-Net++ inference stage into real-world environmental measurements. This step performs deterministic calculations to estimate mangrove coverage, total area, carbon stock, and CO₂ equivalent.

---

## Inputs
- **Segmentation Mask**: Binary mask from Step 4 inference
  - Values: `1` (mangrove), `0` (non-mangrove)
  - Shape: `[H, W]`
  - Source: U-Net++ output after resizing to original image size
- **Image Type**: Supported formats
  - GeoTIFF
  - PNG
  - JPG

---

## Pixel Resolution
- **GeoTIFF**
  - Source: Raster metadata
  - Method: `read_from_rasterio_res`
  - Unit: meters per pixel
- **PNG/JPG**
  - Source: Fixed assumption
  - Value: `0.7` meters per pixel
  - Note: PNG and JPG images do not contain geospatial metadata. A constant pixel size of 0.7 meters per pixel is assumed.

---

## Constants
- **Carbon Density**
  - Value: `150` tons per hectare
  - Description: Average mangrove carbon density used for estimation. Configurable and based on selected literature.
- **CO₂ Conversion Factor**
  - Value: `3.67`
  - Description: IPCC conversion factor from carbon to CO₂ equivalent.

---

## Calculations
- **Pixel Area**
  - Formula: `pixel_size_m * pixel_size_m`
  - Unit: square meters
- **Total Pixels**
  - Formula: `height * width`
  - Unit: pixels
- **Mangrove Pixels**
  - Formula: `sum(segmentation_mask == 1)`
  - Unit: pixels
- **Coverage Percentage**
  - Formula: `(mangrove_pixels / total_pixels) * 100`
  - Unit: percent
- **Mangrove Area (m²)**
  - Formula: `mangrove_pixels * pixel_area`
  - Unit: square meters
- **Mangrove Area (hectares)**
  - Formula: `mangrove_area_m2 / 10000`
  - Unit: hectares
- **Carbon Stock**
  - Formula: `mangrove_area_hectares * carbon_density`
  - Unit: tons of carbon
- **CO₂ Equivalent**
  - Formula: `carbon_stock * co2_conversion_factor`
  - Unit: tons of CO₂

---

## Outputs
- **File**: `step5_results.json` (location: `results/run_<timestamp>/`)
- **Fields**:
  - `pixel_size_m` (meters)
  - `pixel_area_m2` (square meters)
  - `mangrove_pixels` (pixels)
  - `total_pixels` (pixels)
  - `coverage_percent` (percent)
  - `area_m2` (square meters)
  - `area_ha` (hectares)
  - `carbon_density_ton_per_ha` (tons per hectare)
  - `carbon_tons` (tons of carbon)
  - `co2_tons` (tons of CO₂)

---

## System Role
- **Purpose**: Translate segmentation results into interpretable environmental metrics.
- **Scope**:
  - Mangrove area estimation
  - Carbon stock estimation
- **Excluded**:
  - Carbon credit pricing
  - Trading or certification

---

## Notes
- Step 5 does not involve machine learning.
- All computations are transparent and reproducible.
- Results depend on segmentation accuracy and pixel size assumption.
- GeoTIFF inputs produce more reliable area estimates than PNG/JPG.

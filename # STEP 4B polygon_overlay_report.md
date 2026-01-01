# STEP 4B — Polygon Overlay & GeoJSON Export Report

## Overview
This report documents the process of generating a red polygon overlay and exporting polygons as GeoJSON from the predicted mangrove mask, as part of the mangrove segmentation pipeline.

---

## Task Description
- **Objective:** Visualize the predicted mangrove mask as a red overlay on the original tile and export the detected mangrove regions as polygons in GeoJSON format for GIS or reporting.
- **Script Used:** `src/labeling/step4b_polygonize_geojson_overlay.py`
- **Input Files:**
  - Tile: `data/tiles_clean/STL_Langkawi_Mangrove10_43.tif`
  - Predicted Mask: `data/pred_mask.png`
- **Output Files:**
  - Overlay Image: `data/overlay_red.png`
  - Polygon GeoJSON: `data/polygons.geojson`

---

## Processing Details
- The script loads the original tile and the predicted mask.
- The mask is cleaned (removal of small objects, morphology, dilation) and holes are filled.
- A semi-transparent red fill is applied to mangrove regions, with a red outline for clarity.
- The overlay is saved as a PNG image for visual inspection or app display.
- Mask contours are converted to polygons using the tile's georeferencing, and exported as a GeoJSON file for GIS or further analysis.

---

## Example Command
```bash
python src/labeling/step4b_polygonize_geojson_overlay.py
```

---

## Results
- **Overlay image saved to:** `data/overlay_red.png`
- **Polygons exported to:** `data/polygons.geojson`
- The overlay provides a clear, app-ready visualization of predicted mangrove regions.
- The GeoJSON file enables spatial analysis and integration with GIS tools.

---

## Conclusion
The polygon overlay and GeoJSON export step provides both visual and spatial evidence of the model's predictions, supporting quality control, reporting, and downstream geospatial analysis.

---

*Report generated on: 2026-01-01*

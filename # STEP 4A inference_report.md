# STEP 4A — Inference: Predict Mangrove Mask on New Tile

## Overview
This report documents the inference process for predicting mangrove segmentation masks on new satellite image tiles using the trained U-Net++ model.

---

## Task Description
- **Objective:** Use the trained U-Net++ model to generate a binary mangrove mask for a new input tile.
- **Script Used:** `src/inference/predict_tile.py`
- **Model Used:** `models/unetpp_best.pth`
- **Input Tile:** `data/tiles_clean/STL_Langkawi_Mangrove10_43.tif`
- **Output Mask:** `data/pred_mask.png`

---

## Inference Details
- The script loads the trained U-Net++ model and the specified input tile.
- The input image is normalized and padded to the required size for the model.
- The model predicts the probability map for mangrove presence.
- The output is thresholded at 0.5 to create a binary mask (0 = background, 1 = mangrove).
- The predicted mask is saved as a PNG file for further analysis or visualization.

---

## Example Command
```bash
python src/inference/predict_tile.py
```

---

## Results
- **Predicted mask saved to:** `data/pred_mask.png`
- The mask can be visualized or used for further post-processing (e.g., polygon overlay, area/carbon estimation).

---

## Conclusion
The inference step successfully generated a mangrove segmentation mask for the new input tile using the trained U-Net++ model. This process can be repeated for any new tile or batch of tiles to enable rapid, automated mangrove mapping.

---

*Report generated on: 2026-01-01*

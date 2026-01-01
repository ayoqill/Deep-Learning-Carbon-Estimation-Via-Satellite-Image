# STEP 3 — U-Net++ Model Training Report

## Overview

This report documents the training of a U-Net++ model for mangrove segmentation using the refined masks generated in Step 2.

---

## Task Description

- **Objective:** Train a deep learning model (U-Net++) to segment mangroves from satellite imagery, enabling fast and automated predictions without reliance on SAM-2.
- **Script Used:** `src/training/train_unetpp.py`
- **Input Directories:**
  - `data/tiles_clean/` (input images)
  - `data/masks_refined/` (ground truth masks)
- **Output:**  
  - `unetpp_best.pth` (best model weights)

---

## Training Details

- The model was trained for **30 epochs**, which was sufficient for convergence as the validation Dice coefficient stabilised without signs of overfitting.
- Data was split into **train (80%)**, **validation (10%)**, and **test (10%)** sets:
  - **Train:** 2,280 images
  - **Validation:** 285 images
  - **Test:** 286 images
- Training progress and metrics (loss, Dice, IoU) were monitored for each epoch.
- The best model (highest validation Dice) was saved.

---

## Evaluation

The trained model was evaluated on a **held-out test dataset** that was not used during training or validation.  
The final performance was measured using **Dice coefficient** and **Intersection-over-Union (IoU)** to assess segmentation accuracy.

- **Test Dice:** 0.8175
- **Test IoU:** 0.7158

### **Interpretation of Segmentation Scores**

| Dice Score   | Interpretation         |
|--------------|-----------------------|
| 0.6–0.7      | okay                  |
| 0.7–0.8      | good                  |
| 0.8–0.85     | **very good** ✅      |
| >0.85        | excellent (manual)    |

**Result:**  
- The model achieved a **Dice score of 0.8175** on the test set, which is considered **very good** for automatic segmentation.

“The U-Net++ model achieved a validation Dice coefficient of 0.84 and a test Dice coefficient of 0.82, indicating strong segmentation performance and good generalisation on unseen mangrove tiles. The corresponding IoU of 0.72 further confirms accurate spatial overlap between predicted and ground truth mangrove regions.”
---

## Conclusion

The U-Net++ model trained in this step demonstrates strong performance on unseen data, with segmentation accuracy suitable for downstream analysis and application.  
This model can now be used for fast, automated mangrove prediction on new satellite images (see Step 4).

---

*Report generated on: <!-- add date here -->*
# Mangrove Mask Generation Batch Report  STEP 1

## Overview
This report documents the batch generation of binary masks for mangrove images using the SAM2 annotator. The process was executed to support further analysis and training tasks in the mangrove carbon pipeline project.

## Task Description
- **Objective:** Generate draft binary masks for all cleaned mangrove image tiles using the SAM2 annotator in batch mode.
- **Script Used:** `src/labeling/run_sam2_batch.py`
- **Module Referenced:** `sam2` (custom module in project)
- **Input Directory:** `data/tiles_clean/`
- **Output Directory:** `data/masks_raw/`

## Execution Details
- The script was executed with the following command:
  ```bash
  export PYTHONPATH="/Users/amxr666/Desktop/mangrove-carbon-pipeline/sam2:$PYTHONPATH" && /usr/local/bin/python3 /Users/amxr666/Desktop/mangrove-carbon-pipeline/src/labeling/run_sam2_batch.py
  ```
- The environment variable `PYTHONPATH` was set to ensure the `sam2` module could be found.
- The script processed all `.tif` images in the input directory, segmenting each image and saving the resulting mask as a PNG file in the output directory.
- Each mask file is named after the input image, suffixed with `_mask.png` (e.g., `STL_Langkawi_Mangrove32_266_mask.png`).

## Results
- **Total Images Processed:** 5,217
- **Output Masks Generated:** 5,217 PNG files in `data/masks_raw/`
- **Processing:** Each image was segmented with high confidence and coverage, using a bounding box covering the full image as the prompt.
- **No errors or issues** were reported during execution.

## Evidence
- Example output files:
  - `data/masks_raw/STL_Langkawi_Mangrove10_10_mask.png`
  - `data/masks_raw/STL_Langkawi_Mangrove10_100_mask.png`
  - ... (see `data/masks_raw/` for full list)
- Script used: `src/labeling/run_sam2_batch.py`
- Command executed: see above

## Conclusion
The batch mask generation for mangrove images using the SAM2 annotator was completed successfully. All masks were generated and saved as expected, providing a solid foundation for subsequent training or analysis tasks.


## Explaination

What these masks mean (important)

White = SAM-2 thinks “foreground”

Black = background

Many tiles are:

fully white → SAM-2 grabbed the whole tile

partly white → random shapes (water edges, noise)

almost empty → little confidence

⚠️ This is NORMAL for Step 1.
These are pseudo-labels, not final labels.

You are not supposed to train yet.

Why many masks look “bad”

You used:

full-image box prompt

no vegetation filtering yet

no water removal

no morphology

## WHY

Because:

SAM-2 is only a label generator

You already planned Step 2: automatic refinement

U-Net++ is your real model
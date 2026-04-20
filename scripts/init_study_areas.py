#!/usr/bin/env python3
# filepath: scripts/init_study_areas.py
# Initialize precomputed study areas with results

import sys
from pathlib import Path
import json
import numpy as np
import cv2
import torch
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.utils.io import load_image_any, create_run_dir, build_run_paths, save_mask_png, save_overlay_png, save_json
from src.utils.analytics import AnalyticsManager
from src.utils.study_areas import StudyAreaManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("init_study_areas")

# Import from app.py
sys.path.insert(0, str(project_root))
from app import (
    load_model, predict_mask_tiled, step5_calculate, 
    DEVICE, MODEL_PATHS, TILE_H, TILE_W, TILE_OVERLAP, BATCH_TILES,
    DEFAULT_PIXEL_SIZE_M, DEFAULT_CARBON_DENSITY_TON_PER_HA,
    loaded_models, model_in_channels
)

def process_langkawi_images():
    """
    Process all images in TEST IMAGES folder and save as precomputed analyses.
    """
    test_images_path = project_root / "TEST IMAGES"
    results_dir = project_root / "results"
    
    if not test_images_path.exists():
        logger.error(f"TEST IMAGES folder not found: {test_images_path}")
        return False
    
    # Initialize managers
    analytics_mgr = AnalyticsManager(results_dir)
    study_areas_mgr = StudyAreaManager(test_images_path, results_dir, loaded_models, DEVICE)
    
    # Load model
    model_name = "unetpp"
    if not load_model(model_name):
        logger.error(f"Failed to load model: {model_name}")
        return False
    
    logger.info(f"Loaded model: {model_name}")
    
    # Discover images
    image_files = study_areas_mgr.discover_study_area_images("langkawi", test_images_path)
    if not image_files:
        logger.warning("No images found in TEST IMAGES folder")
        return False
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    processed_count = 0
    for img_path in image_files:
        try:
            logger.info(f"Processing: {img_path.name}")
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Load image
            current_channels = model_in_channels.get(model_name, 3)
            model_img, rgb_img, pixel_size_from_tif, tif_source = load_image_any(
                img_path,
                model_in_channels=current_channels
            )
            
            # Determine pixel size
            if pixel_size_from_tif is not None:
                pixel_size_m = pixel_size_from_tif
            else:
                pixel_size_m = DEFAULT_PIXEL_SIZE_M
            
            # Inference
            prob_map = predict_mask_tiled(model_img, model_name)
            DETECTION_THRESHOLD = 0.01
            mask01 = (prob_map >= DETECTION_THRESHOLD).astype(np.uint8)
            
            # Save results
            run_dir = create_run_dir(results_dir, timestamp)
            paths = build_run_paths(run_dir)
            
            save_mask_png(mask01, paths["mask"])
            save_overlay_png(rgb_img, mask01, paths["overlay"])
            
            results = step5_calculate(
                mask01, 
                pixel_size_m=pixel_size_m, 
                carbon_density_ton_per_ha=DEFAULT_CARBON_DENSITY_TON_PER_HA
            )
            save_json(results, paths["json"])
            
            # Save to analytics
            analysis = {
                "type": "precomputed",
                "title": img_path.stem,
                "location": "Langkawi, Kedah, Malaysia",
                "originalImagePath": f"/results/run_{timestamp}/original.png",
                "resultImagePath": f"/results/run_{timestamp}/overlay.png",
                "maskPath": f"/results/run_{timestamp}/pred_mask.png",
                "model": model_name,
                "mangroveCoverage": round(results["coverage_percent"], 2),
                "totalAreaHectares": round(results["area_ha"], 4),
                "totalAreaM2": round(results["area_m2"], 2),
                "carbonStock": round(results["carbon_tons"], 2),
                "co2Equivalent": round(results["co2_tons"], 2),
                "pixelSizeM": results["pixel_size_m"],
            }
            
            # Save original image for display
            original_path = results_dir / f"run_{timestamp}" / "original.png"
            cv2.imwrite(str(original_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            analytics_mgr.save_analysis(analysis)
            logger.info(f"✅ Processed and saved: {img_path.name}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to process {img_path.name}: {e}")
            continue
    
    logger.info(f"\n✅ Successfully processed {processed_count}/{len(image_files)} images")
    
    # Initialize study area config
    study_areas_mgr.initialize_langkawi(test_images_path, model_name)
    logger.info("✅ Langkawi study area initialized")
    
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Initializing Precomputed Study Areas")
    logger.info("=" * 60)
    
    success = process_langkawi_images()
    
    if success:
        logger.info("\n✅ Study areas initialized successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Failed to initialize study areas")
        sys.exit(1)

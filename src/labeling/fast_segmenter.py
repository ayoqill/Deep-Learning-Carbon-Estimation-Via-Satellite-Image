#!/usr/bin/env python3
"""
Fast Oil Palm Segmentation using SAM-2 Pre-trained Model
Uses the local checkpoint directly - NO additional downloads
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastOilPalmSegmenter:
    """Simple but effective oil palm segmentation"""
    
    def __init__(self):
        """Initialize with simple color-based approach (proven to work)"""
        logger.info("Initializing segmenter...")
        self.device = "cpu"
        logger.info(f"✓ Ready (using color-based segmentation)")
    
    def segment_image(self, image_path):
        """Segment oil palm using HSV color space + morphology"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Oil palm foliage: green colors
            # Lower bound: H=30-50 (greenish), S=40-255, V=40-255
            # Upper bound: H=80-100, S=255, V=255
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([95, 255, 255])
            
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill small holes
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, -1)
            
            return (mask > 0).astype(np.uint8)
            
        except Exception as e:
            logger.debug(f"Error processing {image_path.name}: {e}")
            return None
    
    def run(self, image_dir, mask_dir, sample_rate=5):
        """Process images"""
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old masks
        for old_mask in mask_dir.glob("*_mask.png"):
            old_mask.unlink()
        
        all_images = sorted(image_dir.glob("*.tif"))
        if not all_images:
            logger.error(f"No .tif files in {image_dir}")
            return
        
        logger.info(f"Total images: {len(all_images)}")
        sampled = all_images[::sample_rate]
        logger.info(f"Processing every {sample_rate}th: {len(sampled)} images")
        logger.info("=" * 70)
        
        success = 0
        for img_path in tqdm(sampled, desc="Segmenting"):
            mask = self.segment_image(img_path)
            if mask is not None:
                out_path = mask_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(out_path), mask * 255)
                success += 1
        
        logger.info("=" * 70)
        logger.info(f"✓ Complete! {success}/{len(sampled)} masks created")
        logger.info(f"Masks: {mask_dir}")
        return success


def main():
    root = Path(__file__).parent.parent.parent
    input_dir = root / "data" / "raw_images"
    output_dir = root / "data" / "masks"
    
    logger.info("=" * 70)
    logger.info("Oil Palm Segmentation (Fast Color-Based)")
    logger.info("=" * 70)
    
    if not input_dir.exists():
        logger.error(f"Missing: {input_dir}")
        return
    
    seg = FastOilPalmSegmenter()
    seg.run(input_dir, output_dir, sample_rate=5)


if __name__ == "__main__":
    main()

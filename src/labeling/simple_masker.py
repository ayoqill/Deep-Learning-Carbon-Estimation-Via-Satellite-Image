#!/usr/bin/env python3
"""
Oil Palm Masking - Fallback Method
Uses OpenCV and image processing instead of SAM-2 weights
Creates binary masks for oil palm plantation images
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMasker:
    """Create masks using image processing (no weights needed)"""
    
    def mask_image(self, image_path):
        """
        Create mask using color-based segmentation
        Oil palm plantations have distinctive green/brown colors
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to HSV (better for color detection)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define range for green (vegetation)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([90, 255, 255])
            
            # Create mask for green vegetation
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Define range for brown/yellow (plantations)
            lower_brown = np.array([10, 40, 40])
            upper_brown = np.array([25, 255, 255])
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Combine masks
            mask = cv2.bitwise_or(mask_green, mask_brown)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Convert to binary (0 or 255)
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            return mask_binary
            
        except Exception as e:
            logger.warning(f"Error processing {Path(image_path).name}: {e}")
            return None
    
    def run(self, image_dir, mask_dir, sample_rate=5):
        """Process all images"""
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        all_images = sorted(image_dir.glob("*.tif"))
        logger.info(f"Found {len(all_images)} total images")
        
        # Sample
        images = all_images[::sample_rate]
        logger.info(f"Processing every {sample_rate}th image: {len(images)} images")
        logger.info("=" * 60)
        
        success = 0
        for img_path in tqdm(images, desc="Creating masks"):
            mask = self.mask_image(img_path)
            if mask is not None:
                mask_path = mask_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask)
                success += 1
        
        logger.info("=" * 60)
        logger.info(f"✓ Complete! Created {success}/{len(images)} masks")
        
        return success


def main():
    """Main"""
    logger.info("=" * 60)
    logger.info("Oil Palm Masking (Image Processing Method)")
    logger.info("=" * 60)
    logger.info("Method: HSV color segmentation + morphological operations")
    logger.info("(No deep learning weights required)")
    logger.info("=" * 60)
    
    image_dir = Path("data/raw_images")
    mask_dir = Path("data/masks")
    
    masker = SimpleMasker()
    masker.run(image_dir, mask_dir, sample_rate=5)


if __name__ == "__main__":
    main()

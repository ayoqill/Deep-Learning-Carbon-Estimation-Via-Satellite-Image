#!/usr/bin/env python3
"""
Targeted Labeling: Only detect GREEN vegetation areas
Fixes the issue where model detects bare soil instead of plantations
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sam2_path = Path(__file__).parent.parent.parent.parent / "sam2"
sys.path.insert(0, str(sam2_path))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class TargetedLabeler:
    """Label ONLY green vegetation areas (plantations)"""
    
    def __init__(self, checkpoint_path, config_path):
        logger.info("Loading SAM-2...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam2_model = build_sam2(
            config_file=str(config_path),
            ckpt_path=str(checkpoint_path),
            device=device
        )
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device
        logger.info(f"✓ SAM-2 ready on {device}")
    
    def detect_green_vegetation(self, image):
        """
        Detect ONLY green vegetation areas using color filtering
        This will create correct masks for training
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define GREEN color range (vegetation)
        # Adjust these values based on your plantation images
        lower_green = np.array([35, 30, 30])   # Dark green
        upper_green = np.array([90, 255, 255]) # Light green
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Find green areas
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        labels = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Only significant green areas
            if area > 200:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    points.append([cx, cy])
                    labels.append(1)  # Positive prompt for green areas
        
        return np.array(points), np.array(labels)
    
    def label_image(self, image_path, visualize=True):
        """Segment ONLY green vegetation"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None, None, None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get green vegetation points
            points, labels = self.detect_green_vegetation(image_rgb)
            
            if len(points) == 0:
                logger.warning(f"No green vegetation in {image_path.name}")
                # Create empty mask instead of None
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                return mask, None, None
            
            # Use SAM-2 to refine the green areas
            self.predictor.set_image(image_rgb)
            masks, _, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
            
            mask = masks[0].astype(np.uint8)
            
            # Additional filter: Keep only pixels that are actually green
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            green_filter = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([90, 255, 255]))
            
            # Combine SAM-2 mask with green filter
            mask = cv2.bitwise_and(mask, green_filter // 255)
            
            overlay = None
            polygons = None
            
            if visualize:
                overlay = self.create_polygon_overlay(image, mask)
                polygons = self.mask_to_polygons(mask)
            
            return mask, overlay, polygons
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return None, None, None
    
    def create_polygon_overlay(self, image, mask, color=(0, 0, 255), alpha=0.5):
        """Create overlay with RED polygons (changed from green)"""
        overlay = image.copy()
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask_overlay = np.zeros_like(image)
        cv2.drawContours(mask_overlay, contours, -1, color, -1)
        cv2.addWeighted(mask_overlay, alpha, overlay, 1 - alpha, 0, overlay)
        cv2.drawContours(overlay, contours, -1, color, 3)
        
        return overlay
    
    def mask_to_polygons(self, mask, min_area=50):
        """Extract polygons from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                coords = approx.reshape(-1, 2).tolist()
                if len(coords) >= 3:
                    polygons.append({
                        'coordinates': coords,
                        'area_pixels': float(area),
                        'class': 'vegetation'
                    })
        
        return polygons
    
    def run(self, image_dir, output_dir, sample_rate=10):
        """Process images with vegetation-only labeling"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        mask_dir = output_dir / "masks"
        overlay_dir = output_dir / "overlays"
        
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = sorted(image_dir.glob("*.tif"))
        sampled = all_images[::sample_rate]
        
        logger.info(f"Creating GREEN VEGETATION masks for {len(sampled)} images...")
        logger.info("Overlays will be shown in RED for visibility")
        
        success = 0
        
        for img_path in tqdm(sampled, desc="Processing"):
            mask, overlay, polygons = self.label_image(img_path, visualize=True)
            
            if mask is not None:
                base_name = img_path.stem
                
                # Save mask for training
                mask_path = mask_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), mask * 255)
                
                # Save overlay for verification
                if overlay is not None:
                    overlay_path = overlay_dir / f"{base_name}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                
                success += 1
        
        logger.info(f"\n✓ Created {success} training masks")
        logger.info(f"✓ Masks: {mask_dir}")
        logger.info(f"✓ Check overlays in: {overlay_dir}")
        logger.info("\n⚠️  IMPORTANT: Check the overlay images!")
        logger.info("   RED areas = vegetation detected (what model will learn)")


def main():
    checkpoint = Path(__file__).parent.parent.parent.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    config = Path(__file__).parent.parent.parent.parent / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    
    input_dir = Path(__file__).parent.parent.parent / "data" / "raw_images"
    output_dir = Path(__file__).parent.parent.parent / "data" / "labeled_vegetation_only"
    
    logger.info("=" * 60)
    logger.info("VEGETATION-ONLY Labeler (RED Polygon Overlays)")
    logger.info("=" * 60)
    
    labeler = TargetedLabeler(checkpoint, config)
    labeler.run(input_dir, output_dir, sample_rate=10)
    
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("1. Check overlay images - RED areas show detected vegetation")
    logger.info("2. If correct, retrain U-Net with these new masks")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

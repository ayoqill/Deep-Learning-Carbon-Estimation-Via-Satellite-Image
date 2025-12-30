#!/usr/bin/env python3
"""
Quick Start: Batch SAM-2 Labeler with Polygon Visualization
Process a sample of images with visual overlays for web display
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

# Add SAM-2 to path
sam2_path = Path(__file__).parent.parent.parent.parent / "sam2"
sys.path.insert(0, str(sam2_path))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class QuickLabeler:
    """Fast batch labeler with polygon visualization"""
    
    def __init__(self, checkpoint_path, config_path):
        """Initialize SAM-2"""
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
    
    def label_image(self, image_path, visualize=True):
        """Segment single image and optionally create visualization"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None, None, None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Bounding box: 10% margin from edges
            box = np.array([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]], dtype=np.float32)
            
            self.predictor.set_image(image_rgb)
            masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
            
            mask = masks[0].astype(np.uint8)
            
            # Create visualization and polygons
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
        """
        Create visual overlay with RED polygon boundaries
        Perfect for web display!
        """
        overlay = image.copy()
        
        # Find contours (polygon boundaries)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw filled polygons with transparency
        mask_overlay = np.zeros_like(image)
        cv2.drawContours(mask_overlay, contours, -1, color, -1)
        cv2.addWeighted(mask_overlay, alpha, overlay, 1 - alpha, 0, overlay)
        
        # Draw polygon outlines (thick red lines)
        cv2.drawContours(overlay, contours, -1, color, 3)
        
        return overlay
    
    def mask_to_polygons(self, mask, min_area=50):
        """Extract polygon coordinates from mask"""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Simplify polygon
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                coords = approx.reshape(-1, 2).tolist()
                if len(coords) >= 3:
                    polygons.append({
                        'coordinates': coords,
                        'area_pixels': float(area),
                        'perimeter': float(cv2.arcLength(cnt, True)),
                        'class': 'vegetation'  # Change to 'palm_oil' or 'mangrove' as needed
                    })
        
        return polygons
    
    def save_geojson(self, polygons, output_path, image_id):
        """Save polygons as GeoJSON for QGIS"""
        features = []
        for i, poly in enumerate(polygons):
            feature = {
                'type': 'Feature',
                'properties': {
                    'image_id': image_id,
                    'polygon_id': i,
                    'area_pixels': poly['area_pixels'],
                    'perimeter': poly['perimeter'],
                    'class': poly['class']
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [poly['coordinates']]
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'properties': {
                'total_polygons': len(features),
                'total_area_pixels': sum(p['area_pixels'] for p in polygons)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
    
    def run(self, image_dir, output_dir, sample_rate=10, visualize=True):
        """
        Process every Nth image with visualization
        
        Args:
            image_dir: Source directory with .tif files
            output_dir: Output directory for masks and overlays
            sample_rate: Process every Nth image (10 = every 10th)
            visualize: Create polygon overlays (True for web display)
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        # Create subdirectories
        mask_dir = output_dir / "masks"
        overlay_dir = output_dir / "overlays"
        geojson_dir = output_dir / "geojson"
        
        mask_dir.mkdir(parents=True, exist_ok=True)
        if visualize:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            geojson_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        all_images = sorted(image_dir.glob("*.tif"))
        logger.info(f"Total images: {len(all_images)}")
        
        # Sample every Nth
        sampled = all_images[::sample_rate]
        logger.info(f"Sampling every {sample_rate}th image: {len(sampled)} to process")
        
        success = 0
        total_polygons = 0
        
        for img_path in tqdm(sampled, desc="Processing"):
            mask, overlay, polygons = self.label_image(img_path, visualize)
            
            if mask is not None:
                base_name = img_path.stem
                
                # Save binary mask (for training)
                mask_path = mask_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), mask * 255)
                
                # Save overlay (for web display)
                if visualize and overlay is not None:
                    overlay_path = overlay_dir / f"{base_name}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                
                # Save GeoJSON (for GIS analysis)
                if visualize and polygons:
                    geojson_path = geojson_dir / f"{base_name}_polygons.geojson"
                    self.save_geojson(polygons, geojson_path, base_name)
                    total_polygons += len(polygons)
                
                success += 1
        
        logger.info(f"\n✓ Done! Processed {success}/{len(sampled)} images")
        if visualize:
            logger.info(f"✓ Extracted {total_polygons} total polygons")
            logger.info(f"✓ Overlays saved to: {overlay_dir}")
            logger.info(f"✓ GeoJSON saved to: {geojson_dir}")
        logger.info(f"✓ Masks saved to: {mask_dir}")


def main():
    """Quick start with polygon visualization"""
    
    checkpoint = Path(__file__).parent.parent.parent.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    config = Path(__file__).parent.parent.parent.parent / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    
    input_dir = Path(__file__).parent.parent.parent / "data" / "raw_images"
    output_dir = Path(__file__).parent.parent.parent / "data" / "labeled_output"
    
    logger.info("=" * 60)
    logger.info("SAM-2 Labeler with Polygon Visualization")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("=" * 60)
    
    labeler = QuickLabeler(checkpoint, config)
    
    # Process every 10th image with visualization
    # Set visualize=False if you only want masks (faster)
    labeler.run(input_dir, output_dir, sample_rate=10, visualize=True)


if __name__ == "__main__":
    main()


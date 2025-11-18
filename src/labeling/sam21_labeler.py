#!/usr/bin/env python3
"""
SAM 2.1 Official Labeler
Uses official Meta SAM 2.1 checkpoint (facebook/sam2.1-hiera-large)
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


class SAM21Labeler:
    """SAM 2.1 labeler using official Meta checkpoint"""
    
    def __init__(self):
        """Initialize SAM 2.1 from official checkpoint"""
        logger.info("Loading SAM 2.1 official checkpoint...")
        
        try:
            # Import SAM 2 from source
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Checkpoint path
            checkpoint = Path(__file__).parent.parent.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
            
            if not checkpoint.exists():
                logger.error(f"Checkpoint not found: {checkpoint}")
                raise FileNotFoundError(f"Missing: {checkpoint}")
            
            logger.info(f"Checkpoint: {checkpoint.name} ({checkpoint.stat().st_size / 1e9:.2f} GB)")
            
            # Build model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {device}")
            
            model = build_sam2(
                config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                ckpt_path=str(checkpoint),
                device=device
            )
            
            self.predictor = SAM2ImagePredictor(model)
            
            logger.info("✓ SAM 2.1 loaded successfully")
            self.device = device
            
        except Exception as e:
            logger.error(f"Failed to load SAM 2.1: {e}")
            logger.error("Ensure SAM 2 source is properly installed:")
            logger.error("  cd sam2 && pip install -e .")
            raise
    
    def segment_image(self, image_path):
        """Segment image using SAM 2.1"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Set image
            self.predictor.set_image(image_rgb)
            
            # Bounding box: 10% margin (oil palm in center)
            input_box = np.array([
                int(w * 0.1), int(h * 0.1),
                int(w * 0.9), int(h * 0.9)
            ])
            
            # Predict mask
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=False
            )
            
            # Get best mask
            mask = masks[0]  # (H, W)
            binary_mask = (mask > 0).astype(np.uint8)
            
            return binary_mask
            
        except Exception as e:
            logger.debug(f"Error: {image_path.name}: {e}")
            return None
    
    def run(self, image_dir, mask_dir, sample_rate=5):
        """Process images"""
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = sorted(image_dir.glob("*.tif"))
        if not all_images:
            logger.error(f"No .tif files in {image_dir}")
            return
        
        logger.info(f"Total images: {len(all_images)}")
        sampled = all_images[::sample_rate]
        logger.info(f"Processing every {sample_rate}th: {len(sampled)} images")
        logger.info("=" * 70)
        
        success = 0
        failed = []
        
        for img_path in tqdm(sampled, desc="Segmenting"):
            mask = self.segment_image(img_path)
            if mask is not None:
                out_path = mask_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(out_path), mask * 255)
                success += 1
            else:
                failed.append(img_path.name)
        
        logger.info("=" * 70)
        logger.info(f"✓ Complete! {success}/{len(sampled)} masks created")
        
        if failed:
            logger.warning(f"Failed ({len(failed)}): {failed[:5]}")
        
        logger.info(f"Masks: {mask_dir}")
        return success


def main():
    root = Path(__file__).parent.parent.parent
    input_dir = root / "data" / "raw_images"
    output_dir = root / "data" / "masks"
    
    logger.info("=" * 70)
    logger.info("SAM 2.1 Official Labeler")
    logger.info("=" * 70)
    
    if not input_dir.exists():
        logger.error(f"Missing: {input_dir}")
        return
    
    labeler = SAM21Labeler()
    labeler.run(input_dir, output_dir, sample_rate=5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick Start: Batch SAM-2 Labeler with Sampling
Process a sample of images (every Nth image) for quick testing
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add SAM-2 to path
sam2_path = Path(__file__).parent.parent.parent.parent / "sam2"
sys.path.insert(0, str(sam2_path))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class QuickLabeler:
    """Fast batch labeler for oil palm images"""
    
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
    
    def label_image(self, image_path):
        """Segment single image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Bounding box: 10% margin from edges
            box = np.array([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]], dtype=np.float32)
            
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
            
            return masks[0].astype(np.uint8)
        except:
            return None
    
    def run(self, image_dir, mask_dir, sample_rate=20):
        """
        Process every Nth image
        
        Args:
            image_dir: Source directory with .tif files
            mask_dir: Output directory for masks
            sample_rate: Process every Nth image (20 = every 20th)
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        all_images = sorted(image_dir.glob("*.tif"))
        logger.info(f"Total images: {len(all_images)}")
        
        # Sample every Nth
        sampled = all_images[::sample_rate]
        logger.info(f"Sampling every {sample_rate}th image: {len(sampled)} to process")
        
        success = 0
        for img_path in tqdm(sampled, desc="Labeling"):
            mask = self.label_image(img_path)
            if mask is not None:
                mask_path = mask_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask * 255)
                success += 1
        
        logger.info(f"\n✓ Done! Labeled {success}/{len(sampled)} images")


def main():
    """Quick start with sampling"""
    
    checkpoint = Path(__file__).parent.parent.parent.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    config = Path(__file__).parent.parent.parent.parent / "sam2" / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml"
    
    input_dir = Path(__file__).parent.parent.parent / "data" / "raw_images"
    output_dir = Path(__file__).parent.parent.parent / "data" / "masks"
    
    logger.info("=" * 60)
    logger.info("Quick SAM-2 Labeler")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("=" * 60)
    
    labeler = QuickLabeler(checkpoint, config)
    
    # Process every 10th image (290 images from 5900)
    # Change sample_rate to:
    #   1 = all images (slow)
    #   5 = every 5th (fast)
    #   10 = every 10th (faster)
    #   20 = every 20th (fastest)
    
    labeler.run(input_dir, output_dir, sample_rate=10)


if __name__ == "__main__":
    main()

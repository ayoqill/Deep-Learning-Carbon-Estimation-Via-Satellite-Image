#!/usr/bin/env python3
"""
Batch SAM-2 Labeler for Oil Palm Images
Automatically segments all images in a folder using SAM-2
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add SAM-2 to path
sam2_path = Path(__file__).parent.parent.parent.parent / "sam2"
sys.path.insert(0, str(sam2_path))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    logger.error(f"Failed to import SAM-2: {e}")
    sys.exit(1)


class BatchLabeler:
    """Batch process images with SAM-2"""
    
    def __init__(self, checkpoint_path, config_path, device="cuda"):
        """Initialize SAM-2 model"""
        logger.info(f"Initializing SAM-2 on {device}...")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Config: {config_path}")
        
        try:
            # Build model
            sam2_model = build_sam2(
                config_file=str(config_path),
                ckpt_path=str(checkpoint_path),
                device=device
            )
            
            # Create predictor
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.device = device
            logger.info("✓ SAM-2 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM-2: {e}")
            raise
    
    def segment_image(self, image_path, use_gpu=True):
        """
        Segment image using SAM-2
        
        Args:
            image_path: Path to .tif image
            use_gpu: Use GPU for faster processing
            
        Returns:
            np.ndarray: Binary mask (0 or 1)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Cannot load: {Path(image_path).name}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Create bounding box around center area (oil palm likely in middle)
            # Use 80% of image
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            box = np.array([
                [margin_x, margin_y, w - margin_x, h - margin_y]
            ], dtype=np.float32)
            
            # Set image and predict
            self.predictor.set_image(image)
            
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=False
            )
            
            # Get best mask
            mask = masks[0].astype(np.uint8)
            
            return mask
            
        except Exception as e:
            logger.warning(f"Segmentation failed for {Path(image_path).name}: {e}")
            return None
    
    def save_mask(self, mask, output_path):
        """Save mask as PNG"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to 0-255 range
            mask_png = mask * 255
            cv2.imwrite(str(output_path), mask_png)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save mask: {e}")
            return False
    
    def process_folder(self, input_dir, output_dir, max_images=None, sample_rate=1):
        """
        Process all images in folder
        
        Args:
            input_dir: Folder with .tif images
            output_dir: Folder to save masks
            max_images: Process only this many images (None = all)
            sample_rate: Process every Nth image (1 = all, 20 = every 20th)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all .tif files
        image_files = sorted(input_dir.glob("*.tif"))
        
        if not image_files:
            logger.error(f"No .tif files found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images")
        
        # Apply sampling
        if sample_rate > 1:
            image_files = image_files[::sample_rate]
            logger.info(f"Sampling every {sample_rate}th image: {len(image_files)} to process")
        
        # Limit number of images
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Processing {len(image_files)} images...")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 70)
        
        success_count = 0
        error_count = 0
        
        # Process with progress bar
        for image_path in tqdm(image_files, desc="Labeling"):
            # Segment image
            mask = self.segment_image(image_path)
            
            if mask is not None:
                # Save mask
                mask_name = image_path.stem + "_mask.png"
                mask_path = output_dir / mask_name
                
                if self.save_mask(mask, mask_path):
                    success_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1
        
        # Summary
        logger.info("=" * 70)
        logger.info(f"✓ Processing complete!")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Success rate: {(success_count / len(image_files) * 100):.1f}%")
        
        return success_count, error_count


def main():
    """Main entry point"""
    
    # Configuration
    checkpoint_path = Path(__file__).parent.parent.parent.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    config_path = Path(__file__).parent.parent.parent.parent / "sam2" / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml"
    
    # Input/Output directories
    input_dir = Path(__file__).parent.parent.parent / "data" / "raw_images"
    output_dir = Path(__file__).parent.parent.parent / "data" / "masks"
    
    # Verify paths
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Initialize labeler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labeler = BatchLabeler(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device
    )
    
    logger.info(f"Using device: {device}")
    
    # Process images
    # Options:
    #   max_images=500  -> process first 500 images
    #   sample_rate=20  -> process every 20th image (for quick test)
    
    labeler.process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        max_images=None,  # Change to limit (e.g., 100, 500)
        sample_rate=1     # Change to sample (e.g., 5, 10, 20)
    )


if __name__ == "__main__":
    main()

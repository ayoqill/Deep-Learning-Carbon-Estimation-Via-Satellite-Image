#!/usr/bin/env python3
"""
Data Preparation for U-Net Training
Splits 368 labeled images into train/val/test folders
"""

import shutil
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_training_data(raw_images_dir, masks_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split images and masks into train/val/test folders
    
    Args:
        raw_images_dir: Directory with raw .tif images
        masks_dir: Directory with corresponding masks
        output_dir: Output directory for organized data
        train_ratio: Training split (0.7 = 70%)
        val_ratio: Validation split (0.15 = 15%)
        test_ratio: Testing split (0.15 = 15%)
    """
    
    raw_images_dir = Path(raw_images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    
    # Create output structure
    train_images = output_dir / "train" / "images"
    train_masks = output_dir / "train" / "masks"
    val_images = output_dir / "val" / "images"
    val_masks = output_dir / "val" / "masks"
    test_images = output_dir / "test" / "images"
    test_masks = output_dir / "test" / "masks"
    
    for folder in [train_images, train_masks, val_images, val_masks, test_images, test_masks]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Get all mask files (only use images with labels)
    all_masks = sorted(masks_dir.glob("*_mask.png"))
    logger.info(f"Found {len(all_masks)} labeled masks")
    
    if not all_masks:
        logger.error(f"No mask files found in {masks_dir}")
        return
    
    # Extract image names from mask files (remove "_mask.png")
    image_names = [mask.name.replace("_mask.png", "") for mask in all_masks]
    
    # Split into train/val/test
    train_names, temp_names = train_test_split(
        image_names, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    val_names, test_names = train_test_split(
        temp_names,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42
    )
    
    logger.info(f"Split: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")
    
    # Copy files
    def copy_files(names, img_dest, mask_dest, split_name):
        count = 0
        for name in names:
            # Find image file
            img_file = list(raw_images_dir.glob(f"{name}.*"))
            if img_file:
                shutil.copy(img_file[0], img_dest / img_file[0].name)
            
            # Find mask file
            mask_file = masks_dir / f"{name}_mask.png"
            if mask_file.exists():
                shutil.copy(mask_file, mask_dest / mask_file.name)
                count += 1
        
        logger.info(f"✓ {split_name}: Copied {count} image-mask pairs")
    
    logger.info("Copying files...")
    copy_files(train_names, train_images, train_masks, "Training")
    copy_files(val_names, val_images, val_masks, "Validation")
    copy_files(test_names, test_images, test_masks, "Testing")
    
    logger.info("=" * 70)
    logger.info(f"✓ Data preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    return train_names, val_names, test_names


if __name__ == "__main__":
    # Paths - Using vegetation-only masks for v2 training
    raw_images_dir = Path(__file__).parent.parent.parent / "data" / "raw_images"
    masks_dir = Path(__file__).parent.parent.parent / "data" / "labeled_vegetation_only" / "masks"
    output_dir = Path(__file__).parent.parent.parent / "data" / "prepared"
    
    logger.info("=" * 70)
    logger.info("Data Preparation for U-Net Training v2 (Vegetation-Only)")
    logger.info("=" * 70)
    
    prepare_training_data(raw_images_dir, masks_dir, output_dir)

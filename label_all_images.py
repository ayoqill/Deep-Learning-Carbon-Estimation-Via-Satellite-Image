"""
Incremental labeling - only label unlabeled images
Supports both Kuala Lipis and Tawau datasets
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import sys

# Initialize Hydra before any SAM2 imports
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Initialize Hydra globally once
sam2_path = Path(__file__).parent.parent / "sam2"
if not GlobalHydra().is_initialized():
    config_path = str((sam2_path / "sam2" / "configs").absolute())
    initialize_config_dir(config_dir=config_path, version_base=None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IncrementalLabeler:
    """Label only unlabeled images using SAM 2.1"""
    
    def __init__(self):
        """Initialize SAM 2.1"""
        logger.info("Loading SAM 2.1...")
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Checkpoint path (in parent folder's sam2)
            checkpoint = Path(__file__).parent.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
            
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing: {checkpoint}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {device}")
            
            # Build model (use full config path like working script)
            model = build_sam2(
                config_file="sam2.1/sam2.1_hiera_l",
                ckpt_path=str(checkpoint),
                device=device
            )
            
            self.predictor = SAM2ImagePredictor(model)
            self.device = device
            logger.info("✓ SAM 2.1 ready")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 2.1: {e}")
            raise
    
    def segment_image(self, image_path):
        """Segment with SAM 2.1"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            self.predictor.set_image(image_rgb)
            
            # Bounding box (10% margin)
            input_box = np.array([
                int(w * 0.1), int(h * 0.1),
                int(w * 0.9), int(h * 0.9)
            ])
            
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=False
            )
            
            mask = masks[0]
            binary_mask = (mask > 0).astype(np.uint8)
            
            return binary_mask
            
        except Exception as e:
            logger.debug(f"Error: {e}")
            return None
    
    def run(self, image_dir, mask_dir):
        """Label only unlabeled images"""
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        all_images = sorted(image_dir.glob("*.tif"))
        logger.info(f"Total images: {len(all_images)}")
        
        # Check which are already labeled
        existing_masks = set()
        for mask_file in mask_dir.glob("*_mask.png"):
            # Remove '_mask' suffix to get original image name
            original_name = mask_file.stem.replace('_mask', '')
            existing_masks.add(original_name)
        
        logger.info(f"Already labeled: {len(existing_masks)}")
        
        # Find unlabeled images
        unlabeled = [img for img in all_images if img.stem not in existing_masks]
        logger.info(f"Need labeling: {len(unlabeled)}")
        
        if not unlabeled:
            logger.info("✓ All images already labeled!")
            return len(existing_masks)
        
        # Show breakdown by dataset
        kualalipis_unlabeled = [img for img in unlabeled if not img.name.startswith('Tawau_')]
        tawau_unlabeled = [img for img in unlabeled if img.name.startswith('Tawau_')]
        
        logger.info(f"\n📊 Unlabeled breakdown:")
        logger.info(f"   Kuala Lipis: {len(kualalipis_unlabeled)}")
        logger.info(f"   Tawau: {len(tawau_unlabeled)}")
        
        # Ask user
        print("\n" + "=" * 70)
        print("🏷️  Labeling Options:")
        print("=" * 70)
        print(f"  1. Label ALL unlabeled ({len(unlabeled)} images) - Recommended")
        print(f"  2. Label every 2nd unlabeled (~{len(unlabeled)//2} images)")
        print(f"  3. Label every 3rd unlabeled (~{len(unlabeled)//3} images)")
        print(f"  4. Label every 5th unlabeled (~{len(unlabeled)//5} images)")
        print(f"  5. Cancel")
        print("=" * 70)
        
        choice = input("\nYour choice (1-5): ").strip()
        
        if choice == '5':
            logger.info("Cancelled")
            return len(existing_masks)
        
        sample_rates = {'1': 1, '2': 2, '3': 3, '4': 5}
        sample_rate = sample_rates.get(choice, 1)
        
        sampled = unlabeled[::sample_rate]
        
        logger.info(f"\n🚀 Will label {len(sampled)} images (sample rate: {sample_rate})")
        
        # Estimate time
        time_per_image = 20  # seconds
        estimated_hours = (len(sampled) * time_per_image) / 3600
        logger.info(f"⏱️  Estimated time: {estimated_hours:.1f} hours")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Cancelled")
            return len(existing_masks)
        
        logger.info("\n" + "=" * 70)
        logger.info("🏷️  Starting SAM 2.1 Labeling")
        logger.info("=" * 70)
        
        success = 0
        failed = []
        
        for img_path in tqdm(sampled, desc="Labeling", unit="img"):
            mask = self.segment_image(img_path)
            if mask is not None:
                out_path = mask_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(out_path), mask * 255)
                success += 1
            else:
                failed.append(img_path.name)
        
        logger.info("=" * 70)
        logger.info(f"✅ Labeling Complete!")
        logger.info("=" * 70)
        logger.info(f"📊 Results:")
        logger.info(f"   New masks created: {success}")
        logger.info(f"   Total masks now: {len(existing_masks) + success}")
        logger.info(f"   Failed: {len(failed)}")
        logger.info("=" * 70)
        
        if failed:
            logger.warning(f"Failed images (first 10): {failed[:10]}")
        
        return len(existing_masks) + success


def main():
    root = Path(__file__).parent
    image_dir = root / "data" / "raw_images"
    mask_dir = root / "data" / "masks"
    
    logger.info("=" * 70)
    logger.info("🏷️  Incremental Image Labeler")
    logger.info("   (Labels only new/unlabeled images)")
    logger.info("=" * 70)
    
    if not image_dir.exists():
        logger.error(f"Missing: {image_dir}")
        logger.info("\n💡 First run: python merge_datasets.py")
        return
    
    try:
        labeler = IncrementalLabeler()
        total_masks = labeler.run(image_dir, mask_dir)
        
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Prepare training data: python src/training/prepare_data.py")
        logger.info("  2. Train model: python src/training/train_unet_improved.py")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

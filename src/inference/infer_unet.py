"""
U-Net Inference on Unlabeled Images
Segments all oil palm images using trained model
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from unet_model import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Inferencer:
    """Run inference on unlabeled images"""
    
    def __init__(self, model_path, device="cuda", img_size=256):
        """Load trained model"""
        self.device = device
        self.img_size = img_size
        
        self.model = UNet(in_channels=3, num_classes=1)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"✓ Model loaded: {Path(model_path).name}")
    
    def infer_image(self, image_path):
        """Segment single image"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                return None
            
            h, w = image.shape[:2]
            
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            mask = cv2.resize(pred, (w, h))
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            return binary_mask
            
        except Exception as e:
            logger.debug(f"Error: {Path(image_path).name}: {e}")
            return None
    
    def process_folder(self, input_dir, output_dir):
        """Process all images"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(input_dir.glob("*.tif"))
        
        if not image_files:
            logger.error(f"No .tif files found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 70)
        
        success_count = 0
        
        for image_path in tqdm(image_files, desc="Inference"):
            mask = self.infer_image(image_path)
            
            if mask is not None:
                mask_name = image_path.stem + "_mask.png"
                mask_path = output_dir / mask_name
                mask_png = mask * 255
                cv2.imwrite(str(mask_path), mask_png)
                success_count += 1
        
        logger.info("=" * 70)
        logger.info(f"✓ Inference complete!")
        logger.info(f"  Segmented: {success_count}/{len(image_files)}")


def main():
    """Main inference"""
    
    input_dir = Path(__file__).parent.parent.parent / "data" / "raw_images"
    output_dir = Path(__file__).parent.parent.parent / "data" / "masks_inferred"
    model_path = Path(__file__).parent.parent.parent / "models" / "unet_final.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info("=" * 70)
    logger.info("U-Net Inference on All Oil Palm Images")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    inferencer = Inferencer(model_path, device=device)
    inferencer.process_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()

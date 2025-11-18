"""
Evaluate U-Net Model on Test Set
Calculates accuracy metrics and shows results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
import cv2
import numpy as np
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from unet_model import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OilPalmDataset(Dataset):
    """Dataset for oil palm segmentation"""
    
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.image_files = sorted(self.images_dir.glob("*.tif"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        mask_path = self.masks_dir / f"{img_path.stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        image = image.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


class Evaluator:
    """Evaluate U-Net model"""
    
    def __init__(self, model_path, device="cuda"):
        """Load trained model"""
        self.device = device
        
        self.model = UNet(in_channels=3, num_classes=1)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"✓ Model loaded: {Path(model_path).name}")
    
    def calculate_iou(self, pred, target):
        """Calculate Intersection over Union"""
        pred = (pred > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def calculate_dice(self, pred, target):
        """Calculate Dice coefficient"""
        pred = (pred > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred, target).sum()
        dice = 2 * intersection / (pred.sum() + target.sum() + 1e-8)
        
        return dice
    
    def calculate_metrics(self, pred, target):
        """Calculate all metrics"""
        pred_binary = (pred > 0.5).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)
        
        iou = self.calculate_iou(pred, target)
        dice = self.calculate_dice(pred, target)
        
        tp = np.logical_and(pred_binary, target_binary).sum()
        fp = np.logical_and(pred_binary, ~target_binary).sum()
        fn = np.logical_and(~pred_binary, target_binary).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        all_metrics = {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                targets = masks.cpu().numpy()
                
                for pred, target in zip(preds, targets):
                    metrics = self.calculate_metrics(pred, target)
                    for key in all_metrics:
                        all_metrics[key].append(metrics[key])
        
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        return avg_metrics, all_metrics
    
    def print_results(self, avg_metrics, all_metrics):
        """Print evaluation results"""
        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS (Test Set)")
        logger.info("=" * 70)
        logger.info(f"IoU:       {avg_metrics['iou']:.4f}")
        logger.info(f"Dice:      {avg_metrics['dice']:.4f}")
        logger.info(f"Precision: {avg_metrics['precision']:.4f}")
        logger.info(f"Recall:    {avg_metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {avg_metrics['f1']:.4f}")
        logger.info("=" * 70)
        
        results = {
            'average_metrics': avg_metrics,
            'all_metrics': {k: [float(v) for v in vs] for k, vs in all_metrics.items()}
        }
        
        results_path = Path(__file__).parent.parent.parent / "results" / "evaluation_metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Metrics saved: results/evaluation_metrics.json")


def main():
    """Main evaluation"""
    
    test_images_dir = Path(__file__).parent.parent.parent / "data" / "prepared" / "test" / "images"
    test_masks_dir = Path(__file__).parent.parent.parent / "data" / "prepared" / "test" / "masks"
    model_path = Path(__file__).parent.parent.parent / "models" / "unet_final.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info("=" * 70)
    logger.info("U-Net Model Evaluation")
    logger.info("=" * 70)
    
    test_dataset = OilPalmDataset(test_images_dir, test_masks_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(model_path, device=device)
    
    avg_metrics, all_metrics = evaluator.evaluate(test_loader)
    evaluator.print_results(avg_metrics, all_metrics)


if __name__ == "__main__":
    main()

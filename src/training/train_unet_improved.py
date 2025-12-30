"""
Improved U-Net Training with Data Augmentation and Dice Loss
Enhanced version with better performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice Loss for better segmentation"""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class OilPalmDataset(Dataset):
    """Dataset with heavy augmentation"""
    
    def __init__(self, images_dir, masks_dir, img_size=256, augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Define augmentation transforms
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=45, 
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                ], p=0.3),
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        
        self.image_files = sorted(list(self.images_dir.glob("*.tif")))
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_path.stem + "_mask.png"
        mask_path = self.masks_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Ensure binary mask
        mask = (mask > 127).astype(np.uint8)
        
        # Apply augmentation
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0).float()
        
        return image_tensor, mask_tensor


class ImprovedTrainer:
    """Improved trainer with better optimization"""
    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Use combined loss (BCE + Dice)
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        
        # Adam optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=30,  # 30 epochs for CPU
            eta_min=1e-6
        )
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, save_dir="models"):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("Starting Improved Training")
        logger.info("=" * 70)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Loss Function: Combined (BCE + Dice)")
        logger.info(f"Augmentation: Enabled on training set")
        logger.info("=" * 70)
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info("-" * 70)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.best_model_state, save_dir / "unet_best_v2.pt")
                logger.info(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = save_dir / f"unet_v2_epoch_{epoch}.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"✓ Checkpoint saved: {checkpoint_path.name}")
        
        # Save final model as v2
        final_path = save_dir / "unet_final_v2.pt"
        torch.save(self.best_model_state, final_path)
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ Training Complete!")
        logger.info(f"✓ Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"✓ Final model saved: {final_path}")
        logger.info("=" * 70)


def main():
    """Main training function"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "prepared"
    
    # Hyperparameters
    IMG_SIZE = 256
    BATCH_SIZE = 4  # Reduced for CPU training
    LEARNING_RATE = 1e-4
    EPOCHS = 30  # Reduced for CPU (was 100)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create datasets with augmentation
    logger.info("\nLoading datasets...")
    train_dataset = OilPalmDataset(
        data_dir / "train" / "images",
        data_dir / "train" / "masks",
        img_size=IMG_SIZE,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = OilPalmDataset(
        data_dir / "val" / "images",
        data_dir / "val" / "masks",
        img_size=IMG_SIZE,
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = UNet(in_channels=3, num_classes=1)
    
    # Create trainer
    trainer = ImprovedTrainer(model, device, learning_rate=LEARNING_RATE)
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        save_dir=project_root / "models"
    )


if __name__ == "__main__":
    main()

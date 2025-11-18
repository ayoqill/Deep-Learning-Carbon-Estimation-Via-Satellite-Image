#!/usr/bin/env python3
"""
U-Net Training Script for Oil Palm Segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
import numpy as np
import cv2
from tqdm import tqdm

from unet_model import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OilPalmDataset(Dataset):
    """Dataset for oil palm segmentation"""
    
    def __init__(self, images_dir, masks_dir, img_size=256):
        """
        Args:
            images_dir: Directory with training images
            masks_dir: Directory with training masks
            img_size: Resize images to this size
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted(self.images_dir.glob("*.tif"))
        
        logger.info(f"Dataset: Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning(f"Failed to load {img_path}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Load mask
        mask_path = self.masks_dir / f"{img_path.stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            logger.warning(f"Failed to load mask {mask_path}")
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert mask to binary [0, 1]
        mask = (mask > 128).astype(np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask


class Trainer:
    """Trainer class for U-Net"""
    
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            logger.info(f"✓ New best model! Val loss: {avg_loss:.4f}")
            self.save_checkpoint("unet_best.pt")
            return True
        
        return False
    
    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop"""
        logger.info(f"Training for {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info("=" * 70)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            logger.info(f"Val Loss: {self.val_losses[-1]:.4f}")
            
            # Update learning rate
            self.scheduler.step(self.val_losses[-1])
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"unet_epoch_{epoch+1}.pt")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Training complete!")
        logger.info("=" * 70)
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        path = Path(__file__).parent.parent.parent / "models" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint saved: {path}")


def main():
    """Main training function"""
    
    # Paths
    data_dir = Path(__file__).parent.parent.parent / "data" / "prepared"
    train_images = data_dir / "train" / "images"
    train_masks = data_dir / "train" / "masks"
    val_images = data_dir / "val" / "images"
    val_masks = data_dir / "val" / "masks"
    
    # Check paths
    if not train_images.exists():
        logger.error("Please run prepare_data.py first!")
        return
    
    logger.info("=" * 70)
    logger.info("U-Net Training for Oil Palm Segmentation")
    logger.info("=" * 70)
    
    # Create datasets
    train_dataset = OilPalmDataset(train_images, train_masks)
    val_dataset = OilPalmDataset(val_images, val_masks)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, num_classes=1)
    logger.info(f"Created U-Net model (device: {device})")
    
    # Train
    trainer = Trainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=50)
    
    # Save final model
    final_path = Path(__file__).parent.parent.parent / "models" / "unet_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    logger.info(f"✓ Final model saved: {final_path}")


if __name__ == "__main__":
    main()

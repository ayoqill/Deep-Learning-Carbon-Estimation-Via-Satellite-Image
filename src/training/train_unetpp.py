import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import cv2

import segmentation_models_pytorch as smp


# -------------------------
# Config
# -------------------------
TILES_DIR = Path("data/tiles_clean")
MASKS_DIR = Path("data/masks_refined")
SAVE_PATH = "unetpp_best.pth"

PAD_TO = 160            # 157x157 -> pad to 160x160
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
SEED = 42

# ✅ Apple Silicon (M1/M2/M3) support + fallback
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("✅ Using device:", DEVICE)

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1


# -------------------------
# Utils
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_tif_4band(path: Path) -> np.ndarray:
    """Return image as (H,W,4) float32."""
    with rasterio.open(path) as src:
        img = src.read()  # (C,H,W)
    img = np.transpose(img, (1, 2, 0)).astype(np.float32)  # (H,W,C)
    return img

def normalize_per_channel(img: np.ndarray) -> np.ndarray:
    """Normalize each channel to 0..1 using 2-98 percentile."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi - lo < 1e-6:
            out[:, :, c] = 0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out

def read_mask_png(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.float32)  # (H,W) in {0,1}

def pad_to_size(img: np.ndarray, mask: np.ndarray, target: int = PAD_TO):
    """
    img: (H,W,C), mask: (H,W)
    pad to (target,target) without resizing/distortion.
    """
    H, W = img.shape[:2]
    if H == target and W == target:
        return img, mask

    if H > target or W > target:
        img = img[:target, :target, :]
        mask = mask[:target, :target]
        return img, mask

    pad_h = target - H
    pad_w = target - W

    # image: reflect padding preserves texture
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    # mask: constant 0 padding = background
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    return img, mask

def dice_coef(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


# -------------------------
# Dataset
# -------------------------
class MangroveDataset(Dataset):
    def __init__(self, tile_paths, mask_paths):
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        mask_path = self.mask_paths[idx]

        img = read_tif_4band(tile_path)          # (H,W,4)
        img = normalize_per_channel(img)         # 0..1
        mask = read_mask_png(mask_path)          # (H,W)

        # pad 157->160 for stable U-Net++ downsampling
        img, mask = pad_to_size(img, mask, target=PAD_TO)

        # (H,W,C) -> (C,H,W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)          # (4,H,W)
        mask = mask[None, :, :].astype(np.float32)                      # (1,H,W)

        return torch.tensor(img), torch.tensor(mask)


# -------------------------
# Prepare file pairs + split
# -------------------------
def get_pairs():
    tile_files = sorted(TILES_DIR.glob("*.tif"))
    pairs = []
    for t in tile_files:
        m = MASKS_DIR / f"{t.stem}.png"
        if m.exists():
            pairs.append((t, m))
    return pairs

def split_pairs(pairs):
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = pairs[:n_train]
    val   = pairs[n_train:n_train + n_val]
    test  = pairs[n_train + n_val:]
    return train, val, test


# -------------------------
# Train
# -------------------------
def main():
    set_seed()

    pairs = get_pairs()
    print(f"Matched pairs: {len(pairs)}")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    train_ds = MangroveDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs])
    val_ds   = MangroveDataset([p[0] for p in val_pairs],   [p[1] for p in val_pairs])
    test_ds  = MangroveDataset([p[0] for p in test_pairs],  [p[1] for p in test_pairs])

    # ✅ Mac-friendly: num_workers=0 avoids dataloader hangs
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # U-Net++ with 4 input channels (RGB+NIR)
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
        activation=None
    ).to(DEVICE)

    # Loss: BCE + Dice (strong baseline)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def loss_fn(logits, y):
        return 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou  = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_dice += dice_coef(probs, y)
                val_iou  += iou_score(probs, y)

        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)
        val_iou  /= max(len(val_loader), 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"Dice {val_dice:.4f} | IoU {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Saved best model: {SAVE_PATH} (val Dice {best_val_dice:.4f})")

    # ---- test ----
    print("\nTesting best model...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    test_dice = 0.0
    test_iou  = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = torch.sigmoid(model(x))
            test_dice += dice_coef(probs, y)
            test_iou  += iou_score(probs, y)

    test_dice /= max(len(test_loader), 1)
    test_iou  /= max(len(test_loader), 1)
    print(f"✅ Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()

"""
Simple Model Evaluation Script
Tests your trained model on existing data and reports accuracy metrics
"""

from pathlib import Path
import numpy as np
import torch
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model-eval")

project_root = Path(__file__).parent.resolve()
MODEL_PATH = project_root / "models" / "unetpp_best.pth"
TILES_DIR = project_root / "data" / "tiles_clean"
MASKS_DIR = project_root / "data" / "masks_refined"

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.001  # Match your app threshold

def load_model():
    """Load trained model"""
    logger.info(f"Loading model from {MODEL_PATH}")
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    
    # Handle wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    
    # Detect input channels
    for k, v in state.items():
        if "conv1.weight" in k and hasattr(v, "shape"):
            in_channels = int(v.shape[1])
            logger.info(f"Detected in_channels: {in_channels}")
            break
    else:
        in_channels = 4
    
    import segmentation_models_pytorch as smp
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None
    ).to(DEVICE)
    
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("✅ Model loaded")
    return model, in_channels

def load_image(path: Path, channels: int):
    """Load and normalize image tile"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    if len(img.shape) == 2:
        img = np.stack([img] * channels, axis=-1)
    
    # Normalize to 0-1
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    
    # Ensure correct number of channels
    if img.shape[2] < channels:
        pad = np.zeros((img.shape[0], img.shape[1], channels - img.shape[2]))
        img = np.concatenate([img, pad], axis=2)
    elif img.shape[2] > channels:
        img = img[:, :, :channels]
    
    return img.astype(np.float32)

def load_mask(path: Path):
    """Load ground truth mask"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.dtype == np.uint8:
        mask = (mask > 127).astype(np.uint8)
    return mask

def predict_single_tile(model, img):
    """Predict mask for single tile"""
    if img.shape[2] == 1:
        img = np.stack([img[:,:,0]] * 4, axis=-1)
    
    # Store original shape for unpadding
    h, w = img.shape[:2]
    
    # Pad to be divisible by 32
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Unpad to original shape
    probs = probs[:h, :w]
    
    return probs

def evaluate():
    """Main evaluation"""
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load model
    model, in_channels = load_model()
    
    # Get file pairs
    tiles = sorted(TILES_DIR.glob("*.tif"))
    masks = sorted(MASKS_DIR.glob("*.png"))
    
    if len(tiles) == 0 or len(masks) == 0:
        logger.error("❌ No tiles or masks found")
        return
    
    logger.info(f"Found {len(tiles)} tiles and {len(masks)} masks")
    
    # Use first 100 samples for quick evaluation
    sample_size = min(100, len(tiles))
    logger.info(f"Using {sample_size} samples for evaluation")
    
    all_preds = []
    all_gts = []
    
    for i, tile_path in enumerate(tiles[:sample_size]):
        mask_path = MASKS_DIR / tile_path.name.replace(".tif", ".png")
        
        if not mask_path.exists():
            continue
        
        # Load data
        img = load_image(tile_path, in_channels)
        gt = load_mask(mask_path)
        
        if img is None or gt is None:
            continue
        
        # Predict
        prob = predict_single_tile(model, img)
        pred = (prob >= THRESHOLD).astype(np.uint8)
        
        # Collect
        all_preds.extend(pred.flatten())
        all_gts.extend(gt.flatten())
        
        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i + 1}/{sample_size}")
    
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    # Calculate metrics
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    precision = precision_score(all_gts, all_preds, zero_division=0)
    recall = recall_score(all_gts, all_preds, zero_division=0)
    f1 = f1_score(all_gts, all_preds, zero_division=0)
    accuracy = accuracy_score(all_gts, all_preds)
    
    logger.info(f"\n📊 METRICS:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    logger.info(f"  F1-Score:  {f1:.4f}")
    
    # Interpretation
    logger.info(f"\n🎯 INTERPRETATION:")
    if accuracy >= 0.90:
        logger.info("  ✅ EXCELLENT - Model is very accurate")
    elif accuracy >= 0.80:
        logger.info("  ✅ GOOD - Model performs well")
    elif accuracy >= 0.70:
        logger.info("  ⚠️  FAIR - Model is usable but could improve")
    else:
        logger.info("  ❌ POOR - Model needs improvement")
    
    if precision >= 0.80:
        logger.info("  ✅ HIGH PRECISION - Few false positives")
    else:
        logger.info("  ⚠️  MEDIUM PRECISION - Some false positives")
    
    if recall >= 0.80:
        logger.info("  ✅ HIGH RECALL - Finds most mangroves")
    else:
        logger.info("  ⚠️  MEDIUM RECALL - Misses some mangroves")
    
    # Confusion matrix
    cm = confusion_matrix(all_gts, all_preds)
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"  True Negatives:  {cm[0,0]}")
    logger.info(f"  False Positives: {cm[0,1]}")
    logger.info(f"  False Negatives: {cm[1,0]}")
    logger.info(f"  True Positives:  {cm[1,1]}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model is {'✅ GOOD ENOUGH' if accuracy >= 0.80 else '⚠️  NEEDS IMPROVEMENT'}")
    logger.info(f"Best for: {'High coverage (recall={recall:.2%})' if recall > precision else f'High accuracy (precision={precision:.2%})'}")
    logger.info("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate()
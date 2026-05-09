"""
Evaluate Model Accuracy at Specific Threshold (0.01)
Tests your trained DeepLabV3+ model on test data with threshold 0.01
"""

from pathlib import Path
import numpy as np
import torch
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, jaccard_score
import logging
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("threshold-eval")

project_root = Path(__file__).parent.resolve()
MODEL_PATH = project_root / "models" / "deeplabv3" / "deeplabv3_best.pth"
TILES_DIR = project_root / "data" / "tiles_clean"
MASKS_DIR = project_root / "data" / "masks_refined"

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.01  # Your threshold

logger.info(f"Device: {DEVICE}")
logger.info(f"Threshold: {THRESHOLD}")

def load_model():
    """Load trained DeepLabV3+ model"""
    logger.info(f"Loading model from {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return None, None
    
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    
    # Handle wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    
    # Detect input channels
    in_channels = 4
    for k, v in state.items():
        if "conv1.weight" in k and hasattr(v, "shape"):
            in_channels = int(v.shape[1])
            logger.info(f"Detected in_channels: {in_channels}")
            break
    
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None
    ).to(DEVICE)
    
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("✅ Model loaded successfully")
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
    h, w = img.shape[:2]
    
    # Pad to multiple of 16
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    img_tensor = torch.from_numpy(img).float().to(DEVICE)
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    else:
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    
    # Remove padding
    probs = probs[:h, :w]
    
    return probs

def evaluate_threshold(model, in_channels, threshold):
    """Evaluate model at specific threshold"""
    
    if not TILES_DIR.exists() or not MASKS_DIR.exists():
        logger.error(f"Data directories not found. Check {TILES_DIR} and {MASKS_DIR}")
        return None
    
    # Get test images and masks
    image_files = sorted(list(TILES_DIR.glob("*.png")) + list(TILES_DIR.glob("*.jpg")) + list(TILES_DIR.glob("*.tif")))
    
    if not image_files:
        logger.error(f"No images found in {TILES_DIR}")
        return None
    
    # Sample for faster evaluation (use first 100 images for demo)
    sample_size = min(100, len(image_files))
    image_files = image_files[:sample_size]
    logger.info(f"Evaluating {len(image_files)} test images (sample size)")
    
    all_preds = []
    all_gts = []
    
    logger.info(f"Evaluating {len(image_files)} test images...")
    
    for idx, img_path in enumerate(image_files):
        # Convert .tif to .png for mask lookup
        mask_name = img_path.stem + ".png"
        mask_path = MASKS_DIR / mask_name
        
        if not mask_path.exists():
            continue
        
        # Load image and mask
        img = load_image(img_path, in_channels)
        mask = load_mask(mask_path)
        
        if img is None or mask is None:
            continue
        
        # Predict
        prob_map = predict_single_tile(model, img)
        
        # Apply threshold
        pred_mask = (prob_map >= threshold).astype(np.uint8)
        
        # Flatten for metrics
        all_preds.extend(pred_mask.flatten())
        all_gts.extend(mask.flatten())
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(image_files)} images")
    
    if not all_preds:
        logger.error("No predictions made. Check your data.")
        return None
    
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    # Calculate metrics
    accuracy = accuracy_score(all_gts, all_preds)
    precision = precision_score(all_gts, all_preds, zero_division=0)
    recall = recall_score(all_gts, all_preds, zero_division=0)
    f1 = f1_score(all_gts, all_preds, zero_division=0)
    iou = jaccard_score(all_gts, all_preds, zero_division=0)
    
    # Dice score (same as F1 for binary classification)
    dice = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_gts, all_preds).ravel()
    
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "dice_score": dice,
        "iou": iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_pixels": len(all_gts),
        "mangrove_pixels": int(all_preds.sum()),
        "gt_mangrove_pixels": int(all_gts.sum()),
    }

def main():
    logger.info("="*60)
    logger.info("Threshold-Specific Accuracy Evaluation")
    logger.info("="*60)
    
    # Load model
    model, in_channels = load_model()
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Evaluate at threshold 0.01
    logger.info(f"\nEvaluating at threshold: {THRESHOLD}")
    results = evaluate_threshold(model, in_channels, THRESHOLD)
    
    if results is None:
        logger.error("Evaluation failed")
        return
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD 0.01 RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy:        {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"Precision:       {results['precision']:.4f}")
    logger.info(f"Recall:          {results['recall']:.4f}")
    logger.info(f"F1 Score:        {results['f1_score']:.4f}")
    logger.info(f"Dice Score:      {results['dice_score']:.4f}")
    logger.info(f"IoU:             {results['iou']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {results['tn']}")
    logger.info(f"  False Positives: {results['fp']}")
    logger.info(f"  False Negatives: {results['fn']}")
    logger.info(f"  True Positives:  {results['tp']}")
    logger.info(f"\nPixel Statistics:")
    logger.info(f"  Total Pixels:           {results['total_pixels']}")
    logger.info(f"  Predicted Mangrove:     {results['mangrove_pixels']}")
    logger.info(f"  Ground Truth Mangrove:  {results['gt_mangrove_pixels']}")
    logger.info("="*60)
    
    # Save results
    results_file = project_root / "threshold_0.01_results.txt"
    with open(results_file, "w") as f:
        f.write("THRESHOLD 0.01 EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy:        {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:       {results['precision']:.4f}\n")
        f.write(f"Recall:          {results['recall']:.4f}\n")
        f.write(f"F1 Score:        {results['f1_score']:.4f}\n")
        f.write(f"Dice Score:      {results['dice_score']:.4f}\n")
        f.write(f"IoU:             {results['iou']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  True Negatives:  {results['tn']}\n")
        f.write(f"  False Positives: {results['fp']}\n")
        f.write(f"  False Negatives: {results['fn']}\n")
        f.write(f"  True Positives:  {results['tp']}\n")
        f.write(f"\nPixel Statistics:\n")
        f.write(f"  Total Pixels:           {results['total_pixels']}\n")
        f.write(f"  Predicted Mangrove:     {results['mangrove_pixels']}\n")
        f.write(f"  Ground Truth Mangrove:  {results['gt_mangrove_pixels']}\n")
    
    logger.info(f"\n✅ Results saved to {results_file}")

if __name__ == "__main__":
    main()

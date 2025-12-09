"""
Backup current model before training improved version
"""

from pathlib import Path
import shutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_model():
    """Backup existing model and results"""
    
    project_root = Path(__file__).parent
    
    # Create backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / f"backup_baseline_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("BACKING UP BASELINE MODEL")
    logger.info("=" * 70)
    
    # Backup model
    model_src = project_root / "models" / "unet_final.pt"
    if model_src.exists():
        model_dst = backup_dir / "unet_baseline.pt"
        shutil.copy2(model_src, model_dst)
        logger.info(f"✓ Model backed up: {model_dst}")
    
    # Backup results
    results_src = project_root / "results"
    if results_src.exists():
        results_dst = backup_dir / "results_baseline"
        shutil.copytree(results_src, results_dst, dirs_exist_ok=True)
        logger.info(f"✓ Results backed up: {results_dst}")
    
    # Create README
    readme = backup_dir / "README.txt"
    with open(readme, 'w') as f:
        f.write(f"""BASELINE MODEL BACKUP
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Performance Metrics:
- IoU: 0.7366
- Dice: 0.8209
- Precision: 0.4492
- Recall: 0.4497
- F1-Score: 0.4398

Training Configuration:
- Epochs: 50
- Batch Size: 8
- Learning Rate: 1e-4
- Loss: BCEWithLogitsLoss
- Augmentation: None

This backup was created before training the improved model.
""")
    
    logger.info(f"✓ README created: {readme}")
    logger.info("=" * 70)
    logger.info(f"✓ Backup complete: {backup_dir}")
    logger.info("=" * 70)
    
    return backup_dir

if __name__ == "__main__":
    backup_model()

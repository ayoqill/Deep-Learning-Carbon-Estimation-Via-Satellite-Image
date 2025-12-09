"""
Quick Start Guide for Improved Training

Run this to start improved training with:
✅ Data Augmentation (flips, rotations, brightness, noise)
✅ Dice Loss (better for segmentation)
✅ 100 Epochs (instead of 50)
✅ Cosine Annealing LR Scheduler
✅ Gradient Clipping
✅ Weight Decay
"""

import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("IMPROVED U-NET TRAINING")
print("=" * 70)
print()
print("🚀 Improvements over original:")
print("   1. ✅ Heavy data augmentation (6 types)")
print("   2. ✅ Combined BCE + Dice Loss")
print("   3. ✅ 100 epochs (was 50)")
print("   4. ✅ Cosine annealing scheduler")
print("   5. ✅ Gradient clipping")
print("   6. ✅ Weight decay regularization")
print()
print("📊 Expected results:")
print("   - Current IoU: 0.7366")
print("   - Expected IoU: 0.83-0.88 (+10-15%)")
print("   - Training time: ~30-40 hours (CPU)")
print()
print("=" * 70)
print()

response = input("Start improved training? (y/n): ")

if response.lower() == 'y':
    print("\n🚀 Starting training...\n")
    script_path = Path(__file__).parent / "train_unet_improved.py"
    subprocess.run([sys.executable, str(script_path)])
else:
    print("\n❌ Training cancelled")

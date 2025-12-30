#!/usr/bin/env python3
"""
Monitor U-Net training progress
"""

from pathlib import Path
import os

def check_training_status():
    """Check if training is still running"""
    models_dir = Path(__file__).parent.parent / "models"
    
    # Check for model files
    checkpoint_files = list(models_dir.glob("*.pt"))
    
    print("\n" + "=" * 70)
    print("U-Net Training Status")
    print("=" * 70)
    
    if not checkpoint_files:
        print("⏳ Training in progress (no checkpoints yet)")
        print("First checkpoint expected in ~15-20 minutes")
    else:
        print(f"✅ {len(checkpoint_files)} checkpoint(s) found:")
        for f in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_mb = f.stat().st_size / 1e6
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print("\nExpected completion:")
    print("  - 50 epochs × ~18 min/epoch = ~15 hours")
    print("  - Started: ~10:40 AM")
    print("  - Expected finish: ~1:40 AM (next day)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_training_status()

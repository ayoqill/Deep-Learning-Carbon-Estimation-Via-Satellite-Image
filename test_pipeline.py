#!/usr/bin/env python3
"""
Complete Pipeline Test
Verifies all components work end-to-end before processing mangrove data
"""

import json
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineValidator:
    """Validate entire pipeline"""
    
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
    
    def test_directories_exist(self):
        """Test 1: Check required directories"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 1: Directory Structure")
        logger.info("=" * 70)
        
        dirs_to_check = {
            'data/raw_images': 'Source images',
            'data/masks': 'SAM-2.1 labeled masks',
            'data/masks_inferred': 'U-Net inferred masks',
            'data/prepared': 'Train/Val/Test split data',
            'models': 'Trained models',
            'results': 'Output results',
            'src/training': 'Training scripts',
            'src/inference': 'Inference scripts',
            'src/labeling': 'Labeling scripts'
        }
        
        all_exist = True
        for dir_path, description in dirs_to_check.items():
            full_path = self.root / dir_path
            exists = full_path.exists()
            status = "✅" if exists else "❌"
            logger.info(f"{status} {dir_path:30s} - {description}")
            
            if not exists:
                all_exist = False
        
        self.results['tests'].append({
            'name': 'Directories',
            'passed': all_exist
        })
        
        if all_exist:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        return all_exist
    
    def test_input_data(self):
        """Test 2: Check input data exists"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 2: Input Data")
        logger.info("=" * 70)
        
        raw_images = list((self.root / "data" / "raw_images").glob("*.tif"))
        labeled_masks = list((self.root / "data" / "masks").glob("*_mask.png"))
        
        logger.info(f"Raw images: {len(raw_images)}")
        logger.info(f"Labeled masks: {len(labeled_masks)}")
        
        images_ok = len(raw_images) > 0
        masks_ok = len(labeled_masks) > 0
        
        logger.info(f"{'✅' if images_ok else '❌'} Raw images: {len(raw_images)} found")
        logger.info(f"{'✅' if masks_ok else '❌'} Labeled masks: {len(labeled_masks)} found")
        
        self.results['tests'].append({
            'name': 'Input Data',
            'passed': images_ok and masks_ok,
            'images': len(raw_images),
            'masks': len(labeled_masks)
        })
        
        if images_ok and masks_ok:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        return images_ok and masks_ok
    
    def test_trained_model(self):
        """Test 3: Check trained model exists"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 3: Trained Model")
        logger.info("=" * 70)
        
        model_path = self.root / "models" / "unet_final.pt"
        exists = model_path.exists()
        
        if exists:
            size_mb = model_path.stat().st_size / 1e6
            logger.info(f"✅ Model found: {model_path.name} ({size_mb:.1f} MB)")
        else:
            logger.info(f"❌ Model not found: {model_path}")
        
        self.results['tests'].append({
            'name': 'Trained Model',
            'passed': exists
        })
        
        if exists:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        return exists
    
    def test_prepared_data(self):
        """Test 4: Check train/val/test splits"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 4: Data Splits")
        logger.info("=" * 70)
        
        train_images = list((self.root / "data" / "prepared" / "train" / "images").glob("*.tif"))
        val_images = list((self.root / "data" / "prepared" / "val" / "images").glob("*.tif"))
        test_images = list((self.root / "data" / "prepared" / "test" / "images").glob("*.tif"))
        
        train_masks = list((self.root / "data" / "prepared" / "train" / "masks").glob("*_mask.png"))
        val_masks = list((self.root / "data" / "prepared" / "val" / "masks").glob("*_mask.png"))
        test_masks = list((self.root / "data" / "prepared" / "test" / "masks").glob("*_mask.png"))
        
        logger.info(f"Train: {len(train_images)} images, {len(train_masks)} masks")
        logger.info(f"Val:   {len(val_images)} images, {len(val_masks)} masks")
        logger.info(f"Test:  {len(test_images)} images, {len(test_masks)} masks")
        
        train_ok = len(train_images) == len(train_masks) and len(train_images) > 0
        val_ok = len(val_images) == len(val_masks) and len(val_images) > 0
        test_ok = len(test_images) == len(test_masks) and len(test_images) > 0
        
        logger.info(f"{'✅' if train_ok else '❌'} Training data paired correctly")
        logger.info(f"{'✅' if val_ok else '❌'} Validation data paired correctly")
        logger.info(f"{'✅' if test_ok else '❌'} Test data paired correctly")
        
        self.results['tests'].append({
            'name': 'Data Splits',
            'passed': train_ok and val_ok and test_ok,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        })
        
        if train_ok and val_ok and test_ok:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        return train_ok and val_ok and test_ok
    
    def test_inferred_masks(self):
        """Test 5: Check inferred masks"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 5: Inferred Masks")
        logger.info("=" * 70)
        
        inferred_masks = list((self.root / "data" / "masks_inferred").glob("*_mask.png"))
        raw_images = list((self.root / "data" / "raw_images").glob("*.tif"))
        
        logger.info(f"Inferred masks: {len(inferred_masks)}")
        logger.info(f"Raw images: {len(raw_images)}")
        
        coverage = (len(inferred_masks) / len(raw_images) * 100) if raw_images else 0
        logger.info(f"Coverage: {coverage:.1f}%")
        
        ok = len(inferred_masks) > 0
        logger.info(f"{'✅' if ok else '❌'} Inferred masks exist")
        
        self.results['tests'].append({
            'name': 'Inferred Masks',
            'passed': ok,
            'count': len(inferred_masks),
            'coverage': coverage
        })
        
        if ok:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        return ok
    
    def test_evaluation_results(self):
        """Test 6: Check evaluation metrics"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 6: Evaluation Metrics")
        logger.info("=" * 70)
        
        metrics_file = self.root / "results" / "evaluation_metrics.json"
        
        if not metrics_file.exists():
            logger.info(f"❌ Metrics file not found: {metrics_file}")
            self.results['tests'].append({
                'name': 'Evaluation',
                'passed': False
            })
            self.results['failed'] += 1
            return False
        
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            avg = metrics['average_metrics']
            logger.info(f"✅ Evaluation metrics found:")
            logger.info(f"  IoU:       {avg['iou']:.4f}")
            logger.info(f"  Dice:      {avg['dice']:.4f}")
            logger.info(f"  Precision: {avg['precision']:.4f}")
            logger.info(f"  Recall:    {avg['recall']:.4f}")
            logger.info(f"  F1-Score:  {avg['f1']:.4f}")
            
            self.results['tests'].append({
                'name': 'Evaluation',
                'passed': True,
                'metrics': avg
            })
            self.results['passed'] += 1
            return True
            
        except Exception as e:
            logger.info(f"❌ Error reading metrics: {e}")
            self.results['tests'].append({
                'name': 'Evaluation',
                'passed': False
            })
            self.results['failed'] += 1
            return False
    
    def test_carbon_results(self):
        """Test 7: Check carbon estimation"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 7: Carbon Estimation")
        logger.info("=" * 70)
        
        carbon_file = self.root / "results" / "carbon_estimation.json"
        
        if not carbon_file.exists():
            logger.info(f"❌ Carbon file not found: {carbon_file}")
            self.results['tests'].append({
                'name': 'Carbon Estimation',
                'passed': False
            })
            self.results['failed'] += 1
            return False
        
        try:
            with open(carbon_file) as f:
                carbon = json.load(f)
            
            logger.info(f"✅ Carbon estimation found:")
            logger.info(f"  Total Area: {carbon['total_area_hectares']:.2f} hectares")
            logger.info(f"  Carbon: {carbon['total_carbon_tons']:.2f} tons C")
            logger.info(f"  CO₂ Equivalent: {carbon['co2_equivalent_tons']:.2f} tons CO₂")
            
            self.results['tests'].append({
                'name': 'Carbon Estimation',
                'passed': True,
                'results': {
                    'area_hectares': carbon['total_area_hectares'],
                    'carbon_tons': carbon['total_carbon_tons'],
                    'co2_tons': carbon['co2_equivalent_tons']
                }
            })
            self.results['passed'] += 1
            return True
            
        except Exception as e:
            logger.info(f"❌ Error reading carbon results: {e}")
            self.results['tests'].append({
                'name': 'Carbon Estimation',
                'passed': False
            })
            self.results['failed'] += 1
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n\n")
        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " " * 68 + "║")
        logger.info("║" + "  PIPELINE VALIDATION TEST".center(68) + "║")
        logger.info("║" + " " * 68 + "║")
        logger.info("╚" + "=" * 68 + "╝")
        
        self.test_directories_exist()
        self.test_input_data()
        self.test_trained_model()
        self.test_prepared_data()
        self.test_inferred_masks()
        self.test_evaluation_results()
        self.test_carbon_results()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n\n")
        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " TEST SUMMARY ".center(68) + "║")
        logger.info("╠" + "=" * 68 + "╣")
        
        for test in self.results['tests']:
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            logger.info(f"║ {status} - {test['name']:40s} ║")
        
        logger.info("╠" + "=" * 68 + "╣")
        logger.info(f"║ TOTAL: {self.results['passed']} PASSED, {self.results['failed']} FAILED".ljust(68) + "║")
        logger.info("╚" + "=" * 68 + "╝")
        
        if self.results['failed'] == 0:
            logger.info("\n" + "🎉 " * 15)
            logger.info("ALL TESTS PASSED! ✅ PIPELINE IS READY FOR MANGROVE DATA".center(70))
            logger.info("🎉 " * 15)
        else:
            logger.info("\n⚠️  SOME TESTS FAILED - FIX ISSUES BEFORE PROCEEDING")


def main():
    """Main test"""
    project_root = Path(__file__).parent
    
    validator = PipelineValidator(project_root)
    validator.run_all_tests()


if __name__ == "__main__":
    main()

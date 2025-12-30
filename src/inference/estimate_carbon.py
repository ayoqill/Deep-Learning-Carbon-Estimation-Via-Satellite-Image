"""
Carbon Stock Estimation from Segmentation Masks
Calculates total carbon based on segmented oil palm area
"""

import numpy as np
from pathlib import Path
import logging
import cv2
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CarbonEstimator:
    """Estimate carbon stock from masks"""
    
    def __init__(self, pixel_to_meters=10, carbon_density=150):
        """Initialize parameters
        
        Args:
            pixel_to_meters: Resolution of satellite image (10m for typical Sentinel-2)
            carbon_density: Carbon density in tC/hectare for oil palm (150 for mature plantation)
        """
        self.pixel_to_meters = pixel_to_meters
        self.carbon_density = carbon_density
    
    def calculate_area(self, mask):
        """Calculate area from mask (in hectares)"""
        palm_pixels = np.sum(mask > 0)
        
        area_m2 = palm_pixels * (self.pixel_to_meters ** 2)
        area_hectares = area_m2 / 10000
        
        return area_hectares, area_m2, palm_pixels
    
    def estimate_carbon(self, area_hectares):
        """Estimate carbon from area"""
        carbon_tons = area_hectares * self.carbon_density
        return carbon_tons
    
    def process_masks(self, mask_dir):
        """Process all masks and calculate carbon"""
        mask_dir = Path(mask_dir)
        
        mask_files = sorted(mask_dir.glob("*_mask.png"))
        
        if not mask_files:
            logger.error(f"No masks found in {mask_dir}")
            return None
        
        logger.info(f"Found {len(mask_files)} masks")
        logger.info("=" * 70)
        
        results = {
            'total_area_hectares': 0.0,
            'total_area_m2': 0.0,
            'total_palm_pixels': 0,
            'total_carbon_tons': 0.0,
            'co2_equivalent_tons': 0.0,
            'images': []
        }
        
        for mask_path in tqdm(mask_files, desc="Calculating carbon"):
            try:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                area_ha, area_m2, pixels = self.calculate_area(mask)
                carbon = self.estimate_carbon(area_ha)
                
                results['total_area_hectares'] += area_ha
                results['total_area_m2'] += area_m2
                results['total_palm_pixels'] += pixels
                results['total_carbon_tons'] += carbon
                
                results['images'].append({
                    'file': mask_path.name,
                    'area_hectares': float(area_ha),
                    'carbon_tons': float(carbon)
                })
                
            except Exception as e:
                logger.debug(f"Error: {mask_path.name}: {e}")
        
        # Convert to CO2 equivalent (1 ton C = 3.67 tons CO2)
        results['co2_equivalent_tons'] = float(results['total_carbon_tons'] * 3.67)
        results['total_area_hectares'] = float(results['total_area_hectares'])
        results['total_area_m2'] = float(results['total_area_m2'])
        results['total_carbon_tons'] = float(results['total_carbon_tons'])
        results['total_palm_pixels'] = int(results['total_palm_pixels'])
        
        return results
    
    def print_results(self, results):
        """Print and save results"""
        logger.info("=" * 70)
        logger.info("🌴 CARBON STOCK ESTIMATION RESULTS 🌴")
        logger.info("=" * 70)
        logger.info(f"Total Images Processed: {len(results['images'])}")
        logger.info(f"Total Oil Palm Area: {results['total_area_hectares']:.2f} hectares")
        logger.info(f"Total Oil Palm Area: {results['total_area_m2']:.0f} m²")
        logger.info("")
        logger.info("CARBON STOCK:")
        logger.info(f"  Carbon: {results['total_carbon_tons']:.2f} tons C")
        logger.info(f"  CO₂ Equivalent: {results['co2_equivalent_tons']:.2f} tons CO₂")
        logger.info("=" * 70)
        
        output_path = Path(__file__).parent.parent.parent / "results" / "carbon_estimation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved: results/carbon_estimation.json")


def main():
    """Main carbon estimation"""
    
    mask_dir = Path(__file__).parent.parent.parent / "data" / "masks_inferred"
    
    if not mask_dir.exists():
        logger.error(f"Mask directory not found: {mask_dir}")
        logger.info("Run inference first: python src/inference/infer_unet.py")
        return
    
    logger.info("=" * 70)
    logger.info("Oil Palm Carbon Stock Estimation")
    logger.info("=" * 70)
    
    estimator = CarbonEstimator(
        pixel_to_meters=10,
        carbon_density=150
    )
    
    results = estimator.process_masks(mask_dir)
    
    if results:
        estimator.print_results(results)


if __name__ == "__main__":
    main()

"""
Automatic mask refinement for SAM2 pseudo-labels.

This module converts SAM2-generated pseudo-masks into verified ground truth
labels through automatic post-processing, without requiring manual editing.

Refinement Pipeline:
1. Remove small isolated regions (noise filtering)
2. Morphological operations (smooth boundaries)
3. Green color filtering (vegetation only)
4. NDVI filtering (if NIR band available)
5. Fill holes in mangrove regions

Academic Note:
- SAM2 output = Pseudo-labels (may contain errors)
- After refinement = Verified masks (ground truth for training)
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskRefinement:
    """
    Automatic mask refinement for SAM2 pseudo-labels.
    
    This class applies rule-based post-processing to convert SAM2
    auto-generated masks into verified ground truth labels suitable
    for training U-Net++ models.
    
    Refinement steps:
    1. Remove small isolated regions (noise)
    2. Morphological operations (smooth boundaries)
    3. Green color filtering (vegetation only)
    4. NDVI filtering (if NIR available)
    5. Fill holes in mangrove regions
    
    Example:
        refiner = MaskRefinement(min_area=500, green_ratio_min=1.1)
        refined_mask = refiner.refine(image, pseudo_mask)
    """
    
    def __init__(
        self,
        min_area: int = 500,
        ndvi_threshold: float = 0.2,
        green_ratio_min: float = 1.1,
        morph_kernel_size: int = 5,
        fill_holes: bool = True
    ):
        """
        Initialize mask refinement parameters.
        
        Args:
            min_area: Minimum region size in pixels (removes noise)
            ndvi_threshold: NDVI threshold for vegetation (0.2-0.4 typical)
            green_ratio_min: Minimum green/red and green/blue ratio
            morph_kernel_size: Kernel size for morphological operations
            fill_holes: Whether to fill holes in mangrove regions
        """
        self.min_area = min_area
        self.ndvi_threshold = ndvi_threshold
        self.green_ratio_min = green_ratio_min
        self.fill_holes_enabled = fill_holes
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )
    
    def refine(
        self,
        image: np.ndarray,
        pseudo_mask: np.ndarray,
        apply_green_filter: bool = True,
        apply_ndvi_filter: bool = False,
        nir_channel: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply all refinement steps to pseudo-mask.
        
        Args:
            image: Original satellite image (H, W, C) in BGR or RGB format
            pseudo_mask: SAM2 generated mask (H, W), values 0-1 or 0-255
            apply_green_filter: Whether to filter by green color
            apply_ndvi_filter: Whether to filter by NDVI (requires NIR)
            nir_channel: Channel index for NIR band (if available)
        
        Returns:
            Refined mask (H, W), values 0 or 1 (float32)
        
        Example:
            refiner = MaskRefinement(min_area=500)
            refined = refiner.refine(image, sam2_mask, apply_green_filter=True)
        """
        mask = pseudo_mask.copy().astype(np.float32)
        
        # Normalize mask to 0-1 if needed
        if mask.max() > 1:
            mask = mask / 255.0
        
        original_pixels = mask.sum()
        logger.info(f"Starting mask refinement... Original pixels: {original_pixels:.0f}")
        
        # Step 1: Remove small isolated regions
        mask = self._remove_small_regions(mask)
        logger.info(f"After small region removal: {mask.sum():.0f} pixels")
        
        # Step 2: Morphological cleanup
        mask = self._morphological_cleanup(mask)
        logger.info(f"After morphological cleanup: {mask.sum():.0f} pixels")
        
        # Step 3: Green color filtering
        if apply_green_filter and image is not None:
            mask = self._filter_by_green(image, mask)
            logger.info(f"After green filtering: {mask.sum():.0f} pixels")
        
        # Step 4: NDVI filtering
        if apply_ndvi_filter and nir_channel is not None and image is not None:
            mask = self._filter_by_ndvi(image, mask, nir_channel)
            logger.info(f"After NDVI filtering: {mask.sum():.0f} pixels")
        
        # Step 5: Fill holes
        if self.fill_holes_enabled:
            mask = self._fill_holes(mask)
            logger.info(f"After hole filling: {mask.sum():.0f} pixels")
        
        # Final cleanup - remove any remaining small regions
        mask = self._remove_small_regions(mask)
        
        final_pixels = mask.sum()
        logger.info(f"Refinement complete. Final pixels: {final_pixels:.0f} "
                   f"(retained {100*final_pixels/max(original_pixels, 1):.1f}%)")
        
        return mask
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove regions smaller than min_area pixels."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        
        refined = np.zeros_like(mask)
        regions_removed = 0
        regions_kept = 0
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                refined[labels == i] = 1
                regions_kept += 1
            else:
                regions_removed += 1
        
        logger.debug(f"Regions kept: {regions_kept}, removed: {regions_removed}")
        
        return refined
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening and closing to smooth boundaries."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Opening: removes small noise (erode then dilate)
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Closing: fills small holes (dilate then erode)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return (closed > 0).astype(np.float32)
    
    def _filter_by_green(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Keep only regions where green channel indicates vegetation.
        
        This filters out water, soil, and shadows that SAM2 might
        have incorrectly included in the pseudo-mask.
        """
        if len(image.shape) != 3 or image.shape[2] < 3:
            logger.warning("Image does not have 3 channels, skipping green filter")
            return mask
        
        # Assuming BGR format (OpenCV default)
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        
        # Avoid division by zero
        r_safe = np.maximum(r, 1)
        b_safe = np.maximum(b, 1)
        
        # Green should be relatively higher for vegetation
        # Using OR to be less restrictive
        green_dominant = (g > r_safe * self.green_ratio_min) | \
                        (g > b_safe * self.green_ratio_min)
        
        # Also check for reasonable brightness (not too dark = shadows)
        not_too_dark = (g > 20) & (r > 10) & (b > 10)
        
        vegetation_mask = green_dominant & not_too_dark
        refined = mask * vegetation_mask.astype(np.float32)
        
        return refined
    
    def _filter_by_ndvi(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nir_channel: int = 3
    ) -> np.ndarray:
        """
        Filter by NDVI (Normalized Difference Vegetation Index).
        
        NDVI = (NIR - Red) / (NIR + Red)
        - Values > 0.2-0.4 typically indicate vegetation
        - Mangroves usually have NDVI > 0.3
        
        Args:
            image: Multispectral image with NIR band
            mask: Current mask
            nir_channel: Index of NIR channel (default 3, assuming RGBNIR)
        """
        if len(image.shape) != 3 or image.shape[2] <= nir_channel:
            logger.warning(f"NIR channel {nir_channel} not available, skipping NDVI filter")
            return mask
        
        # Assuming channel order: B, G, R, NIR (or R, G, B, NIR)
        red = image[:, :, 2].astype(np.float32)
        nir = image[:, :, nir_channel].astype(np.float32)
        
        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        denominator = nir + red + 1e-8  # Avoid division by zero
        ndvi = np.where(
            denominator > 0,
            (nir - red) / denominator,
            0
        )
        
        # Keep only pixels with NDVI above threshold
        vegetation = ndvi > self.ndvi_threshold
        refined = mask * vegetation.astype(np.float32)
        
        logger.debug(f"NDVI range: [{ndvi.min():.2f}, {ndvi.max():.2f}]")
        
        return refined
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill small holes within mangrove regions."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find external contours and fill them
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        filled = np.zeros_like(mask_uint8)
        cv2.drawContours(filled, contours, -1, 255, -1)  # -1 = fill contours
        
        return (filled > 0).astype(np.float32)
    
    def get_refinement_stats(
        self,
        original_mask: np.ndarray,
        refined_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get statistics about the refinement process.
        
        Returns:
            Dictionary with refinement statistics
        """
        original_pixels = original_mask.sum()
        refined_pixels = refined_mask.sum()
        
        return {
            "original_pixels": int(original_pixels),
            "refined_pixels": int(refined_pixels),
            "pixels_removed": int(original_pixels - refined_pixels),
            "retention_rate": refined_pixels / max(original_pixels, 1) * 100,
            "original_coverage": original_pixels / original_mask.size * 100,
            "refined_coverage": refined_pixels / refined_mask.size * 100,
        }


def refine_single_mask(
    image_path: str,
    mask_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Refine a single SAM2 pseudo-mask.
    
    Args:
        image_path: Path to original satellite image
        mask_path: Path to SAM2 pseudo-mask
        output_path: Path to save refined mask
        config: Optional refinement configuration
    
    Returns:
        Dictionary with refinement statistics
    
    Example:
        stats = refine_single_mask(
            "data/images/tile_001.png",
            "data/sam2_masks/tile_001.png",
            "data/refined_masks/tile_001.png"
        )
    """
    config = config or {}
    
    image = cv2.imread(image_path)
    pseudo_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if pseudo_mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    refiner = MaskRefinement(
        min_area=config.get("min_area", 500),
        green_ratio_min=config.get("green_ratio_min", 1.1),
        morph_kernel_size=config.get("morph_kernel_size", 5),
        ndvi_threshold=config.get("ndvi_threshold", 0.2),
        fill_holes=config.get("fill_holes", True)
    )
    
    refined_mask = refiner.refine(
        image,
        pseudo_mask / 255.0,
        apply_green_filter=config.get("apply_green_filter", True),
        apply_ndvi_filter=config.get("apply_ndvi_filter", False),
        nir_channel=config.get("nir_channel", None)
    )
    
    # Get statistics
    stats = refiner.get_refinement_stats(pseudo_mask / 255.0, refined_mask)
    
    # Save refined mask
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), (refined_mask * 255).astype(np.uint8))
    
    logger.info(f"Refined mask saved to: {output_path}")
    
    return stats


def refine_batch(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Refine all SAM2 pseudo-masks in a directory.
    
    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing SAM2 pseudo-masks
        output_dir: Directory to save refined masks
        config: Optional refinement configuration
    
    Returns:
        Dictionary with batch refinement statistics
    
    Example:
        stats = refine_batch(
            image_dir="data/raw_images",
            mask_dir="data/labeled_output/masks",
            output_dir="data/refined_masks",
            config={"min_area": 500, "green_ratio_min": 1.1}
        )
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mask files
    mask_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    mask_files = []
    for ext in mask_extensions:
        mask_files.extend(list(mask_dir.glob(ext)))
    
    logger.info(f"Found {len(mask_files)} masks to refine")
    
    batch_stats = {
        "total_files": len(mask_files),
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "total_original_pixels": 0,
        "total_refined_pixels": 0,
    }
    
    for mask_path in mask_files:
        # Find corresponding image
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = image_dir / (mask_path.stem + ext)
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            logger.warning(f"Image not found for mask: {mask_path.name}")
            batch_stats["skipped"] += 1
            continue
        
        output_path = output_dir / mask_path.name
        
        try:
            stats = refine_single_mask(
                str(image_path),
                str(mask_path),
                str(output_path),
                config
            )
            batch_stats["processed"] += 1
            batch_stats["total_original_pixels"] += stats["original_pixels"]
            batch_stats["total_refined_pixels"] += stats["refined_pixels"]
            
        except Exception as e:
            logger.error(f"Error refining {mask_path.name}: {e}")
            batch_stats["failed"] += 1
    
    # Calculate overall retention rate
    if batch_stats["total_original_pixels"] > 0:
        batch_stats["overall_retention_rate"] = (
            batch_stats["total_refined_pixels"] / 
            batch_stats["total_original_pixels"] * 100
        )
    else:
        batch_stats["overall_retention_rate"] = 0
    
    logger.info(f"Batch refinement complete: {batch_stats['processed']} processed, "
               f"{batch_stats['failed']} failed, {batch_stats['skipped']} skipped")
    
    return batch_stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine SAM2 pseudo-masks")
    parser.add_argument("--image-dir", required=True, help="Directory with original images")
    parser.add_argument("--mask-dir", required=True, help="Directory with SAM2 masks")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum region size")
    parser.add_argument("--green-ratio", type=float, default=1.1, help="Green ratio threshold")
    
    args = parser.parse_args()
    
    config = {
        "min_area": args.min_area,
        "green_ratio_min": args.green_ratio,
        "morph_kernel_size": 5,
        "apply_green_filter": True,
    }
    
    stats = refine_batch(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    print("\nBatch Refinement Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

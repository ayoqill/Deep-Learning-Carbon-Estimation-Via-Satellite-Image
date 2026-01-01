"""
SAM-2 based annotation module for mangrove segmentation.
Uses Meta's Segment Anything Model 2 to generate segmentation masks.

Requirements:
    - SAM-2 repository cloned from: https://github.com/facebookresearch/sam2.git
    - Weights downloaded to: ../sam2/checkpoints/sam2.1_hiera_large.pt
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import logging
import importlib.util

# Fix dynamic import paths to use the correct absolute path for sam2/sam2
sam2_dir = str(Path(__file__).resolve().parent.parent.parent / "sam2/sam2")
sys.path.insert(0, sam2_dir)
build_sam_spec = importlib.util.spec_from_file_location("build_sam", f"{sam2_dir}/build_sam.py")
build_sam = importlib.util.module_from_spec(build_sam_spec)
build_sam_spec.loader.exec_module(build_sam)
sam2_image_predictor_spec = importlib.util.spec_from_file_location("sam2_image_predictor", f"{sam2_dir}/sam2_image_predictor.py")
sam2_image_predictor = importlib.util.module_from_spec(sam2_image_predictor_spec)
sam2_image_predictor_spec.loader.exec_module(sam2_image_predictor)
build_sam2 = build_sam.build_sam2
SAM2ImagePredictor = sam2_image_predictor.SAM2ImagePredictor

try:
    from utils.config import Config
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    # Fallback if utils not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class Config:
        """Fallback config loader if utils not available"""
        def __init__(self, config_path="config/settings.yaml"):
            import yaml
            self.config_path = Path(config_path)
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.data = yaml.safe_load(f)
            else:
                self.data = {}
        
        def get(self, key, default=None):
            return self.data.get(key, default)


class SAM2Annotator:
    """
    Use SAM-2 (Segment Anything Model 2) to automatically segment mangroves.
    """
    
    def __init__(self, config_path="config/settings.yaml"):
        """
        Initialize SAM-2 model.
        
        Args:
            config_path: Path to settings.yaml configuration file
            
        Raises:
            ImportError: If SAM-2 is not installed
            FileNotFoundError: If checkpoint file not found
        """
        self.config = Config(config_path)
        self.sam2_config = self.config.get('sam2', {})
        
        # Get device (cuda or cpu)
        device_str = self.sam2_config.get('device', 'cuda')
        self.device = torch.device('cuda' if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
        
        logger.info(f"Initializing SAM-2 on {self.device}...")
        logger.info(f"Model type: {self.sam2_config.get('model_type')}")
        
        config_file = self.sam2_config.get('config_file')
        checkpoint_path = self.sam2_config.get('checkpoint_path', 'sam2/checkpoints/sam2_hiera_large.pt')
        
        # Validate paths exist
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt"
            )
        
        # Build model
        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device
        )
        
        # Build predictor wrapper for image input/output handling
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info("✓ SAM-2 loaded successfully")
    
    def segment_with_points(self, image_path, points, point_labels=None):
        """
        Segment image using point prompts.
        
        Method:
            User provides a few points where mangrove exists.
            SAM-2 automatically finds the mangrove boundary using these hints.
        
        Args:
            image_path (str): Path to .tif satellite image
            points (list or np.ndarray): List of [x, y] coordinates where mangrove exists
                                        Shape: (N, 2) where N is number of points
            point_labels (np.ndarray, optional): Label for each point (1=positive/mangrove, 0=negative/background)
                                               Default: all 1s (all points are mangrove)
            
        Returns:
            np.ndarray: Binary mask (0=background, 1=mangrove) with same size as image
            float: Confidence score of segmentation (higher is better)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return None, 0.0
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Segmenting {Path(image_path).name} with {len(points)} point(s)...")
        
        # Convert points to numpy array
        points = np.array(points, dtype=np.float32)
        
        # Default: all points are positive (mangrove)
        if point_labels is None:
            point_labels = np.ones(len(points), dtype=np.int32)
        
        try:
            # Set image for predictor
            self.predictor.set_image(image)
            
            # Run SAM-2 prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=False
            )
            
            # Get single best mask (masks[0])
            mask = masks[0]
            confidence = scores[0]
            
            # Convert to binary (0 or 1)
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Log result
            mangrove_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            coverage = (mangrove_pixels / total_pixels) * 100
            
            logger.info(
                f"✓ Segmentation complete\n"
                f"  Confidence: {confidence:.4f}\n"
                f"  Coverage: {coverage:.2f}% ({mangrove_pixels:,} / {total_pixels:,} pixels)"
            )
            
            return binary_mask, float(confidence)
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return None, 0.0
    
    def segment_with_box(self, image_path, box):
        """
        Segment image using bounding box prompt.
        
        Method:
            User provides a bounding box around the mangrove area.
            SAM-2 automatically finds the mangrove boundary within this box.
        
        Args:
            image_path (str): Path to .tif satellite image
            box (list or np.ndarray): [x1, y1, x2, y2] coordinates of bounding box
                                     (x1, y1) = top-left, (x2, y2) = bottom-right
            
        Returns:
            np.ndarray: Binary mask (0=background, 1=mangrove) with same size as image
            float: Confidence score of segmentation
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return None, 0.0
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Segmenting {Path(image_path).name} with bounding box...")
        
        box = np.array(box, dtype=np.float32)
        
        try:
            # Set image for predictor
            self.predictor.set_image(image)
            
            # Run SAM-2 prediction with box
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=False
            )
            
            mask = masks[0]
            confidence = scores[0]
            
            binary_mask = (mask > 0).astype(np.uint8)
            
            mangrove_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            coverage = (mangrove_pixels / total_pixels) * 100
            
            logger.info(
                f"✓ Segmentation complete\n"
                f"  Confidence: {confidence:.4f}\n"
                f"  Coverage: {coverage:.2f}%"
            )
            
            return binary_mask, float(confidence)
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return None, 0.0
    
    def refine_mask(self, mask, erosion_kernel_size=3, dilation_kernel_size=3):
        """
        Refine binary mask using morphological operations.
        
        Method:
            - Erosion: Remove small noise/artifacts
            - Dilation: Fill small holes, connect nearby regions
        
        Args:
            mask (np.ndarray): Binary mask from SAM-2
            erosion_kernel_size (int): Kernel size for erosion (3, 5, 7, etc.)
            dilation_kernel_size (int): Kernel size for dilation
            
        Returns:
            np.ndarray: Refined binary mask
        """
        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        
        # Erode then dilate (removes small noise)
        refined = cv2.erode(mask, kernel_erosion, iterations=1)
        refined = cv2.dilate(refined, kernel_dilation, iterations=1)
        
        logger.info(f"✓ Mask refined (erosion {erosion_kernel_size}x{erosion_kernel_size} → dilation {dilation_kernel_size}x{dilation_kernel_size})")
        
        return refined
    
    def save_mask(self, mask, output_path, format='png'):
        """
        Save binary mask to file.
        
        Args:
            mask (np.ndarray): Binary mask (0 or 1)
            output_path (str): Path to save mask
            format (str): 'png' or 'npy'
                         'png': 8-bit image (0-255), viewable with image viewers
                         'npy': NumPy binary format, preserves exact values
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'png':
            # Convert to 0-255 range for PNG
            mask_png = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), mask_png)
            logger.info(f"✓ Mask saved (PNG): {output_path}")
            
        elif format.lower() == 'npy':
            # Save as NumPy array
            np.save(str(output_path), mask)
            logger.info(f"✓ Mask saved (NPY): {output_path}")
        else:
            logger.error(f"Unknown format: {format}. Use 'png' or 'npy'")


def batch_annotate(image_dir, output_dir, annotator, prompt_type='box', 
                   point_per_image=None, boxes_per_image=None):
    """
    Process multiple images and generate segmentation masks.
    
    Args:
        image_dir (str): Directory containing .tif images
        output_dir (str): Directory to save .png masks
        annotator (SAM2Annotator): Initialized SAM-2 annotator
        prompt_type (str): 'point' or 'box' (how to guide SAM-2)
        point_per_image (dict): Dict mapping image_filename to [[x, y], [x, y], ...] points
                               Used when prompt_type='point'
        boxes_per_image (dict): Dict mapping image_filename to [x1, y1, x2, y2] box
                               Used when prompt_type='box'
    
    Returns:
        list: Paths to generated mask files
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .tif images
    image_files = sorted(image_dir.glob('*.tif')) + sorted(image_dir.glob('*.tiff'))
    
    if not image_files:
        logger.warning(f"No .tif images found in {image_dir}")
        return []
    
    logger.info(f"Found {len(image_files)} image(s) to annotate")
    
    mask_paths = []
    
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        output_path = output_dir / f"{image_path.stem}_mask.png"
        
        try:
            if prompt_type == 'point':
                if not point_per_image or image_path.name not in point_per_image:
                    logger.warning(f"No points provided for {image_path.name}, skipping...")
                    continue
                
                points = point_per_image[image_path.name]
                mask, confidence = annotator.segment_with_points(str(image_path), points)
                
            elif prompt_type == 'box':
                if not boxes_per_image or image_path.name not in boxes_per_image:
                    logger.warning(f"No box provided for {image_path.name}, skipping...")
                    continue
                
                box = boxes_per_image[image_path.name]
                mask, confidence = annotator.segment_with_box(str(image_path), box)
            
            if mask is not None:
                # Optionally refine mask
                mask = annotator.refine_mask(mask, erosion_kernel_size=3, dilation_kernel_size=3)
                
                # Save mask
                annotator.save_mask(mask, str(output_path), format='png')
                mask_paths.append(str(output_path))
        
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            continue
    
    logger.info(f"\n✓ Batch annotation complete. Generated {len(mask_paths)} mask(s)")
    
    return mask_paths


# Example usage and testing
if __name__ == "__main__":
    # Initialize SAM-2
    try:
        annotator = SAM2Annotator()
        
        logger.info("\n" + "="*60)
        logger.info("SAM-2 Annotator initialized successfully!")
        logger.info("="*60)
        
        logger.info("\nUsage examples:")
        logger.info("  1. Point-based segmentation:")
        logger.info("     points = [[100, 150], [200, 250]]")
        logger.info("     mask, score = annotator.segment_with_points(image_path, points)")
        logger.info("\n  2. Box-based segmentation:")
        logger.info("     box = [50, 50, 300, 300]  # x1, y1, x2, y2")
        logger.info("     mask, score = annotator.segment_with_box(image_path, box)")
        logger.info("\n  3. Batch processing:")
        logger.info("     boxes = {'image1.tif': [50, 50, 300, 300], ...}")
        logger.info("     mask_paths = batch_annotate(image_dir, output_dir, annotator, prompt_type='box', boxes_per_image=boxes)")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)

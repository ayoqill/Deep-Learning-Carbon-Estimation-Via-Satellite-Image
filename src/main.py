# Contents of /mangrove-carbon-pipeline/mangrove-carbon-pipeline/src/main.py

"""
Mangrove Carbon Estimation Pipeline
Main entry point for the project

Workflow:
1. Load preprocessed satellite images (already corrected by SNAP or similar)
2. Annotate/label vegetation areas using SAM-2
3. Prepare masks for model training
4. Train semantic segmentation model (YOLOv8-seg or U-Net)
5. Evaluate model performance
6. Estimate carbon stock from segmentation results
"""

import os
from pathlib import Path

from src.labeling.sam2_annotator import SAM2Annotator, batch_annotate
from src.data.loader import load_data
from src.data.preprocessor import preprocess_data, convert_masks_to_training_format
from src.models.estimator import train_model, predict
from src.satellite.processor import calculate_carbon_stock
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization.plotter import plot_results


def main():
    """Main pipeline execution"""
    
    # Setup logger
    logger = setup_logger(__name__)
    
    # Load configuration
    config = Config()
    logger.info("Configuration loaded")
    
    # ===== PHASE 1: LABELING =====
    if config.run_phase == "label" or config.run_phase == "all":
        logger.info("=" * 60)
        logger.info("PHASE 1: SAM-2 ANNOTATION")
        logger.info("=" * 60)
        
        annotator = SAM2Annotator(model_name=config.sam2_model)
        batch_annotate(
            image_dir=config.images_dir,
            output_dir=config.masks_dir,
            annotator=annotator
        )
        logger.info("Annotation phase completed")
    
    # ===== PHASE 2: DATA PREPARATION =====
    if config.run_phase == "prepare" or config.run_phase == "all":
        logger.info("=" * 60)
        logger.info("PHASE 2: DATA PREPARATION")
        logger.info("=" * 60)
        
        logger.info("Loading satellite images...")
        images, metadata = load_data(config.images_dir)
        
        logger.info("Loading segmentation masks...")
        masks = load_data(config.masks_dir)
        
        logger.info("Converting masks to training format...")
        train_images, train_masks = convert_masks_to_training_format(
            images=images,
            masks=masks,
            format=config.mask_format,  # 'yolo' or 'segmentation'
            output_dir=config.training_data_dir
        )
        logger.info("Data preparation completed")
    
    # ===== PHASE 3: MODEL TRAINING =====
    if config.run_phase == "train" or config.run_phase == "all":
        logger.info("=" * 60)
        logger.info("PHASE 3: MODEL TRAINING")
        logger.info("=" * 60)
        
        logger.info(f"Training {config.model_type} model...")
        model = train_model(
            train_images_dir=config.training_data_dir,
            config=config.model_params,
            model_type=config.model_type  # 'yolov8-seg' or 'unet'
        )
        
        logger.info(f"Model saved to: {config.model_checkpoint_path}")
    
    # ===== PHASE 4: INFERENCE & CARBON ESTIMATION =====
    if config.run_phase == "infer" or config.run_phase == "all":
        logger.info("=" * 60)
        logger.info("PHASE 4: INFERENCE & CARBON ESTIMATION")
        logger.info("=" * 60)
        
        logger.info("Loading trained model...")
        model = load_model(config.model_checkpoint_path, config.model_type)
        
        logger.info("Making predictions on validation set...")
        predictions = predict(model, config.validation_images_dir)
        
        logger.info("Calculating mangrove area and carbon stock...")
        results = calculate_carbon_stock(
            masks=predictions,
            pixel_size_m=config.pixel_size_m,
            carbon_density_kg_ha=config.carbon_density_kg_ha,
            metadata=metadata
        )
        
        logger.info(f"Total mangrove area: {results['area_ha']:.2f} ha")
        logger.info(f"Total carbon stock: {results['carbon_stock_tC']:.2f} tC")
    
    # ===== PHASE 5: VISUALIZATION & REPORTING =====
    if config.run_phase == "visualize" or config.run_phase == "all":
        logger.info("=" * 60)
        logger.info("PHASE 5: VISUALIZATION & REPORTING")
        logger.info("=" * 60)
        
        logger.info("Generating visualizations...")
        plot_results(
            predictions=predictions,
            original_images=images,
            metadata=metadata,
            output_dir=config.output_dir
        )
        logger.info("Visualizations saved")
    
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


def load_model(checkpoint_path: str, model_type: str):
    """Load trained model from checkpoint"""
    # Implementation depends on model_type
    pass


if __name__ == "__main__":
    main()
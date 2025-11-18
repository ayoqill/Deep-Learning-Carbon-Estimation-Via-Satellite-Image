# Contents of /mangrove-carbon-pipeline/mangrove-carbon-pipeline/src/utils/config.py

"""
Configuration management for mangrove carbon estimation pipeline
Loads settings from YAML and provides easy access to all parameters
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration loader for the pipeline"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to settings.yaml
        """
        self.config_path = Path(config_path)
        self.settings = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        dirs = [
            self.images_dir,
            self.masks_dir,
            self.training_data_dir,
            self.output_dir,
            self.logs_dir,
            self.model_dir
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Pipeline
    @property
    def run_phase(self) -> str:
        """Which phase to run: 'label', 'prepare', 'train', 'infer', 'visualize', 'all'"""
        return self.settings['pipeline']['run_phase']
    
    # SAM-2 Settings
    @property
    def sam2_model(self) -> str:
        """SAM-2 model name from Hugging Face"""
        return self.settings['sam2']['model_name']
    
    @property
    def sam2_device(self) -> str:
        """Device for SAM-2: 'cuda' or 'cpu'"""
        return self.settings['sam2']['device']
    
    # Data Paths
    @property
    def images_dir(self) -> str:
        """Directory containing input preprocessed satellite images"""
        return self.settings['data']['images_dir']
    
    @property
    def masks_dir(self) -> str:
        """Directory for SAM-2 generated masks"""
        return self.settings['data']['masks_dir']
    
    @property
    def training_data_dir(self) -> str:
        """Directory for prepared training data"""
        return self.settings['data']['training_data_dir']
    
    @property
    def validation_images_dir(self) -> str:
        """Directory for validation images"""
        return self.settings['data']['validation_images_dir']
    
    @property
    def output_dir(self) -> str:
        """Output directory for results"""
        return self.settings['data']['output_dir']
    
    @property
    def logs_dir(self) -> str:
        """Logging directory"""
        return Path(self.settings['logging']['log_file']).parent
    
    @property
    def model_dir(self) -> str:
        """Directory for model checkpoints"""
        return "models"
    
    # Dataset Settings
    @property
    def mask_format(self) -> str:
        """Mask format: 'segmentation' or 'yolo'"""
        return self.settings['dataset']['mask_format']
    
    @property
    def train_val_split(self) -> float:
        """Training/validation split ratio"""
        return self.settings['dataset']['train_val_split']
    
    # Model Configuration
    @property
    def model_type(self) -> str:
        """Model type: 'unet' or 'yolov8-seg'"""
        return self.settings['model']['type']
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """All model parameters as dictionary"""
        return self.settings['model']
    
    @property
    def model_checkpoint_path(self) -> str:
        """Path to save/load model checkpoint"""
        return self.settings['model']['checkpoint_path']
    
    # Training Settings
    @property
    def batch_size(self) -> int:
        """Batch size for training"""
        return self.settings['model']['batch_size']
    
    @property
    def learning_rate(self) -> float:
        """Learning rate for training"""
        return self.settings['model']['learning_rate']
    
    @property
    def num_epochs(self) -> int:
        """Number of training epochs"""
        return self.settings['model']['num_epochs']
    
    @property
    def early_stopping_patience(self) -> int:
        """Early stopping patience"""
        return self.settings['training']['early_stopping_patience']
    
    # Carbon Estimation Settings
    @property
    def pixel_size_m(self) -> float:
        """Pixel size in meters"""
        return self.settings['carbon']['pixel_size_m']
    
    @property
    def carbon_density_kg_ha(self) -> float:
        """Carbon density in kg/ha"""
        return self.settings['carbon']['carbon_density_kg_ha']
    
    # Logging
    @property
    def log_file(self) -> str:
        """Log file path"""
        return self.settings['logging']['log_file']
    
    @property
    def log_level(self) -> str:
        """Logging level"""
        return self.settings['logging']['log_level']
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Example:
            config.get('model.learning_rate')
            config.get('data.images_dir')
        
        Args:
            key: Configuration key (dot-separated for nested values)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.settings
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def display(self) -> None:
        """Print configuration summary"""
        print("=" * 60)
        print("Pipeline Configuration")
        print("=" * 60)
        print(f"Phase: {self.run_phase}")
        print(f"Model: {self.model_type}")
        print(f"Images: {self.images_dir}")
        print(f"Masks: {self.masks_dir}")
        print(f"SAM-2 Model: {self.sam2_model}")
        print(f"Carbon Density: {self.carbon_density_kg_ha} kg/ha")
        print(f"Pixel Size: {self.pixel_size_m} m")
        print("=" * 60)
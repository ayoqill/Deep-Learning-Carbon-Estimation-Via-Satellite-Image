# filepath: src/utils/study_areas.py
# Precomputed study area management for Langkawi and other regions

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import torch

class StudyAreaManager:
    """
    Manages precomputed study area analysis.
    Handles automatic inference on study area images.
    """
    
    def __init__(self, study_areas_data_path: Path, results_path: Path, models: Dict[str, Any], device: str):
        self.study_areas_data_path = Path(study_areas_data_path)
        self.results_path = Path(results_path)
        self.models = models
        self.device = device
        self.config_file = self.results_path / "study_areas_config.json"
        self._ensure_config()
    
    def _ensure_config(self):
        """Ensure study areas config exists"""
        if not self.config_file.exists():
            default_config = {
                "study_areas": [
                    {
                        "id": "langkawi",
                        "name": "Langkawi",
                        "location": "Kedah, Malaysia",
                        "latitude": 6.3333,
                        "longitude": 99.8333,
                        "description": "Precomputed mangrove study area in Langkawi",
                        "images": []
                    }
                ]
            }
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load study areas config"""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def _save_config(self, config: Dict[str, Any]):
        """Save study areas config"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def discover_study_area_images(self, study_area_id: str, image_folder: Path) -> List[Path]:
        """
        Discover all eligible images in a study area folder.
        Returns list of valid image paths.
        """
        if not image_folder.exists():
            return []
        
        valid_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        images = [f for f in image_folder.glob('*') if f.suffix.lower() in valid_extensions]
        return sorted(images)
    
    def initialize_langkawi(self, test_images_path: Path, model_name: str = "unetpp") -> bool:
        """
        Initialize Langkawi study area with images from TEST IMAGES folder.
        Auto-processes all images found there.
        
        Returns:
            bool: Success status
        """
        config = self._load_config()
        langkawi = next((sa for sa in config['study_areas'] if sa['id'] == 'langkawi'), None)
        
        if not langkawi:
            return False
        
        # Discover images
        image_paths = self.discover_study_area_images('langkawi', test_images_path)
        if not image_paths:
            return False
        
        # Process each image
        langkawi['images'] = []
        for img_path in image_paths:
            image_info = {
                'filename': img_path.name,
                'originalPath': str(img_path),
                'processed': False,
                'analysisId': None
            }
            langkawi['images'].append(image_info)
        
        config['study_areas'] = [langkawi if sa['id'] == 'langkawi' else sa for sa in config['study_areas']]
        self._save_config(config)
        
        return True
    
    def get_study_areas(self) -> List[Dict[str, Any]]:
        """Get all study areas"""
        config = self._load_config()
        return config.get('study_areas', [])
    
    def get_study_area(self, study_area_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific study area"""
        config = self._load_config()
        for sa in config.get('study_areas', []):
            if sa['id'] == study_area_id:
                return sa
        return None
    
    def add_custom_study_area(self, area_id: str, name: str, location: str, 
                             latitude: float, longitude: float, description: str = "") -> bool:
        """
        Add a new custom study area.
        Can be extended later to process custom folders.
        """
        config = self._load_config()
        
        # Check if already exists
        if any(sa['id'] == area_id for sa in config['study_areas']):
            return False
        
        new_area = {
            'id': area_id,
            'name': name,
            'location': location,
            'latitude': latitude,
            'longitude': longitude,
            'description': description,
            'images': []
        }
        
        config['study_areas'].append(new_area)
        self._save_config(config)
        return True

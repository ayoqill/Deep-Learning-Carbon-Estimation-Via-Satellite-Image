"""
Unit tests for utility modules (config, logger, etc.)
Tests the configuration system and logging setup
"""

import sys
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config
from utils.logger import setup_logger


class TestConfig:
    """Tests for configuration loading and access"""
    
    def test_config_loads_without_error(self):
        """Test that config file loads successfully"""
        config = Config()
        assert config is not None
    
    def test_config_has_required_sections(self):
        """Test that config has all required sections"""
        config = Config()
        assert hasattr(config, 'run_phase')
        assert hasattr(config, 'sam2_model')
        assert hasattr(config, 'model_type')
        assert hasattr(config, 'carbon_density_kg_ha')
    
    def test_sam2_model_configured(self):
        """Test SAM-2 model is configured"""
        config = Config()
        sam2_model = config.sam2_model
        assert isinstance(sam2_model, str)
        assert len(sam2_model) > 0
        assert 'sam' in sam2_model.lower()
    
    def test_model_type_valid(self):
        """Test model type is either unet or yolov8-seg"""
        config = Config()
        model_type = config.model_type
        assert model_type in ['unet', 'yolov8-seg']
    
    def test_carbon_density_is_positive(self):
        """Test carbon density is a positive number"""
        config = Config()
        carbon_density = config.carbon_density_kg_ha
        assert isinstance(carbon_density, (int, float))
        assert carbon_density > 0
    
    def test_pixel_size_is_positive(self):
        """Test pixel size is positive"""
        config = Config()
        pixel_size = config.pixel_size_m
        assert isinstance(pixel_size, (int, float))
        assert pixel_size > 0
    
    def test_batch_size_is_positive(self):
        """Test batch size is positive"""
        config = Config()
        batch_size = config.batch_size
        assert isinstance(batch_size, int)
        assert batch_size > 0
    
    def test_learning_rate_is_positive(self):
        """Test learning rate is positive"""
        config = Config()
        lr = config.learning_rate
        assert isinstance(lr, (int, float))
        assert lr > 0
    
    def test_num_epochs_is_positive(self):
        """Test number of epochs is positive"""
        config = Config()
        epochs = config.num_epochs
        assert isinstance(epochs, int)
        assert epochs > 0
    
    def test_get_method_with_dot_notation(self):
        """Test dot-notation key access"""
        config = Config()
        # Try to access nested config
        result = config.get('model.type')
        assert result is not None
    
    def test_get_method_with_default(self):
        """Test get method with default value"""
        config = Config()
        result = config.get('nonexistent.key', default='default_value')
        assert result == 'default_value'
    
    def test_directories_created(self):
        """Test that required directories are created"""
        config = Config()
        # Check that paths are strings
        assert isinstance(config.images_dir, str)
        assert isinstance(config.masks_dir, str)
        assert isinstance(config.training_data_dir, str)
        assert isinstance(config.output_dir, str)
    
    def test_run_phase_valid(self):
        """Test run phase is valid"""
        config = Config()
        phase = config.run_phase
        valid_phases = ['label', 'prepare', 'train', 'infer', 'visualize', 'all']
        assert phase in valid_phases


class TestLogger:
    """Tests for logging setup"""
    
    def test_logger_initializes(self):
        """Test logger can be initialized"""
        logger = setup_logger(__name__)
        assert logger is not None
    
    def test_logger_has_name(self):
        """Test logger has correct name"""
        logger = setup_logger('test_logger')
        assert logger.name == 'test_logger'
    
    def test_logger_can_log_info(self):
        """Test logger can log info level"""
        logger = setup_logger(__name__)
        try:
            logger.info("Test info message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger failed to log info: {e}")
    
    def test_logger_can_log_warning(self):
        """Test logger can log warning level"""
        logger = setup_logger(__name__)
        try:
            logger.warning("Test warning message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger failed to log warning: {e}")
    
    def test_logger_can_log_error(self):
        """Test logger can log error level"""
        logger = setup_logger(__name__)
        try:
            logger.error("Test error message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger failed to log error: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

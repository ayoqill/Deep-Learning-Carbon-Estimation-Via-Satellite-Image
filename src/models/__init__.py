"""
Segmentation models for mangrove detection.

Available models:
- UNetPlusPlus: Enhanced U-Net with nested skip connections (recommended)
- get_model: Factory function to create any supported model
- get_unetpp: Quick function to create U-Net++ model

Example:
    from src.models import get_model, UNetPlusPlus
    
    # Using factory function
    model = get_model("unetpp", encoder_name="resnet34", in_channels=3, classes=1)
    
    # Using class directly
    model = UNetPlusPlus(in_channels=3, classes=1)
"""
from .unetpp import UNetPlusPlus, get_unetpp
from .model_factory import get_model, get_recommended_model, list_available_models
from .estimator import CarbonEstimator

__all__ = [
    # Models
    "UNetPlusPlus",
    "get_unetpp",
    # Factory functions
    "get_model",
    "get_recommended_model",
    "list_available_models",
    # Carbon estimation
    "CarbonEstimator",
]
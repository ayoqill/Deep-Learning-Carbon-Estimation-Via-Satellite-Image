"""
Model Factory for easy switching between segmentation architectures.

This module provides a unified interface to create different segmentation
models using the segmentation_models_pytorch library.

Supported models:
- U-Net: Classic encoder-decoder with skip connections
- U-Net++: Nested skip connections for better feature fusion
- DeepLabV3+: ASPP module for multi-scale features (great for satellite imagery)
- MANet: Multi-scale Attention Network
- FPN: Feature Pyramid Network
- PSPNet: Pyramid Scene Parsing Network
"""
import segmentation_models_pytorch as smp
import torch.nn as nn
from typing import Optional, Dict, List


AVAILABLE_MODELS = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "linknet": smp.Linknet,
    "pan": smp.PAN,
}

AVAILABLE_ENCODERS = [
    # ResNet family
    "resnet18",
    "resnet34",      # Good balance (recommended for most cases)
    "resnet50",      # More capacity
    "resnet101",
    # EfficientNet family
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",  # Good for high accuracy
    # MobileNet (lightweight)
    "mobilenet_v2",
    # VGG
    "vgg16",
    "vgg19",
]

ENCODER_RECOMMENDATIONS = {
    "fast": "resnet18",
    "balanced": "resnet34",
    "accurate": "resnet50",
    "efficient": "efficientnet-b2",
    "lightweight": "mobilenet_v2",
}


def get_model(
    model_name: str = "unetpp",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: str = "sigmoid"
) -> nn.Module:
    """
    Factory function to get different segmentation models.
    
    Args:
        model_name: Model architecture. One of:
            - 'unet': Classic U-Net
            - 'unetpp': U-Net++ (recommended)
            - 'deeplabv3plus': DeepLabV3+ (great for satellite imagery)
            - 'manet': Multi-scale Attention Network
            - 'fpn': Feature Pyramid Network
            - 'pspnet': Pyramid Scene Parsing Network
            - 'linknet': LinkNet
            - 'pan': Pyramid Attention Network
        encoder_name: Backbone encoder. Popular options:
            - 'resnet34': Good balance (recommended)
            - 'resnet50': More capacity
            - 'efficientnet-b4': Efficient and accurate
            - 'mobilenet_v2': Lightweight
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels (3 for RGB, more for multispectral)
        classes: Number of output classes (1 for binary segmentation)
        activation: Output activation ('sigmoid' for binary, 'softmax' for multi-class)
    
    Returns:
        Segmentation model ready for training/inference
    
    Example:
        # Binary mangrove segmentation with U-Net++
        model = get_model("unetpp", encoder_name="resnet34", in_channels=3, classes=1)
        
        # Multi-spectral input with DeepLabV3+
        model = get_model("deeplabv3plus", encoder_name="resnet50", in_channels=8, classes=1)
    
    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model_class = AVAILABLE_MODELS[model_name]
    
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )


def get_recommended_model(
    task: str = "mangrove_segmentation",
    priority: str = "balanced",
    in_channels: int = 3
) -> nn.Module:
    """
    Get a recommended model configuration for specific tasks.
    
    Args:
        task: Task type ('mangrove_segmentation', 'vegetation_mapping')
        priority: Optimization priority:
            - 'fast': Fastest inference
            - 'balanced': Balance between speed and accuracy
            - 'accurate': Best accuracy
            - 'lightweight': Smallest model size
        in_channels: Number of input channels
    
    Returns:
        Configured segmentation model
    
    Example:
        model = get_recommended_model(
            task="mangrove_segmentation",
            priority="balanced",
            in_channels=3
        )
    """
    encoder = ENCODER_RECOMMENDATIONS.get(priority, "resnet34")
    
    # U-Net++ is recommended for mangrove segmentation
    return get_model(
        model_name="unetpp",
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation="sigmoid"
    )


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models and encoders.
    
    Returns:
        Dictionary with 'models', 'encoders', and 'recommendations' keys
    
    Example:
        info = list_available_models()
        print(info['models'])  # ['unet', 'unetpp', ...]
    """
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "encoders": AVAILABLE_ENCODERS,
        "recommendations": ENCODER_RECOMMENDATIONS
    }


def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific model architecture.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        "unet": {
            "name": "U-Net",
            "description": "Classic encoder-decoder with skip connections",
            "best_for": "General segmentation tasks",
            "complexity": "Low",
        },
        "unetpp": {
            "name": "U-Net++",
            "description": "Nested skip connections for better feature fusion",
            "best_for": "Complex boundaries, medical/satellite imagery",
            "complexity": "Medium",
        },
        "deeplabv3plus": {
            "name": "DeepLabV3+",
            "description": "ASPP module for multi-scale features",
            "best_for": "Satellite imagery, aerial photos",
            "complexity": "High",
        },
        "manet": {
            "name": "MA-Net",
            "description": "Multi-scale Attention Network",
            "best_for": "High accuracy segmentation",
            "complexity": "High",
        },
        "fpn": {
            "name": "FPN",
            "description": "Feature Pyramid Network",
            "best_for": "Multi-scale object detection",
            "complexity": "Medium",
        },
        "pspnet": {
            "name": "PSPNet",
            "description": "Pyramid Scene Parsing Network",
            "best_for": "Scene understanding",
            "complexity": "High",
        },
    }
    
    return model_info.get(model_name.lower(), {"error": "Model not found"})


if __name__ == "__main__":
    # Test model creation
    print("Available models and encoders:")
    print(list_available_models())
    
    print("\nCreating U-Net++ model...")
    model = get_model("unetpp", encoder_name="resnet34", in_channels=3, classes=1)
    print(f"Model created: {type(model).__name__}")
    
    # Test with dummy input
    import torch
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

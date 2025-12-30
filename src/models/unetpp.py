"""
U-Net++ Model using segmentation_models_pytorch.
Upgraded from basic U-Net for better mangrove segmentation.

U-Net++ has nested skip connections that improve feature fusion
and provide better segmentation results for complex boundaries.
"""
import segmentation_models_pytorch as smp
import torch.nn as nn


def get_unetpp(
    in_channels: int = 3,
    classes: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet"
) -> nn.Module:
    """
    Create U-Net++ model with pretrained encoder.
    
    Args:
        in_channels: Number of input channels (3 for RGB, more for multispectral)
        classes: Number of output classes (1 for binary segmentation)
        encoder_name: Backbone encoder (resnet34, resnet50, efficientnet-b4, etc.)
        encoder_weights: Pretrained weights ("imagenet" or None)
    
    Returns:
        U-Net++ model ready for training/inference
    
    Example:
        model = get_unetpp(in_channels=3, classes=1)
        output = model(image_tensor)  # Returns binary mask
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation="sigmoid"  # Binary output: 0-1
    )
    return model


class UNetPlusPlus(nn.Module):
    """
    Wrapper class for U-Net++ to maintain consistent API.
    
    This class wraps the segmentation_models_pytorch UnetPlusPlus
    for compatibility with existing training code.
    
    Attributes:
        model: The underlying smp.UnetPlusPlus model
        encoder_name: Name of the encoder backbone
        in_channels: Number of input channels
        classes: Number of output classes
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet"
    ):
        """
        Initialize U-Net++ model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            classes: Number of output classes (1 for binary)
            encoder_name: Backbone encoder name
            encoder_weights: Pretrained weights source
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation="sigmoid"
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, classes, H, W) with values in [0, 1]
        """
        return self.model(x)
    
    def get_encoder(self):
        """Get the encoder part of the model."""
        return self.model.encoder
    
    def get_decoder(self):
        """Get the decoder part of the model."""
        return self.model.decoder

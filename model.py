"""
LDCT Project — Model Definition
==================================
MONAI 2D U-Net configured for pseudo-3D (2.5D) LDCT denoising.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from config import (
    IN_CHANNELS, OUT_CHANNELS, CHANNELS, STRIDES,
    NUM_RES_UNITS, DROPOUT,
)


def build_model(device):
    """
    Build and return the MONAI U-Net model.
    Automatically wraps with DataParallel if multiple GPUs are available.

    Args:
        device: torch.device to place the model on.

    Returns:
        model on the specified device.
    """
    model = UNet(
        spatial_dims=2,
        in_channels=IN_CHANNELS,      # 🔥 pseudo-3D: prev + curr + next
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        strides=STRIDES,
        num_res_units=NUM_RES_UNITS,
        dropout=DROPOUT,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦  Model parameters: {total_params:,}")

    return model

"""
LDCT Project — Model Definition (MW-FA Net)
==============================================
Multi-Window Frequency-Aware Pseudo-3D UNet (MW-FA Net)
Combining 9-channel Multi-Window inputs with High-Frequency Wavelet Attention.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from config import (
    IN_CHANNELS, OUT_CHANNELS, CHANNELS, STRIDES,
    NUM_RES_UNITS, DROPOUT, USE_WAVELET,
)
from wavelet_layers import WaveletHFAttention


class MultiWindowWaveletUNet(nn.Module):
    """
    Multi-Window Frequency-Aware Pseudo-3D UNet (MW-FA Net).
    Combines 9-channel Multi-Window Pseudo-3D inputs with
    a High-Frequency Wavelet Attention Block at the UNet bottleneck.
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        strides=STRIDES,
        num_res_units=NUM_RES_UNITS,
        dropout=DROPOUT,
        use_wavelet=USE_WAVELET,
    ):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
        )

        if self.use_wavelet:
            bottleneck_channels = channels[-1]  # 512
            self.wavelet_attn = WaveletHFAttention(bottleneck_channels)

            # Traversal to the deepest bottleneck inside MONAI UNet SkipConnection structure
            sub = self.unet.model[1]
            for _ in range(len(channels) - 2):
                if hasattr(sub, "submodule") and isinstance(sub.submodule, nn.Sequential) and len(sub.submodule) > 1:
                    sub = sub.submodule[1]

            assert hasattr(sub, "submodule"), (
                "MONAI UNet architecture mismatch: Could not locate 'submodule' "
                "for Wavelet Attention injection!"
            )

            sub.submodule = nn.Sequential(
                sub.submodule,
                self.wavelet_attn,
            )

    def forward(self, x):
        return self.unet(x)


def build_model(device):
    """
    Build and return the MW-FA Net model.
    Automatically wraps with DataParallel if multiple GPUs are available.

    Args:
        device: torch.device to place the model on.

    Returns:
        model on the specified device.
    """
    model = MultiWindowWaveletUNet().to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦  Model parameters: {total_params:,}")

    return model

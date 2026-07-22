"""
LDCT Project — MS-NAFMambaNet Architecture
==============================================
Multi-Scale Non-Linear Activation-Free Mamba Network for LDCT Denoising.

Scientific Contributions:
  1. Activation-Free Restoration Encoder/Decoder (NAF Blocks from Megvii NAFNet).
  2. Anatomy-Guided Attention Skip Gates (AG-Skip) for quantum noise suppression.
  3. 2D Vision State-Space Mamba Bottleneck (NVIDIA MambaVision derived) for global streak artifact removal.
  4. Modular 4-Stage Ablation Framework (controlled via MAMBA_MODE: "basic", "residual", "multiscale", "full").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IN_CHANNELS, OUT_CHANNELS, MAMBA_MODE
from naf_mamba_blocks import (
    LayerNorm2d, NAFBlock, AnatomyAttentionGate2D,
    SS2DMambaBottleneck, ResidualMambaBottleneck,
    MultiScaleSpatialFusion
)


class MSNAFMambaNet(nn.Module):
    """
    Multi-Scale NAF-Mamba Network with Anatomy Attention Skip Gates.
    Supports 4-stage ablation studies via mamba_mode.
    """
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, mamba_mode=MAMBA_MODE):
        super().__init__()
        self.mamba_mode = mamba_mode.lower()
        print(f"🏗️  Initializing MS-NAFMambaNet (Ablation Mode: '{self.mamba_mode.upper()}')")

        # Stem Conv
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=True)

        # Encoder Stages (NAF Blocks)
        self.enc1 = NAFBlock(32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        self.enc2 = NAFBlock(64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.enc3 = NAFBlock(128)
        self.down3 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        self.enc4 = NAFBlock(256)
        self.down4 = nn.Conv2d(256, 512, kernel_size=2, stride=2)

        # Anatomy Attention Gates on Skips
        self.ag1 = AnatomyAttentionGate2D(F_g=64, F_l=32, F_int=32)
        self.ag2 = AnatomyAttentionGate2D(F_g=128, F_l=64, F_int=64)
        self.ag3 = AnatomyAttentionGate2D(F_g=256, F_l=128, F_int=128)
        self.ag4 = AnatomyAttentionGate2D(F_g=512, F_l=256, F_int=256)

        # Bottleneck (SS2D Mamba 2D Selective State-Space Module)
        if self.mamba_mode in ["residual", "full"]:
            self.bottleneck = ResidualMambaBottleneck(512)
        else:
            self.bottleneck = SS2DMambaBottleneck(512)

        # Multi-Scale Spatial Fusion (1/16 Mamba <-> 1/8 NAF)
        if self.mamba_mode in ["multiscale", "full"]:
            self.fusion = MultiScaleSpatialFusion(in_c_low=512, in_c_high=256, out_c=256)

        # Decoder Stages (Upsampling + Concatenation + NAF Blocks)
        self.up4 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), NAFBlock(256))

        self.up3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), NAFBlock(128))

        self.up2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), NAFBlock(64))

        self.up1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), NAFBlock(32))

        # Output Head
        self.head = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Stem
        x_stem = self.stem(x)

        # Encoder Forward
        e1 = self.enc1(x_stem)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        # Mamba Bottleneck
        b_feat = self.bottleneck(d4)

        # Skip Attention Gating
        g4 = self.ag4(g=b_feat, x=e4)
        g3 = self.ag3(g=d3, x=e3)
        g2 = self.ag2(g=d2, x=e2)
        g1 = self.ag1(g=d1, x=e1)

        # Multi-Scale Fusion (Stage 3 & 4)
        if self.mamba_mode in ["multiscale", "full"]:
            g4 = self.fusion(feat_low=b_feat, feat_high=g4)

        # Decoder Forward
        u4 = self.up4(b_feat)
        d4_cat = torch.cat([u4, g4], dim=1)
        dec4_out = self.dec4(d4_cat)

        u3 = self.up3(dec4_out)
        d3_cat = torch.cat([u3, g3], dim=1)
        dec3_out = self.dec3(d3_cat)

        u2 = self.up2(dec3_out)
        d2_cat = torch.cat([u2, g2], dim=1)
        dec2_out = self.dec2(d2_cat)

        u1 = self.up1(dec2_out)
        d1_cat = torch.cat([u1, g1], dim=1)
        dec1_out = self.dec1(d1_cat)

        # Final Residual Prediction
        out = self.head(dec1_out)
        return out


def build_model(device):
    """
    Factory function for building MS-NAFMambaNet model.
    Wraps with DataParallel if multiple GPUs are available.
    """
    model = MSNAFMambaNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, mamba_mode=MAMBA_MODE).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦  Model parameters ({MAMBA_MODE.upper()} mode): {total_params:,}")

    return model

"""
LDCT Project — Model Definition (MS-NAFMambaNet)
=================================================
Multi-Scale Non-Linear Activation-Free Mamba Network (MS-NAFMambaNet)
with Mathematically Rigorous 4-Way Selective Scan S6 and Self-Contained Output.

Supports 4 Modular Ablation Modes via `mamba_mode`:
  1. "basic"      : NAF-Encoder + Multiplicative AG Skips + SS2D Mamba (1/16) + NAF-Decoder
  2. "residual"   : Stage 1 + Residual Dual-Mamba Bottleneck (Mamba -> Add -> Mamba)
  3. "multiscale" : Stage 1 + Adaptive Gated Multi-Scale Fusion (1/16 <-> 1/8)
  4. "full"       : Full MS-NAFMambaNet (Dual-Mamba + Adaptive Gated Fusion + AG Skips)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IN_CHANNELS, OUT_CHANNELS, CHANNELS, DROPOUT, MAMBA_MODE
from naf_mamba_blocks import (
    LayerNorm2d, NAFBlock, StructureAwareAttentionGate,
    Mamba2DSSM, ResidualMambaBottleneck, AdaptiveGatedFusion
)


class MSNAFMambaNet(nn.Module):
    """
    Multi-Scale Non-Linear Activation-Free Mamba Network (MS-NAFMambaNet)
    with 4-Stage Modular Ablation Framework.
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        dropout=DROPOUT,
        mamba_mode=MAMBA_MODE,
    ):
        super().__init__()
        self.mamba_mode = mamba_mode.lower()
        c1, c2, c3, c4, c5 = channels  # (32, 64, 128, 256, 512)

        # ── 1. Initial Projection ──
        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # ── 2. NAF Encoder Path ──
        self.enc1 = NAFBlock(c1, drop_out=dropout)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2)

        self.enc2 = NAFBlock(c2, drop_out=dropout)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2)

        self.enc3 = NAFBlock(c3, drop_out=dropout)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=2, stride=2)

        self.enc4 = NAFBlock(c4, drop_out=dropout)
        self.down4 = nn.Conv2d(c4, c5, kernel_size=2, stride=2)

        # ── 3. Multiplicative Residual Structure-Aware Attention Gates ──
        self.ag1 = StructureAwareAttentionGate(gate_channels=c2, skip_channels=c1)
        self.ag2 = StructureAwareAttentionGate(gate_channels=c3, skip_channels=c2)
        self.ag3 = StructureAwareAttentionGate(gate_channels=c4, skip_channels=c3)
        self.ag4 = StructureAwareAttentionGate(gate_channels=c5, skip_channels=c4)

        # ── 4. True Selective State-Space Bottleneck (1/16 Resolution) ──
        if self.mamba_mode in ["residual", "full"]:
            self.bottleneck = ResidualMambaBottleneck(c5)
        else:
            self.bottleneck = Mamba2DSSM(c5)

        # ── 5. Adaptive Gated Spatial-State Space Fusion (1/16 <-> 1/8) ──
        if self.mamba_mode in ["multiscale", "full"]:
            self.ms_fusion = AdaptiveGatedFusion(low_res_channels=c5, high_res_channels=c4)
        else:
            self.ms_fusion = None

        # ── 6. NAF Decoder Path ──
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = NAFBlock(c4, drop_out=dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = NAFBlock(c3, drop_out=dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = NAFBlock(c2, drop_out=dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = NAFBlock(c1, drop_out=dropout)

        # ── 7. Final Output Projection ──
        self.final_conv = nn.Sequential(
            LayerNorm2d(c1),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Extract central input slice for residual reconstruction
        mid_slice = x[:, 1:2, :, :] if x.shape[1] >= 3 else x[:, 0:1, :, :]

        # Initial Embedding
        x_in = self.init_conv(x)  # [B, 32, H, W]

        # Encoder Stage 1
        e1 = self.enc1(x_in)
        d1 = self.down1(e1)      # [B, 64, H/2, W/2]

        # Encoder Stage 2
        e2 = self.enc2(d1)
        d2 = self.down2(e2)      # [B, 128, H/4, W/4]

        # Encoder Stage 3
        e3 = self.enc3(d2)
        d3 = self.down3(e3)      # [B, 256, H/8, W/8]

        # Encoder Stage 4
        e4 = self.enc4(d3)
        d4 = self.down4(e4)      # [B, 512, H/16, W/16]

        # Bottleneck State-Space Processing
        b_feat = self.bottleneck(d4)  # [B, 512, H/16, W/16]

        # Decoder Stage 4
        g4 = self.ag4(b_feat, e4)
        u4 = self.up4(b_feat) + g4
        d4_out = self.dec4(u4)    # [B, 256, H/8, W/8]

        # Adaptive Gated Multi-Scale Fusion (Stage 3/4)
        if self.ms_fusion is not None:
            d4_out = self.ms_fusion(b_feat, d4_out)

        # Decoder Stage 3
        g3 = self.ag3(d4_out, e3)
        u3 = self.up3(d4_out) + g3
        d3_out = self.dec3(u3)    # [B, 128, H/4, W/4]

        # Decoder Stage 2
        g2 = self.ag2(d3_out, e2)
        u2 = self.up2(d3_out) + g2
        d2_out = self.dec2(u2)    # [B, 64, H/2, W/2]

        # Decoder Stage 1
        g1 = self.ag1(d2_out, e1)
        u1 = self.up1(d2_out) + g1
        d1_out = self.dec1(u1)    # [B, 32, H, W]

        # Predicted Noise Residual + Central Input Slice -> Self-Contained Output
        out_res = self.final_conv(d1_out)
        pred_img = torch.clamp(mid_slice + out_res, 0.0, 1.0)
        return pred_img


def build_model(device, mamba_mode=MAMBA_MODE):
    """
    Build and return the MS-NAFMambaNet model.
    Automatically wraps with DataParallel if multiple GPUs are available.

    Args:
        device: torch.device to place the model on.
        mamba_mode: ablation mode ("basic", "residual", "multiscale", "full").

    Returns:
        model placed on the specified device.
    """
    model = MSNAFMambaNet(mamba_mode=mamba_mode).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦  Model parameters ({mamba_mode.upper()} mode): {total_params:,}")

    return model

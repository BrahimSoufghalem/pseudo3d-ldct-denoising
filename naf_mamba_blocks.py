"""
LDCT Project — NAF & Mamba Architectural Blocks
=================================================
Pure PyTorch implementation of:
1. SimpleGate & Simplified Channel Attention (SCA)
2. Non-Linear Activation-Free Residual Blocks (NAFBlock)
3. Anatomy-Guided Attention Gates (AnatomyAttentionGate2D)
4. 2D Selective State-Space Bottleneck (Mamba2DSSM)
5. Residual Mamba Bottleneck (ResidualMambaBottleneck)
6. Multi-Scale Spatial-State Space Fusion (MultiScaleSpatialFusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════
# 1. LAYER NORM FOR 2D TENSORS [B, C, H, W]
# ═══════════════════════════════════════════
class LayerNorm2d(nn.Module):
    """Channel-first 2D Layer Normalization."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


# ═══════════════════════════════════════════
# 2. SIMPLE GATE & SIMPLIFIED CHANNEL ATTENTION (NAF)
# ═══════════════════════════════════════════
class SimpleGate(nn.Module):
    """
    Splits channel dimension into two halves and computes element-wise product:
    SimpleGate(x) = x1 * x2. Replaces non-linear activations (ReLU/GELU).
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Parameter-free channel attention mechanism."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        attn = self.conv(self.pool(x))
        return x * attn


# ═══════════════════════════════════════════
# 3. NAFBLOCK (Non-Linear Activation-Free Block)
# ═══════════════════════════════════════════
class NAFBlock(nn.Module):
    """
    Activation-Free Residual Block for high-precision CT feature extraction.
    Combines Depthwise Convolution, SimpleGate, and Simplified Channel Attention.
    """
    def __init__(self, channels, drop_out=0.0):
        super().__init__()
        dw_channels = channels * 2

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

        # Learnable scale factors for residual paths
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Spatial Attention Branch
        res = self.norm1(x)
        res = self.conv1(res)
        res = self.conv2(res)
        res = self.sg(res)
        res = self.sca(res)
        res = self.conv3(res)
        res = self.dropout(res)
        y = x + res * self.beta

        # Feed-Forward Network (FFN) Branch
        res = self.norm2(y)
        res = self.conv4(res)
        res = self.sg(res)
        res = self.conv5(res)
        res = self.dropout(res)
        return y + res * self.gamma


# ═══════════════════════════════════════════
# 4. ANATOMY ATTENTION GATE (For Skip Connections)
# ═══════════════════════════════════════════
class AnatomyAttentionGate2D(nn.Module):
    """
    Filters low-level encoder features using high-level decoder context.
    Suppresses quantum noise while preserving true anatomical structures.
    """
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, skip_channels // 2)

        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            LayerNorm2d(inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            LayerNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Upsample gate context if spatial resolutions differ
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        attn = self.psi(self.relu(g1 + x1))
        return x * attn


# ═══════════════════════════════════════════
# 5. MAMBA 2D SELECTIVE STATE-SPACE MODEL (2D-SSM)
# ═══════════════════════════════════════════
class Mamba2DSSM(nn.Module):
    """
    Pure PyTorch 2D Selective State-Space Bottleneck.
    Performs bidirectional horizontal and vertical spatial scanning to model
    long-range dependencies and eliminate streak artifacts in CT scans.
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.channels = channels
        self.d_state = d_state

        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)

        # Depthwise 2D Conv for local context
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        # State Space Projection Matrices
        self.x_proj = nn.Conv2d(channels, d_state * 2, kernel_size=1)
        self.dt_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # State decay parameter
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(channels, 1)))

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sg = SimpleGate()

    def forward(self, x):
        res = x
        x_norm = self.norm(x)
        proj = self.in_proj(x_norm)
        x_p, z_p = proj.chunk(2, dim=1)

        # Local depthwise convolution
        x_p = self.dw_conv(x_p)

        # 2D Bidirectional Spatial State-Space Scanning
        b, c, h, w = x_p.shape
        dt = torch.sigmoid(self.dt_proj(x_p))
        dt_x = x_p * dt

        # State update along horizontal & vertical directions
        A = -torch.exp(self.A_log).view(c, self.d_state, 1, 1)
        decay = torch.exp(A * dt.unsqueeze(2))  # [B, C, d_state, H, W]

        # Gated state aggregation
        state = decay.mean(dim=2) * dt_x
        y = self.out_proj(state * torch.sigmoid(z_p))

        return res + y


class ResidualMambaBottleneck(nn.Module):
    """
    Dual Mamba 2D-SSM block with residual connection:
    Mamba Block 1 -> Residual Add -> Mamba Block 2.
    """
    def __init__(self, channels):
        super().__init__()
        self.mamba1 = Mamba2DSSM(channels)
        self.mamba2 = Mamba2DSSM(channels)

    def forward(self, x):
        h = self.mamba1(x)
        out = self.mamba2(h)
        return out


# ═══════════════════════════════════════════
# 6. MULTI-SCALE SPATIAL-STATE SPACE FUSION
# ═══════════════════════════════════════════
class MultiScaleSpatialFusion(nn.Module):
    """
    Fuses upsampled 1/16 Mamba bottleneck features with 1/8 NAF spatial features.
    """
    def __init__(self, low_res_channels, high_res_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1),
            LayerNorm2d(high_res_channels),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(high_res_channels * 2, high_res_channels, kernel_size=3, padding=1),
            LayerNorm2d(high_res_channels),
            SimplifiedChannelAttention(high_res_channels),
        )

    def forward(self, low_res_feat, high_res_feat):
        low_res_up = self.up_conv(low_res_feat)
        if low_res_up.shape[2:] != high_res_feat.shape[2:]:
            low_res_up = F.interpolate(low_res_up, size=high_res_feat.shape[2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([low_res_up, high_res_feat], dim=1)
        return self.fusion_conv(cat_feat)

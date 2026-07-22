"""
LDCT Project — Mathematically Rigorous NAF & Mamba Architectural Blocks
=========================================================================
Implementation aligned with official literature (Mamba CVPR 2024 & NAFNet CVPR 2022):
1. SimpleGate & Simplified Channel Attention (SCA)
2. Non-Linear Activation-Free Residual Blocks (NAFBlock)
3. Multiplicative Residual Structure-Aware Attention Gates (StructureAwareAttentionGate)
4. JIT-Compiled 4-Way 2D Selective State-Space Scan (SS2DSelectScan & Mamba2DSSM)
5. Residual Mamba Bottleneck with Explicit Skips (ResidualMambaBottleneck)
6. Adaptive Gated Spatial-State Space Fusion (AdaptiveGatedFusion)
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
# 2. SIMPLE GATE & SIMPLIFIED CHANNEL ATTENTION (NAFNet CVPR 2022)
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
    Exact NAFNet Residual Block (Chen et al. CVPR 2022).
    Combines Depthwise Convolution, SimpleGate, SCA, and learnable scale factors (beta, gamma).
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

        # Learnable scale parameters for residual stability
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
# 4. STRUCTURE-AWARE ATTENTION GATE
# ═══════════════════════════════════════════
class StructureAwareAttentionGate(nn.Module):
    """
    Multiplicative Residual Structure-Aware Attention Gate: Output = x * (1 + AttnMask).
    Ensures 100% preservation of structural features even when attention is small.
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
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        attn = self.psi(self.relu(g1 + x1))
        # Multiplicative residual bypass: prevents signal drop
        return x * (1.0 + attn)


# Backward compatibility alias
AnatomyAttentionGate2D = StructureAwareAttentionGate


# ═══════════════════════════════════════════
# 5. JIT-COMPILED 4-WAY 2D SELECTIVE STATE-SPACE SCAN (SS2D / S6)
# ═══════════════════════════════════════════
@torch.jit.script
def _selective_scan_1d_jit(x_seq: torch.Tensor, delta_seq: torch.Tensor, B_seq: torch.Tensor, C_seq: torch.Tensor, A_log: torch.Tensor) -> torch.Tensor:
    """
    Torch JIT Fused Fused 1D Selective Scan:
    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C_t * h_t
    """
    b, c, l = x_seq.shape
    d_state = B_seq.shape[1]
    A = -torch.exp(A_log).view(1, c, d_state, 1)  # [1, C, N, 1]

    delta = F.softplus(delta_seq).unsqueeze(2)    # [B, C, 1, L]
    A_bar = torch.exp(A * delta)                   # [B, C, N, L]
    B_bar = delta * B_seq.unsqueeze(1)             # [B, C, N, L]

    # Recurrent state accumulation over sequence
    h = torch.zeros(b, c, d_state, device=x_seq.device)
    ys = []
    for t in range(l):
        h = A_bar[:, :, :, t] * h + B_bar[:, :, :, t] * x_seq[:, :, t:t+1]
        C_t = C_seq[:, :, t].unsqueeze(1)         # [B, 1, N]
        y_t = (h * C_t).sum(dim=-1)               # [B, C]
        ys.append(y_t)

    return torch.stack(ys, dim=-1)                # [B, C, L]


class SS2DSelectScan(nn.Module):
    """
    PyTorch-native 4-Way 2D Selective State-Space Scan (SS2D / S6) with JIT compilation.
    Scans 2D feature maps along 4 spatial directions to eliminate streak artifacts.
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.channels = channels
        self.d_state = d_state

        # Projections for Selective Parameters (Delta, B, C)
        self.x_proj = nn.Conv2d(channels, (d_state * 2 + channels), kernel_size=1)
        self.dt_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # State matrix A initialization
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(channels, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # 4 directional output fusion
        self.out_fusion = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Compute Selective Parameters
        dt = self.dt_proj(x)
        proj = self.x_proj(x)
        B_proj, C_proj, _ = proj.split([self.d_state, self.d_state, c], dim=1)

        # 4 Directional Scans using safe .reshape():
        # 1. Forward Horizontal
        x_s1 = x.reshape(b, c, h * w)
        dt_s1 = dt.reshape(b, c, h * w)
        B_s1 = B_proj.reshape(b, self.d_state, h * w)
        C_s1 = C_proj.reshape(b, self.d_state, h * w)
        y1 = _selective_scan_1d_jit(x_s1, dt_s1, B_s1, C_s1, self.A_log).reshape(b, c, h, w)

        # 2. Backward Horizontal
        x_s2 = torch.flip(x, dims=[-1]).reshape(b, c, h * w)
        dt_s2 = torch.flip(dt, dims=[-1]).reshape(b, c, h * w)
        B_s2 = torch.flip(B_proj, dims=[-1]).reshape(b, self.d_state, h * w)
        C_s2 = torch.flip(C_proj, dims=[-1]).reshape(b, self.d_state, h * w)
        y2 = torch.flip(_selective_scan_1d_jit(x_s2, dt_s2, B_s2, C_s2, self.A_log).reshape(b, c, h, w), dims=[-1])

        # 3. Forward Vertical
        x_trans = x.transpose(-2, -1).contiguous()
        x_s3 = x_trans.reshape(b, c, h * w)
        dt_s3 = dt.transpose(-2, -1).contiguous().reshape(b, c, h * w)
        B_s3 = B_proj.transpose(-2, -1).contiguous().reshape(b, self.d_state, h * w)
        C_s3 = C_proj.transpose(-2, -1).contiguous().reshape(b, self.d_state, h * w)
        y3 = _selective_scan_1d_jit(x_s3, dt_s3, B_s3, C_s3, self.A_log).reshape(b, c, w, h).transpose(-2, -1).contiguous()

        # 4. Backward Vertical
        x_vflip = torch.flip(x_trans, dims=[-1]).contiguous()
        x_s4 = x_vflip.reshape(b, c, h * w)
        dt_s4 = torch.flip(dt.transpose(-2, -1), dims=[-1]).contiguous().reshape(b, c, h * w)
        B_s4 = torch.flip(B_proj.transpose(-2, -1), dims=[-1]).contiguous().reshape(b, self.d_state, h * w)
        C_s4 = torch.flip(C_proj.transpose(-2, -1), dims=[-1]).contiguous().reshape(b, self.d_state, h * w)
        y4_rec = _selective_scan_1d_jit(x_s4, dt_s4, B_s4, C_s4, self.A_log).reshape(b, c, w, h)
        y4 = torch.flip(y4_rec, dims=[-1]).transpose(-2, -1).contiguous()

        # Combine 4 directional scan representations
        y_concat = torch.cat([y1, y2, y3, y4], dim=1)
        return self.out_fusion(y_concat)


class Mamba2DSSM(nn.Module):
    """
    2D Selective State-Space Bottleneck Layer using JIT-compiled SS2DSelectScan.
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ss2d = SS2DSelectScan(channels, d_state=d_state)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sg = SimpleGate()

    def forward(self, x):
        res = x
        x_norm = self.norm(x)
        proj = self.in_proj(x_norm)
        x_p, z_p = proj.chunk(2, dim=1)

        x_p = self.dw_conv(x_p)
        y_ssm = self.ss2d(x_p)
        out = self.out_proj(y_ssm * torch.sigmoid(z_p))

        return res + out


class ResidualMambaBottleneck(nn.Module):
    """
    Dual Mamba 2D-SSM bottleneck with explicit residual skips:
    h1 = Mamba1(x) + x
    out = Mamba2(h1) + h1
    """
    def __init__(self, channels):
        super().__init__()
        self.mamba1 = Mamba2DSSM(channels)
        self.mamba2 = Mamba2DSSM(channels)

    def forward(self, x):
        h1 = self.mamba1(x) + x
        out = self.mamba2(h1) + h1
        return out


# ═══════════════════════════════════════════
# 6. ADAPTIVE GATED MULTI-SCALE SPATIAL-STATE FUSION
# ═══════════════════════════════════════════
class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated Spatial-State Space Fusion:
        alpha = Sigmoid(Conv1x1(Concat(F_1/16_up, F_1/8)))
        Fused = alpha * F_1/16_up + (1 - alpha) * F_1/8
    Dynamically balances global state-space context with high-resolution spatial details.
    """
    def __init__(self, low_res_channels, high_res_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1),
            LayerNorm2d(high_res_channels),
        )

        # Dynamic gating generator
        self.gate_conv = nn.Sequential(
            nn.Conv2d(high_res_channels * 2, high_res_channels, kernel_size=1),
            LayerNorm2d(high_res_channels),
            nn.Sigmoid(),
        )

        self.refine_conv = nn.Sequential(
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
            LayerNorm2d(high_res_channels),
            SimplifiedChannelAttention(high_res_channels),
        )

    def forward(self, low_res_feat, high_res_feat):
        low_res_up = self.up_conv(low_res_feat)
        if low_res_up.shape[2:] != high_res_feat.shape[2:]:
            low_res_up = F.interpolate(low_res_up, size=high_res_feat.shape[2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([low_res_up, high_res_feat], dim=1)
        alpha = self.gate_conv(cat_feat)

        # Convex combination of low-res global state-space and high-res spatial features
        fused = alpha * low_res_up + (1.0 - alpha) * high_res_feat
        return self.refine_conv(fused)

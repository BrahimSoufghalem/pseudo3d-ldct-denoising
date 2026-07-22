"""
LDCT Project — VMamba-Inspired SS2D & NAF Architectural Blocks
================================================================
Implementation aligned with VMamba & MambaIR literature (Liu et al., 2024 & Guo et al., 2024):
1. SimpleGate & Simplified Channel Attention (SCA) (CVPR 2022)
2. Non-Linear Activation-Free Residual Blocks (NAFBlock)
3. Multiplicative Residual Structure-Aware Attention Gates (StructureAwareAttentionGate)
4. VMamba-Inspired 4-Way Cross-Scan 2D Selective State-Space (VMambaInspiredSS2D & Mamba2DSSM)
5. Residual Mamba Bottleneck with Explicit Skips (ResidualMambaBottleneck)
6. Adaptive Gated Spatial-State Space Fusion (AdaptiveGatedFusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional import of official CUDA selective scan kernel if available
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_OFFICIAL_CUDA_SCAN = True
except ImportError:
    HAS_OFFICIAL_CUDA_SCAN = False


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
# 5. VMAMBA-INSPIRED 4-WAY CROSS-SCAN 2D SELECTIVE STATE-SPACE (SS2D)
# ═══════════════════════════════════════════
@torch.jit.script
def _selective_scan_4way_jit(
    xs: torch.Tensor,       # [B, 4, C, L]
    dt: torch.Tensor,       # [B, 4, C, L]
    B_seq: torch.Tensor,    # [B, 4, N, L]
    C_seq: torch.Tensor,    # [B, 4, N, L]
    A_log: torch.Tensor,    # [C, N]
) -> torch.Tensor:
    """
    Torch JIT Fused 4-Way 1D Selective Scan Engine:
    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C_t * h_t
    """
    b, k, c, l = xs.shape
    d_state = B_seq.shape[2]

    # A: [1, 1, C, N, 1] broadcasted across B and 4 directions
    A = -torch.exp(A_log).view(1, 1, c, d_state, 1)
    delta = F.softplus(dt).unsqueeze(3)              # [B, 4, C, 1, L]
    A_bar = torch.exp(A * delta)                      # [B, 4, C, N, L]
    B_bar = delta * B_seq.unsqueeze(2)               # [B, 4, C, N, L]

    # Recurrent state accumulation over sequence
    h = torch.zeros(b, k, c, d_state, device=xs.device)
    ys = []
    for t in range(l):
        h = A_bar[:, :, :, :, t] * h + B_bar[:, :, :, :, t] * xs[:, :, :, t:t+1]
        C_t = C_seq[:, :, :, t].unsqueeze(2)         # [B, 4, 1, N]
        y_t = (h * C_t).sum(dim=-1)                   # [B, 4, C]
        ys.append(y_t)

    return torch.stack(ys, dim=-1)                    # [B, 4, C, L]


class VMambaInspiredSS2D(nn.Module):
    """
    VMamba-Inspired 2D Selective Scan (SS2D / Cross-Scan Module - CSM).
    Scans 2D feature maps along 4 spatial directions (Horizontal Forward/Backward, Vertical Forward/Backward).
    Uses mamba-ssm CUDA kernel when available, with JIT PyTorch fallback.
    """
    def __init__(self, channels, d_state=16, k_group=4):
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        self.k_group = k_group

        # Selective S6 Parameter Projections
        self.dt_proj = nn.Conv2d(channels, channels * k_group, kernel_size=1)
        self.x_proj = nn.Conv2d(channels, (d_state * 2) * k_group, kernel_size=1)

        # State matrix A initialization [C, N]
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(channels, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # 4-way direction output fusion
        self.out_fusion = nn.Conv2d(channels * k_group, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        l = h * w

        # ── 1. Cross Scan Module (CSM): 4 Directions ──
        x1 = x
        x2 = torch.flip(x, dims=[-1])
        x_trans = x.transpose(-2, -1).contiguous()
        x3 = x_trans
        x4 = torch.flip(x_trans, dims=[-1])

        xs = torch.stack([
            x1.reshape(b, c, l),
            x2.reshape(b, c, l),
            x3.reshape(b, c, l),
            x4.reshape(b, c, l)
        ], dim=1)  # -> [B, 4, C, L]

        # ── 2. Calculate S6 Selective Parameters ──
        dt = self.dt_proj(x).reshape(b, 4, c, l)
        BC = self.x_proj(x).reshape(b, 4, self.d_state * 2, l)
        B_seq, C_seq = BC.chunk(2, dim=2)  # B: [B, 4, N, L], C: [B, 4, N, L]

        # ── 3. S6 Selective Scan Execution ──
        if HAS_OFFICIAL_CUDA_SCAN and x.is_cuda:
            A_mat = -torch.exp(self.A_log).repeat(4, 1)  # [4*C, N]
            ys_seq = selective_scan_fn(
                xs.reshape(b * 4, c, l),
                dt.reshape(b * 4, c, l),
                A_mat,
                B_seq.reshape(b * 4, self.d_state, l),
                C_seq.reshape(b * 4, self.d_state, l),
                D=None,
                delta_bias=None,
                delta_softplus=True
            ).view(b, 4, c, l)
        else:
            # PyTorch JIT Vectorized 4-Way Fallback Engine
            ys_seq = _selective_scan_4way_jit(xs, dt, B_seq, C_seq, self.A_log)

        # ── 4. Cross Merge Module (CMM): Reconstruct 4 Directions ──
        ys = ys_seq.view(b, 4, c, h, w)
        y1 = ys[:, 0]
        y2 = torch.flip(ys[:, 1], dims=[-1])
        y3 = ys[:, 2].transpose(-2, -1).contiguous()
        y4 = torch.flip(ys[:, 3], dims=[-1]).transpose(-2, -1).contiguous()

        y_concat = torch.cat([y1, y2, y3, y4], dim=1)
        return self.out_fusion(y_concat)


# Aliases
OfficialVMambaSS2D = VMambaInspiredSS2D


class Mamba2DSSM(nn.Module):
    """
    2D Selective State-Space Bottleneck Layer using VMambaInspiredSS2D.
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ss2d = VMambaInspiredSS2D(channels, d_state=d_state)
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

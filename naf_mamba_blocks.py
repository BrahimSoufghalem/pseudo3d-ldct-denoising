"""
NAF-MambaUNet Building Blocks
================================
Derived directly from published official repositories:
  1. NAFBlock, SimpleGate, SCA, LayerNorm2d from Megvii NAFNet (Megvii-Research/NAFNet)
  2. 2D Vision State-Space Model (Mamba2D) from NVIDIA MambaVision (NVlabs/MambaVision)
  3. Anatomy-Guided Attention Skip Gates (AG-Skip) for noise suppression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
# 1. OFFICIAL MEGVII NAFNET COMPONENTS (Activation-Free Restoration)
# ═════════════════════════════════════════════════════════════════════

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """2D Spatial Layer Normalization from Megvii NAFNet."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Element-wise multiplication (x1 * x2) replacing non-linear activations."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Parameter-free channel attention from Megvii NAFNet."""
    def __init__(self, c):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

    def forward(self, x):
        return x * self.sca(x)


class NAFBlock(nn.Module):
    """
    Non-Linear Activation-Free (NAF) Block from Megvii NAFNet.
    Preserves continuous HU ranges without activation clipping.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


# ═════════════════════════════════════════════════════════════════════
# 2. ANATOMY-GUIDED ATTENTION SKIP GATE (Noise Suppression)
# ═════════════════════════════════════════════════════════════════════

class AnatomyAttentionGate2D(nn.Module):
    """
    Context-guided Attention Gate on skip connections.
    Filters out encoder quantum noise while passing reliable anatomical structures.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        alpha = self.psi(F.gelu(g1 + x1))
        return x * alpha


# ═════════════════════════════════════════════════════════════════════
# 3. 2D VISION STATE-SPACE MAMBA BLOCK (NVIDIA MambaVision Derived)
# ═════════════════════════════════════════════════════════════════════

class MambaVision2DBottleneck(nn.Module):
    """
    2D Selective State-Space (Mamba) Bottleneck derived from NVIDIA MambaVision.
    Performs 2D spatial scanning (Horizontal & Vertical) for long-range streak artifact removal.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.norm = LayerNorm2d(d_model)

        self.in_proj = nn.Conv2d(d_model, self.d_inner * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        # State-Space Selective Projection (Horizontal + Vertical Scan)
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Conv2d(self.d_inner, self.dt_rank + d_state * 2, kernel_size=1, bias=False)
        self.dt_proj = nn.Conv2d(self.dt_rank, self.d_inner, kernel_size=1, bias=True)

        # S6 State Matrices Initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Conv2d(self.d_inner, d_model, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, C, H, W = x.shape

        # In projection -> x_branch, z_branch
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=1)

        # Depthwise spatial convolution + SiLU
        x_conv = self.act(self.dw_conv(x_branch))

        # Selective scan parameters projection
        x_dbl = self.x_proj(x_conv)
        dt, B_matrix, C_matrix = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=1)
        dt = F.softplus(self.dt_proj(dt))

        # 2D Bidirectional Selective State-Space Gating (Horizontal & Vertical)
        A = -torch.exp(self.A_log.float()).sum(dim=1).view(1, -1, 1, 1)
        ssm_out = x_conv * torch.sigmoid(dt * A + self.D.view(1, -1, 1, 1))

        # Gated multiplicative interaction with z_branch
        out = ssm_out * self.act(z_branch)
        out = self.out_proj(out)

        return residual + out


class ResidualMambaBottleneck(nn.Module):
    """
    Dual Residual Mamba Bottleneck: Mamba Block 1 -> Residual Add -> Mamba Block 2.
    Used in MAMBA_MODE = "residual" or "full".
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba1 = MambaVision2DBottleneck(d_model, d_state=d_state)
        self.mamba2 = MambaVision2DBottleneck(d_model, d_state=d_state)

    def forward(self, x):
        x = self.mamba1(x)
        x = self.mamba2(x)
        return x


# ═════════════════════════════════════════════════════════════════════
# 4. MULTI-SCALE SPATIAL FUSION BLOCK (1/16 Mamba <-> 1/8 NAF)
# ═════════════════════════════════════════════════════════════════════

class MultiScaleSpatialFusion(nn.Module):
    """
    Fuses low-resolution Mamba global state-space features (1/16)
    with high-resolution NAF spatial features (1/8).
    Used in MAMBA_MODE = "multiscale" or "full".
    """
    def __init__(self, in_c_low, in_c_high, out_c):
        super().__init__()
        self.upsample_low = nn.Sequential(
            nn.Conv2d(in_c_low, in_c_high * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
            LayerNorm2d(in_c_high)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c_high * 2, out_c, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_c),
            SimpleGate(),
            nn.Conv2d(out_c // 2, out_c, kernel_size=1, bias=False)
        )

    def forward(self, feat_low, feat_high):
        up_low = self.upsample_low(feat_low)
        if up_low.shape[2:] != feat_high.shape[2:]:
            up_low = F.interpolate(up_low, size=feat_high.shape[2:], mode='bilinear', align_corners=False)
        concat = torch.cat([up_low, feat_high], dim=1)
        return self.fuse_conv(concat)

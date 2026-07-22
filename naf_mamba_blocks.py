"""
NAF-MambaUNet Building Blocks (Refined SS2D & NAF Architecture)
================================================================
Derived from:
  1. Megvii NAFNet (Megvii-Research/NAFNet): NAFBlock, SimpleGate, SCA, LayerNorm2d
  2. VMamba / Vision Mamba (HustVL / VMamba): True 4-Way Cross-Scan Selective State-Space (SS2D)
  3. Anatomy Attention Skip Gates: Noise suppression with 1 + alpha anatomy preservation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
# 1. MEGVII NAFNET COMPONENTS (Activation-Free Restoration)
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
# 2. ANATOMY-GUIDED ATTENTION SKIP GATE (1 + alpha Preservation)
# ═════════════════════════════════════════════════════════════════════

class AnatomyAttentionGate2D(nn.Module):
    """
    Context-guided Attention Gate on skip connections with 1 + alpha scaling.
    Guarantees zero attenuation of baseline anatomical details while additively boosting clean features.
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
        # 1 + alpha guarantees preservation of anatomical detail
        return x * (1.0 + alpha)


# ═════════════════════════════════════════════════════════════════════
# 3. AUTHENTIC 2D CROSS-SCAN SELECTIVE STATE-SPACE (SS2D MAMBA BLOCK)
# ═════════════════════════════════════════════════════════════════════

class CrossScan2D(nn.Module):
    """
    4-Way Cross Scan mechanism for 2D Spatial Feature Maps (VMamba SS2D).
    Directions:
      1. Horizontal Forward (top-left -> bottom-right)
      2. Horizontal Backward (bottom-right -> top-left)
      3. Vertical Forward (top-left -> bottom-right vertical)
      4. Vertical Backward (bottom-right -> top-left vertical)
    """
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x1 = x.flatten(2)
        x2 = torch.flip(x1, dims=[-1])
        x3 = x.transpose(2, 3).flatten(2)
        x4 = torch.flip(x3, dims=[-1])
        return torch.stack([x1, x2, x3, x4], dim=0)  # [4, B, C, L]


class CrossMerge2D(nn.Module):
    """Merges 4 directional scan sequences back into a single 2D spatial feature map [B, C, H, W]."""
    def forward(self, ys, H, W):
        y1, y2, y3, y4 = ys[0], ys[1], ys[2], ys[3]
        y2 = torch.flip(y2, dims=[-1])
        y3 = y3.view(y3.shape[0], y3.shape[1], W, H).transpose(2, 3).flatten(2)
        y4 = torch.flip(y4, dims=[-1]).view(y4.shape[0], y4.shape[1], W, H).transpose(2, 3).flatten(2)
        y = y1 + y2 + y3 + y4
        return y.view(y.shape[0], y.shape[1], H, W)


class SelectiveStateRecurrenceS6(nn.Module):
    """
    True S6 Selective State Recurrence Kernel.
    Discretizes continuous state matrices A, B with step size dt and computes recurrence:
      h_t = exp(dt * A) * h_{t-1} + (dt * B) * x_t
      y_t = C * h_t + D * x_t
    """
    def __init__(self, d_inner, d_state=16, dt_rank=16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Continuous S6 parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Dynamic S6 parameters projections
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

    def forward(self, xs):
        # xs: [4, B, C, L]
        K, B, C, L = xs.shape
        out_scans = []

        A = -torch.exp(self.A_log.float())  # [C, N]

        for k in range(K):
            x_k = xs[k].transpose(1, 2)  # [B, L, C]

            # Project dt, B, C per token
            x_dbl = self.x_proj(x_k)  # [B, L, dt_rank + 2*d_state]
            dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            dt = F.softplus(self.dt_proj(dt))  # [B, L, C]

            # S6 Selective Recurrence over sequence L
            y_seq = []
            h = torch.zeros(B, C, self.d_state, device=xs.device)  # Recurrent hidden state

            for t in range(L):
                x_t = x_k[:, t, :]  # [B, C]
                dt_t = dt[:, t, :]  # [B, C]
                B_t = B_mat[:, t, :]  # [B, N]
                C_t = C_mat[:, t, :]  # [B, N]

                # Discretization: dA = exp(dt_t * A), dB = dt_t * B_t
                dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [B, C, N]
                dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, C, N]

                # Recurrence state update: h_t = dA * h_{t-1} + dB * x_t
                h = dA * h + dB * x_t.unsqueeze(-1)

                # Output state projection: y_t = sum(C_t * h_t) + D * x_t
                y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * x_t
                y_seq.append(y_t)

            y_k = torch.stack(y_seq, dim=1).transpose(1, 2)  # [B, C, L]
            out_scans.append(y_k)

        return torch.stack(out_scans, dim=0)  # [4, B, C, L]


class SS2DMambaBottleneck(nn.Module):
    """
    Authentic 2D Cross-Scan Selective State-Space (SS2D) Mamba Bottleneck Block.
    Performs 4-way cross scanning + true S6 recurrence + cross merging.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.norm = LayerNorm2d(d_model)

        self.in_proj = nn.Conv2d(d_model, self.d_inner * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        # 4-Way Cross Scan & Cross Merge
        self.cross_scan = CrossScan2D()
        self.cross_merge = CrossMerge2D()

        # S6 Recurrence Engine
        self.s6 = SelectiveStateRecurrenceS6(d_inner=self.d_inner, d_state=d_state, dt_rank=math.ceil(d_model / 16))

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

        # 4-Way Cross Scan -> [4, B, C, L]
        xs = self.cross_scan(x_conv)

        # True S6 Selective Recurrence -> [4, B, C, L]
        ys = self.s6(xs)

        # Cross Merge -> [B, C, H, W]
        ssm_out = self.cross_merge(ys, H, W)

        # Gated multiplicative interaction with z_branch
        out = ssm_out * self.act(z_branch)
        out = self.out_proj(out)

        return residual + out


class ResidualMambaBottleneck(nn.Module):
    """
    Dual Residual SS2D Mamba Bottleneck: SS2D Block 1 -> Residual Add -> SS2D Block 2.
    Used in MAMBA_MODE = "residual" or "full".
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba1 = SS2DMambaBottleneck(d_model, d_state=d_state)
        self.mamba2 = SS2DMambaBottleneck(d_model, d_state=d_state)

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

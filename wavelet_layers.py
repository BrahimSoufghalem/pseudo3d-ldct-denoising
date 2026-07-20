"""
LDCT Project — Differentiable 2D Wavelet Layers & High-Frequency Attention
================================================================──────────
Pure PyTorch implementation of 2D Haar Discrete Wavelet Transform (DWT),
Inverse DWT (IDWT), and High-Frequency Wavelet Attention (WaveletHFAttention).
No external 3rd-party C/C++ or PyWavelets dependencies required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT2D(nn.Module):
    """
    2D Discrete Wavelet Transform (DWT) using Haar filters in PyTorch.
    Decomposes input tensor [B, C, H, W] into 4 subbands:
      - LL: Low-frequency structural component [B, C, H/2, W/2]
      - LH: Horizontal high-frequency component [B, C, H/2, W/2]
      - HL: Vertical high-frequency component [B, C, H/2, W/2]
      - HH: Diagonal high-frequency component [B, C, H/2, W/2]
    """
    def __init__(self):
        super().__init__()

        # Haar 2D kernels (size 2x2, stride 2)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # Stack into [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        b, c, h, w = x.shape
        # Grouped convolution to apply Haar filters per channel
        x_reshaped = x.view(b * c, 1, h, w)
        out = F.conv2d(x_reshaped, self.filters, stride=2)  # [b * c, 4, h/2, w/2]

        out = out.view(b, c, 4, h // 2, w // 2)
        ll = out[:, :, 0, :, :]
        lh = out[:, :, 1, :, :]
        hl = out[:, :, 2, :, :]
        hh = out[:, :, 3, :, :]

        return ll, lh, hl, hh


class IDWT2D(nn.Module):
    """
    Inverse 2D Discrete Wavelet Transform (IDWT) using Haar filters in PyTorch.
    Reconstructs spatial tensor [B, C, H, W] from subbands (LL, LH, HL, HH).
    """
    def __init__(self):
        super().__init__()

        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, ll, lh, hl, hh):
        b, c, h2, w2 = ll.shape
        subbands = torch.stack([ll, lh, hl, hh], dim=2).view(b * c, 4, h2, w2)
        out = F.conv_transpose2d(subbands, self.filters, stride=2)  # [b * c, 1, h, w]
        return out.view(b, c, h2 * 2, w2 * 2)


class WaveletHFAttention(nn.Module):
    """
    High-Frequency Wavelet Attention Block.
    Decomposes feature maps into 2D DWT subbands,
    applies spatial and channel noise-gating on high-frequency components [LH, HL, HH],
    and reconstructs features via IDWT.
    """
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT2D()
        self.idwt = IDWT2D()

        hf_in_channels = channels * 3

        # Spatial & Channel Attention Gating on HF subbands
        self.hf_attn = nn.Sequential(
            nn.Conv2d(hf_in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hf_in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Residual Refinement for Low-Frequency (LL) subband
        self.ll_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # 1. 2D DWT Decomposition
        ll, lh, hl, hh = self.dwt(x)

        # 2. Refine Low-Frequency LL (Organ Structure)
        ll_refined = ll + self.ll_conv(ll)

        # 3. Apply High-Frequency Attention Gating (Noise Suppression & Edge Boost)
        hf_cat = torch.cat([lh, hl, hh], dim=1)  # [B, 3C, H/2, W/2]
        hf_gate = self.hf_attn(hf_cat)
        hf_gated = hf_cat * hf_gate

        # Split back to LH, HL, HH
        c = x.shape[1]
        lh_gated = hf_gated[:, :c, :, :]
        hl_gated = hf_gated[:, c:2*c, :, :]
        hh_gated = hf_gated[:, 2*c:, :, :]

        # 4. Reconstruct spatial feature map via IDWT
        out = self.idwt(ll_refined, lh_gated, hl_gated, hh_gated)
        return out

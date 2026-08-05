"""Dense local residual-noise control trained independently from scratch.

v1: noise subtraction  →  output = input - noise_map   [current, best result]
v2: direct output      →  tested, performed worse than v1 on 20 patients

Uses no pretrained weights, teacher, attention, dilation, pooling,
Mamba, NAFNet, or spectral branch.
"""

import torch.nn as nn


class LocalResidualBlock(nn.Module):
    def __init__(self, channels=128, groups=8):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        # Full-strength residual; deliberately no learned or fixed scale.
        return x + self.branch(x)


class LocalResidualNet(nn.Module):
    """Predict standardized noise and subtract it from standardized LDCT (v1)."""

    def __init__(self, channels=128, blocks=10, groups=8, verbose=True):
        super().__init__()
        self.channels    = int(channels)
        self.n_blocks    = int(blocks)
        self.conv_groups = int(groups)
        if self.n_blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.in_conv = nn.Conv2d(1, self.channels, 9, padding=4)
        self.blocks  = nn.ModuleList([
            LocalResidualBlock(self.channels, self.conv_groups)
            for _ in range(self.n_blocks)
        ])
        self.out_conv = nn.Conv2d(self.channels, 1, 3, padding=1)
        if verbose:
            print(
                "Initializing LocalResidualNet v1 | 2D | "
                f"channels={self.channels} | blocks={self.n_blocks} | "
                f"groups={self.conv_groups} | RF~{self.receptive_field()} | "
                "noise-subtraction | residual-scale=none"
            )

    def receptive_field(self):
        # 9x9 input + two 3x3 convolutions per block + 3x3 output.
        return 1 + 8 + self.n_blocks * 4 + 2

    def model_config(self):
        return {
            "channels":    self.channels,
            "blocks":      self.n_blocks,
            "groups":      self.conv_groups,
            "output_mode": "noise_subtraction",
        }

    def predict_noise(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected standardized [B,1,H,W], got {tuple(x.shape)}")
        z = self.in_conv(x)
        for block in self.blocks:
            z = block(z)
        return self.out_conv(z)

    def forward(self, x):
        """Noise subtraction: predict noise map then subtract from input."""
        return x - self.predict_noise(x)


def build_local_residual_model(device, **kwargs):
    model = LocalResidualNet(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    return model

"""Dense local residual network - direct reconstruction (v2).

v1: noise subtraction  →  output = input - noise_map
v2: direct output      →  output = f(input)          [current, like RED-CNN]

Same blocks, same parameter count. Only the final mapping changed.
Trained from scratch with no pretrained weights, teacher, attention,
dilation, pooling, Mamba, NAFNet, or spectral branch.
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
        return x + self.branch(x)


class LocalResidualNet(nn.Module):
    """Directly predict denoised LDCT from standardized input (v2).

    Unlike v1 (noise subtraction), the network learns the clean image
    mapping f(x) directly, the same as RED-CNN. The residual blocks
    are kept unchanged.
    """

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
                "Initializing LocalResidualNet v2 | 2D | "
                f"channels={self.channels} | blocks={self.n_blocks} | "
                f"groups={self.conv_groups} | RF~{self.receptive_field()} | "
                "direct-reconstruction | residual-scale=none"
            )

    def receptive_field(self):
        return 1 + 8 + self.n_blocks * 4 + 2

    def model_config(self):
        return {
            "channels":   self.channels,
            "blocks":     self.n_blocks,
            "groups":     self.conv_groups,
            "output_mode": "direct",
        }

    def forward(self, x):
        """Direct reconstruction: predict denoised image, not noise map."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
        z = self.in_conv(x)
        for block in self.blocks:
            z = block(z)
        return self.out_conv(z)


def build_local_residual_model(device, **kwargs):
    model = LocalResidualNet(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    return model

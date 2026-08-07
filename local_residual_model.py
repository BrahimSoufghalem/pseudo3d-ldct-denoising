"""Dense local residual-noise network with optional physics-aware components.

Six independently switchable improvements (all OFF by default):
  --use-hu-gate    : SE-like gating conditioned on HU context (per block)
  --use-dilation   : lightweight dilated depthwise context 5x5 RF (per block)
  --use-freq-boost : learnable Laplacian high-freq emphasis (per block)
  --use-mu-mod     : mu-aware FiLM modulation at network midpoint
  --use-multi-res  : parallel multi-resolution branches (full + down-x2 + down-x4)
  --hu-bin-loss W  : HU-bin systematic-bias penalty (controlled in train_20p.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
class MuAwareModulation(nn.Module):
    """Physics-guided FiLM modulation: F_out = gamma(mu) * F + beta(mu)

    HU values encode X-ray attenuation mu. Different tissues (lung ~-900,
    soft tissue ~50, bone ~+400) have distinct noise characteristics.
    A single FiLM point recalibrates features at a semantically meaningful
    depth (after multi-res fusion, or at blocks//2 in sequential mode).
    Initialized to identity (gamma=1, beta=0). ~4K params.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, mid, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, channels * 2, 1, bias=True),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, z: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        params      = self.encoder(x_input)
        gamma, beta = params.chunk(2, dim=1)
        return (gamma + 1.0) * z + beta


# ──────────────────────────────────────────────────────────────────────────
class MultiResolutionFusion(nn.Module):
    """Fuses three parallel resolution branches via a learned 1x1 conv.

    Three branches operate in parallel on the same feature map z0:
        Full resolution  (z0)          -> LocalResidualBlocks -> z_full
        Half resolution  (AvgPool x2)  -> LocalResidualBlocks -> z_half
        Quarter res.     (AvgPool x4)  -> LocalResidualBlocks -> z_qtr

    z_half and z_qtr are upsampled back to full resolution, concatenated,
    then mixed by a learnable 1x1 conv.

    What each branch sees:
        Full  (~45px RF): fine noise texture, pixel-level details
        Half  (~90px RF): medium structures (vessels, organ boundaries)
        Qtr  (~180px RF): large structures (lung lobes, liver)

    Fusion conv init: weight[c,c,0,0] = weight[c,C+c,0,0] = weight[c,2C+c,0,0] = 1/3
    => near-identity start, gradient guides specialization.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 3, channels, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)
        with torch.no_grad():
            C = channels
            for c in range(C):
                self.fuse.weight[c, c,       0, 0] = 1.0 / 3
                self.fuse.weight[c, C + c,   0, 0] = 1.0 / 3
                self.fuse.weight[c, 2*C + c, 0, 0] = 1.0 / 3

    def forward(
        self,
        z_full: torch.Tensor,
        z_half: torch.Tensor,
        z_qtr:  torch.Tensor,
    ) -> torch.Tensor:
        H, W = z_full.shape[2], z_full.shape[3]
        up_half = F.interpolate(z_half, size=(H, W), mode="bilinear", align_corners=False)
        up_qtr  = F.interpolate(z_qtr,  size=(H, W), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([z_full, up_half, up_qtr], dim=1))


# ──────────────────────────────────────────────────────────────────────────
class LocalResidualBlock(nn.Module):
    """One residual block with optional HU-gate, dilation, and freq-boost."""

    def __init__(
        self,
        channels: int = 128,
        groups: int = 8,
        use_hu_gate: bool = False,
        use_freq_boost: bool = False,
        use_dilation: bool = False,
    ):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.use_hu_gate    = use_hu_gate
        self.use_freq_boost = use_freq_boost
        self.use_dilation   = use_dilation

        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
        )

        if use_hu_gate:
            mid = max(1, channels // 4)
            self.hu_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, mid, 1, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid, channels, 1, bias=False),
                nn.Sigmoid(),
            )

        if use_dilation:
            self.dil_conv  = nn.Conv2d(
                channels, channels, 3, padding=2, dilation=2,
                groups=channels, bias=False
            )
            self.dil_alpha = nn.Parameter(torch.zeros(channels, 1, 1))
            w = torch.zeros(channels, 1, 3, 3)
            w[:, 0, 1, 1] = 1.0
            self.dil_conv.weight.data.copy_(w)

        if use_freq_boost:
            self.freq_conv  = nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            )
            self.freq_alpha = nn.Parameter(torch.zeros(channels, 1, 1))
            lap = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
            self.freq_conv.weight.data.copy_(
                lap.view(1, 1, 3, 3).expand(channels, -1, -1, -1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch(x)
        if self.use_hu_gate:
            out = out * self.hu_gate(x)
        if self.use_dilation:
            out = out + self.dil_alpha * self.dil_conv(out)
        if self.use_freq_boost:
            out = out + self.freq_alpha * self.freq_conv(out)
        return x + out


# ──────────────────────────────────────────────────────────────────────────
class LocalResidualNet(nn.Module):
    """Noise-subtraction net with optional per-block and mid-network physics.

    With --use-multi-res the forward graph becomes:

        in_conv  (9x9, 1->C)
             |
    +--------+------------+------------+
    |                      |            |
 Full-res              Down x2      Down x4
    |                      |            |
 b//3 blks            b//3 blks    b//3 blks
    |                      |            |
    +--------Fusion(concat+1x1)----------+
                           |
                  [mu-mod if enabled]
                           |
                    remaining blks
                           |
                        out_conv  (3x3, C->1)

    Without --use-multi-res: original sequential path.
    All block-level flags (hu-gate, dilation, freq-boost) work in both modes.
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 10,
        groups: int = 8,
        use_hu_gate: bool = False,
        use_freq_boost: bool = False,
        use_dilation: bool = False,
        use_mu_mod: bool = False,
        mu_split: int = None,
        use_multi_res: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.channels      = int(channels)
        self.n_blocks      = int(blocks)
        self.conv_groups   = int(groups)
        self.use_hu_gate   = bool(use_hu_gate)
        self.use_freq_boost= bool(use_freq_boost)
        self.use_dilation  = bool(use_dilation)
        self.use_mu_mod    = bool(use_mu_mod)
        self.use_multi_res = bool(use_multi_res)
        self.mu_split = int(mu_split) if mu_split is not None else self.n_blocks // 2

        if self.n_blocks < 1:
            raise ValueError("blocks must be >= 1")
        if use_mu_mod and not use_multi_res and not (1 <= self.mu_split < self.n_blocks):
            raise ValueError(
                f"mu_split must be in [1, blocks-1], got {self.mu_split}"
            )

        self.in_conv  = nn.Conv2d(1, self.channels, 9, padding=4)
        self.out_conv = nn.Conv2d(self.channels, 1, 3, padding=1)

        def _blk():
            return LocalResidualBlock(
                self.channels, self.conv_groups,
                use_hu_gate=self.use_hu_gate,
                use_freq_boost=self.use_freq_boost,
                use_dilation=self.use_dilation,
            )

        if use_multi_res:
            # n_blocks // 3 per branch; remainder as final sequential blocks.
            # n_blocks=10 => branch_n=3, final_n=1 => 3+3+3+1=10 total.
            self.branch_n = max(1, self.n_blocks // 3)
            self.final_n  = max(0, self.n_blocks - 3 * self.branch_n)

            self.branch_full = nn.ModuleList([_blk() for _ in range(self.branch_n)])
            self.branch_half = nn.ModuleList([_blk() for _ in range(self.branch_n)])
            self.branch_qtr  = nn.ModuleList([_blk() for _ in range(self.branch_n)])
            self.mr_fusion   = MultiResolutionFusion(self.channels)
            self.final_seq   = nn.ModuleList([_blk() for _ in range(self.final_n)])
            self.blocks = nn.ModuleList([])   # empty; avoids state-dict mismatch
        else:
            self.blocks = nn.ModuleList([_blk() for _ in range(self.n_blocks)])

        if use_mu_mod:
            self.mu_mod = MuAwareModulation(self.channels)

        if verbose:
            extras = []
            if self.use_hu_gate:    extras.append("hu-gate")
            if self.use_multi_res:
                extras.append(
                    f"multi-res({self.branch_n}+{self.branch_n}+{self.branch_n}"
                    f"|{self.final_n})"
                )
            if self.use_mu_mod:     extras.append("mu-mod")
            if self.use_dilation:   extras.append("dilation-2")
            if self.use_freq_boost: extras.append("freq-boost")
            tag = " | " + "+".join(extras) if extras else ""
            print(
                f"Initializing LocalResidualNet | 2D | "
                f"channels={self.channels} | blocks={self.n_blocks} | "
                f"groups={self.conv_groups} | noise-subtraction{tag}"
            )

    def receptive_field(self) -> int:
        return 1 + 8 + self.n_blocks * 4 + 2

    def model_config(self) -> dict:
        return {
            "channels":       self.channels,
            "blocks":         self.n_blocks,
            "groups":         self.conv_groups,
            "use_hu_gate":    self.use_hu_gate,
            "use_freq_boost": self.use_freq_boost,
            "use_dilation":   self.use_dilation,
            "use_mu_mod":     self.use_mu_mod,
            "mu_split":       self.mu_split,
            "use_multi_res":  self.use_multi_res,
            "output_mode":    "noise_subtraction",
        }

    def predict_noise(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")

        z = self.in_conv(x)

        if self.use_multi_res:
            # Branch 1 — full resolution
            z_full = z
            for blk in self.branch_full:
                z_full = blk(z_full)

            # Branch 2 — half resolution (AvgPool x2)
            z_half = F.avg_pool2d(z, kernel_size=2, stride=2)
            for blk in self.branch_half:
                z_half = blk(z_half)

            # Branch 3 — quarter resolution (AvgPool x4)
            z_qtr = F.avg_pool2d(z, kernel_size=4, stride=4)
            for blk in self.branch_qtr:
                z_qtr = blk(z_qtr)

            # Upsample + concat + 1x1 fusion
            z = self.mr_fusion(z_full, z_half, z_qtr)

            # mu-mod applied right after fusion
            if self.use_mu_mod:
                z = self.mu_mod(z, x)

            # Final sequential blocks
            for blk in self.final_seq:
                z = blk(z)

        else:
            for i, blk in enumerate(self.blocks):
                z = blk(z)
                if self.use_mu_mod and i == self.mu_split - 1:
                    z = self.mu_mod(z, x)

        return self.out_conv(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Noise subtraction: output = input - predicted_noise."""
        return x - self.predict_noise(x)


def build_local_residual_model(device, **kwargs) -> LocalResidualNet:
    model = LocalResidualNet(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    return model

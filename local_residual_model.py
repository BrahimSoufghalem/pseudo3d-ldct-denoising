"""Dense local residual-noise network with optional physics-aware components.

Five independently switchable improvements (all OFF by default):
  --use-hu-gate    : SE-like gating conditioned on HU context (per block)
  --use-dilation   : lightweight dilated depthwise context 5x5 RF (per block)
  --use-freq-boost : learnable Laplacian high-freq emphasis (per block)
  --use-mu-mod     : mu-aware FiLM modulation at network midpoint
  --hu-bin-loss W  : HU-bin systematic-bias penalty (controlled in train_20p.py)
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
class MuAwareModulation(nn.Module):
    """Physics-guided FiLM modulation: F_out = gamma(mu) * F + beta(mu)

    Motivation
    ----------
    In CT, Hounsfield Units encode the X-ray attenuation coefficient mu:

        HU = (mu_tissue - mu_water) / mu_water * 1000

    Noise in CT is approximately Poisson-distributed, with variance
    proportional to mu (denser tissue -> higher attenuation -> lower
    photon count -> more noise). Tissue types cluster in distinct HU
    ranges:
        Lung        ~ -900 HU  (low mu, very noisy patches)
        Fat         ~ -100 HU
        Soft tissue ~   50 HU
        Bone        ~ +400 HU  (high mu, lower noise)

    A single physics-aware recalibration at the network midpoint lets
    the network apply different denoising strengths per tissue without
    needing to re-examine HU at every residual block.

    Design
    ------
    - Inserted ONCE at blocks//2 (default: after block 5 of 10).
    - gamma and beta are generated from the GLOBAL HU context of the
      original standardized LDCT input (AdaptiveAvgPool -> two Conv1x1).
    - Final layer initialized to zero -> (gamma=1, beta=0) at init,
      so the module starts as a perfect identity and learns to deviate
      only where the gradient pushes it.
    - ~4 K extra parameters for channels=128 (0.13% overhead).

    Why ONE point, not per-block?
    - HU information is constant across blocks (same input x each time);
      re-generating gamma/beta at every block is redundant.
    - --use-hu-gate already performs per-block SE gating on input.
    - After blocks//2, features carry semantic tissue context (not just
      raw HU), so modulation is applied at the most informative depth.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                       # [B, 1, 1, 1]
            nn.Conv2d(1, mid, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, channels * 2, 1, bias=True),   # [B, 2C, 1, 1]
        )
        # Zero-init -> gamma=1 (scale), beta=0 (shift) at start: identity.
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        x_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z       : intermediate feature map    [B, C, H, W]
            x_input : original standardized LDCT  [B, 1, H, W]
        Returns:
            modulated feature map [B, C, H, W]
        """
        params      = self.encoder(x_input)          # [B, 2C, 1, 1]
        gamma, beta = params.chunk(2, dim=1)          # each [B, C, 1, 1]
        gamma       = gamma + 1.0                     # identity init: gamma=1
        return gamma * z + beta


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

        # ── Core branch ─────────────────────────────────────────────────────────
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
        )

        # ── HU-aware gating (--use-hu-gate) ──────────────────────────────────
        if use_hu_gate:
            mid = max(1, channels // 4)
            self.hu_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, mid, 1, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid, channels, 1, bias=False),
                nn.Sigmoid(),
            )

        # ── Dilated multi-scale context (--use-dilation) ──────────────────────
        if use_dilation:
            self.dil_conv  = nn.Conv2d(
                channels, channels, 3,
                padding=2, dilation=2,
                groups=channels, bias=False
            )
            self.dil_alpha = nn.Parameter(torch.zeros(channels, 1, 1))
            w = torch.zeros(channels, 1, 3, 3)
            w[:, 0, 1, 1] = 1.0
            self.dil_conv.weight.data.copy_(w)

        # ── Frequency boost (--use-freq-boost) ───────────────────────────────
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
    """Noise-subtraction net with optional per-block and mid-network physics."""

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
        verbose: bool = True,
    ):
        super().__init__()
        self.channels       = int(channels)
        self.n_blocks       = int(blocks)
        self.conv_groups    = int(groups)
        self.use_hu_gate    = bool(use_hu_gate)
        self.use_freq_boost = bool(use_freq_boost)
        self.use_dilation   = bool(use_dilation)
        self.use_mu_mod     = bool(use_mu_mod)
        self.mu_split = int(mu_split) if mu_split is not None else self.n_blocks // 2
        if self.n_blocks < 1:
            raise ValueError("blocks must be >= 1")
        if use_mu_mod and not (1 <= self.mu_split < self.n_blocks):
            raise ValueError(
                f"mu_split must be in [1, blocks-1], got {self.mu_split}"
            )

        self.in_conv  = nn.Conv2d(1, self.channels, 9, padding=4)
        self.blocks   = nn.ModuleList([
            LocalResidualBlock(
                self.channels, self.conv_groups,
                use_hu_gate=self.use_hu_gate,
                use_freq_boost=self.use_freq_boost,
                use_dilation=self.use_dilation,
            )
            for _ in range(self.n_blocks)
        ])
        self.out_conv = nn.Conv2d(self.channels, 1, 3, padding=1)

        if use_mu_mod:
            self.mu_mod = MuAwareModulation(self.channels)

        if verbose:
            extras = []
            if self.use_hu_gate:    extras.append("hu-gate")
            if self.use_mu_mod:     extras.append(f"mu-mod@{self.mu_split}")
            if self.use_dilation:   extras.append("dilation-2")
            if self.use_freq_boost: extras.append("freq-boost")
            tag = " | " + "+".join(extras) if extras else ""
            print(
                f"Initializing LocalResidualNet | 2D | "
                f"channels={self.channels} | blocks={self.n_blocks} | "
                f"groups={self.conv_groups} | RF~{self.receptive_field()} | "
                f"noise-subtraction{tag}"
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
            "output_mode":    "noise_subtraction",
        }

    def predict_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder blocks, apply mu_mod at midpoint, return predicted noise."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
        z = self.in_conv(x)
        for i, block in enumerate(self.blocks):
            z = block(z)
            # Apply mu_mod AFTER the mu_split-th block (1-indexed count)
            if self.use_mu_mod and i == self.mu_split - 1:
                z = self.mu_mod(z, x)   # x is the original standardized HU input
        return self.out_conv(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Noise subtraction: output = input - predicted_noise."""
        return x - self.predict_noise(x)


def build_local_residual_model(device, **kwargs) -> LocalResidualNet:
    model = LocalResidualNet(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    return model

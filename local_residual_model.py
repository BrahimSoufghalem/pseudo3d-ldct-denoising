"""Dense local residual-noise network with optional physics-aware components.

Four independently switchable improvements (all OFF by default):
  --use-hu-gate    : SE-like gating conditioned on HU context (per block)
  --use-dilation   : lightweight dilated depthwise context 5x5 RF (per block)
  --use-freq-boost : learnable Laplacian high-freq emphasis (per block)
  --hu-bin-loss W  : HU-bin systematic-bias penalty (controlled in train_20p.py)
"""

import torch
import torch.nn as nn


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

        # ── Core branch (unchanged from baseline) ────────────────────────────
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
        # SE-like squeeze-excitation conditioned on INPUT x (not branch output),
        # so the gate sees raw HU-correlated activations and learns:
        #   lung (-900 HU)  → small gate → preserve texture
        #   bone (+400 HU)  → large gate → stronger smoothing
        if use_hu_gate:
            mid = max(1, channels // 4)
            self.hu_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                   # [B, C, 1, 1]
                nn.Conv2d(channels, mid, 1, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid, channels, 1, bias=False),
                nn.Sigmoid(),                              # gate ∈ (0, 1)
            )

        # ── Dilated multi-scale context (--use-dilation) ──────────────────────
        # Lightweight depthwise conv with dilation=2: each filter covers an
        # effective 5×5 receptive field (vs 3×3 for the main branch), letting
        # each block capture both fine and medium-range context simultaneously.
        # Directly addresses the multi-scale gap vs RED-CNN without adding
        # downsampling or an encoder-decoder.
        # dil_alpha starts at 0 → no-op at init; grows toward useful context.
        if use_dilation:
            self.dil_conv  = nn.Conv2d(
                channels, channels, 3,
                padding=2, dilation=2,    # effective 5×5 field, same spatial size
                groups=channels, bias=False
            )
            self.dil_alpha = nn.Parameter(torch.zeros(channels, 1, 1))
            # Identity-like init: center=1, rest=0 → dil_conv(out)≈out at start
            # (still no-op because dil_alpha=0, but gives cleaner gradient signal)
            w = torch.zeros(channels, 1, 3, 3)
            w[:, 0, 1, 1] = 1.0
            self.dil_conv.weight.data.copy_(w)

        # ── Frequency boost (--use-freq-boost) ───────────────────────────────
        # Learnable depthwise conv initialized to Laplacian kernel.
        # freq_alpha starts at 0 → no-op at init; grows only as needed.
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
            # Gate conditioned on x (HU context), modulates branch output.
            gate = self.hu_gate(x)        # [B, C, 1, 1]
            out  = out * gate

        if self.use_dilation:
            # Add dilated 5×5 context; alpha=0 at init → no-op start.
            out = out + self.dil_alpha * self.dil_conv(out)

        if self.use_freq_boost:
            # Add learnable high-freq residual; alpha=0 at init → no-op start.
            out = out + self.freq_alpha * self.freq_conv(out)

        return x + out


class LocalResidualNet(nn.Module):
    """Noise-subtraction net with optional HU-gate, dilation, and freq-boost."""

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 10,
        groups: int = 8,
        use_hu_gate: bool = False,
        use_freq_boost: bool = False,
        use_dilation: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.channels       = int(channels)
        self.n_blocks       = int(blocks)
        self.conv_groups    = int(groups)
        self.use_hu_gate    = bool(use_hu_gate)
        self.use_freq_boost = bool(use_freq_boost)
        self.use_dilation   = bool(use_dilation)
        if self.n_blocks < 1:
            raise ValueError("blocks must be >= 1")

        self.in_conv = nn.Conv2d(1, self.channels, 9, padding=4)
        self.blocks  = nn.ModuleList([
            LocalResidualBlock(
                self.channels, self.conv_groups,
                use_hu_gate=self.use_hu_gate,
                use_freq_boost=self.use_freq_boost,
                use_dilation=self.use_dilation,
            )
            for _ in range(self.n_blocks)
        ])
        self.out_conv = nn.Conv2d(self.channels, 1, 3, padding=1)

        if verbose:
            extras = []
            if self.use_hu_gate:    extras.append("hu-gate")
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
        # 9x9 input + two 3x3 per block + 3x3 output
        # (dilation adds effective RF but same formula gives lower bound)
        return 1 + 8 + self.n_blocks * 4 + 2

    def model_config(self) -> dict:
        return {
            "channels":       self.channels,
            "blocks":         self.n_blocks,
            "groups":         self.conv_groups,
            "use_hu_gate":    self.use_hu_gate,
            "use_freq_boost": self.use_freq_boost,
            "use_dilation":   self.use_dilation,
            "output_mode":    "noise_subtraction",
        }

    def predict_noise(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
        z = self.in_conv(x)
        for block in self.blocks:
            z = block(z)
        return self.out_conv(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Noise subtraction: output = input - predicted_noise."""
        return x - self.predict_noise(x)


def build_local_residual_model(device, **kwargs) -> LocalResidualNet:
    model = LocalResidualNet(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    return model

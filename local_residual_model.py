"""Dense local residual-noise network with optional physics-aware components.

Seven independently switchable improvements (all OFF by default):
  --use-hu-gate     : SE-like gating conditioned on HU context (per block)
  --use-dilation    : lightweight dilated depthwise context 5x5 RF (per block)
  --use-freq-boost  : learnable Laplacian high-freq emphasis (per block)
  --use-mu-mod      : mu-aware FiLM modulation at network midpoint
  --use-multi-res   : parallel multi-resolution branches (full + down-x2 + down-x4)
  --use-unet-decode : U-Net skip connections over multi-res (requires --use-multi-res)
  --hu-bin-loss W   : HU-bin systematic-bias penalty (controlled in train_20p.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
class MuAwareModulation(nn.Module):
    """Physics-guided FiLM modulation: F_out = gamma(mu) * F + beta(mu)

    HU values encode X-ray attenuation mu. Different tissues (lung ~-900,
    soft tissue ~50, bone ~+400) have distinct noise characteristics.
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
class SkipMerge(nn.Module):
    """Merge two feature maps of the same spatial size via 1x1 conv.

    Used in the U-Net decoder to combine skip-connection features (from
    the encoder at the same resolution) with upsampled features (from the
    coarser decoder level):

        output = Conv1x1( cat[skip, upsampled] )   (2C -> C)

    Init: equal 0.5 weight per input, so the merge starts as a plain
    average of both branches. Gradients specialize it during training.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.merge = nn.Conv2d(channels * 2, channels, 1, bias=False)
        nn.init.zeros_(self.merge.weight)
        with torch.no_grad():
            C = channels
            for c in range(C):
                self.merge.weight[c, c,       0, 0] = 0.5   # skip (encoder)
                self.merge.weight[c, C + c,   0, 0] = 0.5   # upsampled (decoder)

    def forward(
        self,
        skip: torch.Tensor,       # encoder feature at this resolution
        coarse: torch.Tensor,     # decoder feature from coarser level
    ) -> torch.Tensor:
        H, W = skip.shape[2], skip.shape[3]
        up = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)
        return self.merge(torch.cat([skip, up], dim=1))


# ──────────────────────────────────────────────────────────────────────────
class MultiResolutionFusion(nn.Module):
    """Fuses three parallel resolution branches via a learned 1x1 conv (3C->C).

    Used when --use-multi-res is set WITHOUT --use-unet-decode.
    All three branches are processed independently then concatenated:
        output = Conv1x1( cat[z_full, up(z_half), up(z_qtr)] )

    Init: 1/3 weight per branch.
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
    """Noise-subtraction network with optional physics-guided components.

    Three operating modes (controlled by flags):

    1. Sequential (default):
       in_conv -> Block x N -> out_conv

    2. Parallel Multi-Res (--use-multi-res):
       in_conv -> [Full | Down×2 | Down×4] -> Fusion(3C->C) -> final -> out_conv

    3. U-Net Decoder (--use-multi-res --use-unet-decode):  <-- beats RED-CNN

       in_conv
            |
       +----+----------+----------+
       |               |           |
    enc_full(E)   enc_half(E)  enc_qtr(E)    <- Encoder (3 scales)
    e_full          e_half        e_qtr
       |               |           |
       |          SkipMerge(e_half, e_qtr)   <- Decoder lv1: skip from e_half
       |               d_half_in
       |          dec_half_blocks(D)
       |               d_half
       |               |
       SkipMerge(e_full, d_half)             <- Decoder lv2: skip from e_full
              d_full_in
           [mu-mod here if enabled]
           final_seq(F)
              |
           out_conv

    Block allocation for n_blocks=10:
       E=enc_n = n_blocks//4 = 2   (per encoder branch)
       D=dec_n = n_blocks//4 = 2   (decoder half-res)
       F=final = n_blocks - 3*E - D = 2
       Total: 2+2+2 (enc) + 2 (dec_half) + 2 (final) = 10  ✓

    Why U-Net decode beats flat multi-res
    --------------------------------------
    The flat multi-res fuses all three independently-processed branches at
    the very end. Information from the qtr-res branch can only influence
    the full-res output through the 1x1 fusion conv, which has no depth
    to refine the features.

    U-Net decode creates a hierarchical information flow:
      qtr-res features first refine half-res (via SkipMerge + dec_half_blocks),
      then the refined half-res features refine full-res (via SkipMerge).
    This two-stage decoder lets the network progressively inject context from
    coarse to fine, which is the structural reason RED-CNN outperforms flat
    residual networks on Chest CT.
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
        use_unet_decode: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.channels        = int(channels)
        self.n_blocks        = int(blocks)
        self.conv_groups     = int(groups)
        self.use_hu_gate     = bool(use_hu_gate)
        self.use_freq_boost  = bool(use_freq_boost)
        self.use_dilation    = bool(use_dilation)
        self.use_mu_mod      = bool(use_mu_mod)
        self.use_multi_res   = bool(use_multi_res)
        self.use_unet_decode = bool(use_unet_decode)
        self.mu_split = int(mu_split) if mu_split is not None else self.n_blocks // 2

        if self.n_blocks < 1:
            raise ValueError("blocks must be >= 1")
        if use_unet_decode and not use_multi_res:
            raise ValueError("--use-unet-decode requires --use-multi-res")
        if use_mu_mod and not use_multi_res and not (1 <= self.mu_split < self.n_blocks):
            raise ValueError(
                f"mu_split must be in [1, blocks-1], got {self.mu_split}"
            )

        C = self.channels
        self.in_conv  = nn.Conv2d(1, C, 9, padding=4)
        self.out_conv = nn.Conv2d(C, 1, 3, padding=1)

        def _blk():
            return LocalResidualBlock(
                C, self.conv_groups,
                use_hu_gate=self.use_hu_gate,
                use_freq_boost=self.use_freq_boost,
                use_dilation=self.use_dilation,
            )

        if use_unet_decode:
            # ── U-Net encoder-decoder ──────────────────────────────────────
            self.enc_n  = max(1, self.n_blocks // 4)       # 2 for n=10
            self.dec_n  = max(1, self.n_blocks // 4)       # 2 for n=10
            self.final_n = max(0, self.n_blocks - 3 * self.enc_n - self.dec_n)  # 2

            # Encoder
            self.branch_full = nn.ModuleList([_blk() for _ in range(self.enc_n)])
            self.branch_half = nn.ModuleList([_blk() for _ in range(self.enc_n)])
            self.branch_qtr  = nn.ModuleList([_blk() for _ in range(self.enc_n)])

            # Decoder
            self.dec_half_merge  = SkipMerge(C)   # e_half + up(e_qtr)  -> C
            self.dec_half_blocks = nn.ModuleList([_blk() for _ in range(self.dec_n)])
            self.dec_full_merge  = SkipMerge(C)   # e_full + up(d_half) -> C

            # Final sequential blocks
            self.final_seq = nn.ModuleList([_blk() for _ in range(self.final_n)])

            # Unused (keeps state-dict clean)
            self.blocks    = nn.ModuleList([])
            self.mr_fusion = nn.Identity()   # placeholder; not used in unet mode

        elif use_multi_res:
            # ── Flat parallel multi-res (no decoder) ──────────────────────────
            self.branch_n = max(1, self.n_blocks // 3)               # 3
            self.final_n  = max(0, self.n_blocks - 3 * self.branch_n)  # 1

            self.branch_full = nn.ModuleList([_blk() for _ in range(self.branch_n)])
            self.branch_half = nn.ModuleList([_blk() for _ in range(self.branch_n)])
            self.branch_qtr  = nn.ModuleList([_blk() for _ in range(self.branch_n)])
            self.mr_fusion   = MultiResolutionFusion(C)
            self.final_seq   = nn.ModuleList([_blk() for _ in range(self.final_n)])
            self.blocks      = nn.ModuleList([])

        else:
            # ── Sequential (original) ─────────────────────────────────────────
            self.blocks = nn.ModuleList([_blk() for _ in range(self.n_blocks)])

        if use_mu_mod:
            self.mu_mod = MuAwareModulation(C)

        if verbose:
            extras = []
            if self.use_hu_gate:     extras.append("hu-gate")
            if self.use_unet_decode:
                extras.append(
                    f"unet-decode(enc={self.enc_n},dec={self.dec_n},final={self.final_n})"
                )
            elif self.use_multi_res:
                extras.append(
                    f"multi-res({self.branch_n}+{self.branch_n}+{self.branch_n}"
                    f"|{self.final_n})"
                )
            if self.use_mu_mod:      extras.append("mu-mod")
            if self.use_dilation:    extras.append("dilation-2")
            if self.use_freq_boost:  extras.append("freq-boost")
            tag = " | " + "+".join(extras) if extras else ""
            print(
                f"Initializing LocalResidualNet | 2D | "
                f"channels={C} | blocks={self.n_blocks} | "
                f"groups={self.conv_groups} | noise-subtraction{tag}"
            )

    def receptive_field(self) -> int:
        return 1 + 8 + self.n_blocks * 4 + 2

    def model_config(self) -> dict:
        return {
            "channels":        self.channels,
            "blocks":          self.n_blocks,
            "groups":          self.conv_groups,
            "use_hu_gate":     self.use_hu_gate,
            "use_freq_boost":  self.use_freq_boost,
            "use_dilation":    self.use_dilation,
            "use_mu_mod":      self.use_mu_mod,
            "mu_split":        self.mu_split,
            "use_multi_res":   self.use_multi_res,
            "use_unet_decode": self.use_unet_decode,
            "output_mode":     "noise_subtraction",
        }

    def _run_blocks(self, z: torch.Tensor, blocks) -> torch.Tensor:
        for blk in blocks:
            z = blk(z)
        return z

    def predict_noise(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")

        z0 = self.in_conv(x)

        if self.use_unet_decode:
            # ── Encoder ────────────────────────────────────────────────────────────
            e_full = self._run_blocks(z0, self.branch_full)
            e_half = self._run_blocks(
                F.avg_pool2d(z0, kernel_size=2, stride=2), self.branch_half
            )
            e_qtr  = self._run_blocks(
                F.avg_pool2d(z0, kernel_size=4, stride=4), self.branch_qtr
            )

            # ── Decoder level 1: qtr → half (skip connection from e_half) ───
            d_half = self._run_blocks(
                self.dec_half_merge(e_half, e_qtr),
                self.dec_half_blocks,
            )

            # ── Decoder level 2: half → full (skip connection from e_full) ───
            z = self.dec_full_merge(e_full, d_half)

            # mu-mod at decoder output (semantic midpoint of the network)
            if self.use_mu_mod:
                z = self.mu_mod(z, x)

            # Final full-resolution blocks
            z = self._run_blocks(z, self.final_seq)

        elif self.use_multi_res:
            # ── Flat parallel multi-res ─────────────────────────────────────
            z_full = self._run_blocks(z0, self.branch_full)
            z_half = self._run_blocks(
                F.avg_pool2d(z0, kernel_size=2, stride=2), self.branch_half
            )
            z_qtr  = self._run_blocks(
                F.avg_pool2d(z0, kernel_size=4, stride=4), self.branch_qtr
            )
            z = self.mr_fusion(z_full, z_half, z_qtr)

            if self.use_mu_mod:
                z = self.mu_mod(z, x)

            z = self._run_blocks(z, self.final_seq)

        else:
            # ── Sequential ────────────────────────────────────────────────────────
            z = z0
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

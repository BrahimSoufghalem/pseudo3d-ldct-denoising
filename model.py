"""
LDCT Project - MS-NAFMambaNet Architecture
==============================================
Multi-Scale Non-Linear Activation-Free Mamba Network for LDCT denoising.

Components:
  1. Activation-free restoration encoder/decoder (NAF blocks, Megvii NAFNet).
  2. Anatomy-guided attention skip gates driven by bottleneck/decoder context.
  3. 2D cross-scan selective state-space (SS2D) Mamba bottleneck.
  4. Two orthogonal ablation axes:
       INPUT_MODE : "2d" | "2.5d"
       MAMBA_MODE : "basic" | "residual" | "multiscale" | "full"

The network predicts a RESIDUAL. The centre low-dose slice is added back
outside the model (see train.py / evaluate.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import config as cfg
from naf_mamba_blocks import (
    AnatomyAttentionGate2D,
    MultiScaleSpatialFusion,
    ResidualMambaBottleneck,
    SS2DMambaBottleneck,
    build_stem,
    make_naf_stage,
    warn_if_slow_scan,
)


def _up_block(c_in):
    """PixelShuffle upsampler halving the channel count and doubling the size."""
    c_out = c_in // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out * 4, kernel_size=1, bias=False),
        nn.PixelShuffle(2),
    )


class MSNAFMambaNet(nn.Module):
    """Multi-scale NAF-Mamba network with anatomy attention skip gates."""

    def __init__(
        self,
        in_channels=None,
        out_channels=cfg.OUT_CHANNELS,
        mamba_mode=None,
        input_mode=None,
        width=cfg.MODEL_WIDTH,
        enc_blocks=cfg.ENC_BLOCKS,
        dec_blocks=cfg.DEC_BLOCKS,
        d_state=cfg.D_STATE,
        n_directions=cfg.N_SCAN_DIRECTIONS,
        scan_backend=cfg.SCAN_BACKEND,
        scan_chunk_size=cfg.SCAN_CHUNK_SIZE,
        use_checkpoint=cfg.USE_GRAD_CHECKPOINT,
        size_divisor=cfg.SIZE_DIVISOR,
        verbose=True,
    ):
        super().__init__()
        self.input_mode = cfg.normalize_input_mode(input_mode)
        self.mamba_mode = cfg.normalize_mamba_mode(mamba_mode)
        self.in_channels = in_channels or cfg.in_channels_for(self.input_mode)
        self.centre_index = self.in_channels // 2
        self.use_checkpoint = use_checkpoint
        self.size_divisor = size_divisor

        w = width
        c1, c2, c3, c4, c5 = w, 2 * w, 4 * w, 8 * w, 16 * w

        scan_kwargs = dict(
            d_state=d_state,
            n_directions=n_directions,
            scan_backend=scan_backend,
            scan_chunk_size=scan_chunk_size,
        )

        # ---- Stem (2D vs pseudo-3D z-axis modelling) --------------------
        self.stem = build_stem(self.in_channels, c1, self.input_mode)

        # ---- Encoder ----------------------------------------------------
        self.enc1 = make_naf_stage(c1, enc_blocks[0])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2)
        self.enc2 = make_naf_stage(c2, enc_blocks[1])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2)
        self.enc3 = make_naf_stage(c3, enc_blocks[2])
        self.down3 = nn.Conv2d(c3, c4, kernel_size=2, stride=2)
        self.enc4 = make_naf_stage(c4, enc_blocks[3])
        self.down4 = nn.Conv2d(c4, c5, kernel_size=2, stride=2)

        # ---- Bottleneck --------------------------------------------------
        if self.mamba_mode in ("residual", "full"):
            self.bottleneck = ResidualMambaBottleneck(c5, **scan_kwargs)
        else:
            self.bottleneck = SS2DMambaBottleneck(c5, **scan_kwargs)

        # ---- Multi-scale fusion (1/16 Mamba <-> 1/8 NAF) -----------------
        self.use_fusion = self.mamba_mode in ("multiscale", "full")
        if self.use_fusion:
            self.fusion = MultiScaleSpatialFusion(in_c_low=c5, in_c_high=c4, out_c=c4)

        # ---- Attention gates --------------------------------------------
        # g comes from the bottleneck (ag4) or the decoder path (ag3..ag1),
        # so each gate sees genuinely deeper context than the skip it gates.
        self.ag4 = AnatomyAttentionGate2D(F_g=c5, F_l=c4, F_int=c4)
        self.ag3 = AnatomyAttentionGate2D(F_g=c4, F_l=c3, F_int=c3)
        self.ag2 = AnatomyAttentionGate2D(F_g=c3, F_l=c2, F_int=c2)
        self.ag1 = AnatomyAttentionGate2D(F_g=c2, F_l=c1, F_int=c1)

        # ---- Decoder ------------------------------------------------------
        self.up4 = _up_block(c5)
        self.dec4 = nn.Sequential(nn.Conv2d(c5, c4, kernel_size=1), make_naf_stage(c4, dec_blocks[0]))
        self.up3 = _up_block(c4)
        self.dec3 = nn.Sequential(nn.Conv2d(c4, c3, kernel_size=1), make_naf_stage(c3, dec_blocks[1]))
        self.up2 = _up_block(c3)
        self.dec2 = nn.Sequential(nn.Conv2d(c3, c2, kernel_size=1), make_naf_stage(c2, dec_blocks[2]))
        self.up1 = _up_block(c2)
        self.dec1 = nn.Sequential(nn.Conv2d(c2, c1, kernel_size=1), make_naf_stage(c1, dec_blocks[3]))

        for up in (self.up4, self.up3, self.up2, self.up1):
            self._init_icnr(up[0], scale=2)

        # ---- Output head (zero-init -> exact identity residual at step 0) --
        self.head = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        if verbose:
            warn_if_slow_scan()
            print(
                f"Initializing MS-NAFMambaNet | input={self.input_mode} "
                f"({self.in_channels}ch) | mamba={self.mamba_mode} | width={w}"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _init_icnr(conv, scale=2):
        """ICNR init for PixelShuffle (removes checkerboard artefacts).

        `nonlinearity='linear'` (gain 1) is used because the network is
        activation-free; the default leaky_relu gain of sqrt(2) would inflate
        the variance.
        """
        oc, ic, kh, kw = conv.weight.data.shape
        sub_kernel = torch.empty(oc // (scale ** 2), ic, kh, kw)
        nn.init.kaiming_normal_(sub_kernel, nonlinearity='linear')
        sub_kernel = sub_kernel.repeat_interleave(scale ** 2, dim=0)
        conv.weight.data.copy_(sub_kernel)

    def _run_bottleneck(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self.bottleneck, x, use_reentrant=False)
        return self.bottleneck(x)

    # ------------------------------------------------------------------
    def _forward_features(self, x):
        x_stem = self.stem(x)

        e1 = self.enc1(x_stem)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)
        d3 = self.down3(e3)
        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        b_feat = self._run_bottleneck(d4)

        # Stage 4: gate with the global bottleneck context
        g4 = self.ag4(g=b_feat, x=e4)
        if self.use_fusion:
            g4 = self.fusion(feat_low=b_feat, feat_high=g4)
        dec4_out = self.dec4(torch.cat([self.up4(b_feat), g4], dim=1))

        # Stages 3..1: gate with the decoder context of the previous stage
        g3 = self.ag3(g=dec4_out, x=e3)
        dec3_out = self.dec3(torch.cat([self.up3(dec4_out), g3], dim=1))

        g2 = self.ag2(g=dec3_out, x=e2)
        dec2_out = self.dec2(torch.cat([self.up2(dec3_out), g2], dim=1))

        g1 = self.ag1(g=dec2_out, x=e1)
        dec1_out = self.dec1(torch.cat([self.up1(dec2_out), g1], dim=1))

        return self.head(dec1_out)

    def forward(self, x):
        """Predict the residual to add to the centre low-dose slice."""
        h, w = x.shape[-2], x.shape[-1]
        div = self.size_divisor
        pad_h = (-h) % div
        pad_w = (-w) % div
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        out = self._forward_features(x)

        if pad_h or pad_w:
            out = out[..., :h, :w]
        return out


# ══════════════════════════════════════════════════════════════
def build_model(device, mamba_mode=None, input_mode=None, use_checkpoint=None,
                data_parallel=True, verbose=True):
    """
    Factory for MS-NAFMambaNet.

    Note: nn.DataParallel is legacy and only used as a convenience when several
    GPUs are visible. Prefer torchrun + DistributedDataParallel for real
    multi-GPU training.
    """
    model = MSNAFMambaNet(
        mamba_mode=mamba_mode,
        input_mode=input_mode,
        use_checkpoint=cfg.USE_GRAD_CHECKPOINT if use_checkpoint is None else use_checkpoint,
        verbose=verbose,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model parameters: {total_params:,}")

    if data_parallel and torch.cuda.device_count() > 1:
        if verbose:
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    return model

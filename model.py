"""
LDCT Project - MS-NAFMambaNet Architecture
==============================================
Multi-Scale Non-Linear Activation-Free Mamba Network for LDCT denoising.

Components:
  1. Activation-free restoration encoder/decoder (NAF blocks, Megvii NAFNet).
  2. Anatomy-guided attention skip gates driven by bottleneck/decoder context.
  3. 2D cross-scan selective state-space (SS2D) Mamba bottleneck.
  4. Three orthogonal ablation axes:
       INPUT_MODE : "2d" | "2.5d"
       MAMBA_MODE : "basic" | "residual" | "multiscale" | "full"
       NUM_STAGES : depth of the resolution pyramid (downsampling = 2**NUM_STAGES)

The network predicts a RESIDUAL. The centre low-dose slice is added back
outside the model (see train.py / evaluate.py).

A note on NUM_STAGES
--------------------
The encoder/decoder used to be written out four times by hand, which made the
downsampling depth impossible to vary. It is now built in a loop. At the
default NUM_STAGES=4 the result is IDENTICAL to the hand-written version -
same channel widths, same construction order, and the submodules are registered
under the same attribute names (enc1..enc4, down1..down4, ag1..ag4, up1..up4,
dec1..dec4) - so every existing checkpoint still loads.
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
        num_stages=cfg.NUM_STAGES,
        enc_blocks=cfg.ENC_BLOCKS,
        dec_blocks=cfg.DEC_BLOCKS,
        d_state=cfg.D_STATE,
        n_directions=cfg.N_SCAN_DIRECTIONS,
        scan_backend=cfg.SCAN_BACKEND,
        scan_chunk_size=cfg.SCAN_CHUNK_SIZE,
        use_checkpoint=cfg.USE_GRAD_CHECKPOINT,
        size_divisor=None,
        verbose=True,
    ):
        super().__init__()
        self.input_mode = cfg.normalize_input_mode(input_mode)
        self.mamba_mode = cfg.normalize_mamba_mode(mamba_mode)
        self.in_channels = in_channels or cfg.in_channels_for(self.input_mode)
        self.centre_index = self.in_channels // 2
        self.use_checkpoint = use_checkpoint

        n = int(num_stages)
        if n < 1:
            raise ValueError(f"num_stages must be >= 1, got {n}")
        if len(enc_blocks) != n or len(dec_blocks) != n:
            raise ValueError(
                f"enc_blocks/dec_blocks must have exactly num_stages={n} entries, "
                f"got {len(enc_blocks)} and {len(dec_blocks)}"
            )
        self.num_stages = n

        # The network downsamples 2**n times, so inputs must be padded up to a
        # multiple of that. Derived unless explicitly overridden.
        self.size_divisor = size_divisor if size_divisor is not None else 2 ** n

        w = width
        # chans[i] is the channel count at pyramid level i.
        # For n=4 this is exactly the old c1..c5 = w, 2w, 4w, 8w, 16w.
        chans = [w * (2 ** i) for i in range(n + 1)]
        self.chans = chans

        scan_kwargs = dict(
            d_state=d_state,
            n_directions=n_directions,
            scan_backend=scan_backend,
            scan_chunk_size=scan_chunk_size,
        )

        # ---- Stem (2D vs pseudo-3D z-axis modelling) --------------------
        self.stem = build_stem(self.in_channels, chans[0], self.input_mode)

        # ---- Encoder ----------------------------------------------------
        # Registered as enc1..encN / down1..downN to keep old state_dict keys.
        for s in range(1, n + 1):
            setattr(self, f"enc{s}", make_naf_stage(chans[s - 1], enc_blocks[s - 1]))
            setattr(
                self,
                f"down{s}",
                nn.Conv2d(chans[s - 1], chans[s], kernel_size=2, stride=2),
            )

        # ---- Bottleneck --------------------------------------------------
        if self.mamba_mode in ("residual", "full"):
            self.bottleneck = ResidualMambaBottleneck(chans[n], **scan_kwargs)
        else:
            self.bottleneck = SS2DMambaBottleneck(chans[n], **scan_kwargs)

        # ---- Multi-scale fusion (deepest Mamba level <-> level below) ----
        self.use_fusion = self.mamba_mode in ("multiscale", "full")
        if self.use_fusion:
            self.fusion = MultiScaleSpatialFusion(
                in_c_low=chans[n], in_c_high=chans[n - 1], out_c=chans[n - 1]
            )

        # ---- Attention gates --------------------------------------------
        # g for the deepest gate comes from the bottleneck; for the others it
        # comes from the decoder output of the stage below, so every gate sees
        # genuinely deeper context than the skip it gates. In both cases the
        # context has chans[s] channels.
        for s in range(1, n + 1):
            setattr(
                self,
                f"ag{s}",
                AnatomyAttentionGate2D(
                    F_g=chans[s], F_l=chans[s - 1], F_int=chans[s - 1]
                ),
            )

        # ---- Decoder ------------------------------------------------------
        # dec_blocks is ordered DEEPEST FIRST: decN <- dec_blocks[0].
        for s in range(n, 0, -1):
            setattr(self, f"up{s}", _up_block(chans[s]))
            setattr(
                self,
                f"dec{s}",
                nn.Sequential(
                    nn.Conv2d(chans[s], chans[s - 1], kernel_size=1),
                    make_naf_stage(chans[s - 1], dec_blocks[n - s]),
                ),
            )
            self._init_icnr(getattr(self, f"up{s}")[0], scale=2)

        # ---- Output head (zero-init -> exact identity residual at step 0) --
        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        if verbose:
            warn_if_slow_scan()
            print(
                f"Initializing MS-NAFMambaNet | input={self.input_mode} "
                f"({self.in_channels}ch) | mamba={self.mamba_mode} | width={w} | "
                f"stages={n} (downsample {2 ** n}x, bottleneck {chans[n]}ch)"
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
        n = self.num_stages

        x = self.stem(x)

        skips = []
        for s in range(1, n + 1):
            e = getattr(self, f"enc{s}")(x)
            skips.append(e)
            x = getattr(self, f"down{s}")(e)

        b_feat = self._run_bottleneck(x)

        # Deepest stage is gated by the global bottleneck context; each stage
        # below is gated by the decoder output of the stage above it.
        ctx = b_feat
        for s in range(n, 0, -1):
            g = getattr(self, f"ag{s}")(g=ctx, x=skips[s - 1])
            if s == n and self.use_fusion:
                g = self.fusion(feat_low=b_feat, feat_high=g)
            up = getattr(self, f"up{s}")(ctx)
            ctx = getattr(self, f"dec{s}")(torch.cat([up, g], dim=1))

        return self.head(ctx)

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


# ═══════════════════════════════════════════════════════════
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

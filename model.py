"""
LDCT Project — MS-NAFMambaNet Architecture
==============================================
Multi-Scale Non-Linear Activation-Free Mamba Network for LDCT Denoising.

Scientific Contributions:
  1. Activation-Free Restoration Encoder/Decoder (NAF Blocks from Megvii NAFNet).
  2. Anatomy-Guided Attention Skip Gates (AG-Skip) for quantum noise suppression.
  3. 2D Vision State-Space Mamba Bottleneck (NVIDIA MambaVision derived) for global streak artifact removal.
  4. Modular 4-Stage Ablation Framework (controlled via MAMBA_MODE: "basic", "residual", "multiscale", "full").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IN_CHANNELS, OUT_CHANNELS, MAMBA_MODE, USE_ANATOMY_CONDITIONING, ANATOMY_EMBED_DIM
from naf_mamba_blocks import (
    LayerNorm2d, NAFBlock, AnatomyAttentionGate2D, AnatomyCondition,
    SS2DMambaBottleneck, ResidualMambaBottleneck,
    MultiScaleSpatialFusion
)


class MSNAFMambaNet(nn.Module):
    """
    Multi-Scale NAF-Mamba Network with Anatomy Attention Skip Gates.
    Supports 4-stage ablation studies via mamba_mode.
    """
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, mamba_mode=MAMBA_MODE,
                 use_anatomy=USE_ANATOMY_CONDITIONING, anatomy_embed_dim=ANATOMY_EMBED_DIM):
        super().__init__()
        self.mamba_mode = mamba_mode.lower()
        self.use_anatomy = use_anatomy
        print(f"🏗️  Initializing MS-NAFMambaNet (Ablation Mode: '{self.mamba_mode.upper()}')")

        # Stem Conv
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=True)

        # Encoder Stages (NAF Blocks)
        self.enc1 = NAFBlock(32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        self.enc2 = NAFBlock(64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.enc3 = NAFBlock(128)
        self.down3 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        self.enc4 = NAFBlock(256)
        self.down4 = nn.Conv2d(256, 512, kernel_size=2, stride=2)

        # Anatomy Attention Gates on Skips
        ag_embed = anatomy_embed_dim if self.use_anatomy else 0
        self.ag1 = AnatomyAttentionGate2D(F_g=64, F_l=32, F_int=32, embed_dim=ag_embed)
        self.ag2 = AnatomyAttentionGate2D(F_g=128, F_l=64, F_int=64, embed_dim=ag_embed)
        self.ag3 = AnatomyAttentionGate2D(F_g=256, F_l=128, F_int=128, embed_dim=ag_embed)
        self.ag4 = AnatomyAttentionGate2D(F_g=512, F_l=256, F_int=256, embed_dim=ag_embed)

        # Bottleneck (SS2D Mamba 2D Selective State-Space Module)
        if self.mamba_mode in ["residual", "full"]:
            self.bottleneck = ResidualMambaBottleneck(512)
        else:
            self.bottleneck = SS2DMambaBottleneck(512)

        # Multi-Scale Spatial Fusion (1/16 Mamba <-> 1/8 NAF)
        if self.mamba_mode in ["multiscale", "full"]:
            self.fusion = MultiScaleSpatialFusion(in_c_low=512, in_c_high=256, out_c=256)

        # Decoder Stages (Upsampling + Concatenation + NAF Blocks)
        self.up4 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), NAFBlock(256))

        self.up3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), NAFBlock(128))

        self.up2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), NAFBlock(64))

        self.up1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), NAFBlock(32))

        # Apply ICNR initialization to eliminate checkerboard artifacts in CT images
        for up in [self.up4, self.up3, self.up2, self.up1]:
            self._init_icnr(up[0], scale=2)

        # Output Head (Zero-initialized for pure identity residual baseline at step 0)
        self.head = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Anatomy Conditioning Modules (FiLM modulation at strategic locations)
        if self.use_anatomy:
            self.anatomy_embedding = nn.Embedding(2, anatomy_embed_dim)
            self.cond_enc2 = AnatomyCondition(anatomy_embed_dim, 64)
            self.cond_enc3 = AnatomyCondition(anatomy_embed_dim, 128)
            self.cond_bottleneck = AnatomyCondition(anatomy_embed_dim, 512)
            self.cond_dec3 = AnatomyCondition(anatomy_embed_dim, 128)
            self.cond_dec2 = AnatomyCondition(anatomy_embed_dim, 64)

    @staticmethod
    def _init_icnr(conv, scale=2):
        oc, ic, kh, kw = conv.weight.data.shape
        sub_kernel = torch.randn(oc // (scale ** 2), ic, kh, kw)
        nn.init.kaiming_normal_(sub_kernel)
        sub_kernel = sub_kernel.repeat_interleave(scale ** 2, dim=0)
        conv.weight.data.copy_(sub_kernel)

    def forward(self, x, anatomy_id=None):
        # Compute anatomy embedding (if enabled and anatomy_id provided)
        anat_emb = None
        if self.use_anatomy and anatomy_id is not None:
            anat_emb = self.anatomy_embedding(anatomy_id)  # [B, embed_dim]

        # Stem
        x_stem = self.stem(x)

        # Encoder Forward
        e1 = self.enc1(x_stem)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        if anat_emb is not None:
            e2 = self.cond_enc2(e2, anat_emb)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        if anat_emb is not None:
            e3 = self.cond_enc3(e3, anat_emb)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        # Mamba Bottleneck
        b_feat = self.bottleneck(d4)
        if anat_emb is not None:
            b_feat = self.cond_bottleneck(b_feat, anat_emb)

        # Skip Attention Gating (anatomy-aware)
        g4 = self.ag4(g=d4, x=e4, anatomy_emb=anat_emb)
        g3 = self.ag3(g=d3, x=e3, anatomy_emb=anat_emb)
        g2 = self.ag2(g=d2, x=e2, anatomy_emb=anat_emb)
        g1 = self.ag1(g=d1, x=e1, anatomy_emb=anat_emb)

        # Multi-Scale Fusion (Stage 3 & 4)
        if self.mamba_mode in ["multiscale", "full"]:
            g4 = self.fusion(feat_low=b_feat, feat_high=g4)

        # Decoder Forward
        u4 = self.up4(b_feat)
        d4_cat = torch.cat([u4, g4], dim=1)
        dec4_out = self.dec4(d4_cat)

        u3 = self.up3(dec4_out)
        d3_cat = torch.cat([u3, g3], dim=1)
        dec3_out = self.dec3(d3_cat)
        if anat_emb is not None:
            dec3_out = self.cond_dec3(dec3_out, anat_emb)

        u2 = self.up2(dec3_out)
        d2_cat = torch.cat([u2, g2], dim=1)
        dec2_out = self.dec2(d2_cat)
        if anat_emb is not None:
            dec2_out = self.cond_dec2(dec2_out, anat_emb)

        u1 = self.up1(dec2_out)
        d1_cat = torch.cat([u1, g1], dim=1)
        dec1_out = self.dec1(d1_cat)

        # Final Residual Prediction
        out = self.head(dec1_out)
        return out


def build_model(device):
    """
    Factory function for building MS-NAFMambaNet model.
    Wraps with DataParallel if multiple GPUs are available.
    """
    model = MSNAFMambaNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, mamba_mode=MAMBA_MODE).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦  Model parameters ({MAMBA_MODE.upper()} mode): {total_params:,}")

    if USE_ANATOMY_CONDITIONING:
        cond_params = sum(
            p.numel() for n, p in model.named_parameters()
            if "cond_" in n or "anatomy" in n or "W_a" in n
        )
        print(f"🧬 Anatomy Conditioning: ENABLED (embed_dim={ANATOMY_EMBED_DIM})")
        print(f"🧬 Conditioning parameters: {cond_params:,} ({cond_params / total_params * 100:.2f}%)")
    else:
        print(f"🧬 Anatomy Conditioning: DISABLED")

    return model

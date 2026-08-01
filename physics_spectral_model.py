"""Metadata-free physics-spectral full-resolution CT denoiser.

This model is trained from scratch. It does not use Mamba, NAFNet, pretrained
weights, patient metadata, or another model's output.

The public forward contract matches the rest of this repository: ``forward``
returns a correction residual and callers form ``denoised = LDCT + residual``.
Internally the network predicts noise, therefore residual = -predicted_noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedGaussianBlur(nn.Module):
    """Depthwise Gaussian blur with a fixed, registered kernel."""

    def __init__(self, sigma: float, kernel_size: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d.view(1, 1, kernel_size, kernel_size))
        self.pad = radius

    def forward(self, x):
        c = x.shape[1]
        weight = self.kernel.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        return F.conv2d(x, weight, groups=c)


class FixedSpectralDecomposition(nn.Module):
    """Undecimated low/mid/high decomposition: x = low + mid + high."""

    def __init__(self, sigma_fine=1.0, sigma_coarse=2.5,
                 kernel_fine=7, kernel_coarse=15):
        super().__init__()
        if sigma_coarse <= sigma_fine:
            raise ValueError("sigma_coarse must exceed sigma_fine")
        self.fine = FixedGaussianBlur(sigma_fine, kernel_fine)
        self.coarse = FixedGaussianBlur(sigma_coarse, kernel_coarse)
        self.settings = dict(
            sigma_fine=float(sigma_fine), sigma_coarse=float(sigma_coarse),
            kernel_fine=int(kernel_fine), kernel_coarse=int(kernel_coarse),
        )

    def forward(self, x):
        smooth_fine = self.fine(x)
        low = self.coarse(x)
        mid = smooth_fine - low
        high = x - smooth_fine
        return low, mid, high


class BandEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class FullResolutionBlock(nn.Module):
    """Dilated context + local mixing, without any intensity normalization."""

    def __init__(self, channels, dilation, residual_scale=0.1):
        super().__init__()
        self.context = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation,
        )
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_scale))
        )

    def forward(self, x):
        z = F.gelu(self.context(x))
        z = self.local(z)
        return x + self.scale * z


class ResidualGroup(nn.Module):
    """A sequence of full-resolution blocks with no second attenuation layer.

    The first implementation wrapped blocks that already used a 0.1 residual
    scale in another 0.1-scaled group residual. That reduced the effective deep
    path to roughly one percent and produced a shallow-filter failure signature
    (chest/abdomen VIF 0.1551/0.4029). One scale per block is sufficient.
    """

    def __init__(self, channels, dilations, residual_scale=0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            FullResolutionBlock(channels, d, residual_scale)
            for d in dilations
        ])

    def forward(self, x):
        return self.blocks(x)


class PhysicsSpectralNet(nn.Module):
    """Full-resolution, metadata-free, physics-spectral residual denoiser."""

    def __init__(
        self,
        channels=64,
        band_channels=16,
        groups=4,
        dilations=(1, 2, 3, 4),
        residual_scale=0.1,
        spectral=True,
        sigma_fine=1.0,
        sigma_coarse=2.5,
        kernel_fine=7,
        kernel_coarse=15,
        verbose=True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.band_channels = int(band_channels)
        self.n_groups = int(groups)
        self.dilations = tuple(int(d) for d in dilations)
        self.spectral_enabled = bool(spectral)

        if self.channels < 8 or self.band_channels < 1 or self.n_groups < 1:
            raise ValueError("channels>=8, band_channels>=1 and groups>=1 are required")
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("dilations must be positive integers")

        self.stem = nn.Conv2d(1, self.channels, 5, padding=2)

        if self.spectral_enabled:
            self.decomposition = FixedSpectralDecomposition(
                sigma_fine, sigma_coarse, kernel_fine, kernel_coarse,
            )
            self.low_encoder = BandEncoder(self.band_channels)
            self.mid_encoder = BandEncoder(self.band_channels)
            self.high_encoder = BandEncoder(self.band_channels)
            self.spectral_projection = nn.Conv2d(
                self.band_channels * 3, self.channels, 1,
            )
            self.spectral_scales = nn.Parameter(
                torch.full((self.n_groups,), 0.1)
            )

        self.groups = nn.ModuleList([
            ResidualGroup(self.channels, self.dilations, residual_scale)
            for _ in range(self.n_groups)
        ])

        # Predict noise; zero init makes the complete denoiser identity at step 0.
        self.noise_head = nn.Conv2d(self.channels, 1, 5, padding=2)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

        if verbose:
            print(
                "Initializing PhysicsSpectralNet | input=2d (1ch) | "
                f"channels={self.channels} | groups={self.n_groups} | "
                f"blocks={self.n_groups * len(self.dilations)} | "
                f"spectral={'on' if self.spectral_enabled else 'off'} | "
                f"RF~{self.receptive_field()} | group-scale=removed"
            )

    def receptive_field(self):
        # stem 5x5 + per block (one dilated 3x3 + one local 3x3) + head 5x5
        return 1 + 4 + self.n_groups * sum(2 * d + 2 for d in self.dilations) + 4

    def model_config(self):
        cfg = dict(
            channels=self.channels,
            band_channels=self.band_channels,
            groups=self.n_groups,
            dilations=list(self.dilations),
            spectral=self.spectral_enabled,
        )
        if self.spectral_enabled:
            cfg.update(self.decomposition.settings)
        return cfg

    def scale_diagnostics(self):
        block_values = torch.cat([
            block.scale.detach().float().reshape(-1)
            for group in self.groups
            for block in group.blocks
        ])
        result = {
            "block_scale_mean": float(block_values.mean()),
            "block_scale_abs_mean": float(block_values.abs().mean()),
            "head_weight_norm": float(self.noise_head.weight.detach().float().norm()),
        }
        if self.spectral_enabled:
            spectral = self.spectral_scales.detach().float()
            result["spectral_scale_mean"] = float(spectral.mean())
            result["spectral_scale_abs_mean"] = float(spectral.abs().mean())
        return result

    def spectral_features(self, x):
        low, mid, high = self.decomposition(x)
        encoded = torch.cat([
            self.low_encoder(low),
            self.mid_encoder(mid),
            self.high_encoder(high),
        ], dim=1)
        return self.spectral_projection(encoded)

    def predict_noise(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"PhysicsSpectralNet is a strict 2D model and expects [B,1,H,W], got {tuple(x.shape)}"
            )
        features = self.stem(x)
        spectral = self.spectral_features(x) if self.spectral_enabled else None
        for i, group in enumerate(self.groups):
            if spectral is not None:
                features = features + self.spectral_scales[i] * spectral
            features = group(features)
        return self.noise_head(features)

    def forward(self, x):
        # Repository contract: caller computes x + correction.
        return -self.predict_noise(x)


def build_physics_model(device, **kwargs):
    model = PhysicsSpectralNet(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    return model

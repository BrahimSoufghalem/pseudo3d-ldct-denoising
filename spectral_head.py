"""Physics-informed spectral residual head (Option A: architectural physics).

Physics background
------------------
In CT, noise texture is set by the FBP reconstruction kernel: a radially
symmetric filter in the 2-D frequency plane (ramp + apodization). Because
backprojection averages over all view angles, the image-domain noise power
spectrum (NPS) is approximately isotropic: a function of the radial spatial
frequency |f| only.

Mechanism
---------
``SpectralResidualHead`` applies a LEARNABLE RADIAL GAIN ``G(|f|)`` to the
noise a base denoiser wants to remove::

    y_hat  = base(x)                       # any denoiser (e.g. RED-CNN)
    n_hat  = x - y_hat                     # noise the base wants to remove
    n'     = IFFT2( G(|f|) * FFT2(n_hat) ) # physics-shaped noise
    output = x - n'

Structural constraints (what a plain CNN cannot bypass):

* ``G`` depends only on the radial frequency ``|f|`` -- the same rotational
  symmetry that FBP physics imposes on the NPS. The head cannot build a
  directional or spatially varying filter.
* ``G`` is parameterized by ``n_bins`` (~32) interpolated log-gain knots and
  initialized to ``G == 1``, so the wrapped model starts EXACTLY equal to
  the base model (safe start; cannot begin worse than the trunk).
* Frequencies are normalized by the Nyquist frequency, so the same curve
  serves 128x128 training patches and 512x512 test slices (no train/test
  extent shift).

Interpretability: after training, plot the learned curve with
``plot_spectral_gain.py``. ``G < 1`` at a band means the head RETURNS part
of the energy the trunk wanted to remove there (protects detail /
noise-texture); ``G > 1`` means it removes more.

Limitations: the radial-symmetry constraint is exact for FBP
reconstructions (our benchmark data); for iterative/deep reconstructions
the NPS is less isotropic and a low-parameter 2-D gain would be the natural
extension.
"""

import torch
import torch.nn as nn


class SpectralResidualHead(nn.Module):
    """Learnable radial gain applied in the Fourier domain.

    Parameters
    ----------
    n_bins : int
        Number of radial gain knots. The knots span radial frequencies
        ``[0, sqrt(2)]`` in units of the (axis) Nyquist frequency; values
        between knots are linearly interpolated.
    """

    _RMAX = 2.0 ** 0.5  # corner of the 2-D frequency plane, Nyquist units

    def __init__(self, n_bins: int = 32):
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = int(n_bins)
        # Log-parameterization: G = exp(log_gain) > 0; init log_gain = 0
        # -> G == 1 -> the head is the identity at initialization.
        self.log_gain = nn.Parameter(torch.zeros(self.n_bins))
        # Cache of interpolation grids keyed by (H, W, device); tensors are
        # not parameters and are rebuilt lazily per resolution/device.
        self._grid_cache = {}

    def gain_curve(self) -> torch.Tensor:
        """Learned radial gain G at the knot positions, shape (n_bins,)."""
        return torch.exp(self.log_gain)

    def _interp_grid(self, h: int, w: int, device):
        key = (h, w, str(device))
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(h, device=device)      # cycles/pixel, (H,)
        fx = torch.fft.rfftfreq(w, device=device)     # cycles/pixel, (W//2+1,)
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        # r = 0.5 cycles/pixel is the Nyquist along an axis. Normalizing by
        # it makes the curve resolution-independent; the 2-D plane extends
        # to sqrt(2) Nyquist units at the corners.
        pos = (r / 0.5) / self._RMAX * (self.n_bins - 1)
        idx0 = pos.floor().long().clamp(0, self.n_bins - 1)
        idx1 = (idx0 + 1).clamp(0, self.n_bins - 1)
        frac = (pos - idx0.to(pos.dtype)).clamp(0.0, 1.0)
        cached = (idx0, idx1, frac)
        self._grid_cache[key] = cached
        return cached

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Shape the spectrum of ``noise`` (B, C, H, W) with G(|f|)."""
        h, w = noise.shape[-2], noise.shape[-1]
        idx0, idx1, frac = self._interp_grid(h, w, noise.device)
        g = self.gain_curve()
        gain2d = (1.0 - frac) * g[idx0] + frac * g[idx1]  # (H, W//2+1)
        spec = torch.fft.rfft2(noise, norm="ortho")
        return torch.fft.irfft2(spec * gain2d, s=(h, w), norm="ortho")


class SpectralResidualModel(nn.Module):
    """Wrap any image-domain denoiser with the spectral residual head.

    ``forward``: ``x - head(x - base(x))``. With the head at initialization
    (G == 1) this is exactly ``base(x)``.
    """

    def __init__(self, base: nn.Module, n_bins: int = 32):
        super().__init__()
        self.base = base
        self.head = SpectralResidualHead(n_bins=n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.base(x)
        noise = x - y_hat
        return x - self.head(noise)


def wrap_with_spectral_head(base: nn.Module, n_bins: int = 32) -> SpectralResidualModel:
    """Convenience constructor used by train/eval scripts."""
    return SpectralResidualModel(base, n_bins=n_bins)

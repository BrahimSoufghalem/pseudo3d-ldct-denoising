"""
LDCT Project - Loss Functions
================================
Charbonnier (robust L1) + (multi-scale) SSIM + Sobel edge, optionally evaluated
a second time inside the clinical diagnostic window.

Naming note: the weight is historically called `lambda_l1`, but the term it
multiplies is a CHARBONNIER loss, not a plain L1. The returned logging dict
reports "Charbonnier" as the primary key and keeps "L1" only as a backward
compatible alias so existing TensorBoard/tqdm code keeps working.

Why multi-scale and why windowed - see the long comment above USE_MS_SSIM in
config.py. Short version: the model reaches 89-112% of the required PSNR/SSIM
gain but only 52-57% of the required VIF gain, and VIF is the only one of the
three that integrates information across several scales; separately, all three
metrics are measured inside a clinical window while the loss was measured over
the full [0, 1] span.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss

from config import (
    LAMBDA_L1, LAMBDA_SSIM, LAMBDA_EDGE,
    USE_MS_SSIM, MS_SSIM_BETAS, MS_SSIM_KERNEL_SIZE,
    WINDOW_LOSS_MODE, LAMBDA_WINDOW,
    CLINICAL_WINDOWS, HU_OFFSET, A_MIN, A_MAX,
)

try:
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure as _ms_ssim,
    )
    _HAS_MS_SSIM = True
except Exception:                                    # torchmetrics missing/old
    _ms_ssim = None
    _HAS_MS_SSIM = False


# ═══════════════════════════════════════════
# CHARBONNIER LOSS
# ═══════════════════════════════════════════
class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2): smooth, robust L1 variant."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ═══════════════════════════════════════════
# SOBEL EDGE LOSS
# ═══════════════════════════════════════════
class SobelEdgeLoss(nn.Module):
    """L1 loss between Sobel gradient magnitudes; preserves sharp boundaries."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def get_edges(self, x):
        c = x.shape[1]
        sobel_x = self.sobel_x.repeat(c, 1, 1, 1).to(x.dtype)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1).to(x.dtype)
        # Reflect padding avoids fake edges along the image border.
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(x, sobel_x, groups=c)
        gy = F.conv2d(x, sobel_y, groups=c)
        return torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

    def forward(self, pred, target):
        return F.l1_loss(self.get_edges(pred), self.get_edges(target))


# ═══════════════════════════════════════════
# MULTI-SCALE SSIM
# ═══════════════════════════════════════════
class MultiScaleSSIMLoss(nn.Module):
    """1 - MS-SSIM, with graceful degradation to single-scale SSIM.

    Why this exists
    ---------------
    VIF decomposes the image into sub-bands and sums the visual information
    preserved in each one. A single-scale SSIM term only constrains structure at
    one window size, so coarse-scale texture is free to be smoothed away without
    the loss noticing. This term restores the missing scales.

    Two implementation details that are not optional:

    * `normalize="relu"`. Without it the per-scale contrast/structure terms can
      go negative at coarse scales, and raising a negative number to a
      fractional beta produces NaN. torchmetrics documents this mode as the one
      to use during training.
    * The number of levels must satisfy
          min(H, W) > (kernel_size - 1) * 2 ** (levels - 1)
      or torchmetrics raises. Rather than crash on a small crop we drop levels
      until it fits, and fall back to single-scale if even two levels do not.
    """

    def __init__(self, data_range=1.0, betas=MS_SSIM_BETAS,
                 kernel_size=MS_SSIM_KERNEL_SIZE, fallback=None):
        super().__init__()
        self.data_range = float(data_range)
        self.betas = tuple(betas)
        self.kernel_size = int(kernel_size)
        self.fallback = fallback or SSIMLoss(spatial_dims=2, data_range=data_range)
        self._warned = False
        self.available = _HAS_MS_SSIM

    def _levels_for(self, height, width):
        """Largest number of pyramid levels this image size can support."""
        smallest = min(int(height), int(width))
        for levels in range(len(self.betas), 1, -1):
            if smallest > (self.kernel_size - 1) * 2 ** (levels - 1):
                return levels
        return 0

    def _warn_once(self, message):
        if not self._warned:
            self._warned = True
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    def forward(self, pred, target):
        if not self.available:
            self._warn_once(
                "MS-SSIM requested but torchmetrics is unavailable; falling back "
                "to single-scale SSIM. Install torchmetrics to enable it."
            )
            return self.fallback(pred, target)

        levels = self._levels_for(pred.shape[-2], pred.shape[-1])
        if levels < 2:
            self._warn_once(
                f"Input {tuple(pred.shape[-2:])} is too small for multi-scale SSIM "
                f"with kernel_size={self.kernel_size}; using single-scale SSIM."
            )
            return self.fallback(pred, target)

        betas = self.betas[:levels]
        try:
            value = _ms_ssim(
                pred, target,
                gaussian_kernel=True,
                sigma=1.5,
                kernel_size=self.kernel_size,
                data_range=self.data_range,
                betas=betas,
                normalize="relu",
            )
        except Exception as err:                       # never kill a run for this
            self._warn_once(f"MS-SSIM failed ({err}); using single-scale SSIM.")
            return self.fallback(pred, target)

        return 1.0 - value


# ═══════════════════════════════════════════
# CLINICAL WINDOW HELPERS
# ═══════════════════════════════════════════
def window_bounds_normalized(body_type):
    """Clinical window (center, width) expressed in the normalized [0, 1] domain.

    metrics.py windows the image in the HU+1024 domain:
        w(x) = clamp((x - (center - width/2)) / width, 0, 1)

    Training tensors live in [0, 1] after (HU - A_MIN) / (A_MAX - A_MIN), so the
    same window becomes a plain affine transform there. Doing it this way keeps
    the loss and the metric bit-for-bit consistent without denormalising inside
    the training loop.
    """
    center, width = CLINICAL_WINDOWS[body_type]
    span = A_MAX - A_MIN
    lo_hu = center - width / 2.0 - HU_OFFSET
    hi_hu = center + width / 2.0 - HU_OFFSET
    return (lo_hu - A_MIN) / span, (hi_hu - A_MIN) / span


WINDOW_BOUNDS = {bt: window_bounds_normalized(bt) for bt in CLINICAL_WINDOWS}


def canonical_body_type(value):
    """Map anything ('Chest', 'C121', 'c', ...) onto 'Chest' or 'Abdomen'."""
    return "Chest" if str(value).strip().lower().startswith("c") else "Abdomen"


def batch_window_bounds(body_types, batch_size, device, dtype=torch.float32):
    """Per-sample (lo, hi) tensors shaped [B, 1, 1, 1].

    A batch mixes chest and abdomen slices, and their windows differ by almost
    4x in width, so a single scalar window would be wrong for most of the batch.
    """
    if not isinstance(body_types, (list, tuple)):
        body_types = [body_types] * batch_size

    los, his = [], []
    for i in range(batch_size):
        raw = body_types[i] if i < len(body_types) else body_types[-1]
        lo, hi = WINDOW_BOUNDS[canonical_body_type(raw)]
        los.append(lo)
        his.append(hi)

    shape = (batch_size, 1, 1, 1)
    lo_t = torch.tensor(los, device=device, dtype=dtype).view(shape)
    hi_t = torch.tensor(his, device=device, dtype=dtype).view(shape)
    return lo_t, hi_t


def apply_window_normalized(x, lo, hi, eps=1e-6):
    """Rescale [lo, hi] to [0, 1] and clamp, mirroring metrics.apply_center_width."""
    return ((x - lo) / (hi - lo).clamp_min(eps)).clamp(0.0, 1.0)


# ═══════════════════════════════════════════
# HYBRID LOSS
# ═══════════════════════════════════════════
class MONAIHybridLoss(nn.Module):
    """
    Charbonnier + SSIM (single- or multi-scale) + Sobel edge, optionally repeated
    inside the clinical diagnostic window.

    The loss is always evaluated in FP32 (AMP-safe) and expects predictions in
    the normalized [0, 1] domain.

    Clamping policy (this matters, and an earlier revision got it wrong)
    -------------------------------------------------------------------
    * Charbonnier and the Sobel edge term run on the UNCLAMPED prediction, so a
      pixel that has drifted outside [0, 1] still receives a restoring gradient.
    * SSIM runs on a CLAMPED copy. `SSIMLoss(data_range=1.0)` is only defined
      for inputs inside [0, 1]; once the prediction blows up, the variance term
      in the denominator dominates, the loss saturates at 1.0 and its gradient
      vanishes. With lambda_ssim = 0.6 that silently removed most of the
      training signal and turned a transient divergence into a permanent
      collapse (the model froze at SSIM ~= 0.06 and never recovered).
      Clamping only the SSIM input keeps the term meaningful while Charbonnier
      still pulls the prediction back into range.

    Set `ssim_clamp=False` to restore the old (unstable) behaviour.

    Window modes
    ------------
    * "off"   : loss on the full [0, 1] range only (previous behaviour).
    * "extra" : full-range loss PLUS lambda_window * (windowed loss). Default.
                The windowed transform clamps, so out-of-window pixels get zero
                gradient from it; keeping the global term is what stops bone and
                air from drifting.
    * "only"  : windowed loss alone.

    `body_types` must be supplied for the windowed modes. Without it the window
    term is skipped and a warning is issued once - guessing a window would
    quietly apply the 1500 HU lung window to abdomen slices, which is worse than
    not applying it at all.
    """

    def __init__(
        self,
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_edge=LAMBDA_EDGE,
        spatial_dims=2,
        charbonnier_eps=1e-3,
        ssim_clamp=True,
        use_ms_ssim=USE_MS_SSIM,
        window_mode=WINDOW_LOSS_MODE,
        lambda_window=LAMBDA_WINDOW,
    ):
        super().__init__()
        self.lambda_charbonnier = lambda_l1
        self.lambda_l1 = lambda_l1          # alias, kept for compatibility
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.ssim_clamp = ssim_clamp
        self.window_mode = str(window_mode).strip().lower()
        self.lambda_window = float(lambda_window)

        self.charbonnier_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.edge_loss = SobelEdgeLoss()

        single_scale = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0)
        if use_ms_ssim and spatial_dims == 2:
            self.ssim_loss = MultiScaleSSIMLoss(data_range=1.0, fallback=single_scale)
            self.ms_ssim = True
        else:
            self.ssim_loss = single_scale
            self.ms_ssim = False

        self._warned_no_body_types = False

    # ---- description used by the training scripts' banner ----------
    def describe(self):
        ssim_name = "MS-SSIM" if self.ms_ssim else "SSIM"
        parts = [
            f"{self.lambda_charbonnier:g}*Charbonnier",
            f"{self.lambda_ssim:g}*{ssim_name}",
            f"{self.lambda_edge:g}*Edge",
        ]
        base = " + ".join(parts)
        if self.window_mode == "off":
            return f"{base}  (full range)"
        if self.window_mode == "only":
            return f"{base}  (clinical window only)"
        return f"{base}  +  {self.lambda_window:g} * [same, clinical window]"

    # ---- the three terms, computed on whatever domain is passed ----
    def _triple(self, pred, target):
        charb = self.charbonnier_loss(pred, target)
        edge = self.edge_loss(pred, target)
        if self.ssim_clamp:
            ssim = self.ssim_loss(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
        else:
            ssim = self.ssim_loss(pred, target)
        weighted = (
            self.lambda_charbonnier * charb
            + self.lambda_ssim * ssim
            + self.lambda_edge * edge
        )
        return weighted, charb, ssim, edge

    def forward(self, pred_img, target_img, body_types=None):
        pred_img = pred_img.float()
        target_img = target_img.float()

        want_window = self.window_mode in ("extra", "only")
        if want_window and body_types is None:
            if not self._warned_no_body_types:
                self._warned_no_body_types = True
                warnings.warn(
                    f"WINDOW_LOSS_MODE='{self.window_mode}' but no body_types were "
                    f"passed to the loss, so the windowed term is disabled for this "
                    f"run. Pass batch['body_type'] through to enable it.",
                    RuntimeWarning, stacklevel=2,
                )
            want_window = False

        info = {}
        total = None

        if self.window_mode != "only":
            weighted, charb, ssim, edge = self._triple(pred_img, target_img)
            total = weighted
            info["Charbonnier"] = charb.detach().item()
            info["SSIM"] = ssim.detach().item()
            info["Edge"] = edge.detach().item()

        if want_window:
            lo, hi = batch_window_bounds(
                body_types, pred_img.shape[0], pred_img.device, pred_img.dtype,
            )
            pred_w = apply_window_normalized(pred_img, lo, hi)
            target_w = apply_window_normalized(target_img, lo, hi)
            weighted_w, charb_w, ssim_w, edge_w = self._triple(pred_w, target_w)

            scale = 1.0 if self.window_mode == "only" else self.lambda_window
            total = weighted_w * scale if total is None else total + weighted_w * scale

            info["Window"] = weighted_w.detach().item()
            info["Charbonnier_W"] = charb_w.detach().item()
            info["SSIM_W"] = ssim_w.detach().item()
            info["Edge_W"] = edge_w.detach().item()
            if self.window_mode == "only":
                # Keep the standard keys populated so logging code stays simple.
                info.setdefault("Charbonnier", info["Charbonnier_W"])
                info.setdefault("SSIM", info["SSIM_W"])
                info.setdefault("Edge", info["Edge_W"])

        if total is None:      # window_mode == "only" without body_types
            weighted, charb, ssim, edge = self._triple(pred_img, target_img)
            total = weighted
            info["Charbonnier"] = charb.detach().item()
            info["SSIM"] = ssim.detach().item()
            info["Edge"] = edge.detach().item()

        info["Total"] = total.detach().item()
        info["L1"] = info["Charbonnier"]     # deprecated alias
        return total, info

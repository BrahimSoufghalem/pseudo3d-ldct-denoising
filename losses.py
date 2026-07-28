"""
LDCT Project - Loss Functions
================================
Charbonnier (robust L1) + SSIM + Sobel edge loss.

Naming note: the weight is historically called `lambda_l1`, but the term it
multiplies is a CHARBONNIER loss, not a plain L1. The returned logging dict now
reports "Charbonnier" as the primary key and keeps "L1" only as a backward
compatible alias so existing TensorBoard/tqdm code keeps working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss

from config import LAMBDA_L1, LAMBDA_SSIM, LAMBDA_EDGE


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
# HYBRID LOSS
# ═══════════════════════════════════════════
class MONAIHybridLoss(nn.Module):
    """
    Charbonnier + SSIM + Sobel edge loss with configurable weights.

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
    """

    def __init__(
        self,
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_edge=LAMBDA_EDGE,
        spatial_dims=2,
        charbonnier_eps=1e-3,
        ssim_clamp=True,
    ):
        super().__init__()
        self.lambda_charbonnier = lambda_l1
        self.lambda_l1 = lambda_l1          # alias, kept for compatibility
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.ssim_clamp = ssim_clamp

        self.charbonnier_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.ssim_loss = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0)
        self.edge_loss = SobelEdgeLoss()

    def forward(self, pred_img, target_img):
        pred_img = pred_img.float()
        target_img = target_img.float()

        charb = self.charbonnier_loss(pred_img, target_img)
        edge = self.edge_loss(pred_img, target_img)

        if self.ssim_clamp:
            ssim = self.ssim_loss(pred_img.clamp(0.0, 1.0), target_img.clamp(0.0, 1.0))
        else:
            ssim = self.ssim_loss(pred_img, target_img)

        total = (
            self.lambda_charbonnier * charb
            + self.lambda_ssim * ssim
            + self.lambda_edge * edge
        )

        info = {
            "Charbonnier": charb.detach().item(),
            "SSIM": ssim.detach().item(),
            "Edge": edge.detach().item(),
            "Total": total.detach().item(),
        }
        info["L1"] = info["Charbonnier"]     # deprecated alias
        return total, info

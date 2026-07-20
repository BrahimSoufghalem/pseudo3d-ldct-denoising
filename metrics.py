"""
LDCT Project — Evaluation Metrics
====================================
PSNR, RMSE, and VIF (Visual Information Fidelity).
"""

import torch
from torchmetrics.image import VisualInformationFidelity
  

# ═══════════════════════════════════════════ 
# PSNR
# ═══════════════════════════════════════════
def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio (↑ higher = better).
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))


# ═══════════════════════════════════════════
# RMSE
# ═══════════════════════════════════════════
def rmse(pred, target):
    """
    Root Mean Squared Error (↓ lower = better).
    """
    return torch.sqrt(torch.mean((pred - target) ** 2))


# ═══════════════════════════════════════════
# VIF METRIC
# ═══════════════════════════════════════════
class VIFMetric:
    """
    Visual Information Fidelity (↑ higher = better, range [0, 1]).
    Measures the amount of visual information preserved in the enhanced image.
    More sensitive than SSIM for fine medical details (nodules, edges).

    Input:  [B, 1, H, W] in range [0, 1]
    Output: scalar (mean across batch)
    """
    def __init__(self, device='cpu'):
        self.metric = VisualInformationFidelity(
            sigma_n_sq=2.0,
            reduction='mean'
        ).to(device)
        self.device = device
        self._scores = []

    @torch.no_grad()
    def update(self, pred, target):
        score = self.metric(pred.to(self.device), target.to(self.device))
        self._scores.append(score.item())

    def aggregate(self):
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def reset(self):
        self._scores = []

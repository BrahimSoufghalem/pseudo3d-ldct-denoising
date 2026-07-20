"""
LDCT Project — Evaluation Metrics
====================================
Physically and clinically rigorous metric calculations for Low-Dose CT Denoising:
- PSNR and SSIM: Computed on clinical diagnostic windows (Chest: Lung Window, Abdomen: Soft Tissue Window).
- RMSE: Computed directly in physical Hounsfield Units (HU).
- VIF: Computed on physical HU scale.
"""

import torch
from torchmetrics.image import VisualInformationFidelity

# ═══════════════════════════════════════════
# CLINICAL WINDOW DEFINITIONS (HU)
# ═══════════════════════════════════════════
# Clinical Windows: (Center, Width) in Hounsfield Units
CLINICAL_WINDOWS = {
    "Chest": (-600.0, 1500.0),    # Lung window: C=-600 HU, W=1500 HU
    "Abdomen": (50.0, 400.0),     # Soft tissue window: C=50 HU, W=400 HU
}


def apply_clinical_window(hu_tensor, body_type="Abdomen"):
    """
    Apply anatomy-specific clinical windowing (Center & Width) to an image in HU scale.
    - Chest -> Lung Window (Center=-600 HU, Width=1500 HU)
    - Abdomen -> Soft Tissue Window (Center=50 HU, Width=400 HU)

    Output: Tensor normalized to [0.0, 1.0] within the diagnostic window.
    """
    center, width = CLINICAL_WINDOWS.get(body_type, (50.0, 400.0))
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    windowed = torch.clamp((hu_tensor - lower) / (upper - lower), 0.0, 1.0)
    return windowed


def denormalize_hu(tensor, a_min=-1024.0, a_max=1600.0):
    """Convert normalized [0, 1] tensor back to physical Hounsfield Units (HU)."""
    return tensor * (a_max - a_min) + a_min


# ═══════════════════════════════════════════ 
# PSNR
# ═══════════════════════════════════════════
def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio (↑ higher = better).
    Calculated on windowed images [0, 1] with max_val=1.0.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))


# ═══════════════════════════════════════════
# RMSE (in Hounsfield Units - HU)
# ═══════════════════════════════════════════
def rmse(pred_hu, target_hu):
    """
    Root Mean Squared Error directly in physical Hounsfield Units (HU) (↓ lower = better).
    """
    return torch.sqrt(torch.mean((pred_hu - target_hu) ** 2))


# ═══════════════════════════════════════════
# VIF METRIC
# ═══════════════════════════════════════════
class VIFMetric:
    """
    Visual Information Fidelity (↑ higher = better, range [0, 1]).
    Measures the amount of visual information preserved in the enhanced image.
    More sensitive than SSIM for fine medical details (nodules, edges).

    Input:  [B, 1, H, W] tensor
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

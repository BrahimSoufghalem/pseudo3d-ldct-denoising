"""
LDCT Project — Physically Accurate Benchmark Evaluation Metrics (ldct-benchmark Standard)
========================================================================================
- RMSE: Measured in physical Hounsfield Units (HU) clipped to [0, 2924] (HU + 1024 offset).
- PSNR & SSIM: Measured after applying Clinical Diagnostic Windowing (Chest: Lung Window, Abdomen: Soft Tissue Window).
- VIF: Measured on physical HU scale.
"""

import numpy as np
import torch
from skimage.metrics import mean_squared_error, structural_similarity
from torchmetrics.functional.image import visual_information_fidelity

# ═══════════════════════════════════════════
# CONSTANTS & CLINICAL WINDOW DEFINITIONS
# ═══════════════════════════════════════════
DATA_RANGE = 2924.0  # Maximum HU of bone (1900) + DICOM offset (1024) -> 2924

# Center and Width in HU + 1024 offset domain
CW = {
    "Chest": (1024 - 600, 1500),    # Lung window: C=-600 HU, W=1500 HU
    "Abdomen": (1024 + 50, 400),    # Soft tissue window: C=50 HU, W=400 HU
}


def apply_center_width(x: np.ndarray, center: float, width: float, out_range=(0.0, 1.0)) -> np.ndarray:
    """
    Apply clinical center and width windowing to a 2D numpy array (in HU + 1024 domain).
    Clips and scales pixel values to out_range [0.0, 1.0].
    """
    center = float(center)
    width = float(width)
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    res = np.empty(x.shape, dtype=np.float32)
    res[x <= lower] = out_range[0]
    mask = (x > lower) & (x <= upper)
    res[mask] = ((x[mask] - (center - 0.5)) / (width - 1.0) + 0.5) * (out_range[1] - out_range[0]) + out_range[0]
    res[x > upper] = out_range[1]
    return res


def denormalize_to_hu_offset(norm_tensor, a_min=-1024.0, a_max=1600.0):
    """
    Convert model output normalized in [0, 1] back to HU + 1024 offset domain (float32 numpy).
    """
    if isinstance(norm_tensor, torch.Tensor):
        norm_tensor = norm_tensor.detach().cpu().numpy()
    hu = norm_tensor * (a_max - a_min) + a_min
    return (hu + 1024.0).astype(np.float32)


# ═══════════════════════════════════════════
# METRIC COMPUTATION FUNCTIONS (ldct-benchmark)
# ═══════════════════════════════════════════
def compute_psnr_windowed(pred_hu_offset: np.ndarray, target_hu_offset: np.ndarray, body_type: str = "Abdomen") -> float:
    """
    Peak Signal-to-Noise Ratio (dB) calculated after applying clinical diagnostic windowing.
    """
    center, width = CW.get(body_type, CW["Abdomen"])
    t_win = apply_center_width(target_hu_offset, center, width)
    p_win = apply_center_width(pred_hu_offset, center, width)
    mse = mean_squared_error(t_win, p_win)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim_windowed(pred_hu_offset: np.ndarray, target_hu_offset: np.ndarray, body_type: str = "Abdomen") -> float:
    """
    Structural Similarity Index (SSIM) calculated after applying clinical diagnostic windowing (data_range=1.0).
    """
    center, width = CW.get(body_type, CW["Abdomen"])
    t_win = apply_center_width(target_hu_offset, center, width)
    p_win = apply_center_width(pred_hu_offset, center, width)
    return float(structural_similarity(t_win, p_win, data_range=1.0))


def compute_rmse_hu(pred_hu_offset: np.ndarray, target_hu_offset: np.ndarray) -> float:
    """
    Root Mean Squared Error directly in physical Hounsfield Units (HU) clipped to [0, 2924].
    """
    t_clip = np.clip(target_hu_offset, 0.0, DATA_RANGE)
    p_clip = np.clip(pred_hu_offset, 0.0, DATA_RANGE)
    return float(np.sqrt(mean_squared_error(t_clip, p_clip)))


def compute_vif_hu(pred_hu_offset: np.ndarray, target_hu_offset: np.ndarray) -> float:
    """
    Visual Information Fidelity (VIF) on physical HU scale clipped to [0, 2924].
    """
    t_clip = np.clip(target_hu_offset, 0.0, DATA_RANGE)
    p_clip = np.clip(pred_hu_offset, 0.0, DATA_RANGE)
    t_tensor = torch.from_numpy(t_clip).unsqueeze(0).unsqueeze(0)
    p_tensor = torch.from_numpy(p_clip).unsqueeze(0).unsqueeze(0)
    return float(visual_information_fidelity(p_tensor, t_tensor, sigma_n_sq=2.0).numpy())


class VIFMetric:
    """
    VIF metric accumulator class for evaluating slice batches.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self._scores = []

    def update(self, pred_hu_offset, target_hu_offset):
        score = compute_vif_hu(pred_hu_offset, target_hu_offset)
        self._scores.append(score)

    def aggregate(self):
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def reset(self):
        self._scores = []

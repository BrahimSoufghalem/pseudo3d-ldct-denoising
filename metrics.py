"""
LDCT Project — Physically Accurate Benchmark Evaluation Metrics (ldct-benchmark Standard)
========================================================================================
All constants, diagnostic windows, and evaluation settings are imported directly from `config.py`.
- RMSE: Measured in physical Hounsfield Units (HU) clipped to [0, 2924] (HU + 1024 offset).
- PSNR & SSIM: Measured after applying Clinical Diagnostic Windowing (Chest: Lung Window, Abdomen: Soft Tissue Window).
- VIF: Measured on physical HU scale.
"""

import numpy as np
import torch
from skimage.metrics import mean_squared_error, structural_similarity
from torchmetrics.functional.image import visual_information_fidelity

from config import EVAL_DATA_RANGE, CLINICAL_WINDOWS, A_MIN, A_MAX

# ═══════════════════════════════════════════
# CONSTANTS & CLINICAL WINDOW DEFINITIONS FROM CONFIG
# ═══════════════════════════════════════════
DATA_RANGE = EVAL_DATA_RANGE
CW = CLINICAL_WINDOWS


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


def denormalize_to_hu_offset(norm_tensor, a_min=A_MIN, a_max=A_MAX):
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


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR for PyTorch tensors in range [0, 1]."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute RMSE for PyTorch tensors."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_rmse_hu(pred_hu_offset, target_hu_offset) -> float:
    """
    Root Mean Squared Error directly in physical Hounsfield Units (HU) clipped to [0, 2924].
    """
    if isinstance(pred_hu_offset, torch.Tensor):
        pred_hu_offset = pred_hu_offset.detach().cpu().numpy()
    if isinstance(target_hu_offset, torch.Tensor):
        target_hu_offset = target_hu_offset.detach().cpu().numpy()

    t_clip = np.clip(target_hu_offset, 0.0, DATA_RANGE)
    p_clip = np.clip(pred_hu_offset, 0.0, DATA_RANGE)
    return float(np.sqrt(mean_squared_error(t_clip, p_clip)))


def compute_vif_hu(pred_hu_offset, target_hu_offset) -> float:
    """
    Visual Information Fidelity (VIF) on physical HU scale clipped to [0, 2924].
    """
    if isinstance(pred_hu_offset, torch.Tensor):
        pred_hu_offset = pred_hu_offset.detach().cpu().numpy()
    if isinstance(target_hu_offset, torch.Tensor):
        target_hu_offset = target_hu_offset.detach().cpu().numpy()

    # Convert normalized [0, 1] inputs to HU + 1024 offset if needed
    if pred_hu_offset.max() <= 1.5 and pred_hu_offset.min() >= -0.5:
        pred_hu_offset = (pred_hu_offset * (A_MAX - A_MIN) + A_MIN + 1024.0).astype(np.float32)
        target_hu_offset = (target_hu_offset * (A_MAX - A_MIN) + A_MIN + 1024.0).astype(np.float32)

    t_clip = np.clip(target_hu_offset, 0.0, DATA_RANGE)
    p_clip = np.clip(pred_hu_offset, 0.0, DATA_RANGE)

    t_tensor = torch.from_numpy(t_clip)
    p_tensor = torch.from_numpy(p_clip)

    while t_tensor.dim() < 4:
        t_tensor = t_tensor.unsqueeze(0)
    while p_tensor.dim() < 4:
        p_tensor = p_tensor.unsqueeze(0)

    try:
        val = visual_information_fidelity(p_tensor, t_tensor, sigma_n_sq=2.0)
        return float(val.detach().cpu().numpy())
    except Exception:
        return 0.0


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


"""
LDCT Project — Utility Functions
==================================
Reproducibility setup and DICOM sorting helpers.
"""

import os
import re 

import torch 
import pydicom
from monai.utils import set_determinism

from config import SEED

 
def setup_reproducibility():
    """Set all random seeds and deterministic flags for reproducibility."""
    set_determinism(seed=SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sort_by_instance_number(dcm_paths):
    """
    Sort a list of DICOM file paths by their InstanceNumber metadata.
    Falls back to extracting numbers from the filename if metadata is missing.
    """
    def get_instance(path):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            return int(ds.InstanceNumber)
        except Exception:
            nums = re.findall(r'\d+', os.path.basename(path))
            return int(nums[-1]) if nums else 0

    return sorted(dcm_paths, key=get_instance)


def get_device():
    """Return the best available torch device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    return device


def apply_window_tensor(img_tensor, center, width):
    """Apply clinical windowing C/W to a raw HU PyTorch tensor and scale to [0, 1]."""
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    return torch.clamp((img_tensor - lower) / (upper - lower), 0.0, 1.0)


def build_multi_window_input(prev_hu, curr_hu, next_hu,
                             a_min=-1024.0, a_max=1600.0,
                             lung_center=-600.0, lung_width=1500.0,
                             soft_center=50.0, soft_width=400.0):
    """
    Constructs a 9-channel tensor [1, 9, H, W] from raw HU slice tensors:
      - 3 channels Full HU range [-1024, 1600]
      - 3 channels Lung Window (C=-600, W=1500)
      - 3 channels Soft Tissue Window (C=50, W=400)
    """
    if not isinstance(prev_hu, torch.Tensor):
        prev_hu = torch.from_numpy(prev_hu)
    if not isinstance(curr_hu, torch.Tensor):
        curr_hu = torch.from_numpy(curr_hu)
    if not isinstance(next_hu, torch.Tensor):
        next_hu = torch.from_numpy(next_hu)

    # Ensure float32
    prev_hu = prev_hu.float()
    curr_hu = curr_hu.float()
    next_hu = next_hu.float()

    # 1. Full Range Window [0, 1]
    prev_full = torch.clamp((prev_hu - a_min) / (a_max - a_min), 0.0, 1.0)
    curr_full = torch.clamp((curr_hu - a_min) / (a_max - a_min), 0.0, 1.0)
    next_full = torch.clamp((next_hu - a_min) / (a_max - a_min), 0.0, 1.0)

    # 2. Lung Window [0, 1]
    prev_lung = apply_window_tensor(prev_hu, lung_center, lung_width)
    curr_lung = apply_window_tensor(curr_hu, lung_center, lung_width)
    next_lung = apply_window_tensor(next_hu, lung_center, lung_width)

    # 3. Soft Tissue Window [0, 1]
    prev_soft = apply_window_tensor(prev_hu, soft_center, soft_width)
    curr_soft = apply_window_tensor(curr_hu, soft_center, soft_width)
    next_soft = apply_window_tensor(next_hu, soft_center, soft_width)

    channels = [
        prev_full, curr_full, next_full,
        prev_lung, curr_lung, next_lung,
        prev_soft, curr_soft, next_soft
    ]

    # Clean channel dimensions to [9, H, W]
    cleaned = []
    for c in channels:
        while c.dim() > 2:
            c = c.squeeze(0)
        cleaned.append(c)

    inp = torch.stack(cleaned, dim=0).unsqueeze(0)  # → [1, 9, H, W]
    return inp


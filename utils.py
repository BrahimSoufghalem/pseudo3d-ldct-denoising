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


def build_pseudo3d_input(prev_hu, curr_hu, next_hu, a_min=-1024.0, a_max=1600.0):
    """
    Constructs a 3-channel tensor [1, 3, H, W] from raw HU slice tensors:
      - 3 channels Full HU range [-1024, 1600] normalized to [0, 1].
    """
    if not isinstance(prev_hu, torch.Tensor):
        prev_hu = torch.from_numpy(prev_hu)
    if not isinstance(curr_hu, torch.Tensor):
        curr_hu = torch.from_numpy(curr_hu)
    if not isinstance(next_hu, torch.Tensor):
        next_hu = torch.from_numpy(next_hu)

    prev_hu = prev_hu.float()
    curr_hu = curr_hu.float()
    next_hu = next_hu.float()

    prev_full = torch.clamp((prev_hu - a_min) / (a_max - a_min), 0.0, 1.0)
    curr_full = torch.clamp((curr_hu - a_min) / (a_max - a_min), 0.0, 1.0)
    next_full = torch.clamp((next_hu - a_min) / (a_max - a_min), 0.0, 1.0)

    channels = [prev_full, curr_full, next_full]
    cleaned = []
    for c in channels:
        while c.dim() > 2:
            c = c.squeeze(0)
        cleaned.append(c)

    inp = torch.stack(cleaned, dim=0).unsqueeze(0)  # → [1, 3, H, W]
    return inp


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

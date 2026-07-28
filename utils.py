"""
LDCT Project - Shared Utilities
=================================
Reproducibility, device selection, DICOM ordering, mode-aware model input
construction and checkpoint/state-dict helpers.
"""

import os
import random

import numpy as np
import pydicom
import torch
import torch.nn as nn

import config as cfg


# ═══════════════════════════════════════════
# REPRODUCIBILITY / BACKEND FLAGS
# ═══════════════════════════════════════════
def setup_reproducibility(seed=cfg.SEED, deterministic=False, allow_tf32=None):
    """Seed every RNG used by the project and set fast backend defaults.

    `deterministic=True` forces deterministic cuDNN kernels and disables TF32,
    which is slower but makes runs bit-for-bit reproducible.

    TF32 (Ampere and newer) is enabled by default for matmul. PyTorch disables
    matmul TF32 out of the box, which is a large, silent slowdown here: the SS2D
    bottleneck is dominated by matmul/einsum, not convolutions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if allow_tf32 is None:
        allow_tf32 = getattr(cfg, "ALLOW_TF32", True)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    return seed


def get_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════
# DICOM HELPERS
# ═══════════════════════════════════════════
def sort_by_instance_number(file_paths):
    """Sort DICOM paths by InstanceNumber (falls back to the filename)."""
    def key(path):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0
    return sorted(file_paths, key=lambda p: (key(p), os.path.basename(p)))


# ═══════════════════════════════════════════
# MODEL INPUT CONSTRUCTION (mode aware)
# ═══════════════════════════════════════════
def normalize_hu(hu_tensor, a_min=None, a_max=None):
    """Clip HU to [a_min, a_max] and scale to [0, 1] using the config preset."""
    a_min = cfg.A_MIN if a_min is None else a_min
    a_max = cfg.A_MAX if a_max is None else a_max
    return ((hu_tensor.clamp(a_min, a_max) - a_min) / (a_max - a_min)).float()


def build_model_input(prev_hu, curr_hu, next_hu, input_mode=None, a_min=None, a_max=None):
    """
    Build a normalized model input tensor [1, C, H, W] from HU slices.

      input_mode == "2d"   -> C = 1  (centre slice only; prev/next ignored)
      input_mode == "2.5d" -> C = 3  (prev, curr, next)

    HU limits always come from config (A_MIN / A_MAX) unless explicitly
    overridden, so preprocessing can never silently disagree with the dataset
    pipeline.
    """
    mode = cfg.normalize_input_mode(input_mode)
    curr = normalize_hu(curr_hu, a_min, a_max)
    if mode == "2d":
        return curr.reshape(1, 1, *curr.shape[-2:])
    prev = normalize_hu(prev_hu if prev_hu is not None else curr_hu, a_min, a_max)
    nxt = normalize_hu(next_hu if next_hu is not None else curr_hu, a_min, a_max)
    stack = torch.stack([prev.squeeze(), curr.squeeze(), nxt.squeeze()], dim=0)
    return stack.unsqueeze(0)


def build_pseudo3d_input(prev_hu, curr_hu, next_hu, a_min=None, a_max=None):
    """DEPRECATED: kept for backward compatibility.

    Use `build_model_input(..., input_mode=...)`. The old default limits
    (-1024, 1600) contradicted config.A_MIN / config.A_MAX; the config values
    are now the single source of truth.
    """
    return build_model_input(prev_hu, curr_hu, next_hu, input_mode="2.5d",
                             a_min=a_min, a_max=a_max)


def extract_centre_slice(x):
    """Return the centre channel of a [B, C, H, W] input, keeping the dim.

    Works for C == 1 (2D mode) and C == 3 (2.5D mode) alike, so training code
    no longer has to hardcode `x[:, 1:2]`.
    """
    c = x.shape[1] // 2
    return x[:, c:c + 1]


# ═══════════════════════════════════════════
# STATE-DICT HELPERS (DataParallel-safe)
# ═══════════════════════════════════════════
def unwrap_model(model):
    """Return the underlying module for DataParallel / DDP wrappers."""
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def get_state_dict(model):
    """Always return an UNWRAPPED state dict (no `module.` prefixes).

    The previous code saved `model.module.state_dict()` in checkpoints but
    `model.state_dict()` for the best model, so the two files had incompatible
    key names under DataParallel.
    """
    return unwrap_model(model).state_dict()


def load_state_into(model, state, strict=True):
    """Load a state dict saved either with or without a `module.` prefix."""
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    return unwrap_model(model).load_state_dict(cleaned, strict=strict)

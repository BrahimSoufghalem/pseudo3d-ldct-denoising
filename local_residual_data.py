"""Benchmark-aligned 2D data path for the local residual control.

This module is intentionally separate from ``dataset.py``. Existing Mamba and
physics-spectral runs keep their historical HU->[0,1] preprocessing unchanged.
The control here reproduces ldct-benchmark's mean/std convention in the stored
CT pixel domain: pixel = HU + 1024.
"""

import os
import random
from collections import Counter

import torch
from monai.data import CacheDataset, Dataset, DataLoader, PydicomReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropSamplesd,
    ResizeWithPadOrCropd, ToTensord,
)
from monai.utils import set_determinism
from torch.utils.data import WeightedRandomSampler

import config as cfg
from dataset import collect_files


# Exact constants in eeulig/ldct-benchmark ldctbench/data/info.yml.
BENCHMARK_PIXEL_MEAN = 481.45419786099086
BENCHMARK_PIXEL_STD = 502.18507379395044
BENCHMARK_PIXEL_OFFSET = 1024.0


def standardize_hu(hu):
    """Physical HU -> benchmark-standardized stored-pixel values, without clipping."""
    if not isinstance(hu, torch.Tensor):
        hu = torch.as_tensor(hu, dtype=torch.float32)
    pixel = hu.float() + BENCHMARK_PIXEL_OFFSET
    return (pixel - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD


def denormalize_to_pixel(z):
    """Benchmark-standardized tensor -> physical non-negative pixel/HU+1024 domain."""
    return z * BENCHMARK_PIXEL_STD + BENCHMARK_PIXEL_MEAN


class BenchmarkMeanStdd:
    """MONAI dictionary transform using the exact benchmark global statistics."""

    def __init__(self, keys=("image", "label")):
        self.keys = tuple(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = standardize_hu(data[key])
        return data


def _benchmark_reader():
    """Match pydicom.pixel_array orientation used by ldct-benchmark.

    MONAI's PydicomReader applies RescaleSlope/Intercept correctly, but its
    default ``swap_ij=True`` transposes the spatial axes. The benchmark and our
    full-resolution evaluator use pydicom arrays without that swap. Keeping the
    default here would train on transposed anatomy and test on unswapped anatomy.
    """
    return PydicomReader(swap_ij=False)


def _train_transform(patch_size):
    return Compose([
        LoadImaged(keys=["image", "label"], reader=_benchmark_reader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        BenchmarkMeanStdd(),
        RandSpatialCropSamplesd(
            keys=["image", "label"], roi_size=(patch_size, patch_size),
            num_samples=1,
        ),
        ToTensord(keys=["image", "label"]),
    ])


def _val_transform(patch_size):
    return Compose([
        LoadImaged(keys=["image", "label"], reader=_benchmark_reader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        BenchmarkMeanStdd(),
        ResizeWithPadOrCropd(
            keys=["image", "label"], spatial_size=(patch_size, patch_size),
        ),
        ToTensord(keys=["image", "label"]),
    ])


def prepare_local_residual_data(
    in_dir=cfg.DATA_DIR,
    train_patch_size=64,
    val_patch_size=128,
    train_batch_size=64,
    val_batch_size=64,
    iterations_before_val=1000,
    num_workers=cfg.NUM_WORKERS,
    cache=True,
    cache_rate=1.0,
):
    """Create patient-balanced train batches and deterministic validation crops.

    Each training cycle contains exactly ``iterations_before_val`` batches, as
    in ldct-benchmark. Every patient has equal total sampling mass regardless of
    slice count. Validation remains deterministic and visits every validation
    slice, which is more stable than randomly resampling validation patients.
    """
    set_determinism(seed=cfg.SEED)
    random.seed(cfg.SEED)

    all_patients = sorted([
        p for p in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, p))
    ])
    train_patients = [p for p in all_patients if p in cfg.EXPECTED_TRAIN]
    val_patients = [p for p in all_patients if p in cfg.EXPECTED_VAL]
    if not train_patients or not val_patients:
        raise RuntimeError(
            "The benchmark-aligned control requires the explicit patient split "
            "from config.py; train or validation patients were not found."
        )

    train_files = collect_files(train_patients, in_dir, input_mode="2d")
    val_files = collect_files(val_patients, in_dir, input_mode="2d")
    counts = Counter(item["patient"] for item in train_files)
    weights = torch.tensor(
        [1.0 / counts[item["patient"]] for item in train_files],
        dtype=torch.double,
    )
    samples_per_cycle = int(train_batch_size) * int(iterations_before_val)
    sampler_generator = torch.Generator().manual_seed(cfg.SEED)
    sampler = WeightedRandomSampler(
        weights, num_samples=samples_per_cycle, replacement=True,
        generator=sampler_generator,
    )

    train_transform = _train_transform(int(train_patch_size))
    val_transform = _val_transform(int(val_patch_size))
    cache_rate = float(min(max(cache_rate, 0.0), 1.0))
    if cache and cache_rate > 0:
        train_ds = CacheDataset(
            train_files, train_transform, cache_rate=cache_rate,
        )
        val_ds = CacheDataset(
            val_files, val_transform, cache_rate=cache_rate,
        )
    else:
        train_ds = Dataset(train_files, train_transform)
        val_ds = Dataset(val_files, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0,
    )

    chest_train = sum(p.lower().startswith("c") for p in train_patients)
    chest_val = sum(p.lower().startswith("c") for p in val_patients)
    print("\nLocal residual benchmark data:")
    print(f"Train patients : {len(train_patients)} ({chest_train} chest, {len(train_patients)-chest_train} abdomen)")
    print(f"Val patients   : {len(val_patients)} ({chest_val} chest, {len(val_patients)-chest_val} abdomen)")
    print(f"Train slices   : {len(train_files)} | patient-balanced replacement sampling")
    print(f"Val slices     : {len(val_files)} | deterministic center crop")
    print(f"Patches        : train {train_patch_size} | val {val_patch_size}")
    print(f"Train cycle    : {iterations_before_val} iterations x batch {train_batch_size}")
    print("DICOM orientation: PydicomReader(swap_ij=False), aligned with benchmark/evaluation")
    print(
        "Standardization : (HU + 1024 - "
        f"{BENCHMARK_PIXEL_MEAN:.12f}) / {BENCHMARK_PIXEL_STD:.12f}"
    )
    return train_loader, val_loader

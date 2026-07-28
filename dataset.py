"""
LDCT Project - Dataset & Data Pipeline
=========================================
MONAI transforms and DataLoaders for both experiment input modes:

  INPUT_MODE == "2d"   -> only the current low-dose slice is read  -> image [1, H, W]
  INPUT_MODE == "2.5d" -> (prev, curr, next) are read and stacked  -> image [3, H, W]

In "2d" mode the neighbouring files are never loaded, so that ablation is also
about 3x cheaper in I/O and cache memory - not just a masked-out input.
"""

import os
import random
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from monai.utils import set_determinism
from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandSpatialCropSamplesd, ResizeWithPadOrCropd, ToTensord,
)

import config as cfg
from config import (
    DATA_DIR, SPATIAL_SIZE, A_MIN, A_MAX,
    CACHE_DATA, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS,
    SEED, SPLIT_RANDOM_STATE, SPLIT_TEST_SIZE,
    EXPECTED_VAL, EXPECTED_TRAIN,
)
from utils import sort_by_instance_number


def image_keys_for(input_mode=None):
    """Dictionary keys that must be loaded from disk for a given input mode."""
    if cfg.normalize_input_mode(input_mode) == "2d":
        return ["image", "label"]
    return ["image_prev", "image", "image_next", "label"]


# ═══════════════════════════════════════════
# NORMALIZE / STACK TRANSFORM
# ═══════════════════════════════════════════
class StackSlicesd:
    """
    MONAI-style dictionary transform.

    Normalizes HU to [0, 1] with the config window [A_MIN, A_MAX] and, in 2.5D
    mode, concatenates (prev, curr, next) into a 3-channel tensor.
    In 2D mode it only normalizes the current slice, leaving `image` at [1, H, W].
    """

    def __init__(self, input_mode=None, a_min=A_MIN, a_max=A_MAX):
        self.input_mode = cfg.normalize_input_mode(input_mode)
        self.a_min = a_min
        self.a_max = a_max

    def _norm(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        return torch.clamp((x - self.a_min) / (self.a_max - self.a_min), 0.0, 1.0)

    def __call__(self, data):
        data["label"] = self._norm(data["label"])

        if self.input_mode == "2d":
            data["image"] = self._norm(data["image"])
            data.pop("image_prev", None)
            data.pop("image_next", None)
            return data

        prev_full = self._norm(data["image_prev"])
        curr_full = self._norm(data["image"])
        next_full = self._norm(data["image_next"])
        data["image"] = torch.cat([prev_full, curr_full, next_full], dim=0)
        data.pop("image_prev", None)
        data.pop("image_next", None)
        return data


# ═══════════════════════════════════════════
# FILE COLLECTION
# ═══════════════════════════════════════════
def collect_files(patient_list, in_dir=DATA_DIR, input_mode=None):
    """
    Build one record per slice.

    2.5D records carry (image_prev, image, image_next, label); 2D records carry
    only (image, label). Slice neighbours are clamped at volume boundaries.
    """
    mode = cfg.normalize_input_mode(input_mode)
    files = []

    for patient in patient_list:
        low_dir = os.path.join(in_dir, patient, "Low_Dose")
        full_dir = os.path.join(in_dir, patient, "Full_Dose")

        low_imgs = sort_by_instance_number(glob(os.path.join(low_dir, "*.dcm")))
        full_imgs = sort_by_instance_number(glob(os.path.join(full_dir, "*.dcm")))

        assert len(low_imgs) == len(full_imgs), \
            f"Mismatch for patient {patient}: {len(low_imgs)} vs {len(full_imgs)}"

        n = len(low_imgs)
        for i in range(n):
            record = {
                "image": low_imgs[i],
                "label": full_imgs[i],
                "patient": patient,
                "body_type": "Chest" if patient.lower().startswith("c") else "Abdomen",
            }
            if mode == "2.5d":
                record["image_prev"] = low_imgs[max(i - 1, 0)]
                record["image_next"] = low_imgs[min(i + 1, n - 1)]
            files.append(record)

    return files


# ═══════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════
def get_train_transforms(spatial_size=SPATIAL_SIZE, input_mode=None):
    """Training transforms (random spatial crop augmentation)."""
    keys = image_keys_for(input_mode)
    return Compose([
        LoadImaged(keys=keys, reader="PydicomReader"),
        EnsureChannelFirstd(keys=keys),
        StackSlicesd(input_mode=input_mode),
        RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=spatial_size,
            num_samples=1,
        ),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(spatial_size=SPATIAL_SIZE, input_mode=None):
    """Validation transforms (deterministic pad/crop, no augmentation)."""
    keys = image_keys_for(input_mode)
    return Compose([
        LoadImaged(keys=keys, reader="PydicomReader"),
        EnsureChannelFirstd(keys=keys),
        StackSlicesd(input_mode=input_mode),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        ToTensord(keys=["image", "label"]),
    ])


# ═══════════════════════════════════════════
# MAIN DATA PREPARATION
# ═══════════════════════════════════════════
def prepareCT2D(
    in_dir=DATA_DIR,
    spatial_size=SPATIAL_SIZE,
    cache=CACHE_DATA,
    input_mode=None,
    train_batch_size=TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Full data pipeline:
      1. Explicit (or stratified fallback) patient split
      2. Per-slice record collection for the chosen input mode
      3. MONAI datasets & DataLoaders

    Returns (train_loader, val_loader).
    """
    mode = cfg.normalize_input_mode(input_mode)
    set_determinism(seed=SEED)
    random.seed(SEED)

    all_patients = sorted([
        p for p in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, p))
    ])

    train_patients = [p for p in all_patients if p in EXPECTED_TRAIN]
    val_patients = [p for p in all_patients if p in EXPECTED_VAL]

    # Fallback if the dataset has not been split explicitly yet
    if not train_patients or not val_patients:
        chest_patients = [p for p in all_patients if p.lower().startswith("c")]
        abdomen_patients = [p for p in all_patients if p.lower().startswith("l")]
        chest_train, chest_val = train_test_split(
            chest_patients, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
        )
        abdomen_train, abdomen_val = train_test_split(
            abdomen_patients, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
        )
        train_patients = chest_train + abdomen_train
        val_patients = chest_val + abdomen_val

    chest_train = [p for p in train_patients if p.lower().startswith("c")]
    abdomen_train = [p for p in train_patients if p.lower().startswith("l")]
    chest_val = [p for p in val_patients if p.lower().startswith("c")]
    abdomen_val = [p for p in val_patients if p.lower().startswith("l")]

    random.shuffle(train_patients)
    random.shuffle(val_patients)

    print(f"\nSplit (input_mode={mode}):")
    print(f"Train Chest   : {len(chest_train)}")
    print(f"Train Abdomen : {len(abdomen_train)}")
    print(f"Val Chest     : {len(chest_val)}")
    print(f"Val Abdomen   : {len(abdomen_val)}")

    train_files = collect_files(train_patients, in_dir, input_mode=mode)
    val_files = collect_files(val_patients, in_dir, input_mode=mode)

    print(f"Train slices  : {len(train_files)}")
    print(f"Val slices    : {len(val_files)}")

    train_transforms = get_train_transforms(spatial_size, input_mode=mode)
    val_transforms = get_val_transforms(spatial_size, input_mode=mode)

    if cache:
        train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(val_files, val_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(train_files, train_transforms)
        val_ds = Dataset(val_files, val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader

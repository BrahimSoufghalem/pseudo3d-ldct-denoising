"""
LDCT Project — Dataset & Data Pipeline
=========================================
Pseudo-3D (2.5D) data preparation with MONAI transforms and DataLoaders.
"""

import os
import random

import torch
from glob import glob
from sklearn.model_selection import train_test_split
from monai.utils import set_determinism
from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, RandSpatialCropSamplesd,
    ResizeWithPadOrCropd, ToTensord,
)

from config import (
    DATA_DIR, SPATIAL_SIZE, A_MIN, A_MAX, B_MIN, B_MAX,
    CACHE_DATA, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS,
    SEED, SPLIT_RANDOM_STATE, SPLIT_TEST_SIZE,
)
from utils import sort_by_instance_number


# ═══════════════════════════════════════════
# STACK SLICES (Pseudo-3D Transform)
# ═══════════════════════════════════════════
class StackSlicesd:
    """
    Custom MONAI-style dictionary transform.
    Stacks three adjacent slices (prev, curr, next) into a single
    3-channel tensor to provide pseudo-3D context to 2D models.
    """
    def __call__(self, data):
        prev = data["image_prev"]
        curr = data["image"]
        nxt = data["image_next"]
        data["image"] = torch.cat([prev, curr, nxt], dim=0)  # → [3, H, W]
        del data["image_prev"]
        del data["image_next"]
        return data


# ═══════════════════════════════════════════
# FILE COLLECTION
# ═══════════════════════════════════════════
def collect_files(patient_list, in_dir=DATA_DIR):
    """
    For each patient, builds a list of dicts with pseudo-3D triplets:
      image_prev, image (current), image_next, label (full-dose).
    """
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
            prev_i = max(i - 1, 0)
            next_i = min(i + 1, n - 1)
            files.append({
                "image_prev": low_imgs[prev_i],
                "image":      low_imgs[i],
                "image_next": low_imgs[next_i],
                "label":      full_imgs[i],
                "patient":    patient,
                "body_type":  "Chest" if patient.lower().startswith("c") else "Abdomen",
            })
    return files


# ═══════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════
def get_train_transforms(spatial_size=SPATIAL_SIZE):
    """Training transforms with random spatial cropping for augmentation."""
    return Compose([
        LoadImaged(
            keys=["image_prev", "image", "image_next", "label"],
            reader="PydicomReader",
        ),
        EnsureChannelFirstd(
            keys=["image_prev", "image", "image_next", "label"],
        ),
        StackSlicesd(),
        ScaleIntensityRanged(
            keys=["image", "label"],
            a_min=A_MIN, a_max=A_MAX,
            b_min=B_MIN, b_max=B_MAX,
            clip=True,
        ),
        RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=spatial_size,
            num_samples=1,
        ),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(spatial_size=SPATIAL_SIZE):
    """Validation transforms with deterministic pad/crop (no random augmentation)."""
    return Compose([
        LoadImaged(
            keys=["image_prev", "image", "image_next", "label"],
            reader="PydicomReader",
        ),
        EnsureChannelFirstd(
            keys=["image_prev", "image", "image_next", "label"],
        ),
        StackSlicesd(),
        ScaleIntensityRanged(
            keys=["image", "label"],
            a_min=A_MIN, a_max=A_MAX,
            b_min=B_MIN, b_max=B_MAX,
            clip=True,
        ),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=spatial_size,
        ),
        ToTensord(keys=["image", "label"]),
    ])


# ═══════════════════════════════════════════
# MAIN DATA PREPARATION FUNCTION
# ═══════════════════════════════════════════
def prepareCT2D(
    in_dir=DATA_DIR,
    spatial_size=SPATIAL_SIZE,
    cache=CACHE_DATA,
):
    """
    Full data pipeline:
      1. Stratified patient split (Chest / Abdomen)
      2. Collect pseudo-3D file triplets
      3. Build MONAI datasets & DataLoaders

    Returns:
        (train_loader, val_loader)
    """
    set_determinism(seed=SEED)

    # ── Patient split ──
    all_patients = sorted([
        p for p in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, p))
    ])

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

    random.shuffle(train_patients)
    random.shuffle(val_patients)

    print("\n📊 Split:")
    print(f"Train Chest   : {len(chest_train)}")
    print(f"Train Abdomen : {len(abdomen_train)}")
    print(f"Val Chest     : {len(chest_val)}")
    print(f"Val Abdomen   : {len(abdomen_val)}")

    # ── Collect files ──
    train_files = collect_files(train_patients, in_dir)
    val_files = collect_files(val_patients, in_dir)

    print(f"\n✅ Train slices: {len(train_files)}")
    print(f"✅ Val slices  : {len(val_files)}")

    # ── Datasets & Loaders ──
    train_transforms = get_train_transforms(spatial_size)
    val_transforms = get_val_transforms(spatial_size)

    if cache:
        train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(val_files, val_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(train_files, train_transforms)
        val_ds = Dataset(val_files, val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader

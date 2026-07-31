"""
LDCT Project - Inference & DICOM Exporter
===========================================
Runs a trained MS-NAFMambaNet over the low-dose test data and writes the results
as a new DICOM series, preserving the original metadata.

Usage:
    python inference_dicom.py --input-mode 2.5d --mamba-mode full
    python inference_dicom.py --model runs/25d_full/best_model.pt --output-dir Output_DICOM
"""

import os
import argparse
from pathlib import Path
from glob import glob

import torch
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from tqdm import tqdm

import config as cfg
from config import TEST_DIR, A_MIN, A_MAX
from utils import (
    setup_reproducibility, get_device, sort_by_instance_number,
    build_model_input, extract_centre_slice, load_state_into,
)
from model import build_model


# ═══════════════════════════════════════════
# PRE / POST PROCESSING
# ═══════════════════════════════════════════
def load_dicom_tensor(path):
    """Read a DICOM file and return a float32 HU tensor."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return torch.from_numpy(arr)


def normalize(tensor, a_min=A_MIN, a_max=A_MAX):
    """Clip and scale HU -> [0, 1]."""
    return (tensor.clamp(a_min, a_max) - a_min) / (a_max - a_min)


def denormalize_to_hu(tensor, a_min=A_MIN, a_max=A_MAX):
    """Map [0, 1] back to Hounsfield Units."""
    return tensor * (a_max - a_min) + a_min


# ═══════════════════════════════════════════
# DICOM WRITER
# ═══════════════════════════════════════════
def save_as_dicom(ref_dicom_path, output_path, denoised_hu_tensor, series_uid):
    """Write the denoised slice into a copy of the reference DICOM file."""
    ds = pydicom.dcmread(ref_dicom_path)
    denoised_hu = denoised_hu_tensor.squeeze().detach().cpu().numpy()

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixel_array = (denoised_hu - intercept) / slope

    orig_dtype = ds.pixel_array.dtype
    dtype_min = np.iinfo(orig_dtype).min
    dtype_max = np.iinfo(orig_dtype).max
    pixel_array = np.clip(np.rint(pixel_array), dtype_min, dtype_max).astype(orig_dtype)

    ds.PixelData = pixel_array.tobytes()

    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = "Denoised (AI)"
    ds.SeriesNumber = int(getattr(ds, "SeriesNumber", 0) or 0) + 1000
    ds.SOPInstanceUID = generate_uid()
    if hasattr(ds, "file_meta"):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    ds.save_as(output_path)


# ═══════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════
@torch.no_grad()
def process_patient(pid, patient_dir, output_dir, model, device,
                    input_mode="2.5d", use_amp=True):
    """Denoise every slice of one patient and export a new DICOM series."""
    low_dir = patient_dir / "Low_Dose"
    out_patient_dir = output_dir / pid / "Denoised_AI"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    low_imgs = sort_by_instance_number(glob(str(low_dir / "*.dcm")))
    n = len(low_imgs)
    if n == 0:
        print(f"  No images found for patient {pid}.")
        return

    series_uid = generate_uid()
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    for i in tqdm(range(n), desc=f"  Processing [{pid}]", leave=False, unit="slice"):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, n - 1)

        raw_prev = load_dicom_tensor(low_imgs[prev_i])
        raw_curr = load_dicom_tensor(low_imgs[i])
        raw_next = load_dicom_tensor(low_imgs[next_i])

        # [1, C, H, W]; C = 1 in 2D mode, 3 in 2.5D mode
        inp = build_model_input(raw_prev, raw_curr, raw_next, input_mode=input_mode).to(device)
        mid = extract_centre_slice(inp)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred_res = model(inp)
        pred_normalized = torch.clamp(mid.float() + pred_res.float(), 0.0, 1.0)

        pred_hu = denormalize_to_hu(pred_normalized)

        filename = os.path.basename(low_imgs[i])
        save_as_dicom(low_imgs[i], out_patient_dir / filename, pred_hu, series_uid)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Run the LDCT model and export denoised DICOM files.")
    parser.add_argument("--input-mode", default=cfg.INPUT_MODE, choices=list(cfg.VALID_INPUT_MODES))
    parser.add_argument("--mamba-mode", default=cfg.MAMBA_MODE, choices=list(cfg.VALID_MAMBA_MODES))
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained weights (.pt). Defaults to the run folder for the chosen modes.")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR)
    parser.add_argument("--output-dir", type=str, default="Output_DICOM")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    input_mode = cfg.normalize_input_mode(args.input_mode)
    mamba_mode = cfg.normalize_mamba_mode(args.mamba_mode)
    model_path = args.model or cfg.run_paths(mamba_mode=mamba_mode, input_mode=input_mode)["best_model"]

    setup_reproducibility()
    device = get_device()
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {model_path}")
    print(f"input_mode={input_mode} | mamba_mode={mamba_mode} | HU range [{A_MIN}, {A_MAX}]")
    model = build_model(device, mamba_mode=mamba_mode, input_mode=input_mode, data_parallel=False)

    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    load_state_into(model, state)
    model.eval()
    print("Model loaded successfully.\n")

    test_path = Path(args.test_dir)
    patients = sorted([p for p in test_path.iterdir() if p.is_dir() and (p / "Low_Dose").exists()])
    if not patients:
        print(f"No data found in {args.test_dir}.")
        return

    print(f"Found {len(patients)} patients, starting processing...\n")
    for patient_dir in patients:
        process_patient(
            patient_dir.name, patient_dir, output_base_dir, model, device,
            input_mode=input_mode, use_amp=not args.no_amp,
        )

    print("\n" + "=" * 60)
    print(f"Done. Enhanced DICOM files saved in '{args.output_dir}'.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

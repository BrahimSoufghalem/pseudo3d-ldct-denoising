"""
LDCT Project — Inference & DICOM Exporter
===========================================
Reads test data (Low Dose), applies a 2.5D U-Net model to enhance it,
and saves the results as new DICOM files while preserving the original Metadata.
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
from torch.cuda.amp import autocast

from config import TEST_DIR, BEST_MODEL_PATH, A_MIN, A_MAX
from utils import setup_reproducibility, get_device, sort_by_instance_number
from model import build_model


# ═══════════════════════════════════════════
# PREPROCESSING & POSTPROCESSING
# ═══════════════════════════════════════════
def load_dicom_tensor(path):
    """Read DICOM file and convert to Tensor, applying Rescale Slope and Intercept to get HU."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    return torch.from_numpy(arr)


def normalize(tensor, a_min=A_MIN, a_max=A_MAX):
    """Normalize values to the range [0, 1] to fit the model input requirements."""
    tensor = tensor.clamp(a_min, a_max)
    tensor = (tensor - a_min) / (a_max - a_min)
    return tensor


def denormalize_to_hu(tensor, a_min=A_MIN, a_max=A_MAX):
    """Convert values back from [0, 1] to the original Hounsfield Units (HU) range."""
    return tensor * (a_max - a_min) + a_min


# ═══════════════════════════════════════════
# DICOM SAVING HELPER
# ═══════════════════════════════════════════
def save_as_dicom(ref_dicom_path, output_path, denoised_hu_tensor, series_uid):
    """
    Take the enhanced Tensor and rewrite it into a new DICOM file
    using the original file to preserve the Metadata.
    """
    ds = pydicom.dcmread(ref_dicom_path)

    # Convert the Tensor to a 2D Numpy array
    denoised_hu = denoised_hu_tensor.squeeze().cpu().numpy()

    # Retrieve original pixel values based on the equation: HU = pixel * slope + intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    pixel_array = (denoised_hu - intercept) / slope

    # Revert the array to its original data type (usually int16 for CT scans)
    orig_dtype = ds.pixel_array.dtype

    # Ensure pixel values do not exceed the allowed data type boundaries
    dtype_min = np.iinfo(orig_dtype).min
    dtype_max = np.iinfo(orig_dtype).max
    pixel_array = np.clip(pixel_array, dtype_min, dtype_max).astype(orig_dtype)

    # Update pixel data in the DICOM object
    ds.PixelData = pixel_array.tobytes()

    # Create a new DICOM Series
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = "Denoised (AI)"

    if hasattr(ds, "SeriesNumber"):
        ds.SeriesNumber = int(ds.SeriesNumber) + 1000
    else:
        ds.SeriesNumber = 1000

    # Create a unique SOP UID for each slice
    ds.SOPInstanceUID = generate_uid()

    if hasattr(ds, "file_meta"):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    # Save the file
    ds.save_as(output_path)

# ═══════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════
@torch.no_grad()
def process_patient(pid, patient_dir, output_dir, model, device):
    """Process all slices for a patient and save them."""
    low_dir = patient_dir / "Low_Dose"
    out_patient_dir = output_dir / pid / "Denoised_AI"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    low_imgs = sort_by_instance_number(glob(str(low_dir / "*.dcm")))
    n = len(low_imgs)
    
    if n == 0:
        print(f"  ⚠ No images found for patient {pid}.")
        return
    
    series_uid = generate_uid()

    for i in tqdm(range(n), desc=f"  Processing Patient [{pid}]", leave=False, unit="slice"):
        # 2.5D technique (previous, current, and next slice)
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, n - 1)

        # Load and normalize slices
        t_prev = normalize(load_dicom_tensor(low_imgs[prev_i]))
        t_curr = normalize(load_dicom_tensor(low_imgs[i]))
        t_next = normalize(load_dicom_tensor(low_imgs[next_i]))

        # Prepare input for the model [1, 3, H, W]
        inp = torch.stack([t_prev, t_curr, t_next], dim=0).unsqueeze(0).to(device)
        mid = inp[:, 1:2, :, :]

        # Apply model to get the enhanced slice (output is between 0 and 1)
        with autocast():
            pred_res = model(inp)
            pred_normalized = torch.clamp(mid + pred_res, 0.0, 1.0)

        # Revert values back to Hounsfield Units
        pred_hu = denormalize_to_hu(pred_normalized)

        # Prepare save path (using the same original filename)
        filename = os.path.basename(low_imgs[i])
        out_path = out_patient_dir / filename

        # Save the file
        save_as_dicom(low_imgs[i], out_path, pred_hu, series_uid)


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Apply LDCT model and save results as DICOM files.")
    parser.add_argument("--model", type=str, default=BEST_MODEL_PATH, help="Path to the trained model weights (.pt)")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR, help="Path to the test directory")
    parser.add_argument("--output-dir", type=str, default="Output_DICOM", help="Path to save enhanced DICOM files")
    args = parser.parse_args()

    setup_reproducibility()
    device = get_device()
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Model ──
    print(f"\n📂 Loading model from: {args.model}")
    model = build_model(device)
    state = torch.load(args.model, map_location=device)

    # Handle DataParallel state dict if the model was trained on multiple GPUs
    if isinstance(state, dict) and "module." in list(state.keys())[0]:
        state = {k.replace("module.", ""): v for k, v in state.items()}
        
    model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded successfully.\n")

    # ── 2. Fetch patients from test directory ──
    test_path = Path(args.test_dir)
    patients = sorted([p for p in test_path.iterdir() if p.is_dir() and (p / "Low_Dose").exists()])

    if not patients:
        print(f"❌ No data found in {args.test_dir}.")
        return

    print(f"🔍 Found {len(patients)} patients, starting processing...\n")

    # ── 3. Processing and exporting results ──
    for patient_dir in patients:
        pid = patient_dir.name
        process_patient(pid, patient_dir, output_base_dir, model, device)

    print("\n" + "=" * 60)
    print(f"🎉 Process completed successfully! Enhanced DICOM files saved in '{args.output_dir}' directory.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

"""
LDCT Project — Evaluation Script (Full Image Resolution & ldct-benchmark Physics Standard)
==========================================================================================
All parameters, paths, clinical windows, and evaluation settings are imported from `config.py`.
Runs the trained model on the `test/` folder using FULL original resolution
without any cropping or padding.

Calculates physically & clinically accurate metrics using the exact ldct-benchmark standard:
- RMSE: Measured in physical Hounsfield Units (HU) clipped to [0, 2924] (HU + 1024 offset).
- PSNR & SSIM: Measured on Clinical Diagnostic Windows (Lung Window for Chest, Soft Tissue Window for Abdomen).
- VIF: Measured on physical HU scale.

Usage:
    python evaluate.py
    python evaluate.py --model FinalCT_2.5D-UNET-DATASET/best_model.pt 
    python evaluate.py --save-images
"""

import os
import argparse
from pathlib import Path
from glob import glob

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import (
    TEST_DIR, BEST_MODEL_PATH, EVAL_OUTPUT_DIR,
    A_MIN, A_MAX, B_MIN, B_MAX,
    WINDOW_LUNG_CENTER, WINDOW_LUNG_WIDTH,
    WINDOW_SOFT_CENTER, WINDOW_SOFT_WIDTH,
)
from utils import setup_reproducibility, get_device, sort_by_instance_number, build_multi_window_input
from model import build_model
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed,
    compute_rmse_hu, compute_vif_hu,
    denormalize_to_hu_offset, apply_center_width, CW
)

import pydicom


# ═══════════════════════════════════════════
# DICOM LOADER & NORMALIZATION
# ═══════════════════════════════════════════
def load_dicom_tensor(path):
    """Read one DICOM file and return a float32 tensor in HU."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply RescaleSlope / RescaleIntercept if available
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    return torch.from_numpy(arr)


def normalize(tensor, a_min=A_MIN, a_max=A_MAX):
    """Clip and scale HU → [0, 1]."""
    tensor = tensor.clamp(a_min, a_max)
    tensor = (tensor - a_min) / (a_max - a_min)
    return tensor


# ═══════════════════════════════════════════
# PATIENT EVALUATION
# ═══════════════════════════════════════════
@torch.no_grad()
def evaluate_patient(pid, patient_dir, model, device, save_images=False, output_dir=None):
    """
    Evaluate one patient — runs the model on every slice using FULL resolution
    and returns a dict with average benchmark-aligned metrics.
    """
    low_dir = patient_dir / "Low_Dose"
    full_dir = patient_dir / "Full_Dose"

    low_imgs = sort_by_instance_number(glob(str(low_dir / "*.dcm")))
    full_imgs = sort_by_instance_number(glob(str(full_dir / "*.dcm")))

    assert len(low_imgs) == len(full_imgs), \
        f"[{pid}] Mismatch: {len(low_imgs)} low vs {len(full_imgs)} full"

    n = len(low_imgs)
    body_type = "Chest" if pid[0].upper() == "C" else "Abdomen"

    psnr_scores, ssim_scores, rmse_scores, vif_scores = [], [], [], []
    baseline_psnr_scores = []

    # Save one visualization per patient (middle slice)
    viz_slice_idx = n // 2
    viz_triplet = None

    for i in tqdm(range(n), desc=f"  [{pid}]", leave=False, unit="slice"):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, n - 1)

        # Load raw HU tensors
        raw_prev = load_dicom_tensor(low_imgs[prev_i])
        raw_curr = load_dicom_tensor(low_imgs[i])
        raw_next = load_dicom_tensor(low_imgs[next_i])
        raw_full = load_dicom_tensor(full_imgs[i])

        # Build 9-channel Multi-Window input [1, 9, H, W]
        inp = build_multi_window_input(
            raw_prev, raw_curr, raw_next,
            a_min=A_MIN, a_max=A_MAX,
            lung_center=WINDOW_LUNG_CENTER, lung_width=WINDOW_LUNG_WIDTH,
            soft_center=WINDOW_SOFT_CENTER, soft_width=WINDOW_SOFT_WIDTH
        ).to(device)

        lbl = normalize(raw_full).unsqueeze(0).unsqueeze(0).to(device)
        mid = inp[:, 1:2, :, :]                              # current low-dose slice in full HU range [0, 1]

        with autocast():
            pred_res = model(inp)
            pred = torch.clamp(mid + pred_res, 0.0, 1.0)

        # ── 1. Convert tensors to HU + 1024 offset domain (ldct-benchmark standard) ──
        pred_hu_offset = denormalize_to_hu_offset(pred.squeeze(), A_MIN, A_MAX)
        lbl_hu_offset  = denormalize_to_hu_offset(lbl.squeeze(),  A_MIN, A_MAX)
        mid_hu_offset  = denormalize_to_hu_offset(mid.squeeze(),  A_MIN, A_MAX)

        # ── 2. Compute ldct-benchmark metrics ──
        p_val = compute_psnr_windowed(pred_hu_offset, lbl_hu_offset, body_type)
        b_val = compute_psnr_windowed(mid_hu_offset,  lbl_hu_offset, body_type)
        s_val = compute_ssim_windowed(pred_hu_offset, lbl_hu_offset, body_type)
        r_val = compute_rmse_hu(pred_hu_offset, lbl_hu_offset)
        v_val = compute_vif_hu(pred_hu_offset, lbl_hu_offset)

        psnr_scores.append(p_val)
        ssim_scores.append(s_val)
        rmse_scores.append(r_val)
        vif_scores.append(v_val)
        baseline_psnr_scores.append(b_val)

        if i == viz_slice_idx:
            center, width = CW.get(body_type, CW["Abdomen"])
            viz_triplet = (
                apply_center_width(mid_hu_offset, center, width),
                apply_center_width(lbl_hu_offset, center, width),
                apply_center_width(pred_hu_offset, center, width),
            )

    avg = lambda lst: sum(lst) / max(len(lst), 1)

    result = {
        "PatientID":      pid,
        "BodyType":       body_type,
        "NumSlices":      n,
        "PSNR":           round(avg(psnr_scores), 4),
        "Baseline_PSNR":  round(avg(baseline_psnr_scores), 4),
        "Delta_PSNR":     round(avg(psnr_scores) - avg(baseline_psnr_scores), 4),
        "SSIM":           round(avg(ssim_scores), 4),
        "RMSE_HU":        round(avg(rmse_scores), 4),
        "VIF":            round(avg(vif_scores), 4),
    }

    # Save visualization
    if save_images and viz_triplet is not None and output_dir is not None:
        save_patient_viz(pid, body_type, viz_triplet, result, output_dir)

    return result


# ═══════════════════════════════════════════
# VISUALIZATION HELPER
# ═══════════════════════════════════════════
def save_patient_viz(pid, body_type, viz_triplet, metrics, output_dir):
    """Save a side-by-side triplet: LDCT | NDCT | Denoised (using clinical window)."""
    ldct, ndct, denoised = viz_triplet
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    window_name = "Lung Window (C=-600, W=1500)" if body_type == "Chest" else "Soft Tissue Window (C=50, W=400)"

    titles = [
        f"LDCT (Input)\nBaseline PSNR: {metrics['Baseline_PSNR']:.2f} dB",
        f"NDCT (Ground Truth)\n{window_name}",
        f"Denoised (Output)\nPSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f} | RMSE: {metrics['RMSE_HU']:.2f} HU",
    ]
    imgs = [ldct, ndct, denoised]
    cmap = "gray"

    for j, (img, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[j])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Patient: {pid} | Type: {body_type}", fontsize=12, fontweight="bold")
    out_path = output_dir / f"{pid}_viz.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════
# PRINT SUMMARY TABLE
# ═══════════════════════════════════════════
def print_summary(df):
    """Print per-body-type and overall average metrics."""
    print("\n" + "=" * 75)
    print("📊  EVALUATION RESULTS (ldct-benchmark Physical Standard)")
    print("=" * 75)
    print(f"\n{'Patient':<14} {'Type':<9} {'Slices':>6}  {'ΔPSNR':>8}  {'PSNR':>8}  {'SSIM':>8}  {'RMSE(HU)':>10}  {'VIF':>8}")
    print("-" * 75)

    for _, row in df.iterrows():
        print(
            f"{row['PatientID']:<14} {row['BodyType']:<9} {row['NumSlices']:>6}  "
            f"{row['Delta_PSNR']:>+8.2f}  {row['PSNR']:>8.2f}  "
            f"{row['SSIM']:>8.4f}  {row['RMSE_HU']:>10.2f}  {row['VIF']:>8.4f}"
        )

    print("=" * 75)

    for body_type in ["Chest", "Abdomen", "Overall"]:
        sub = df if body_type == "Overall" else df[df["BodyType"] == body_type]
        if sub.empty:
            continue
        label = f"  {body_type} avg " if body_type != "Overall" else "  Overall avg"
        print(
            f"\n{label:<20} "
            f"ΔPSNR: {sub['Delta_PSNR'].mean():>+6.2f} dB  |  "
            f"PSNR: {sub['PSNR'].mean():>6.2f} dB  |  "
            f"SSIM: {sub['SSIM'].mean():>6.4f}  |  "
            f"RMSE: {sub['RMSE_HU'].mean():>6.2f} HU  |  "
            f"VIF: {sub['VIF'].mean():>6.4f}"
        )

    print("=" * 75)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate LDCT denoising model on test set using ldct-benchmark physical metrics.")
    parser.add_argument("--model", type=str, default=BEST_MODEL_PATH,
                        help="Path to the trained model weights (.pt file)")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR,
                        help="Path to test patients directory")
    parser.add_argument("--save-images", action="store_true",
                        help="Save sample LDCT/NDCT/Denoised triplet images")
    parser.add_argument("--output", type=str, default=EVAL_OUTPUT_DIR,
                        help="Output folder for CSV report and images")
    args = parser.parse_args()

    # ── Setup ──
    setup_reproducibility()
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"\n📂  Loading model from: {args.model}")
    model = build_model(device)
    state = torch.load(args.model, map_location=device)

    # Handle DataParallel wrapper
    if isinstance(state, dict) and "module." in list(state.keys())[0]:
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("✅  Model loaded successfully.\n")

    # ── Discover test patients ──
    test_path = Path(args.test_dir)
    patients = sorted([
        p for p in test_path.iterdir()
        if p.is_dir()
        and (p / "Low_Dose").exists()
        and (p / "Full_Dose").exists()
    ])

    if not patients:
        print(f"❌  No valid patients found in '{args.test_dir}'. "
              "Each patient folder must contain 'Low_Dose/' and 'Full_Dose/' subdirectories.")
        return

    chest_patients  = [p for p in patients if p.name[0].upper() == "C"]
    abdomen_patients = [p for p in patients if p.name[0].upper() == "L"]

    print(f"🔍  Found {len(patients)} patients: "
          f"{len(chest_patients)} Chest, {len(abdomen_patients)} Abdomen\n")

    # ── Evaluate ──
    all_results = []
    for patient_dir in patients:
        pid = patient_dir.name
        print(f"⚙️   Evaluating [{pid}] ({'Chest' if pid[0].upper() == 'C' else 'Abdomen'}) ...")
        try:
            result = evaluate_patient(
                pid, patient_dir, model, device,
                save_images=args.save_images,
                output_dir=output_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    if not all_results:
        print("❌  No results collected.")
        return

    # ── Report ──
    df = pd.DataFrame(all_results)
    df = df.sort_values(["BodyType", "PatientID"])

    csv_path = output_dir / "evaluation_report.csv"
    df.to_csv(csv_path, index=False)

    print_summary(df)
    print(f"\n📄  Full report saved → {csv_path}")
    if args.save_images:
        print(f"🖼️   Images saved     → {output_dir}/")


if __name__ == "__main__":
    main()

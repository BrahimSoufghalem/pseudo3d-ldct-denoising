"""
LDCT Project — Benchmark Models Evaluation & Comparison Script
================================================================
Evaluates state-of-the-art benchmark models from `ldct-benchmark`
(RED-CNN, WGAN-VGG, DU-GAN, TransCT, QAE, ResNet, CNN10) on your `test/` dataset.

All metrics (PSNR, SSIM, RMSE, VIF) are computed using YOUR exact normalization
and evaluation formulas, enabling a 100% fair side-by-side comparison.

Usage:
    python evaluate_benchmark_models.py
    python evaluate_benchmark_models.py --models redcnn wganvgg transct
"""

import os
import argparse
from pathlib import Path
from glob import glob

import torch
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from monai.metrics import SSIMMetric
from tqdm import tqdm
import pydicom

from config import (
    TEST_DIR, BEST_MODEL_PATH,
    A_MIN, A_MAX,
)
from utils import setup_reproducibility, get_device, sort_by_instance_number
from model import build_model
from metrics import psnr, rmse, VIFMetric

# Import benchmark loader
from ldctbench.hub import load_model as load_benchmark_model
from ldctbench.evaluate.utils import DATA_INFO


# ═══════════════════════════════════════════
# PREPROCESSING HELPER FOR BENCHMARK MODELS
# ═══════════════════════════════════════════
MEAN_HU_OFFSET = float(DATA_INFO["mean"])
STD_HU_OFFSET  = float(DATA_INFO["std"])

def load_dicom_raw_hu(path):
    """Read DICOM and return raw Hounsfield Units (HU)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def hu_to_user_norm(hu_tensor):
    """Convert raw HU tensor to user's [0, 1] normalized scale."""
    tensor = hu_tensor.clamp(A_MIN, A_MAX)
    return (tensor - A_MIN) / (A_MAX - A_MIN)


@torch.no_grad()
def run_benchmark_model_slice(model, slice_hu_tensor, device):
    """
    Run 2D single-slice benchmark model on raw HU tensor.
    `slice_hu_tensor` is in HU scale.
    Returns predicted slice in user's [0, 1] normalized domain.
    """
    # ldctbench models expect input as (HU + 1024) normalized by (mean, std)
    hu_offset = slice_hu_tensor + 1024.0
    inp_norm = (hu_offset - MEAN_HU_OFFSET) / STD_HU_OFFSET
    inp_t = inp_norm.unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, H, W]

    with autocast():
        out_norm = model(inp_t)                             # [1, 1, H, W]

    # Denormalize output back to HU
    out_hu_offset = out_norm.squeeze() * STD_HU_OFFSET + MEAN_HU_OFFSET
    out_hu = out_hu_offset - 1024.0

    # Scale to user's [0, 1] range
    return hu_to_user_norm(out_hu).to(device)


# ═══════════════════════════════════════════
# EVALUATE ONE PATIENT FOR A BENCHMARK MODEL
# ═══════════════════════════════════════════
@torch.no_grad()
def evaluate_patient_benchmark(model, model_name, pid, patient_dir, device):
    low_dir = patient_dir / "Low_Dose"
    full_dir = patient_dir / "Full_Dose"

    low_imgs = sort_by_instance_number(glob(str(low_dir / "*.dcm")))
    full_imgs = sort_by_instance_number(glob(str(full_dir / "*.dcm")))
    n = len(low_imgs)
    body_type = "Chest" if pid[0].upper() == "C" else "Abdomen"

    psnr_scores, ssim_scores, rmse_scores, vif_scores = [], [], [], []
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0, reduction="mean")

    for i in range(n):
        low_hu = torch.from_numpy(load_dicom_raw_hu(low_imgs[i]))
        full_hu = torch.from_numpy(load_dicom_raw_hu(full_imgs[i]))

        # User normalized target [1, 1, H, W]
        lbl = hu_to_user_norm(full_hu).unsqueeze(0).unsqueeze(0).to(device)

        # Predicted slice in user [1, 1, H, W] domain
        pred_2d = run_benchmark_model_slice(model, low_hu, device)
        pred = pred_2d.unsqueeze(0).unsqueeze(0)

        p_val = psnr(pred.float(), lbl.float()).item()
        r_val = rmse(pred.float(), lbl.float()).item()

        ssim_metric.reset()
        ssim_metric(pred.float(), lbl.float())
        s_val = ssim_metric.aggregate().item()

        vif_m = VIFMetric(device=device)
        vif_m.update(pred.float(), lbl.float())
        v_val = vif_m.aggregate()

        psnr_scores.append(p_val)
        ssim_scores.append(s_val)
        rmse_scores.append(r_val)
        vif_scores.append(v_val)

    avg = lambda lst: sum(lst) / max(len(lst), 1)

    return {
        "Model":          model_name,
        "PatientID":      pid,
        "BodyType":       body_type,
        "NumSlices":      n,
        "PSNR":           round(avg(psnr_scores), 4),
        "SSIM":           round(avg(ssim_scores), 4),
        "RMSE":           round(avg(rmse_scores), 6),
        "VIF":            round(avg(vif_scores), 4),
    }


# ═══════════════════════════════════════════
# EVALUATE USER'S PSEUDO-3D MODEL
# ═══════════════════════════════════════════
@torch.no_grad()
def evaluate_patient_user_model(user_model, pid, patient_dir, device):
    low_dir = patient_dir / "Low_Dose"
    full_dir = patient_dir / "Full_Dose"

    low_imgs = sort_by_instance_number(glob(str(low_dir / "*.dcm")))
    full_imgs = sort_by_instance_number(glob(str(full_dir / "*.dcm")))
    n = len(low_imgs)
    body_type = "Chest" if pid[0].upper() == "C" else "Abdomen"

    psnr_scores, ssim_scores, rmse_scores, vif_scores = [], [], [], []
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0, reduction="mean")

    for i in range(n):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, n - 1)

        t_prev = hu_to_user_norm(torch.from_numpy(load_dicom_raw_hu(low_imgs[prev_i])))
        t_curr = hu_to_user_norm(torch.from_numpy(load_dicom_raw_hu(low_imgs[i])))
        t_next = hu_to_user_norm(torch.from_numpy(load_dicom_raw_hu(low_imgs[next_i])))
        t_full = hu_to_user_norm(torch.from_numpy(load_dicom_raw_hu(full_imgs[i])))

        inp = torch.stack([t_prev, t_curr, t_next], dim=0).unsqueeze(0).to(device)
        lbl = t_full.unsqueeze(0).unsqueeze(0).to(device)
        mid = inp[:, 1:2, :, :]

        with autocast():
            pred_res = user_model(inp)
            pred = torch.clamp(mid + pred_res, 0.0, 1.0)

        p_val = psnr(pred.float(), lbl.float()).item()
        r_val = rmse(pred.float(), lbl.float()).item()

        ssim_metric.reset()
        ssim_metric(pred.float(), lbl.float())
        s_val = ssim_metric.aggregate().item()

        vif_m = VIFMetric(device=device)
        vif_m.update(pred.float(), lbl.float())
        v_val = vif_m.aggregate()

        psnr_scores.append(p_val)
        ssim_scores.append(s_val)
        rmse_scores.append(r_val)
        vif_scores.append(v_val)

    avg = lambda lst: sum(lst) / max(len(lst), 1)

    return {
        "Model":          "Pseudo-3D UNet (Ours)",
        "PatientID":      pid,
        "BodyType":       body_type,
        "NumSlices":      n,
        "PSNR":           round(avg(psnr_scores), 4),
        "SSIM":           round(avg(ssim_scores), 4),
        "RMSE":           round(avg(rmse_scores), 6),
        "VIF":            round(avg(vif_scores), 4),
    }


# ═══════════════════════════════════════════
# MAIN COMPARISON LOOP
# ═══════════════════════════════════════════
BENCHMARK_MODELS = ["redcnn", "wganvgg", "dugan", "transct", "qae", "resnet", "cnn10"]

def main():
    parser = argparse.ArgumentParser(description="Compare Benchmark Models with Pseudo-3D UNet using your metrics.")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR, help="Path to test directory")
    parser.add_argument("--user-model", type=str, default=BEST_MODEL_PATH, help="Path to your best_model.pt")
    parser.add_argument("--models", nargs="+", default=BENCHMARK_MODELS, help="List of benchmark models to test")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    args = parser.parse_args()

    setup_reproducibility()
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = Path(args.test_dir)
    patients = sorted([
        p for p in test_path.iterdir()
        if p.is_dir() and (p / "Low_Dose").exists() and (p / "Full_Dose").exists()
    ])

    if not patients:
        print(f"❌  No test patients found in '{args.test_dir}'.")
        return

    print(f"🔍  Found {len(patients)} test patients across {len(args.models)} benchmark models + Ours.\n")

    all_rows = []

    # 1. Evaluate User's Model first if weights exist
    if os.path.exists(args.user_model):
        print(f"⚙️   [1/1] Evaluating Pseudo-3D UNet (Ours) ...")
        user_net = build_model(device)
        state = torch.load(args.user_model, map_location=device)
        if isinstance(state, dict) and "module." in list(state.keys())[0]:
            state = {k.replace("module.", ""): v for k, v in state.items()}
        user_net.load_state_dict(state)
        user_net.eval()

        for p in patients:
            res = evaluate_patient_user_model(user_net, p.name, p, device)
            all_rows.append(res)
    else:
        print(f"⚠️   User model checkpoint not found at {args.user_model}, skipping user model.")

    # 2. Evaluate Benchmark Models
    for idx, model_key in enumerate(args.models, start=1):
        print(f"\n⚙️   [{idx}/{len(args.models)}] Loading benchmark model: '{model_key}' ...")
        try:
            bench_net = load_benchmark_model(model_key, device=device)
            bench_net.eval()

            for p in tqdm(patients, desc=f"  Evaluating {model_key}"):
                res = evaluate_patient_benchmark(bench_net, f"ldctbench-{model_key.upper()}", p.name, p, device)
                all_rows.append(res)
        except Exception as e:
            print(f"❌  Failed evaluating benchmark model '{model_key}': {e}")

    if not all_rows:
        print("❌  No evaluation results collected.")
        return

    # ── Export & Summary ──
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "benchmark_models_comparison.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("🏆  MODEL COMPARISON SUMMARY TABLE (User Metric Standard)")
    print("=" * 80)
    
    summary_df = df.groupby("Model")[["PSNR", "SSIM", "RMSE", "VIF"]].mean().reset_index()
    summary_df = summary_df.sort_values(by="PSNR", ascending=False)

    print(f"{'Model Name':<28} {'PSNR (dB) ↑':>12} {'SSIM ↑':>12} {'RMSE ↓':>12} {'VIF ↑':>12}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        name_str = f"⭐ {row['Model']}" if "Ours" in row['Model'] else row['Model']
        print(f"{name_str:<28} {row['PSNR']:>12.2f} {row['SSIM']:>12.4f} {row['RMSE']:>12.6f} {row['VIF']:>12.4f}")
    print("=" * 80)
    print(f"\n📄  Full detailed report saved → {csv_path}")


if __name__ == "__main__":
    main()

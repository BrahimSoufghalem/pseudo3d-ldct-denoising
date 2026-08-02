"""Evaluate and compare RED-CNN, ResNet, and LocalResidual on the 20-patient test set.

Usage
-----
    HU_RANGE_PRESET=benchmark python evaluate_20p.py

Requires that models have been trained first via train_20p.py:
    runs_20p/redcnn/best_model.pt
    runs_20p/resnet/best_model.pt
    runs_20p/local_residual/best_model.pt
"""

import argparse
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
from benchmark_architectures import RedCNN, ResNet
from evaluate import load_dicom_tensor, print_summary
from local_residual_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, standardize_hu,
)
from local_residual_model import build_local_residual_model
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed,
    compute_rmse_hu, compute_vif_hu,
)
from twenty_patient_split import TEST_20P
from utils import setup_reproducibility, get_device, sort_by_instance_number


ARCH_MAP = {
    "redcnn": "RED-CNN",
    "resnet": "ResNet",
    "local_residual": "LocalResidual (Ours)",
}


def load_checkpoint(path, arch: str, device):
    state = torch.load(path, map_location=device, weights_only=False)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    if abs(float(meta.get("pixel_mean", BENCHMARK_PIXEL_MEAN)) - BENCHMARK_PIXEL_MEAN) > 1e-6:
        raise RuntimeError(f"[{arch}] pixel_mean mismatch in checkpoint")
    if abs(float(meta.get("pixel_std", BENCHMARK_PIXEL_STD)) - BENCHMARK_PIXEL_STD) > 1e-6:
        raise RuntimeError(f"[{arch}] pixel_std mismatch in checkpoint")
    if arch == "redcnn":
        model = RedCNN().to(device)
    elif arch == "resnet":
        model = ResNet().to(device)
    elif arch == "local_residual":
        model = build_local_residual_model(device, channels=128, blocks=10, groups=8)
    else:
        raise ValueError(arch)
    weights = state.get("model_state_dict", state)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_patient(pid, patient_dir, model, device):
    low = sort_by_instance_number(glob(str(patient_dir / "Low_Dose" / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")
    body = "Chest" if pid.upper().startswith("C") else "Abdomen"
    scores = {k: [] for k in ("psnr", "ssim", "rmse", "vif",
                               "base_psnr", "base_ssim", "base_rmse", "base_vif")}
    for low_path, full_path in tqdm(zip(low, full), total=len(low), desc=f"  {pid}", leave=False):
        low_hu = load_dicom_tensor(low_path).to(device)
        full_hu = load_dicom_tensor(full_path).to(device)
        x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        pred_z = model(x)
        pred_px = denormalize_to_pixel(pred_z.squeeze()).clamp(0.0, cfg.EVAL_DATA_RANGE)
        full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
        low_px  = (low_hu  + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
        scores["psnr"].append(compute_psnr_windowed(pred_px, full_px, body))
        scores["ssim"].append(compute_ssim_windowed(pred_px, full_px, body))
        scores["rmse"].append(compute_rmse_hu(pred_px, full_px))
        scores["vif"].append(compute_vif_hu(pred_px, full_px))
        scores["base_psnr"].append(compute_psnr_windowed(low_px, full_px, body))
        scores["base_ssim"].append(compute_ssim_windowed(low_px, full_px, body))
        scores["base_rmse"].append(compute_rmse_hu(low_px, full_px))
        scores["base_vif"].append(compute_vif_hu(low_px, full_px))
    avg = lambda k: sum(scores[k]) / max(1, len(scores[k]))
    return {
        "PatientID": pid, "BodyType": body, "NumSlices": len(low),
        "PSNR":          avg("psnr"),
        "SSIM":          avg("ssim"),
        "RMSE_HU":       avg("rmse"),
        "VIF":           avg("vif"),
        "Baseline_PSNR": avg("base_psnr"),
        "Baseline_VIF":  avg("base_vif"),
        "Delta_PSNR":    avg("psnr") - avg("base_psnr"),
        "Delta_SSIM":    avg("ssim") - avg("base_ssim"),
        "Delta_VIF":     avg("vif")  - avg("base_vif"),
    }


def print_comparison(all_dfs: dict):
    print("\n" + "=" * 72)
    print("  20-PATIENT FAIR COMPARISON")
    print("  (benchmark mean/std | same data | same protocol | MSE loss)")
    print("=" * 72)
    metrics = ["PSNR", "SSIM", "RMSE_HU", "VIF"]
    header = f"  {'Model':<24}" + "".join(f"{m:>11}" for m in metrics)
    for body in ["Chest", "Abdomen", "Overall"]:
        print(f"\n  [{body.upper()}]")
        print(header)
        print("  " + "-" * 68)
        for arch, df in all_dfs.items():
            sub = df if body == "Overall" else df[df["BodyType"] == body]
            if sub.empty:
                continue
            row = sub[metrics].mean()
            label = ARCH_MAP.get(arch, arch)
            marker = "* " if arch == "local_residual" else "  "
            print(f"  {marker}{label:<24}" + "".join(f"{row[m]:>11.4f}" for m in metrics))
        print("  " + "-" * 68)
    print("\n  * = Our model (LocalResidual)")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs_20p")
    p.add_argument("--test-dir", default=cfg.TEST_DIR)
    p.add_argument("--output", default="eval_20p")
    args = p.parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Run with HU_RANGE_PRESET=benchmark")
    setup_reproducibility()
    device = get_device()
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    test_patients = sorted([
        p for p in Path(args.test_dir).iterdir()
        if p.is_dir() and p.name in TEST_20P
        and (p / "Low_Dose").exists() and (p / "Full_Dose").exists()
    ])
    print(f"Test patients: {[p.name for p in test_patients]}")
    all_dfs = {}
    for arch in ["redcnn", "resnet", "local_residual"]:
        ckpt = Path(args.runs_root) / arch / "best_model.pt"
        if not ckpt.exists():
            print(f"  Skipping {arch}: {ckpt} not found")
            continue
        print(f"\nEvaluating {ARCH_MAP[arch]} ...")
        model = load_checkpoint(str(ckpt), arch, device)
        rows = [evaluate_patient(p.name, p, model, device) for p in test_patients]
        df = pd.DataFrame(rows)
        df["Model"] = ARCH_MAP[arch]
        all_dfs[arch] = df
        df.to_csv(out_path / f"{arch}_results.csv", index=False)
    if not all_dfs:
        print("No checkpoints found. Train first with train_20p.py.")
        return
    print_comparison(all_dfs)
    pd.concat(all_dfs.values(), ignore_index=True).to_csv(
        out_path / "comparison_20p.csv", index=False
    )
    print(f"\nFull report -> {out_path / 'comparison_20p.csv'}")


if __name__ == "__main__":
    main()

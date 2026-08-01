"""Full-resolution evaluation of the mean/std local residual control."""

import argparse
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
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
from utils import setup_reproducibility, get_device, load_state_into, sort_by_instance_number


@torch.no_grad()
def evaluate_patient_local(pid, patient_dir, model, device):
    low = sort_by_instance_number(glob(str(patient_dir / "Low_Dose" / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")
    body = "Chest" if pid.upper().startswith("C") else "Abdomen"
    scores = {k: [] for k in (
        "psnr", "ssim", "rmse", "vif",
        "base_psnr", "base_ssim", "base_rmse", "base_vif",
    )}
    for low_path, full_path in tqdm(
        zip(low, full), total=len(low), desc=f"  [{pid}]", leave=False,
    ):
        low_hu = load_dicom_tensor(low_path).to(device)
        full_hu = load_dicom_tensor(full_path).to(device)
        x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        pred_z = model(x)
        pred_px = denormalize_to_pixel(pred_z.squeeze()).clamp(0.0, cfg.EVAL_DATA_RANGE)
        low_px = (low_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
        full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)

        scores["psnr"].append(compute_psnr_windowed(pred_px, full_px, body))
        scores["ssim"].append(compute_ssim_windowed(pred_px, full_px, body))
        scores["rmse"].append(compute_rmse_hu(pred_px, full_px))
        scores["vif"].append(compute_vif_hu(pred_px, full_px))
        scores["base_psnr"].append(compute_psnr_windowed(low_px, full_px, body))
        scores["base_ssim"].append(compute_ssim_windowed(low_px, full_px, body))
        scores["base_rmse"].append(compute_rmse_hu(low_px, full_px))
        scores["base_vif"].append(compute_vif_hu(low_px, full_px))

    avg = lambda key: sum(scores[key]) / max(1, len(scores[key]))
    return {
        "PatientID": pid,
        "BodyType": body,
        "NumSlices": len(low),
        "Blend": 1.0,
        "PSNR": avg("psnr"),
        "Baseline_PSNR": avg("base_psnr"),
        "Delta_PSNR": avg("psnr") - avg("base_psnr"),
        "SSIM": avg("ssim"),
        "Baseline_SSIM": avg("base_ssim"),
        "Delta_SSIM": avg("ssim") - avg("base_ssim"),
        "RMSE_HU": avg("rmse"),
        "Baseline_RMSE_HU": avg("base_rmse"),
        "Delta_RMSE_HU": avg("base_rmse") - avg("rmse"),
        "VIF": avg("vif"),
        "Baseline_VIF": avg("base_vif"),
        "Delta_VIF": avg("vif") - avg("base_vif"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test-dir", default=cfg.TEST_DIR)
    p.add_argument("--output", default="eval_local_residual")
    args = p.parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Run evaluation with HU_RANGE_PRESET=benchmark")

    setup_reproducibility()
    device = get_device()
    try:
        state = torch.load(args.model, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    if meta.get("normalization") != "benchmark_meanstd":
        raise RuntimeError("Checkpoint is not marked as benchmark mean/std trained")
    if abs(float(meta.get("pixel_mean", 0)) - BENCHMARK_PIXEL_MEAN) > 1e-9:
        raise RuntimeError("Checkpoint mean does not match the benchmark constant")
    if abs(float(meta.get("pixel_std", 0)) - BENCHMARK_PIXEL_STD) > 1e-9:
        raise RuntimeError("Checkpoint std does not match the benchmark constant")

    model = build_local_residual_model(device, **meta.get("model_config", {}))
    load_state_into(model, state)
    model.eval()
    patients = sorted([
        p for p in Path(args.test_dir).iterdir()
        if p.is_dir() and (p / "Low_Dose").exists() and (p / "Full_Dose").exists()
    ])
    print(f"Checkpoint       : {args.model}")
    print("Normalization    : benchmark mean/std in un-clipped HU+1024 domain")
    print(f"Mean / std       : {BENCHMARK_PIXEL_MEAN:.12f} / {BENCHMARK_PIXEL_STD:.12f}")
    print(f"Evaluation range : [0, {cfg.EVAL_DATA_RANGE:g}] after model inference")
    print(f"Patients         : {len(patients)}")

    rows = [evaluate_patient_local(p.name, p, model, device) for p in patients]
    df = pd.DataFrame(rows).sort_values(["BodyType", "PatientID"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "evaluation_report_local_residual_meanstd.csv"
    df.to_csv(csv_path, index=False)
    print_summary(df)
    print(f"\nFull report saved -> {csv_path}")


if __name__ == "__main__":
    main()

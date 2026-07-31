"""Evaluate PhysicsSpectralNet with the exact ldct-benchmark contract."""

import argparse
from pathlib import Path

import pandas as pd
import torch

import config as cfg
from evaluate import evaluate_patient, print_summary
from physics_spectral_model import build_physics_model
from utils import setup_reproducibility, get_device, load_state_into


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test-dir", default=cfg.TEST_DIR)
    p.add_argument("--output", default="eval_physics")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--allow-preset-mismatch", action="store_true")
    args = p.parse_args()

    setup_reproducibility()
    device = get_device()
    try:
        state = torch.load(args.model, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    trained_preset = meta.get("hu_preset")
    if trained_preset and trained_preset != cfg.HU_RANGE_PRESET and not args.allow_preset_mismatch:
        raise RuntimeError(
            f"HU preset mismatch: checkpoint was trained with '{trained_preset}', but evaluation "
            f"is running with '{cfg.HU_RANGE_PRESET}'. Set HU_RANGE_PRESET={trained_preset} or "
            "pass --allow-preset-mismatch only for an intentional stress test."
        )
    model_cfg = dict(meta.get("model_config", {}))
    model = build_physics_model(device, **model_cfg)
    load_state_into(model, state)
    model.eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    patients = sorted([
        pth for pth in Path(args.test_dir).iterdir()
        if pth.is_dir() and (pth / "Low_Dose").exists() and (pth / "Full_Dose").exists()
    ])
    print(f"Checkpoint: {args.model}")
    print(f"HU range: [{cfg.A_MIN}, {cfg.A_MAX}] (preset '{cfg.HU_RANGE_PRESET}')")
    print(f"Benchmark contract: {meta.get('benchmark_contract', 'not recorded')}")
    print(f"Found {len(patients)} patients")

    rows = []
    for patient in patients:
        rows.extend(evaluate_patient(
            patient.name, patient, model, device,
            input_mode="2d", save_images=args.save_images,
            output_dir=output, use_amp=False, blends=(1.0,),
        ))
    df = pd.DataFrame(rows).sort_values(["BodyType", "PatientID"])
    csv_path = output / "evaluation_report_physics_spectral_2d.csv"
    df.to_csv(csv_path, index=False)
    print_summary(df)
    print(f"\nFull report saved -> {csv_path}")


if __name__ == "__main__":
    main()

"""Unified 20-patient trainer for RED-CNN, ResNet, and LocalResidual.

Usage
-----
# Train on Kaggle T4 (groups=1 default, fast):
    HU_RANGE_PRESET=benchmark python train_20p.py \
        --arch redcnn --data-dir /kaggle/input/...

    HU_RANGE_PRESET=benchmark python train_20p.py \
        --arch local_residual --data-dir /kaggle/input/...

# Resume if interrupted:
    ... same command + --resume

# On Blackwell/Ampere (restore groups=8):
    ... --groups 8
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from benchmark_architectures import build_benchmark_model
from local_residual_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, prepare_local_residual_data,
)
from local_residual_model import build_local_residual_model
from metrics import compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu
from twenty_patient_split import TRAIN_20P, VAL_20P
from utils import setup_reproducibility, get_device, get_state_dict


# ── Override the patient split ────────────────────────────────────────
cfg.EXPECTED_TRAIN = TRAIN_20P
cfg.EXPECTED_VAL = VAL_20P


def parse_args():
    p = argparse.ArgumentParser(description="20-patient fair comparison trainer")
    p.add_argument(
        "--arch", required=True,
        choices=["redcnn", "resnet", "local_residual"],
    )
    p.add_argument(
        "--data-dir", default=cfg.DATA_DIR,
        help="Root folder with train+val patient sub-dirs",
    )
    p.add_argument("--max-iterations", type=int, default=20_000)
    p.add_argument("--iterations-before-val", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--val-patch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--output-root", default="runs_20p")
    p.add_argument(
        "--groups", type=int, default=1,
        help="Groups for LocalResidual middle conv. "
             "Default=1 (fast on T4). Use 8 on Blackwell/Ampere.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoint.pt if it exists",
    )
    return p.parse_args()


def build_model(arch: str, device, groups: int = 1):
    if arch == "local_residual":
        return build_local_residual_model(device, channels=128, blocks=10, groups=groups)
    return build_benchmark_model(arch, device)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums = dict(mse=0.0, psnr=0.0, ssim=0.0, rmse=0.0, baseline_psnr=0.0)
    batches = samples = 0
    for batch in tqdm(loader, desc="  Val", leave=False, dynamic_ncols=True):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        pred = model(x)
        sums["mse"] += float(F.mse_loss(pred, y))
        batches += 1
        pred_px = denormalize_to_pixel(pred).clamp(0.0, cfg.EVAL_DATA_RANGE)
        y_px    = denormalize_to_pixel(y).clamp(0.0, cfg.EVAL_DATA_RANGE)
        x_px    = denormalize_to_pixel(x).clamp(0.0, cfg.EVAL_DATA_RANGE)
        body = batch.get("body_type", ["Abdomen"] * pred.shape[0])
        for i in range(pred.shape[0]):
            bt = "Chest" if str(body[i]).lower().startswith("c") else "Abdomen"
            sums["psnr"]          += compute_psnr_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["baseline_psnr"] += compute_psnr_windowed(x_px[i].squeeze(),  y_px[i].squeeze(), bt)
            sums["ssim"]          += compute_ssim_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["rmse"]          += compute_rmse_hu(pred_px[i].squeeze(),       y_px[i].squeeze())
            samples += 1
    n_b = max(1, batches)
    n_s = max(1, samples)
    return {
        "mse":   sums["mse"] / n_b,
        "psnr":  sums["psnr"] / n_s,
        "dpsnr": (sums["psnr"] - sums["baseline_psnr"]) / n_s,
        "ssim":  sums["ssim"] / n_s,
        "rmse":  sums["rmse"] / n_s,
    }


def train_cycle(model, loader, optimizer, device, iteration, max_iter):
    model.train()
    total = count = 0.0
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if iteration >= max_iter:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        iteration += 1
        total += float(loss.detach())
        count += 1
        bar.set_postfix(iter=iteration, mse=f"{loss.item():.6f}")
    return iteration, total / max(1, count)


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError(
            "Set HU_RANGE_PRESET=benchmark.\n"
            "Example: HU_RANGE_PRESET=benchmark python train_20p.py --arch redcnn"
        )

    setup_reproducibility()
    device = get_device()
    out_dir = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    print(f"\n{'='*60}")
    print(f"  20-patient experiment | arch={args.arch.upper()}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Output   : {out_dir}")
    if args.arch == "local_residual":
        print(f"  Groups   : {args.groups} {'(T4-optimised)' if args.groups == 1 else '(benchmark groups)'}")
    print(f"{'='*60}\n")

    model     = build_model(args.arch, device, groups=args.groups)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ── Resume ───────────────────────────────────────────────────
    iteration = 0
    best_ssim = -float("inf")
    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        iteration = int(ckpt.get("iteration", 0))
        best_ssim = float(ckpt.get("ssim", -float("inf")))
        print(f"  Resumed at iteration {iteration} | best SSIM {best_ssim:.5f}")
    elif args.resume:
        print(f"  --resume set but no checkpoint at {ckpt_path}. Starting fresh.")

    if iteration >= args.max_iterations:
        print(f"  Training already complete ({iteration}/{args.max_iterations}).")
        return

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader = prepare_local_residual_data(
        in_dir=args.data_dir,
        train_patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        iterations_before_val=args.iterations_before_val,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
    )

    print(f"Loss             : MSE")
    print(f"Optimizer        : Adam(lr={args.lr:.2e}, b1=0.9, b2=0.999)")
    print(f"Precision        : FP32")
    print(f"Checkpoint metric: validation SSIM")
    if iteration > 0:
        print(f"Starting from    : iteration {iteration}/{args.max_iterations}")

    start = time.time()
    cycle = iteration // args.iterations_before_val

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        iteration, train_mse = train_cycle(
            model, train_loader, optimizer, device, iteration, args.max_iterations
        )
        val = validate(model, val_loader, device)

        meta = {
            "architecture": args.arch,
            "groups": args.groups,
            "normalization": "benchmark_meanstd",
            "pixel_mean": BENCHMARK_PIXEL_MEAN,
            "pixel_std": BENCHMARK_PIXEL_STD,
            "pixel_domain": "HU+1024",
            "hu_preset": cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss": "MSE",
            "input_mode": "2d",
            "n_train_patients": len(TRAIN_20P),
            "n_val_patients": len(VAL_20P),
        }
        payload = {
            "model_state_dict": get_state_dict(model),
            "meta": meta,
            "iteration": iteration,
            "ssim": val["ssim"],
            "psnr": val["psnr"],
            "val_mse": val["mse"],
        }
        torch.save(payload, os.path.join(out_dir, "last_model.pt"))
        if val["ssim"] > best_ssim:
            best_ssim = val["ssim"]
            torch.save(payload, os.path.join(out_dir, "best_model.pt"))
        torch.save(
            {**payload, "optimizer_state": optimizer.state_dict()},
            ckpt_path,
        )

        elapsed = time.time() - t0
        print(
            f"Cycle {cycle:02d} | Iter {iteration:06d}/{args.max_iterations} | "
            f"Train MSE {train_mse:.6f} | Val MSE {val['mse']:.6f} | "
            f"PSNR {val['psnr']:.3f} | dPSNR {val['dpsnr']:+.3f} | "
            f"SSIM {val['ssim']:.5f} | RMSE {val['rmse']:.2f} | "
            f"{elapsed:.1f}s"
        )

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{args.arch.upper()}] in {total_time} | best val SSIM={best_ssim:.5f}")
    print(f"Checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()

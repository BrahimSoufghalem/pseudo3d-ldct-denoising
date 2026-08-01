"""Train the benchmark-aligned dense local residual control."""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from local_residual_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, prepare_local_residual_data,
)
from local_residual_model import build_local_residual_model
from metrics import compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu
from utils import setup_reproducibility, get_device, get_state_dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark-aligned local residual 2D control")
    p.add_argument("--max-iterations", type=int, default=20000)
    p.add_argument("--iterations-before-val", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--val-patch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--output-root", default="runs_local_residual")
    return p.parse_args()


def output_dir(args):
    return os.path.join(args.output_root, "2d_local_residual_meanstd")


def train_cycle(model, loader, optimizer, device, iteration, max_iterations):
    model.train()
    total = count = 0.0
    bar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if iteration >= max_iterations:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        iteration += 1
        total += float(loss.detach())
        count += 1
        bar.set_postfix(iter=iteration, mse=f"{loss.item():.6f}")
    return iteration, total / max(1, count)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums = dict(mse=0.0, psnr=0.0, ssim=0.0, rmse=0.0, baseline=0.0)
    batches = samples = 0
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        pred = model(x)
        sums["mse"] += float(F.mse_loss(pred, y))
        batches += 1
        pred_px = denormalize_to_pixel(pred).clamp(0.0, cfg.EVAL_DATA_RANGE)
        y_px = denormalize_to_pixel(y).clamp(0.0, cfg.EVAL_DATA_RANGE)
        x_px = denormalize_to_pixel(x).clamp(0.0, cfg.EVAL_DATA_RANGE)
        body = batch.get("body_type", ["Abdomen"] * pred.shape[0])
        for i in range(pred.shape[0]):
            bt = "Chest" if str(body[i]).lower().startswith("c") else "Abdomen"
            sums["psnr"] += compute_psnr_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["baseline"] += compute_psnr_windowed(x_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["ssim"] += compute_ssim_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["rmse"] += compute_rmse_hu(pred_px[i].squeeze(), y_px[i].squeeze())
            samples += 1
    return {
        "mse": sums["mse"] / max(1, batches),
        "psnr": sums["psnr"] / max(1, samples),
        "baseline": sums["baseline"] / max(1, samples),
        "ssim": sums["ssim"] / max(1, samples),
        "rmse": sums["rmse"] / max(1, samples),
    }


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Run this comparison with HU_RANGE_PRESET=benchmark")
    setup_reproducibility()
    device = get_device()
    root = output_dir(args)
    os.makedirs(root, exist_ok=True)
    writer = SummaryWriter(os.path.join(root, "logs"))

    model = build_local_residual_model(
        device, channels=args.channels, blocks=args.blocks, groups=args.groups,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    train_loader, val_loader = prepare_local_residual_data(
        train_patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        iterations_before_val=args.iterations_before_val,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
    )

    print("Loss             : MSE")
    print("Precision        : FP32")
    print("Scheduler        : none (fixed learning rate)")
    print("Checkpoint metric: validation SSIM, matching ldct-benchmark BaseTrainer")

    iteration = 0
    best_ssim = -float("inf")
    start = time.time()
    cycle = 0
    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        iteration, train_mse = train_cycle(
            model, train_loader, optimizer, device, iteration, args.max_iterations,
        )
        val = validate(model, val_loader, device)
        dpsnr = val["psnr"] - val["baseline"]
        meta = {
            "architecture": "local_residual_control",
            "model_config": model.model_config(),
            "normalization": "benchmark_meanstd",
            "pixel_mean": BENCHMARK_PIXEL_MEAN,
            "pixel_std": BENCHMARK_PIXEL_STD,
            "pixel_domain": "HU+1024",
            "hu_preset": cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss": "MSE",
            "input_mode": "2d",
            "patient_balanced": True,
            "train_patch_size": args.patch_size,
            "train_batch_size": args.batch_size,
            "benchmark_contract": "windowed PSNR/SSIM; unwindowed clipped RMSE/VIF",
        }
        payload = {
            "model_state_dict": get_state_dict(model),
            "meta": meta,
            "iteration": iteration,
            "ssim": val["ssim"],
            "psnr": val["psnr"],
            "val_mse": val["mse"],
        }
        torch.save(payload, os.path.join(root, "last_model.pt"))
        if val["ssim"] > best_ssim:
            best_ssim = val["ssim"]
            torch.save(payload, os.path.join(root, "best_model.pt"))
        torch.save({
            **payload,
            "optimizer_state": optimizer.state_dict(),
        }, os.path.join(root, "checkpoint.pt"))

        elapsed = time.time() - t0
        print(
            f"Cycle {cycle:02d} | Iter {iteration:06d}/{args.max_iterations} | "
            f"Train MSE {train_mse:.6f} | Val MSE {val['mse']:.6f} | "
            f"PSNR {val['psnr']:.3f} | dPSNR {dpsnr:+.3f} | "
            f"SSIM {val['ssim']:.5f} | RMSE {val['rmse']:.2f} | "
            f"LR {args.lr:.2e} | {elapsed:.1f}s"
        )
        writer.add_scalars("Loss", {"train": train_mse, "val": val["mse"]}, iteration)
        writer.add_scalar("Metrics/PSNR", val["psnr"], iteration)
        writer.add_scalar("Metrics/SSIM", val["ssim"], iteration)
        writer.add_scalar("Metrics/dPSNR", dpsnr, iteration)

    writer.close()
    print(f"Training complete in {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")
    print(f"Best validation SSIM: {best_ssim:.5f} | outputs: {root}")


if __name__ == "__main__":
    main()

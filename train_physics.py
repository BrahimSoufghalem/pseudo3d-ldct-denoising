"""Train PhysicsSpectralNet from scratch on the fair ldct-benchmark pipeline."""

import argparse
import math
import os
import statistics
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from dataset import prepareCT2D
from metrics import compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu, denormalize_to_hu_offset
from physics_losses import PhysicsInformedCTLoss
from physics_spectral_model import build_physics_model
from utils import setup_reproducibility, get_device, get_state_dict


def args_parser():
    p = argparse.ArgumentParser(description="Train metadata-free physics-spectral 2D CT denoiser")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=cfg.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-root", default="runs_physics")
    p.add_argument("--lambda-nps", type=float, default=0.0)
    p.add_argument("--lambda-hu", type=float, default=0.0)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--band-channels", type=int, default=16)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--no-spectral", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--overfit-samples", type=int, default=0,
        help="materialize exactly this many augmented training samples and reuse "
             "them for both train and validation; 0 disables the sanity test",
    )
    return p.parse_args()


def run_dir(args):
    tag = "spatial" if args.no_spectral else "physics_spectral"
    suffix = f"_overfit{args.overfit_samples}" if args.overfit_samples > 0 else ""
    return os.path.join(
        args.output_root,
        f"2d_{tag}_nps{args.lambda_nps:g}_hu{args.lambda_hu:g}{suffix}",
    )


def _slice_batch(batch, take):
    """Copy the first `take` samples while preserving MONAI metadata values."""
    batch_size = int(batch["image"].shape[0])
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value[:take].detach().cpu().clone()
        elif isinstance(value, list) and len(value) == batch_size:
            result[key] = value[:take]
        elif isinstance(value, tuple) and len(value) == batch_size:
            result[key] = value[:take]
        else:
            result[key] = value
    return result


def materialize_overfit_batches(loader, n_samples):
    """Freeze exact random crops so memorisation, not augmentation, is tested."""
    if n_samples <= 0:
        return loader
    frozen = []
    remaining = int(n_samples)
    for batch in loader:
        take = min(remaining, int(batch["image"].shape[0]))
        frozen.append(_slice_batch(batch, take))
        remaining -= take
        if remaining <= 0:
            break
    actual = sum(int(batch["image"].shape[0]) for batch in frozen)
    if actual == 0:
        raise RuntimeError("Could not materialize any overfit samples")
    print(
        f"OVERFIT SANITY MODE: froze {actual} exact augmented samples in "
        f"{len(frozen)} batches; the same batches are used for train and val."
    )
    return frozen


def step_loss(loss_fn, pred_res, inp, target):
    pred = inp + pred_res
    return pred, loss_fn(pred, target, inp)


def train_epoch(model, loader, loss_fn, optimizer, device, clip):
    model.train()
    losses, norms = [], []
    component_sums = {"MSE": 0.0, "NPS": 0.0, "HU": 0.0}
    bar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred, (loss, info) = step_loss(loss_fn, model(x), x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        losses.append(float(loss.detach()))
        norms.append(float(norm))
        for key in component_sums:
            component_sums[key] += info[key]
        bar.set_postfix(loss=f"{loss.item():.5f}", mse=f"{info['MSE']:.5f}",
                        nps=f"{info['NPS']:.3f}", hu=f"{info['HU']:.5f}")
    n = max(1, len(losses))
    return {
        "loss": sum(losses) / n,
        "gnorm": statistics.median(norms) if norms else 0.0,
        **{k: v / n for k, v in component_sums.items()},
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    sums = dict(loss=0.0, psnr=0.0, ssim=0.0, rmse=0.0, baseline=0.0)
    count = batches = 0
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        pred, (loss, _) = step_loss(loss_fn, model(x), x, y)
        pred = pred.clamp(0, 1)
        sums["loss"] += float(loss)
        batches += 1
        body = batch.get("body_type", ["Abdomen"] * pred.shape[0])
        for i in range(pred.shape[0]):
            bt = "Chest" if str(body[i]).lower().startswith("c") else "Abdomen"
            ph = denormalize_to_hu_offset(pred[i].squeeze(), cfg.A_MIN, cfg.A_MAX)
            yh = denormalize_to_hu_offset(y[i].squeeze(), cfg.A_MIN, cfg.A_MAX)
            xh = denormalize_to_hu_offset(x[i].squeeze(), cfg.A_MIN, cfg.A_MAX)
            sums["psnr"] += compute_psnr_windowed(ph, yh, bt)
            sums["baseline"] += compute_psnr_windowed(xh, yh, bt)
            sums["ssim"] += compute_ssim_windowed(ph, yh, bt)
            sums["rmse"] += compute_rmse_hu(ph, yh)
            count += 1
    return {
        "loss": sums["loss"] / max(1, batches),
        "psnr": sums["psnr"] / max(1, count),
        "baseline": sums["baseline"] / max(1, count),
        "ssim": sums["ssim"] / max(1, count),
        "rmse": sums["rmse"] / max(1, count),
    }


def main():
    args = args_parser()
    if cfg.HU_RANGE_PRESET != "benchmark":
        print("WARNING: publication-facing comparison requires HU_RANGE_PRESET=benchmark.")
    setup_reproducibility()
    device = get_device()
    root = run_dir(args)
    os.makedirs(root, exist_ok=True)
    writer = SummaryWriter(os.path.join(root, "logs"))

    model = build_physics_model(
        device, channels=args.channels, band_channels=args.band_channels,
        groups=args.groups, spectral=not args.no_spectral,
    )
    loss_fn = PhysicsInformedCTLoss(
        cfg.A_MIN, cfg.A_MAX, args.lambda_nps, args.lambda_hu,
    ).to(device)
    print(f"HU range: [{cfg.A_MIN}, {cfg.A_MAX}] (preset '{cfg.HU_RANGE_PRESET}')")
    print(f"Loss: {loss_fn.describe()}")
    print("Precision: FP32 (mandatory for the first controlled comparison)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1,
    )
    train_loader, val_loader = prepareCT2D(
        input_mode="2d", train_batch_size=args.batch_size,
    )
    if args.overfit_samples > 0:
        frozen = materialize_overfit_batches(train_loader, args.overfit_samples)
        train_loader = frozen
        val_loader = frozen

    best_psnr = -float("inf")
    start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = train_epoch(model, train_loader, loss_fn, optimizer, device, args.grad_clip)
        va = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        diag = model.scale_diagnostics()
        meta = {
            "architecture": "physics_spectral",
            "model_config": model.model_config(),
            "hu_preset": cfg.HU_RANGE_PRESET,
            "hu_range": [cfg.A_MIN, cfg.A_MAX],
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss": loss_fn.describe(),
            "input_mode": "2d",
            "overfit_samples": args.overfit_samples,
            "benchmark_contract": "ldct-benchmark: windowed PSNR/SSIM, unwindowed clipped RMSE/VIF",
        }
        payload = {
            "model_state_dict": get_state_dict(model), "meta": meta,
            "epoch": epoch, "psnr": va["psnr"], "ssim": va["ssim"],
            "diagnostics": diag,
        }
        torch.save(payload, os.path.join(root, "last_model.pt"))
        if va["psnr"] > best_psnr:
            best_psnr = va["psnr"]
            torch.save(payload, os.path.join(root, "best_model.pt"))
        torch.save({
            **payload, "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, os.path.join(root, "checkpoint.pt"))

        dpsnr = va["psnr"] - va["baseline"]
        elapsed = time.time() - t0
        spec_text = (
            f" spec={diag['spectral_scale_mean']:.3f}"
            if "spectral_scale_mean" in diag else ""
        )
        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] Train {tr['loss']:.6f} | "
            f"Val {va['loss']:.6f} | PSNR {va['psnr']:.3f} | dPSNR {dpsnr:+.3f} | "
            f"SSIM {va['ssim']:.5f} | RMSE {va['rmse']:.2f} | "
            f"MSE {tr['MSE']:.6f} NPS {tr['NPS']:.3f} HU {tr['HU']:.6f} | "
            f"|g|med {tr['gnorm']:.4f} | scales block={diag['block_scale_mean']:.3f}" 
            f"{spec_text} head={diag['head_weight_norm']:.3f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )
        writer.add_scalars("Loss", {"train": tr["loss"], "val": va["loss"]}, epoch + 1)
        writer.add_scalar("Metrics/PSNR", va["psnr"], epoch + 1)
        writer.add_scalar("Metrics/SSIM", va["ssim"], epoch + 1)
        writer.add_scalar("Physics/NPS", tr["NPS"], epoch + 1)
        writer.add_scalar("Physics/HU", tr["HU"], epoch + 1)
        writer.add_scalar("Diagnostics/BlockScaleMean", diag["block_scale_mean"], epoch + 1)
        writer.add_scalar("Diagnostics/HeadWeightNorm", diag["head_weight_norm"], epoch + 1)
        if "spectral_scale_mean" in diag:
            writer.add_scalar("Diagnostics/SpectralScaleMean", diag["spectral_scale_mean"], epoch + 1)

    print(f"Training complete in {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")
    print(f"Best PSNR: {best_psnr:.3f} | outputs: {root}")
    writer.close()


if __name__ == "__main__":
    main()

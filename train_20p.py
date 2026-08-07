"""Unified trainer supporting six independently toggleable improvements.

Usage
-----
# Baseline (pure MSE):
    HU_RANGE_PRESET=benchmark python train_20p.py --arch local_residual \\
        --data-dir /path/to/data

# L1 + SSIM loss (recommended, breaks MSE ceiling):
    ... --ssim-weight 0.5 --l1-weight 0.5

# Only parallel multi-resolution branches:
    ... --use-multi-res

# All improvements + new loss:
    ... --use-hu-gate --use-mu-mod --use-multi-res --use-dilation \\
        --use-freq-boost --hu-bin-loss 0.1 \\
        --ssim-weight 0.5 --l1-weight 0.5

# 20-patient Kaggle experiment:
    ... --split 20p

# Resume after interruption:
    ... --resume

Install (once on Kaggle):
    pip install pytorch-msssim   # required only when --ssim-weight > 0
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

# Optional SSIM loss dependency
try:
    from pytorch_msssim import ssim as _pytorch_ssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Fair comparison trainer with six optional physics improvements"
    )
    p.add_argument("--arch", required=True,
                   choices=["redcnn", "resnet", "local_residual"])
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--max-iterations",        type=int,   default=100_000)
    p.add_argument("--iterations-before-val", type=int,   default=2_500)
    p.add_argument("--batch-size",            type=int,   default=64)
    p.add_argument("--patch-size",            type=int,   default=64)
    p.add_argument("--val-patch-size",        type=int,   default=128)
    p.add_argument("--lr",                    type=float, default=1e-4)
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--cache-rate",            type=float, default=1.0)
    p.add_argument("--output-root",           default="runs")
    p.add_argument("--groups",                type=int,   default=1,
                   help="Groups for middle conv. Default=1 (T4).")
    p.add_argument("--resume", action="store_true")

    # ── Improvement flags — all default OFF ─────────────────────────────────
    p.add_argument(
        "--use-hu-gate", action="store_true",
        help="[1] SE-like HU-context gating inside each block.",
    )
    p.add_argument(
        "--use-mu-mod", action="store_true",
        help="[2] mu-aware FiLM modulation at network midpoint.",
    )
    p.add_argument(
        "--mu-split", type=int, default=None,
        help="Block after which mu-mod is applied (default: blocks//2=5).",
    )
    p.add_argument(
        "--use-multi-res", action="store_true",
        help="[3] Parallel multi-resolution branches: full + down-x2 + down-x4.",
    )
    p.add_argument(
        "--use-dilation", action="store_true",
        help="[4] Lightweight dilated depthwise conv (dilation=2) per block.",
    )
    p.add_argument(
        "--use-freq-boost", action="store_true",
        help="[5] Learnable Laplacian high-freq boost inside each block.",
    )
    p.add_argument(
        "--hu-bin-loss", type=float, default=0.0, metavar="WEIGHT",
        help="[6] HU-bin systematic-bias penalty weight. 0.0 = disabled.",
    )
    p.add_argument("--hu-bin-bins", type=int, default=16)

    # ── Loss function flags ───────────────────────────────────────────────
    p.add_argument(
        "--ssim-weight", type=float, default=0.0, metavar="W",
        help="[7] Weight of SSIM loss component. 0.0 = disabled (pure MSE). "
             "Example: --ssim-weight 0.5 --l1-weight 0.5 => pure L1+SSIM. "
             "Requires: pip install pytorch-msssim",
    )
    p.add_argument(
        "--l1-weight", type=float, default=0.0, metavar="W",
        help="[8] Weight of L1 (MAE) loss component. 0.0 = disabled. "
             "Can be combined with --ssim-weight. "
             "Remaining weight (1 - ssim_w - l1_w) goes to MSE.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.0,
    l1_weight: float = 0.0,
    hu_bin_loss_weight: float = 0.0,
    hu_bin_bins: int = 16,
) -> torch.Tensor:
    """Combined loss: MSE + L1 + SSIM (+ optional HU-bin bias penalty).

    Weights are automatically normalized so MSE+L1+SSIM components sum to 1:
        mse_weight = max(0, 1.0 - ssim_weight - l1_weight)

    Examples
    --------
    Pure MSE (default):       ssim_weight=0.0, l1_weight=0.0
    Pure L1+SSIM (50/50):     ssim_weight=0.5, l1_weight=0.5
    MSE+SSIM blend:           ssim_weight=0.3, l1_weight=0.0  (mse=0.7)
    L1+SSIM dominant + HU:    ssim_weight=0.5, l1_weight=0.5, hu_bin_loss_weight=0.1
    """
    ssim_weight = float(ssim_weight)
    l1_weight   = float(l1_weight)
    mse_weight  = max(0.0, 1.0 - ssim_weight - l1_weight)

    loss = pred.new_zeros(())

    if mse_weight > 0.0:
        loss = loss + mse_weight * F.mse_loss(pred, target)

    if l1_weight > 0.0:
        loss = loss + l1_weight * F.l1_loss(pred, target)

    if ssim_weight > 0.0:
        if not _HAS_MSSSIM:
            raise RuntimeError(
                "pytorch-msssim is required for --ssim-weight > 0.\n"
                "Install with: pip install pytorch-msssim"
            )
        # Dynamic data_range from target batch for standardized HU inputs
        with torch.no_grad():
            data_range = float(
                (target.max() - target.min()).clamp(min=1e-6).item()
            )
        ssim_val = _pytorch_ssim(
            pred, target,
            data_range=data_range,
            size_average=True,
            nonnegative_ssim=True,
        )
        loss = loss + ssim_weight * (1.0 - ssim_val)

    if hu_bin_loss_weight > 0.0:
        loss = loss + hu_bin_loss_weight * hu_bin_bias_loss(
            pred, target, n_bins=hu_bin_bins
        )

    return loss


# ──────────────────────────────────────────────────────────────────────────
def hu_bin_bias_loss(pred, target, n_bins=16):
    t_min = target.detach().min()
    t_max = target.detach().max()
    if (t_max - t_min).item() < 1e-6:
        return pred.new_zeros(())
    edges = torch.linspace(t_min.item(), t_max.item(), n_bins + 1,
                           device=target.device)
    loss  = pred.new_zeros(())
    count = 0
    for i in range(n_bins):
        mask = (target >= edges[i]) & (target < edges[i + 1])
        if mask.sum() < 10:
            continue
        bias  = (pred[mask] - target[mask]).mean()
        loss  = loss + bias * bias
        count += 1
    return loss / max(1, count)


def apply_split(split):
    if split == "20p":
        cfg.EXPECTED_TRAIN = TRAIN_20P
        cfg.EXPECTED_VAL   = VAL_20P
        return len(TRAIN_20P), len(VAL_20P)
    return len(cfg.EXPECTED_TRAIN), len(cfg.EXPECTED_VAL)


def build_model(arch, device, args):
    if arch == "local_residual":
        return build_local_residual_model(
            device,
            channels=128,
            blocks=10,
            groups=args.groups,
            use_hu_gate=args.use_hu_gate,
            use_freq_boost=args.use_freq_boost,
            use_dilation=args.use_dilation,
            use_mu_mod=args.use_mu_mod,
            mu_split=args.mu_split,
            use_multi_res=args.use_multi_res,
        )
    return build_benchmark_model(arch, device)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums    = dict(mse=0.0, psnr=0.0, ssim=0.0, rmse=0.0, baseline_psnr=0.0)
    batches = samples = 0
    for batch in tqdm(loader, desc="  Val", leave=False, dynamic_ncols=True):
        x    = batch["image"].to(device, non_blocking=True)
        y    = batch["label"].to(device, non_blocking=True)
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
            sums["baseline_psnr"] += compute_psnr_windowed(x_px[i].squeeze(),   y_px[i].squeeze(), bt)
            sums["ssim"]          += compute_ssim_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["rmse"]          += compute_rmse_hu(pred_px[i].squeeze(),        y_px[i].squeeze())
            samples += 1
    n_b = max(1, batches)
    n_s = max(1, samples)
    return {
        "mse":   sums["mse"]  / n_b,
        "psnr":  sums["psnr"] / n_s,
        "dpsnr": (sums["psnr"] - sums["baseline_psnr"]) / n_s,
        "ssim":  sums["ssim"] / n_s,
        "rmse":  sums["rmse"] / n_s,
    }


def train_cycle(
    model, loader, optimizer, device, iteration, max_iter,
    ssim_weight=0.0, l1_weight=0.0,
    hu_bin_loss_weight=0.0, hu_bin_bins=16,
):
    model.train()
    total = count = 0.0
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if iteration >= max_iter:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = compute_loss(
            pred, y,
            ssim_weight=ssim_weight,
            l1_weight=l1_weight,
            hu_bin_loss_weight=hu_bin_loss_weight,
            hu_bin_bins=hu_bin_bins,
        )
        loss.backward()
        optimizer.step()
        iteration += 1
        total += float(loss.detach())
        count += 1
        bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}")
    return iteration, total / max(1, count)


# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError(
            "Set HU_RANGE_PRESET=benchmark.\n"
            "Example: HU_RANGE_PRESET=benchmark python train_20p.py --arch redcnn"
        )

    # Validate loss weights
    if args.ssim_weight < 0 or args.l1_weight < 0:
        raise ValueError("--ssim-weight and --l1-weight must be >= 0")
    if args.ssim_weight + args.l1_weight > 1.0:
        raise ValueError(
            f"--ssim-weight ({args.ssim_weight}) + --l1-weight ({args.l1_weight}) "
            f"must not exceed 1.0"
        )
    if args.ssim_weight > 0 and not _HAS_MSSSIM:
        raise RuntimeError(
            "pytorch-msssim is required for --ssim-weight > 0.\n"
            "Install with: pip install pytorch-msssim"
        )

    n_train, n_val = apply_split(args.split)
    setup_reproducibility()
    device    = get_device()
    out_dir   = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    # ── Build loss description
    mse_w  = max(0.0, 1.0 - args.ssim_weight - args.l1_weight)
    loss_parts = []
    if mse_w > 0.0:          loss_parts.append(f"{mse_w:.2f}*MSE")
    if args.l1_weight > 0.0: loss_parts.append(f"{args.l1_weight:.2f}*L1")
    if args.ssim_weight > 0.0: loss_parts.append(f"{args.ssim_weight:.2f}*SSIM")
    loss_desc = " + ".join(loss_parts) if loss_parts else "1.00*MSE"
    if args.hu_bin_loss > 0.0:
        loss_desc += f" + {args.hu_bin_loss}*HU-bin-bias({args.hu_bin_bins}bins)"

    # ── Banner
    active = []
    if args.use_hu_gate:       active.append("HU-gate")
    if args.use_mu_mod:        active.append(f"mu-mod@{args.mu_split or 'auto'}")
    if args.use_multi_res:     active.append("Multi-Res(3+3+3|1)")
    if args.use_dilation:      active.append("Dilation-2")
    if args.use_freq_boost:    active.append("Freq-boost")
    if args.hu_bin_loss > 0.0: active.append(f"HU-bin(w={args.hu_bin_loss})")
    active_str = ", ".join(active) if active else "none (baseline)"

    print(f"\n{'='*64}")
    print(f"  arch={args.arch.upper()} | split={args.split} | improvements: {active_str}")
    print(f"  Train patients : {n_train}  |  Val patients: {n_val}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output         : {out_dir}")
    print(f"  Loss           : {loss_desc}")
    print(f"{'='*64}\n")

    model     = build_model(args.arch, device, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    iteration = 0
    best_ssim = -float("inf")
    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        iteration = int(ckpt.get("iteration", 0))
        best_ssim = float(ckpt.get("ssim", -float("inf")))
        print(f"  Resumed at iter {iteration} | best SSIM {best_ssim:.5f}")
    elif args.resume:
        print(f"  --resume: no checkpoint at {ckpt_path}, starting fresh.")

    if iteration >= args.max_iterations:
        print(f"  Training already complete ({iteration}/{args.max_iterations}).")
        return

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

    print(f"Loss      : {loss_desc}")
    print(f"Optimizer : Adam(lr={args.lr:.2e})")

    start = time.time()
    cycle = iteration // args.iterations_before_val
    eff_mu_split = args.mu_split if args.mu_split is not None else 10 // 2

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        iteration, train_loss = train_cycle(
            model, train_loader, optimizer, device,
            iteration, args.max_iterations,
            ssim_weight=args.ssim_weight,
            l1_weight=args.l1_weight,
            hu_bin_loss_weight=args.hu_bin_loss,
            hu_bin_bins=args.hu_bin_bins,
        )
        val = validate(model, val_loader, device)

        meta = {
            "architecture":    args.arch,
            "split":           args.split,
            "groups":          args.groups,
            "use_hu_gate":     args.use_hu_gate,
            "use_freq_boost":  args.use_freq_boost,
            "use_dilation":    args.use_dilation,
            "use_mu_mod":      args.use_mu_mod,
            "mu_split":        eff_mu_split,
            "use_multi_res":   args.use_multi_res,
            "hu_bin_loss":     args.hu_bin_loss,
            "ssim_weight":     args.ssim_weight,
            "l1_weight":       args.l1_weight,
            "mse_weight":      mse_w,
            "normalization":   "benchmark_meanstd",
            "pixel_mean":      BENCHMARK_PIXEL_MEAN,
            "pixel_std":       BENCHMARK_PIXEL_STD,
            "pixel_domain":    "HU+1024",
            "hu_preset":       cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss":            loss_desc,
            "input_mode":      "2d",
            "n_train_patients": n_train,
            "n_val_patients":   n_val,
        }
        payload = {
            "model_state_dict": get_state_dict(model),
            "meta":      meta,
            "iteration": iteration,
            "ssim":      val["ssim"],
            "psnr":      val["psnr"],
            "val_mse":   val["mse"],
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
            f"Loss {train_loss:.6f} | Val MSE {val['mse']:.6f} | "
            f"PSNR {val['psnr']:.3f} | dPSNR {val['dpsnr']:+.3f} | "
            f"SSIM {val['ssim']:.5f} | RMSE {val['rmse']:.2f} | "
            f"{elapsed:.1f}s"
        )

    total = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{args.arch.upper()}] in {total} | best SSIM={best_ssim:.5f}")
    print(f"Checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()

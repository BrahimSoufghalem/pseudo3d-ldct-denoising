"""Unified trainer supporting seven independently toggleable improvements.

Usage
-----
# Baseline (pure MSE):
    HU_RANGE_PRESET=benchmark python train_20p.py --arch local_residual \\
        --data-dir /path/to/data

# U-Net decoder (best for Chest VIF, beats RED-CNN):
    ... --use-multi-res --use-unet-decode

# U-Net + best loss:
    ... --use-multi-res --use-unet-decode \\
        --ssim-weight 0.5 --l1-weight 0.3 --grad-weight 0.2

# All improvements:
    ... --use-hu-gate --use-mu-mod --use-multi-res --use-unet-decode \\
        --use-dilation --use-freq-boost --hu-bin-loss 0.1 \\
        --ssim-weight 0.5 --l1-weight 0.3 --grad-weight 0.2

# 20-patient Kaggle experiment:
    ... --split 20p

# Resume after interruption:
    ... --resume

Install (once on Kaggle):
    pip install pytorch-msssim   # required only when --ssim-weight > 0

Block budget
------------
Baseline / multi-res  : blocks=10  (10 full-res blocks)
U-Net decode          : blocks=20  (5 enc_full + 5 enc_half + 5 enc_qtr
                                    + 5 dec_half + 0 final)
  - With blocks=10 every branch had only 2 blocks -> too shallow to develop
    hierarchical features -> zero measurable difference vs flat multi-res.
  - With blocks=20 each branch has 5 blocks and the decoder gains real depth.
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

try:
    from pytorch_msssim import ssim as _pytorch_ssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


# ──────────────────────────────────────────────────────────────────────────
# Block budget constants
# ──────────────────────────────────────────────────────────────────────────
_BLOCKS_DEFAULT    = 10   # sequential / flat multi-res
_BLOCKS_UNET       = 20   # U-Net decode: 5 enc x3 + 5 dec_half = 20 blocks
                          # (enc_n = 20//4 = 5  per branch, dec_n = 5, final = 0)


# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Fair comparison trainer with seven optional improvements"
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
    p.add_argument("--groups",                type=int,   default=1)
    p.add_argument("--resume", action="store_true")

    # ── Architecture flags ─────────────────────────────────────────────────
    p.add_argument("--use-hu-gate",    action="store_true",
                   help="[1] SE-like HU-context gating per block.")
    p.add_argument("--use-mu-mod",     action="store_true",
                   help="[2] mu-aware FiLM modulation at midpoint.")
    p.add_argument("--mu-split",       type=int, default=None)
    p.add_argument("--use-multi-res",  action="store_true",
                   help="[3] Parallel multi-resolution branches.")
    p.add_argument(
        "--use-unet-decode", action="store_true",
        help="[4] U-Net skip-connection decoder over multi-res "
             "(requires --use-multi-res). "
             "Uses blocks=20 automatically: enc_n=5 per branch, dec_n=5, final=0. "
             "With only 10 blocks (2 per branch) there is zero measurable "
             "difference vs flat multi-res because the branches are too shallow.",
    )
    p.add_argument("--use-dilation",   action="store_true",
                   help="[5] Dilated depthwise conv per block.")
    p.add_argument("--use-freq-boost", action="store_true",
                   help="[6] Laplacian high-freq boost per block.")
    p.add_argument("--hu-bin-loss",    type=float, default=0.0, metavar="W",
                   help="[7] HU-bin bias penalty weight.")
    p.add_argument("--hu-bin-bins",    type=int, default=16)

    # ── Loss flags ─────────────────────────────────────────────────────────
    p.add_argument("--ssim-weight", type=float, default=0.0, metavar="W",
                   help="SSIM loss weight (requires pytorch-msssim).")
    p.add_argument("--l1-weight",   type=float, default=0.0, metavar="W",
                   help="L1 loss weight. Remainder goes to MSE.")
    p.add_argument("--grad-weight", type=float, default=0.0, metavar="W",
                   help="Gradient edge loss weight (finite diff L1). Additive.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
def image_gradient_loss(pred, target):
    dx_pred   = pred  [:, :, :, 1:] - pred  [:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred   = pred  [:, :, 1:, :] - pred  [:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def compute_loss(pred, target, ssim_weight=0.0, l1_weight=0.0, grad_weight=0.0,
                 hu_bin_loss_weight=0.0, hu_bin_bins=16):
    mse_w = max(0.0, 1.0 - float(ssim_weight) - float(l1_weight))
    loss  = pred.new_zeros(())
    if mse_w > 0.0:
        loss = loss + mse_w * F.mse_loss(pred, target)
    if l1_weight > 0.0:
        loss = loss + float(l1_weight) * F.l1_loss(pred, target)
    if ssim_weight > 0.0:
        if not _HAS_MSSSIM:
            raise RuntimeError("pip install pytorch-msssim")
        with torch.no_grad():
            dr = float((target.max() - target.min()).clamp(min=1e-6).item())
        loss = loss + float(ssim_weight) * (
            1.0 - _pytorch_ssim(pred, target, data_range=dr,
                                size_average=True, nonnegative_ssim=True)
        )
    if grad_weight > 0.0:
        loss = loss + float(grad_weight) * image_gradient_loss(pred, target)
    if hu_bin_loss_weight > 0.0:
        loss = loss + hu_bin_loss_weight * hu_bin_bias_loss(
            pred, target, n_bins=hu_bin_bins)
    return loss


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


def _unet_block_desc(n_blocks):
    """Return a human-readable block allocation string for unet-decode mode."""
    enc_n   = max(1, n_blocks // 4)
    dec_n   = max(1, n_blocks // 4)
    final_n = max(0, n_blocks - 3 * enc_n - dec_n)
    return f"UNet-Decode(enc={enc_n},dec={dec_n},final={final_n})"


def build_model(arch, device, args):
    if arch == "local_residual":
        # U-Net decode needs more blocks so each branch has real depth.
        # With blocks=10 every branch gets only enc_n=2 blocks, which is
        # identical in practice to flat multi-res -> zero measurable gain.
        # blocks=20 gives enc_n=5 per branch, dec_n=5, final_n=0.
        n_blocks = _BLOCKS_UNET if args.use_unet_decode else _BLOCKS_DEFAULT
        return build_local_residual_model(
            device,
            channels=128, blocks=n_blocks, groups=args.groups,
            use_hu_gate=args.use_hu_gate,
            use_freq_boost=args.use_freq_boost,
            use_dilation=args.use_dilation,
            use_mu_mod=args.use_mu_mod,
            mu_split=args.mu_split,
            use_multi_res=args.use_multi_res,
            use_unet_decode=args.use_unet_decode,
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
    n_b, n_s = max(1, batches), max(1, samples)
    return {
        "mse":   sums["mse"]  / n_b,
        "psnr":  sums["psnr"] / n_s,
        "dpsnr": (sums["psnr"] - sums["baseline_psnr"]) / n_s,
        "ssim":  sums["ssim"] / n_s,
        "rmse":  sums["rmse"] / n_s,
    }


def train_cycle(model, loader, optimizer, device, iteration, max_iter,
                ssim_weight=0.0, l1_weight=0.0, grad_weight=0.0,
                hu_bin_loss_weight=0.0, hu_bin_bins=16):
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
        loss = compute_loss(pred, y, ssim_weight=ssim_weight, l1_weight=l1_weight,
                            grad_weight=grad_weight,
                            hu_bin_loss_weight=hu_bin_loss_weight,
                            hu_bin_bins=hu_bin_bins)
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
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    if args.use_unet_decode and not args.use_multi_res:
        raise ValueError("--use-unet-decode requires --use-multi-res")
    if args.ssim_weight + args.l1_weight > 1.0:
        raise ValueError("--ssim-weight + --l1-weight must not exceed 1.0")
    if args.ssim_weight > 0 and not _HAS_MSSSIM:
        raise RuntimeError("pip install pytorch-msssim")

    n_train, n_val = apply_split(args.split)
    setup_reproducibility()
    device    = get_device()
    out_dir   = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    mse_w  = max(0.0, 1.0 - args.ssim_weight - args.l1_weight)
    loss_parts = []
    if mse_w > 0.0:             loss_parts.append(f"{mse_w:.2f}*MSE")
    if args.l1_weight > 0.0:    loss_parts.append(f"{args.l1_weight:.2f}*L1")
    if args.ssim_weight > 0.0:  loss_parts.append(f"{args.ssim_weight:.2f}*SSIM")
    if args.grad_weight > 0.0:  loss_parts.append(f"{args.grad_weight:.2f}*Grad")
    loss_desc = " + ".join(loss_parts) if loss_parts else "1.00*MSE"
    if args.hu_bin_loss > 0.0:
        loss_desc += f" + {args.hu_bin_loss}*HU-bin"

    n_blocks = _BLOCKS_UNET if args.use_unet_decode else _BLOCKS_DEFAULT
    active = []
    if args.use_hu_gate:       active.append("HU-gate")
    if args.use_mu_mod:        active.append(f"mu-mod@{args.mu_split or 'auto'}")
    if args.use_unet_decode:   active.append(_unet_block_desc(n_blocks))
    elif args.use_multi_res:   active.append("Multi-Res(3+3+3|1)")
    if args.use_dilation:      active.append("Dilation-2")
    if args.use_freq_boost:    active.append("Freq-boost")
    if args.hu_bin_loss > 0.0: active.append(f"HU-bin(w={args.hu_bin_loss})")
    active_str = ", ".join(active) if active else "none (baseline)"

    print(f"\n{'='*68}")
    print(f"  arch={args.arch.upper()} | split={args.split} | blocks={n_blocks}")
    print(f"  improvements: {active_str}")
    print(f"  Train patients : {n_train}  |  Val patients: {n_val}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output         : {out_dir}")
    print(f"  Loss           : {loss_desc}")
    print(f"{'='*68}\n")

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
    eff_mu_split = args.mu_split if args.mu_split is not None else n_blocks // 2

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        iteration, train_loss = train_cycle(
            model, train_loader, optimizer, device,
            iteration, args.max_iterations,
            ssim_weight=args.ssim_weight,
            l1_weight=args.l1_weight,
            grad_weight=args.grad_weight,
            hu_bin_loss_weight=args.hu_bin_loss,
            hu_bin_bins=args.hu_bin_bins,
        )
        val = validate(model, val_loader, device)

        meta = {
            "architecture":    args.arch,
            "split":           args.split,
            "groups":          args.groups,
            "n_blocks":        n_blocks,
            "use_hu_gate":     args.use_hu_gate,
            "use_freq_boost":  args.use_freq_boost,
            "use_dilation":    args.use_dilation,
            "use_mu_mod":      args.use_mu_mod,
            "mu_split":        eff_mu_split,
            "use_multi_res":   args.use_multi_res,
            "use_unet_decode": args.use_unet_decode,
            "hu_bin_loss":     args.hu_bin_loss,
            "ssim_weight":     args.ssim_weight,
            "l1_weight":       args.l1_weight,
            "mse_weight":      mse_w,
            "grad_weight":     args.grad_weight,
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
        torch.save({**payload, "optimizer_state": optimizer.state_dict()}, ckpt_path)

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

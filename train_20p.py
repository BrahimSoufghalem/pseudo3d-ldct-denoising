"""Unified trainer supporting seven independently toggleable improvements.

See TRAINING_FIXES.md for the list of protocol fixes applied to this file
(fixed SSIM data range, fixed HU-bin boundaries, per-region validation,
--select-by checkpoint selection, cosine LR decay, mu-mod local pooling,
U-Net final-block reallocation).

Usage
-----
# Baseline (pure MSE):
    HU_RANGE_PRESET=benchmark python train_20p.py --arch local_residual \\
        --data-dir /path/to/data

# EXACT ldct-benchmark paper protocol for RED-CNN.
# Hyperparameters are the official configs/redcnn.yaml of
# github.com/eeulig/ldct-benchmark (found by their hpopt), and the
# checkpoint criterion (--select-by bench_ssim) replicates
# ldctbench/methods/base.py save_checkpoint(to_optimize=\"SSIM\"):
# best OVERALL validation SSIM computed WITHOUT clinical windowing.
    HU_RANGE_PRESET=benchmark python train_20p.py --arch redcnn \\
        --data-dir /path/to/data --split 100p \\
        --max-iterations 92994 --iterations-before-val 1000 \\
        --batch-size 73 --patch-size 128 --val-patch-size 128 \\
        --lr 9.583417460320728e-05 --lr-schedule constant \\
        --select-by bench_ssim

# PILOT MODE: fast config screening on a small deterministic patient subset
# (chest/abdomen balanced). Ranking only -- NOT reportable numbers. See also
# run_pilot.py to sweep several configs sequentially with a summary table:
    ... --train-patients 8 --val-patients 4 --max-iterations 8000

# Physics-informed spectral residual head (Option A: ARCHITECTURAL physics
# addition, see spectral_head.py). Works with ANY --arch; the wrapped model
# starts EXACTLY equal to the base model (G == 1 at init):
    ... --arch redcnn --use-spectral-head --select-by bench_ssim

# U-Net decoder (best for Chest VIF, beats RED-CNN):
    ... --use-multi-res --use-unet-decode

# U-Net + balanced loss (keep MSE weight > 0!):
    ... --use-multi-res --use-unet-decode \\
        --ssim-weight 0.3 --l1-weight 0.3 --grad-weight 0.2

# All improvements:
    ... --use-hu-gate --use-mu-mod --use-multi-res --use-unet-decode \\
        --use-dilation --use-freq-boost --hu-bin-loss 0.1 \\
        --ssim-weight 0.3 --l1-weight 0.3 --grad-weight 0.2

# Physics additions on a benchmark trunk (Option-1 contribution path,
# see PHYSICS_CONTRIBUTION_PLAN.md; works with ANY --arch):
    ... --arch redcnn --nps-weight 0.05 --hu-bin-loss 0.1 \\
        --select-by bench_ssim

# Chest-VIF-oriented checkpoint selection with cosine decay:
    ... --select-by chest_vif --val-vif --lr-schedule cosine --min-lr 1e-6

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
  - NOTE: the default allocation leaves final=0 (no full-resolution blocks
    after the decoder). Use --unet-final-blocks 4 to reallocate to
    enc 4+4+4 / dec 4 / final 4 within the same 20-block budget.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as _sk_ssim
from tqdm import tqdm

import config as cfg
from benchmark_architectures import build_benchmark_model
from local_residual_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, prepare_local_residual_data,
)
from local_residual_model import build_local_residual_model
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu,
    compute_vif_hu,
)
from physics_losses import BatchRadialNPSLoss
from spectral_head import SpectralResidualModel
from twenty_patient_split import TRAIN_20P, VAL_20P
from utils import setup_reproducibility, get_device, get_state_dict

try:
    from pytorch_msssim import ssim as _pytorch_ssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


# ──────────────────────────────────────────────────────────────────────
# Block budget constants
# ──────────────────────────────────────────────────────────────────────
_BLOCKS_DEFAULT    = 10   # sequential / flat multi-res
_BLOCKS_UNET       = 20   # U-Net decode: 5 enc x3 + 5 dec_half = 20 blocks
                          # (enc_n = 20//4 = 5  per branch, dec_n = 5, final = 0)

# FIX: constant SSIM data range in benchmark-standardized units.
# Previously data_range was computed per batch as target.max() - target.min(),
# which rescales the SSIM stability constants C1/C2 on every step and makes
# the loss batch-dependent. The physical evaluation range is EVAL_DATA_RANGE
# (= 2924 for the benchmark preset) in the HU+1024 pixel domain; dividing by
# the benchmark std expresses the same range in standardized units.
_SSIM_DATA_RANGE = float(cfg.EVAL_DATA_RANGE) / float(BENCHMARK_PIXEL_STD)

# FIX: fixed physical tissue boundaries for the HU-bin bias loss (same
# intervals as physics_losses.HUBinBiasLoss). Previously the bins were derived
# from each batch's min/max, so the loss compared different intensity ranges
# on every step.
_HU_BIN_BOUNDARIES_HU = (-1024.0, -500.0, -200.0, 200.0, 600.0, 1900.0)

_SELECT_CHOICES = ("ssim", "psnr", "vif", "chest_ssim", "chest_vif",
                   "bench_ssim")

# Shared radial-NPS loss instance (no parameters; window tensors are created
# on the input device, so a single lazily built module is safe).
_NPS_LOSS = None


def _get_nps_loss():
    global _NPS_LOSS
    if _NPS_LOSS is None:
        _NPS_LOSS = BatchRadialNPSLoss()
    return _NPS_LOSS


# ──────────────────────────────────────────────────────────────────────
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
    p.add_argument("--lr-schedule", choices=["constant", "cosine"],
                   default="constant",
                   help="LR schedule. 'cosine' decays from --lr to --min-lr "
                        "over --max-iterations (resume-safe: LR is a pure "
                        "function of the iteration counter). The ldct-benchmark "
                        "paper uses 'constant'.")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="Final LR for --lr-schedule cosine.")
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--cache-rate",            type=float, default=1.0)
    p.add_argument("--output-root",           default="runs")
    p.add_argument("--groups",                type=int,   default=1)
    p.add_argument("--resume", action="store_true")

    # ── Pilot mode ────────────────────────────────────────────────────────
    p.add_argument("--train-patients", type=int, default=None, metavar="N",
                   help="PILOT MODE: train on only N patients (deterministic "
                        "chest/abdomen-balanced subset). For fast config "
                        "screening: the RANKING of configs is meaningful, "
                        "the absolute numbers are NOT reportable. Retrain "
                        "the winning config on the full split before "
                        "reporting.")
    p.add_argument("--val-patients", type=int, default=None, metavar="N",
                   help="PILOT MODE: validate on only N patients "
                        "(deterministic chest/abdomen-balanced subset). "
                        "Speeds up the per-cycle validation pass.")

    # ── Checkpoint selection ──────────────────────────────────────────────
    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="ssim",
                   help="Validation metric used to select best_model.pt. "
                        "'bench_ssim' replicates the ldct-benchmark paper "
                        "saving criterion EXACTLY: overall validation SSIM "
                        "computed WITHOUT clinical windowing "
                        "(skimage structural_similarity, data_range=2924, on "
                        "denormalized images clipped to [0, 2924]), as in "
                        "ldctbench/methods/base.py "
                        "save_checkpoint(to_optimize='SSIM'). "
                        "'ssim'/'psnr' use clinically WINDOWED metrics and "
                        "are therefore NOT the paper criterion. "
                        "'chest_*' options fall back to the overall metric "
                        "when the validation set has no chest slices. "
                        "Default 'ssim' reproduces the old behavior.")
    p.add_argument("--val-vif", action="store_true",
                   help="Compute VIF during validation even when --select-by "
                        "does not require it. VIF on validation crops is "
                        "indicative only; full-resolution evaluate_20p.py "
                        "remains the ground truth.")

    # ── Architecture flags ─────────────────────────────────────────────────
    p.add_argument("--use-spectral-head", action="store_true",
                   help="[9] Physics-informed spectral residual head "
                        "(spectral_head.py): a learnable RADIAL gain G(|f|) "
                        "applied in the Fourier domain to the noise the base "
                        "model removes. The radial-only constraint mirrors "
                        "FBP noise physics (isotropic NPS); G is initialized "
                        "to 1 so training starts exactly at the base model. "
                        "ARCHITECTURAL addition: works with any --arch, "
                        "including redcnn.")
    p.add_argument("--spectral-bins", type=int, default=32,
                   help="Number of radial gain knots for --use-spectral-head.")
    p.add_argument("--use-hu-gate",    action="store_true",
                   help="[1] SE-like HU-context gating per block.")
    p.add_argument("--use-mu-mod",     action="store_true",
                   help="[2] mu-aware FiLM modulation at midpoint.")
    p.add_argument("--mu-split",       type=int, default=None)
    p.add_argument("--mu-mod-mode", choices=["global", "local"],
                   default="global",
                   help="mu-mod context pooling. 'global' (old behavior) uses "
                        "a single image-wide mean, which differs between "
                        "128-px training patches and 512-px test slices. "
                        "'local' uses fixed-size windows so train and test "
                        "see statistics of the same physical extent.")
    p.add_argument("--mu-local-window", type=int, default=64,
                   help="Window size in pixels for --mu-mod-mode local.")
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
    p.add_argument("--unet-final-blocks", type=int, default=None,
                   help="Reallocate the 20-block U-Net budget to reserve this "
                        "many full-resolution blocks after the decoder "
                        "(e.g. 4 -> enc 4+4+4 / dec 4 / final 4). The default "
                        "allocation has final=0, i.e. no full-resolution "
                        "refinement after the decoder, which limits "
                        "high-frequency (chest/lung) noise modeling.")
    p.add_argument("--use-dilation",   action="store_true",
                   help="[5] Dilated depthwise conv per block.")
    p.add_argument("--use-freq-boost", action="store_true",
                   help="[6] Laplacian high-freq boost per block.")
    p.add_argument("--hu-bin-loss",    type=float, default=0.0, metavar="W",
                   help="[7] HU-bin bias penalty weight. "
                        "Architecture-independent (works with any --arch).")
    p.add_argument("--hu-bin-bins",    type=int, default=16,
                   help="DEPRECATED and ignored: bins are now the fixed "
                        "physical tissue boundaries "
                        "(-1024/-500/-200/200/600/1900 HU).")
    p.add_argument("--nps-weight", type=float, default=0.0, metavar="W",
                   help="[8] Radial noise-power-spectrum matching loss weight "
                        "(physics_losses.BatchRadialNPSLoss). Matches the "
                        "spectrum of the noise the model removes to the "
                        "paired LDCT-NDCT residual, preserving noise "
                        "texture. Additive on top of the base loss. "
                        "Architecture-independent (works with --arch redcnn). "
                        "Pairs naturally with --use-spectral-head: the head "
                        "provides the mechanism, this loss provides the "
                        "training signal.")

    # ── Loss flags ─────────────────────────────────────────────────────────────
    p.add_argument("--ssim-weight", type=float, default=0.0, metavar="W",
                   help="SSIM loss weight (requires pytorch-msssim).")
    p.add_argument("--l1-weight",   type=float, default=0.0, metavar="W",
                   help="L1 loss weight. Remainder goes to MSE.")
    p.add_argument("--grad-weight", type=float, default=0.0, metavar="W",
                   help="Gradient edge loss weight (finite diff L1). Additive.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
def image_gradient_loss(pred, target):
    dx_pred   = pred  [:, :, :, 1:] - pred  [:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred   = pred  [:, :, 1:, :] - pred  [:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def compute_loss(pred, target, ssim_weight=0.0, l1_weight=0.0, grad_weight=0.0,
                 hu_bin_loss_weight=0.0, hu_bin_bins=16,
                 input_img=None, nps_weight=0.0):
    mse_w = max(0.0, 1.0 - float(ssim_weight) - float(l1_weight))
    loss  = pred.new_zeros(())
    if mse_w > 0.0:
        loss = loss + mse_w * F.mse_loss(pred, target)
    if l1_weight > 0.0:
        loss = loss + float(l1_weight) * F.l1_loss(pred, target)
    if ssim_weight > 0.0:
        if not _HAS_MSSSIM:
            raise RuntimeError("pip install pytorch-msssim")
        # FIX: constant benchmark-derived data range instead of the previous
        # per-batch (target.max() - target.min()), which made the SSIM
        # stability constants batch-dependent.
        loss = loss + float(ssim_weight) * (
            1.0 - _pytorch_ssim(pred, target, data_range=_SSIM_DATA_RANGE,
                                size_average=True, nonnegative_ssim=True)
        )
    if grad_weight > 0.0:
        loss = loss + float(grad_weight) * image_gradient_loss(pred, target)
    if hu_bin_loss_weight > 0.0:
        loss = loss + hu_bin_loss_weight * hu_bin_bias_loss(pred, target)
    if nps_weight > 0.0:
        if input_img is None:
            raise ValueError("--nps-weight requires the network input image")
        # Match the spectrum of what the model removed (input - pred) to the
        # paired LDCT-NDCT residual (input - target): texture preservation.
        loss = loss + float(nps_weight) * _get_nps_loss()(
            input_img - pred, input_img - target
        )
    return loss


def hu_bin_bias_loss(pred, target, n_bins=None):
    """Mean-bias penalty inside FIXED physical tissue intervals.

    FIX: previously the bin edges were derived from each batch's min/max, so
    the loss compared different tissue ranges on every step. The boundaries
    are now the fixed physical intervals also used by
    physics_losses.HUBinBiasLoss, converted to the benchmark-standardized
    domain. ``n_bins`` is accepted for backward compatibility and ignored.
    """
    edges = [
        (b + 1024.0 - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD
        for b in _HU_BIN_BOUNDARIES_HU
    ]
    loss  = pred.new_zeros(())
    count = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (target >= lo) & (target < hi)
        if int(mask.sum()) < 10:
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


def _unet_block_alloc(n_blocks, final_blocks=None):
    """Return (enc_n, dec_n, final_n) for unet-decode mode."""
    if final_blocks is None:
        enc_n   = max(1, n_blocks // 4)
        dec_n   = max(1, n_blocks // 4)
        final_n = max(0, n_blocks - 3 * enc_n - dec_n)
        return enc_n, dec_n, final_n
    final_n = int(final_blocks)
    rem = n_blocks - final_n
    if final_n < 0 or rem < 4:
        raise ValueError(
            f"--unet-final-blocks must be in [0, {n_blocks - 4}], got {final_n}"
        )
    enc_n = max(1, rem // 4)
    dec_n = max(1, rem - 3 * enc_n)
    return enc_n, dec_n, final_n


def _unet_block_desc(n_blocks, final_blocks=None):
    enc_n, dec_n, final_n = _unet_block_alloc(n_blocks, final_blocks)
    return f"UNet-Decode(enc={enc_n},dec={dec_n},final={final_n})"


def build_model(arch, device, args):
    if arch == "local_residual":
        # U-Net decode needs more blocks so each branch has real depth.
        n_blocks = _BLOCKS_UNET if args.use_unet_decode else _BLOCKS_DEFAULT
        model = build_local_residual_model(
            device,
            channels=128, blocks=n_blocks, groups=args.groups,
            use_hu_gate=args.use_hu_gate,
            use_freq_boost=args.use_freq_boost,
            use_dilation=args.use_dilation,
            use_mu_mod=args.use_mu_mod,
            mu_split=args.mu_split,
            mu_mod_mode=args.mu_mod_mode,
            mu_local_window=args.mu_local_window,
            use_multi_res=args.use_multi_res,
            use_unet_decode=args.use_unet_decode,
            unet_final_blocks=args.unet_final_blocks,
        )
    else:
        model = build_benchmark_model(arch, device)
    if args.use_spectral_head:
        model = SpectralResidualModel(model, n_bins=args.spectral_bins).to(device)
        print(f"  Spectral head : ON ({args.spectral_bins} radial gain knots, "
              f"init G=1 -> starts exactly at the base model)")
    return model


@torch.no_grad()
def validate(model, loader, device, with_vif=False):
    """Validate with overall AND per-region (Chest/Abdomen) metrics.

    FIX: the previous version only reported the overall mean, which is
    dominated by abdomen slices; the chest gap was invisible during training
    and best_model.pt was selected on a metric that hides it.

    ``bench_ssim`` is the ldct-benchmark paper metric: overall SSIM computed
    WITHOUT clinical windowing, exactly as in ldctbench/utils/metrics.py
    (skimage structural_similarity(target, pred, data_range=2924) on
    denormalized images clipped to [0, 2924]).
    """
    model.eval()
    sums    = dict(mse=0.0, psnr=0.0, ssim=0.0, rmse=0.0,
                   baseline_psnr=0.0, vif=0.0, bench_ssim=0.0)
    region  = {
        "Chest":   dict(psnr=0.0, ssim=0.0, vif=0.0, n=0),
        "Abdomen": dict(psnr=0.0, ssim=0.0, vif=0.0, n=0),
    }
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
            ps = compute_psnr_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            ss = compute_ssim_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            vf = compute_vif_hu(pred_px[i].squeeze(), y_px[i].squeeze()) if with_vif else 0.0
            # ldct-benchmark paper SSIM: unwindowed, data_range = 2924, on
            # denormalized clipped images (see ldctbench/utils/metrics.py).
            t_np = y_px[i].squeeze().detach().cpu().numpy()
            p_np = pred_px[i].squeeze().detach().cpu().numpy()
            bss  = float(_sk_ssim(t_np, p_np,
                                  data_range=float(cfg.EVAL_DATA_RANGE)))
            sums["psnr"]          += ps
            sums["ssim"]          += ss
            sums["vif"]           += vf
            sums["bench_ssim"]    += bss
            sums["baseline_psnr"] += compute_psnr_windowed(x_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["rmse"]          += compute_rmse_hu(pred_px[i].squeeze(), y_px[i].squeeze())
            r = region[bt]
            r["psnr"] += ps
            r["ssim"] += ss
            r["vif"]  += vf
            r["n"]    += 1
            samples += 1
    n_b, n_s = max(1, batches), max(1, samples)
    out = {
        "mse":        sums["mse"]  / n_b,
        "psnr":       sums["psnr"] / n_s,
        "dpsnr":      (sums["psnr"] - sums["baseline_psnr"]) / n_s,
        "ssim":       sums["ssim"] / n_s,
        "rmse":       sums["rmse"] / n_s,
        "vif":        sums["vif"]  / n_s,
        "bench_ssim": sums["bench_ssim"] / n_s,
    }
    for name, r in region.items():
        key = name.lower()
        n   = max(1, r["n"])
        out[f"{key}_psnr"] = r["psnr"] / n
        out[f"{key}_ssim"] = r["ssim"] / n
        out[f"{key}_vif"]  = r["vif"]  / n
        out[f"{key}_n"]    = r["n"]
    return out


def selection_score(val, select_by):
    """Score used to pick best_model.pt. chest_* falls back to overall when
    the validation set contains no chest slices. 'bench_ssim' is the exact
    ldct-benchmark paper criterion (overall unwindowed SSIM)."""
    if select_by == "ssim":
        return val["ssim"]
    if select_by == "psnr":
        return val["psnr"]
    if select_by == "vif":
        return val["vif"]
    if select_by == "bench_ssim":
        return val["bench_ssim"]
    if select_by == "chest_ssim":
        return val["chest_ssim"] if val["chest_n"] > 0 else val["ssim"]
    if select_by == "chest_vif":
        return val["chest_vif"] if val["chest_n"] > 0 else val["vif"]
    raise ValueError(f"Unknown --select-by: {select_by}")


def lr_at(iteration, base_lr, min_lr, max_iter, schedule):
    """LR as a pure function of the iteration counter (resume-safe)."""
    if schedule != "cosine":
        return base_lr
    t = min(1.0, max(0.0, iteration / max(1, max_iter)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def train_cycle(model, loader, optimizer, device, iteration, max_iter,
                ssim_weight=0.0, l1_weight=0.0, grad_weight=0.0,
                hu_bin_loss_weight=0.0, hu_bin_bins=16,
                nps_weight=0.0,
                base_lr=1e-4, min_lr=1e-6, lr_schedule="constant"):
    model.train()
    total = count = 0.0
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if iteration >= max_iter:
            break
        lr_now = lr_at(iteration, base_lr, min_lr, max_iter, lr_schedule)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = compute_loss(pred, y, ssim_weight=ssim_weight, l1_weight=l1_weight,
                            grad_weight=grad_weight,
                            hu_bin_loss_weight=hu_bin_loss_weight,
                            hu_bin_bins=hu_bin_bins,
                            input_img=x, nps_weight=nps_weight)
        loss.backward()
        optimizer.step()
        iteration += 1
        total += float(loss.detach())
        count += 1
        bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}",
                        lr=f"{lr_now:.2e}")
    return iteration, total / max(1, count)


# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    if args.use_unet_decode and not args.use_multi_res:
        raise ValueError("--use-unet-decode requires --use-multi-res")
    if args.unet_final_blocks is not None and not args.use_unet_decode:
        raise ValueError("--unet-final-blocks requires --use-unet-decode")
    if args.ssim_weight + args.l1_weight > 1.0:
        raise ValueError("--ssim-weight + --l1-weight must not exceed 1.0")
    if args.ssim_weight > 0 and not _HAS_MSSSIM:
        raise RuntimeError("pip install pytorch-msssim")
    if args.nps_weight < 0.0:
        raise ValueError("--nps-weight must be >= 0")
    if args.spectral_bins < 2:
        raise ValueError("--spectral-bins must be >= 2")
    if args.train_patients is not None and args.train_patients < 1:
        raise ValueError("--train-patients must be >= 1")
    if args.val_patients is not None and args.val_patients < 1:
        raise ValueError("--val-patients must be >= 1")

    n_train, n_val = apply_split(args.split)
    if args.train_patients is not None:
        n_train = min(n_train, args.train_patients)
    if args.val_patients is not None:
        n_val = min(n_val, args.val_patients)
    pilot = args.train_patients is not None or args.val_patients is not None
    setup_reproducibility()
    device    = get_device()
    out_dir   = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    with_vif = bool(args.val_vif or args.select_by in ("vif", "chest_vif"))

    mse_w  = max(0.0, 1.0 - args.ssim_weight - args.l1_weight)
    if mse_w == 0.0 and (args.ssim_weight > 0.0 or args.l1_weight > 0.0):
        print(
            "\n" + "!" * 68 + "\n"
            "  WARNING: ssim_weight + l1_weight = 1.0 -> MSE weight is 0.\n"
            "  The benchmark protocol that reaches Chest VIF 0.221 (RED-CNN)\n"
            "  trains with pure MSE. Dropping MSE entirely is an untested\n"
            "  protocol deviation; keep MSE weight >= 0.3 unless this is a\n"
            "  deliberate, pre-registered ablation.\n"
            + "!" * 68 + "\n"
        )
    loss_parts = []
    if mse_w > 0.0:             loss_parts.append(f"{mse_w:.2f}*MSE")
    if args.l1_weight > 0.0:    loss_parts.append(f"{args.l1_weight:.2f}*L1")
    if args.ssim_weight > 0.0:  loss_parts.append(f"{args.ssim_weight:.2f}*SSIM")
    if args.grad_weight > 0.0:  loss_parts.append(f"{args.grad_weight:.2f}*Grad")
    loss_desc = " + ".join(loss_parts) if loss_parts else "1.00*MSE"
    if args.hu_bin_loss > 0.0:
        loss_desc += f" + {args.hu_bin_loss}*HU-bin"
    if args.nps_weight > 0.0:
        loss_desc += f" + {args.nps_weight}*NPS"

    n_blocks = _BLOCKS_UNET if args.use_unet_decode else _BLOCKS_DEFAULT
    active = []
    if args.use_spectral_head:
        active.append(f"Spectral-Head(bins={args.spectral_bins})")
    if args.use_hu_gate:       active.append("HU-gate")
    if args.use_mu_mod:
        mu_tag = f"mu-mod@{args.mu_split or 'auto'}"
        if args.mu_mod_mode == "local":
            mu_tag += f"(local,w={args.mu_local_window})"
        active.append(mu_tag)
    if args.use_unet_decode:   active.append(_unet_block_desc(n_blocks, args.unet_final_blocks))
    elif args.use_multi_res:   active.append("Multi-Res(3+3+3|1)")
    if args.use_dilation:      active.append("Dilation-2")
    if args.use_freq_boost:    active.append("Freq-boost")
    if args.hu_bin_loss > 0.0: active.append(f"HU-bin(w={args.hu_bin_loss})")
    if args.nps_weight > 0.0:  active.append(f"NPS(w={args.nps_weight})")
    active_str = ", ".join(active) if active else "none (baseline)"

    print(f"\n{'='*68}")
    print(f"  arch={args.arch.upper()} | split={args.split} | blocks={n_blocks}")
    print(f"  improvements: {active_str}")
    print(f"  Train patients : {n_train}  |  Val patients: {n_val}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output         : {out_dir}")
    print(f"  Loss           : {loss_desc}")
    print(f"  LR schedule    : {args.lr_schedule} (lr={args.lr:.2e}"
          + (f" -> {args.min_lr:.2e}" if args.lr_schedule == "cosine" else "")
          + ")")
    print(f"  Select best by : {args.select_by}"
          + (" [ldct-benchmark paper criterion]" if args.select_by == "bench_ssim" else "")
          + (" (+VIF in val)" if with_vif else ""))
    if pilot:
        print("  *** PILOT MODE : reduced patient subset. Use for config "
              "screening/ranking ONLY -- retrain the winner on the full "
              "split before reporting numbers. ***")
    print(f"{'='*68}\n")

    model     = build_model(args.arch, device, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    iteration  = 0
    best_score = -float("inf")
    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        iteration  = int(ckpt.get("iteration", 0))
        old_select = ckpt.get("select_by", "ssim")
        if old_select == args.select_by:
            best_score = float(ckpt.get("score", ckpt.get("ssim", -float("inf"))))
        else:
            print(f"  NOTE: checkpoint used --select-by {old_select}; "
                  f"resetting best score for {args.select_by}.")
        print(f"  Resumed at iter {iteration} | best {args.select_by} {best_score:.5f}")
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
        max_train_patients=args.train_patients,
        max_val_patients=args.val_patients,
    )

    print(f"Loss      : {loss_desc}")
    print(f"Optimizer : Adam(lr={args.lr:.2e}, schedule={args.lr_schedule})")

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
            nps_weight=args.nps_weight,
            base_lr=args.lr,
            min_lr=args.min_lr,
            lr_schedule=args.lr_schedule,
        )
        val   = validate(model, val_loader, device, with_vif=with_vif)
        score = selection_score(val, args.select_by)

        meta = {
            "architecture":    args.arch,
            "split":           args.split,
            "groups":          args.groups,
            "n_blocks":        n_blocks,
            "use_spectral_head": args.use_spectral_head,
            "spectral_bins":   args.spectral_bins,
            "use_hu_gate":     args.use_hu_gate,
            "use_freq_boost":  args.use_freq_boost,
            "use_dilation":    args.use_dilation,
            "use_mu_mod":      args.use_mu_mod,
            "mu_split":        eff_mu_split,
            "mu_mod_mode":     args.mu_mod_mode,
            "mu_local_window": args.mu_local_window,
            "use_multi_res":   args.use_multi_res,
            "use_unet_decode": args.use_unet_decode,
            "unet_final_blocks": args.unet_final_blocks,
            "hu_bin_loss":     args.hu_bin_loss,
            "nps_weight":      args.nps_weight,
            "ssim_weight":     args.ssim_weight,
            "l1_weight":       args.l1_weight,
            "mse_weight":      mse_w,
            "grad_weight":     args.grad_weight,
            "ssim_data_range": _SSIM_DATA_RANGE,
            "lr_schedule":     args.lr_schedule,
            "min_lr":          args.min_lr,
            "select_by":       args.select_by,
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
            "max_train_patients": args.train_patients,
            "max_val_patients":   args.val_patients,
            "pilot_mode":      pilot,
        }
        payload = {
            "model_state_dict": get_state_dict(model),
            "meta":      meta,
            "iteration": iteration,
            "ssim":      val["ssim"],
            "psnr":      val["psnr"],
            "val_mse":   val["mse"],
            "score":     score,
            "select_by": args.select_by,
            "val_detail": {k: v for k, v in val.items()},
        }
        torch.save(payload, os.path.join(out_dir, "last_model.pt"))
        if score > best_score:
            best_score = score
            torch.save(payload, os.path.join(out_dir, "best_model.pt"))
        torch.save({**payload, "optimizer_state": optimizer.state_dict()}, ckpt_path)

        elapsed = time.time() - t0
        region_str = (
            f"C-PSNR {val['chest_psnr']:.2f} C-SSIM {val['chest_ssim']:.4f} | "
            f"A-PSNR {val['abdomen_psnr']:.2f} A-SSIM {val['abdomen_ssim']:.4f}"
            if val["chest_n"] > 0 and val["abdomen_n"] > 0 else ""
        )
        vif_str = (
            f" | VIF {val['vif']:.4f}"
            + (f" (C {val['chest_vif']:.4f})" if val["chest_n"] > 0 else "")
            if with_vif else ""
        )
        print(
            f"Cycle {cycle:02d} | Iter {iteration:06d}/{args.max_iterations} | "
            f"Loss {train_loss:.6f} | Val MSE {val['mse']:.6f} | "
            f"PSNR {val['psnr']:.3f} | dPSNR {val['dpsnr']:+.3f} | "
            f"SSIM {val['ssim']:.5f} | bSSIM {val['bench_ssim']:.5f} | "
            f"RMSE {val['rmse']:.2f}"
            f"{vif_str} | {args.select_by} {score:.5f}"
            f"{(' | ' + region_str) if region_str else ''} | "
            f"{elapsed:.1f}s"
        )

    total = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{args.arch.upper()}] in {total} | "
          f"best {args.select_by}={best_score:.5f}")
    print(f"Checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")

    if pilot:
        print("\nPILOT MODE reminder: these numbers are for config "
              "ranking only. Retrain the winning config on the full "
              "split with the paper budget before reporting.")


if __name__ == "__main__":
    main()

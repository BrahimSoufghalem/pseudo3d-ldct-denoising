"""
LDCT Project - Training Script
=================================
Training loop, validation, checkpointing and TensorBoard logging.

Ablation axes are selected from the command line and each combination gets its
own run directory (runs/<input>_<mamba>/):

    python train.py --input-mode 2.5d --mamba-mode full
    python train.py --input-mode 2d   --mamba-mode basic

Mixed precision is DISABLED by default: bfloat16 was measured to cap this model
at about dPSNR +3.7 and to collapse around epoch 4 (see USE_AMP in config.py).

Stability notes
---------------
1. Gradient CLIPPING is not a safety net for Adam/AdamW: the optimiser
   normalises by the second moment, so a clipped spike still takes a full
   ~lr-sized step in a corrupted direction and pollutes the running variance.
   Spiking steps are therefore SKIPPED outright (--grad-skip-norm).

2. The skip guard is a seatbelt, NOT a cure. In the diverged wide-HU run it
   dropped 498/1113 steps in epoch 3 and the model collapsed anyway through the
   steps that stayed under the threshold. If you are seeing many spikes, the
   real cause is upstream - see the HU_RANGE_PRESET notes in config.py.

Objective notes
---------------
The loss can now be multi-scale (--no-ms-ssim to disable) and can be evaluated a
second time inside the clinical diagnostic window (--window-loss). Both are
motivated by the baseline-relative measurement documented in config.py: the
model reaches 89-112% of the required PSNR/SSIM gain but only 52-57% of the
required VIF gain.
"""

import argparse
import os
import statistics
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from config import (
    TOTAL_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP_MAX_NORM, WARMUP_EPOCHS,
    LAMBDA_L1, LAMBDA_SSIM, LAMBDA_EDGE,
    USE_MS_SSIM, WINDOW_LOSS_MODE, LAMBDA_WINDOW, VALID_WINDOW_LOSS_MODES,
    SCHEDULER_MIN_LR,
    A_MIN, A_MAX,
)
from utils import (
    setup_reproducibility, get_device,
    extract_centre_slice, get_state_dict, load_state_into,
)
from dataset import prepareCT2D
from model import build_model
from losses import MONAIHybridLoss
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed,
    compute_rmse_hu,
    denormalize_to_hu_offset,
)

# Abort the run when more than this fraction of an epoch's steps are skipped:
# at that point the optimiser is barely moving and the log is misleading.
SPIKE_RATE_ABORT = 0.5


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Train MS-NAFMambaNet for LDCT denoising")
    p.add_argument("--input-mode", default=cfg.INPUT_MODE, choices=list(cfg.VALID_INPUT_MODES),
                   help="2d = centre slice only, 2.5d = (prev, curr, next)")
    p.add_argument("--mamba-mode", default=cfg.MAMBA_MODE, choices=list(cfg.VALID_MAMBA_MODES))
    p.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS,
                   help="linear LR warmup length (clamped to epochs - 1)")
    p.add_argument("--grad-skip-norm", type=float, default=100.0,
                   help="skip the optimiser step when the pre-clip gradient norm "
                        "exceeds this value; 0 disables the guard")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute the Mamba bottleneck in backward to save VRAM")
    p.add_argument("--output-root", default=cfg.OUTPUT_ROOT)

    # ---- objective ----------------------------------------------------
    p.add_argument("--no-ms-ssim", action="store_true",
                   help="use single-scale SSIM instead of multi-scale SSIM")
    p.add_argument("--window-loss", default=WINDOW_LOSS_MODE,
                   choices=list(VALID_WINDOW_LOSS_MODES),
                   help="evaluate the loss inside the clinical window too "
                        "('extra'), only there ('only'), or not at all ('off')")
    p.add_argument("--lambda-window", type=float, default=LAMBDA_WINDOW,
                   help="weight of the windowed term when --window-loss=extra")
    return p.parse_args()


# ═══════════════════════════════════════════
# OPTIMISER PARAM GROUPS
# ═══════════════════════════════════════════
def build_param_groups(model, weight_decay):
    """Split parameters into decayed and non-decayed groups.

    The SSM tensors (`A_logs`, `Ds`, `dt_projs_bias`) carry a `_no_weight_decay`
    flag, and every 1-D tensor (biases, LayerNorm scales, the NAF beta/gamma
    residual gates) should be excluded too. Passing `model.parameters()`
    directly to AdamW silently ignored all of that and decayed them anyway.
    """
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, "_no_weight_decay", False) or param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def top_grad_params(model, k=5):
    """Return the k parameters carrying the largest gradient norm.

    Used to name the unstable tensor when a spike is detected, instead of
    guessing which part of the network blew up.
    """
    scored = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        scored.append((float(param.grad.detach().norm()), name))
    scored.sort(reverse=True)
    return scored[:k]


def body_types_of(batch, batch_size):
    """Extract 'Chest'/'Abdomen' per sample from a batch, or None if absent.

    The windowed loss term needs this to pick the lung window (1500 HU) or the
    soft tissue window (400 HU) for each slice. Returning None rather than a
    guess is deliberate: the loss then skips the windowed term and warns, which
    is far better than applying the wrong window to half the batch.
    """
    raw = batch.get("body_type", None) if hasattr(batch, "get") else None
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raw = [raw] * batch_size
    return [str(raw[i] if i < len(raw) else raw[-1]) for i in range(batch_size)]


# ═══════════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════════
def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, total_epochs,
                    scaler=None, amp_dtype=None, use_amp=False,
                    grad_skip_norm=100.0, spike_report_budget=None):
    """Run one training epoch.

    Returns a dict with the average loss and the gradient statistics needed to
    diagnose instability.
    """
    model.train()
    train_loss = 0.0
    skipped_nonfinite = 0
    spikes = 0
    max_gnorm = 0.0
    gnorm_samples = []
    if spike_report_budget is None:
        spike_report_budget = [5]

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs} [Train]",
        unit="batch", leave=False, dynamic_ncols=True,
    )

    for batch in train_bar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mid_slice = extract_centre_slice(images)
        body_types = body_types_of(batch, images.shape[0])

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred_res = model(images)
            # The loss runs on the UNCLAMPED prediction so saturated pixels keep
            # a restoring gradient. MONAIHybridLoss clamps only its SSIM term,
            # which is undefined outside [0, 1] - see losses.py.
            pred_img = mid_slice + pred_res
            loss, loss_info = loss_fn(pred_img.float(), labels.float(), body_types)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # clip_grad_norm_ returns the norm BEFORE clipping, which is the signal
        # we actually care about.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        finite = bool(torch.isfinite(gnorm))
        gnorm_val = float(gnorm) if finite else float("inf")
        spiking = grad_skip_norm > 0 and finite and gnorm_val > grad_skip_norm

        if spiking and spike_report_budget[0] > 0:
            spike_report_budget[0] -= 1
            offenders = ", ".join(f"{n}={v:.1f}" for v, n in top_grad_params(model))
            tqdm.write(
                f"  [spike] epoch {epoch + 1} | |g|={gnorm_val:.1f} > {grad_skip_norm:.0f} "
                f"| loss={loss.item():.4f} | step skipped | top grads: {offenders}"
            )

        if not finite:
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
        elif spiking:
            spikes += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            max_gnorm = max(max_gnorm, gnorm_val)
            gnorm_samples.append(gnorm_val)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
            else:
                optimizer.step()

        if scaler is not None and scaler.is_enabled():
            scaler.update()

        train_loss += loss.item()
        postfix = {
            "loss": f"{loss.item():.4f}",
            "L1": f"{loss_info['L1']:.4f}",
            "SSIM": f"{loss_info['SSIM']:.4f}",
            "gn": f"{gnorm_val:.1f}",
            "sp": spikes,
        }
        if "Window" in loss_info:
            postfix["win"] = f"{loss_info['Window']:.4f}"
        train_bar.set_postfix(**postfix)

    n_steps = max(1, len(train_loader))
    return {
        "avg_train": train_loss / n_steps,
        "skipped": skipped_nonfinite,
        "spikes": spikes,
        "spike_rate": (spikes + skipped_nonfinite) / n_steps,
        "max_gnorm": max_gnorm,
        "median_gnorm": statistics.median(gnorm_samples) if gnorm_samples else 0.0,
    }


# ═══════════════════════════════════════════
# VALIDATE ONE EPOCH
# ═══════════════════════════════════════════
@torch.no_grad()
def validate_one_epoch(model, val_loader, loss_fn, device, epoch, total_epochs,
                       amp_dtype=None, use_amp=False):
    """
    One validation epoch with ldct-benchmark physical HU metrics.

    The LOSS uses the same unclamped prediction as training (so train/val curves
    are comparable); the METRICS use the clamped image, which is what a real
    viewer would display.
    """
    model.eval()
    val_loss = 0.0
    val_psnr_sum = val_ssim_sum = val_rmse_sum = baseline_psnr_sum = 0.0
    total_samples = 0
    psnr_chest, psnr_abd, ssim_chest, ssim_abd = [], [], [], []
    viz_images = None

    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs} [Val]",
        unit="batch", leave=False, dynamic_ncols=True,
    )

    for i, batch in enumerate(val_bar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mid_slice = extract_centre_slice(images)
        body_types = body_types_of(batch, images.shape[0])

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred_res = model(images)

        pred_img = (mid_slice + pred_res).float()
        loss, _ = loss_fn(pred_img, labels.float(), body_types)   # unclamped, like training
        preds = pred_img.clamp(0.0, 1.0)                          # clamped, for metrics/viz

        val_loss += loss.item()

        for b_idx in range(preds.shape[0]):
            pred_hu = denormalize_to_hu_offset(preds[b_idx:b_idx + 1], A_MIN, A_MAX).squeeze()
            lbl_hu = denormalize_to_hu_offset(labels[b_idx:b_idx + 1], A_MIN, A_MAX).squeeze()
            mid_hu = denormalize_to_hu_offset(mid_slice[b_idx:b_idx + 1], A_MIN, A_MAX).squeeze()

            bt = "Abdomen"
            if body_types is not None:
                bt = "Chest" if body_types[b_idx].strip().lower().startswith("c") else "Abdomen"

            p_val = compute_psnr_windowed(pred_hu, lbl_hu, bt)
            b_val = compute_psnr_windowed(mid_hu, lbl_hu, bt)
            s_val = compute_ssim_windowed(pred_hu, lbl_hu, bt)
            r_val = compute_rmse_hu(pred_hu, lbl_hu)

            val_psnr_sum += p_val
            baseline_psnr_sum += b_val
            val_ssim_sum += s_val
            val_rmse_sum += r_val
            total_samples += 1

            if bt == "Chest":
                psnr_chest.append(p_val)
                ssim_chest.append(s_val)
            else:
                psnr_abd.append(p_val)
                ssim_abd.append(s_val)

        if i == 0:
            viz_images = (mid_slice.float().cpu(), labels.float().cpu(), preds.float().cpu())

        val_bar.set_postfix(loss=f"{loss.item():.4f}")

    n_val = max(1, len(val_loader))
    n_samples = max(1, total_samples)

    return {
        "avg_val": val_loss / n_val,
        "avg_psnr": val_psnr_sum / n_samples,
        "avg_rmse": val_rmse_sum / n_samples,
        "avg_baseline": baseline_psnr_sum / n_samples,
        "avg_ssim": val_ssim_sum / n_samples,
        "avg_psnr_chest": sum(psnr_chest) / max(1, len(psnr_chest)),
        "avg_psnr_abd": sum(psnr_abd) / max(1, len(psnr_abd)),
        "avg_ssim_chest": sum(ssim_chest) / max(1, len(ssim_chest)),
        "avg_ssim_abd": sum(ssim_abd) / max(1, len(ssim_abd)),
        "viz_images": viz_images,
    }


# ═══════════════════════════════════════════
# CHECKPOINT HELPERS
# ═══════════════════════════════════════════
def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, best_ssim, best_psnr, patience_counter, meta=None):
    """Save a resumable checkpoint. Weights are always stored unwrapped."""
    torch.save({
        "epoch": epoch,
        "model_state": get_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_ssim": best_ssim,
        "best_psnr": best_psnr,
        "patience_counter": patience_counter,
        "meta": meta or {},
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Resume from `path` if it exists."""
    if not os.path.exists(path):
        return 0, float("inf"), -float("inf"), -float("inf"), 0

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    load_state_into(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    best_ssim = checkpoint.get("best_ssim", -float("inf"))
    best_psnr = checkpoint.get("best_psnr", -float("inf"))
    patience_counter = checkpoint.get("patience_counter", 0)
    print(f"Resumed from epoch {start_epoch} | best SSIM={best_ssim:.4f} | best PSNR={best_psnr:.2f} dB")
    return start_epoch, best_val_loss, best_ssim, best_psnr, patience_counter


# ═══════════════════════════════════════════
# TENSORBOARD LOGGING
# ═══════════════════════════════════════════
def log_to_tensorboard(writer, epoch, train_stats, metrics, current_lr, epoch_time):
    writer.add_scalars("Loss", {"Train": train_stats["avg_train"], "Val": metrics["avg_val"]}, epoch + 1)
    writer.add_scalar("Metrics/PSNR", metrics["avg_psnr"], epoch + 1)
    writer.add_scalar("Metrics/DELTA_PSNR", metrics["avg_psnr"] - metrics["avg_baseline"], epoch + 1)
    writer.add_scalar("Metrics/SSIM", metrics["avg_ssim"], epoch + 1)
    writer.add_scalar("Metrics/RMSE", metrics["avg_rmse"], epoch + 1)
    writer.add_scalar("Chest/PSNR", metrics["avg_psnr_chest"], epoch + 1)
    writer.add_scalar("Chest/SSIM", metrics["avg_ssim_chest"], epoch + 1)
    writer.add_scalar("Abdomen/PSNR", metrics["avg_psnr_abd"], epoch + 1)
    writer.add_scalar("Abdomen/SSIM", metrics["avg_ssim_abd"], epoch + 1)
    writer.add_scalar("Training/LR", current_lr, epoch + 1)
    writer.add_scalar("Training/EpochTime", epoch_time, epoch + 1)
    writer.add_scalar("Training/GradNormMax", train_stats["max_gnorm"], epoch + 1)
    writer.add_scalar("Training/GradNormMedian", train_stats["median_gnorm"], epoch + 1)
    writer.add_scalar("Training/GradSpikes", train_stats["spikes"], epoch + 1)

    if (epoch + 1) % 10 == 0 and metrics["viz_images"] is not None:
        inp, lbl, out = metrics["viz_images"]
        writer.add_images("Images/LDCT_Input", inp.repeat(1, 3, 1, 1), epoch + 1)
        writer.add_images("Images/NDCT_Target", lbl.repeat(1, 3, 1, 1), epoch + 1)
        writer.add_images("Images/Enhanced_Output", out.repeat(1, 3, 1, 1), epoch + 1)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    args = parse_args()
    input_mode = cfg.normalize_input_mode(args.input_mode)
    mamba_mode = cfg.normalize_mamba_mode(args.mamba_mode)
    total_epochs = args.epochs
    warmup_epochs = max(1, min(args.warmup_epochs, total_epochs - 1))
    if warmup_epochs != args.warmup_epochs:
        print(f"Note: warmup clamped from {args.warmup_epochs} to {warmup_epochs} "
              f"(needs to stay below --epochs).")

    paths = cfg.run_paths(mamba_mode=mamba_mode, input_mode=input_mode, output_root=args.output_root)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)

    setup_reproducibility()
    device = get_device()

    print(f"Run: input_mode={input_mode} | mamba_mode={mamba_mode} | device={device}")
    print(f"Outputs -> {paths['run_dir']}")
    print(f"HU range: [{A_MIN}, {A_MAX}] (preset '{cfg.HU_RANGE_PRESET}')")
    print(f"LR: {args.lr:.2e} | warmup: {warmup_epochs} | wd: {WEIGHT_DECAY} | "
          f"grad clip: {GRAD_CLIP_MAX_NORM} | grad skip: {args.grad_skip_norm}")

    # ---- AMP setup --------------------------------------------------
    use_amp = cfg.USE_AMP and (not args.no_amp) and device.type == "cuda"
    amp_dtype = None
    scaler = None
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))
        print(f"Mixed precision: {amp_dtype} (GradScaler={'on' if scaler.is_enabled() else 'off'})")
    else:
        print("Mixed precision: disabled (FP32)")

    # ---- Model / loss / optimiser -----------------------------------
    model = build_model(
        device,
        mamba_mode=mamba_mode,
        input_mode=input_mode,
        use_checkpoint=args.grad_checkpoint or cfg.USE_GRAD_CHECKPOINT,
    )

    loss_fn = MONAIHybridLoss(
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_edge=LAMBDA_EDGE,
        spatial_dims=2,
        use_ms_ssim=USE_MS_SSIM and not args.no_ms_ssim,
        window_mode=args.window_loss,
        lambda_window=args.lambda_window,
    ).to(device)
    print(f"Loss: {loss_fn.describe()}")

    optimizer = torch.optim.AdamW(build_param_groups(model, WEIGHT_DECAY), lr=args.lr)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=SCHEDULER_MIN_LR,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
    )

    writer = SummaryWriter(log_dir=paths["logs"])
    print(f"TensorBoard logs -> {paths['logs']}")

    start_epoch, best_val_loss, best_ssim, best_psnr, patience_counter = load_checkpoint(
        paths["checkpoint"], model, optimizer, scheduler, scaler, device
    )

    train_loader, val_loader = prepareCT2D(
        input_mode=input_mode,
        train_batch_size=args.batch_size,
    )

    meta = {"input_mode": input_mode, "mamba_mode": mamba_mode,
            "hu_range": [A_MIN, A_MAX], "hu_preset": cfg.HU_RANGE_PRESET,
            "loss": loss_fn.describe()}

    spike_report_budget = [8]      # shared across epochs: report the first few only
    training_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, total_epochs,
            scaler=scaler, amp_dtype=amp_dtype, use_amp=use_amp,
            grad_skip_norm=args.grad_skip_norm, spike_report_budget=spike_report_budget,
        )
        metrics = validate_one_epoch(
            model, val_loader, loss_fn, device, epoch, total_epochs,
            amp_dtype=amp_dtype, use_amp=use_amp,
        )

        delta_psnr = metrics["avg_psnr"] - metrics["avg_baseline"]
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        scheduler.step()

        print(
            f"Epoch [{epoch + 1:03d}/{total_epochs}] "
            f"Train: {train_stats['avg_train']:.5f} | Val: {metrics['avg_val']:.5f} | "
            f"PSNR: {metrics['avg_psnr']:.3f} dB | dPSNR: {delta_psnr:+.3f} dB | "
            f"SSIM: {metrics['avg_ssim']:.5f} | RMSE: {metrics['avg_rmse']:.3f} | "
            f"|g|med: {train_stats['median_gnorm']:.2f} | |g|max: {train_stats['max_gnorm']:.1f} | "
            f"spikes: {train_stats['spikes']}/{len(train_loader)} | "
            f"nonfinite: {train_stats['skipped']} | "
            f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
        )

        log_to_tensorboard(writer, epoch, train_stats, metrics, current_lr, epoch_time)

        if metrics["avg_psnr"] > best_psnr:
            best_psnr = metrics["avg_psnr"]
            best_ssim = metrics["avg_ssim"]
            best_val_loss = metrics["avg_val"]
            patience_counter = 0
            torch.save({"model_state_dict": get_state_dict(model), "meta": meta,
                        "psnr": best_psnr, "ssim": best_ssim, "epoch": epoch},
                       paths["best_model"])
            print(f"  Best model saved: PSNR={best_psnr:.2f} | SSIM={best_ssim:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(paths["checkpoint"], epoch, model, optimizer, scheduler, scaler,
                        best_val_loss, best_ssim, best_psnr, patience_counter, meta=meta)

        # A mostly-skipped epoch means the optimiser is frozen. Continuing would
        # print progress lines that look like training but change nothing.
        if train_stats["spike_rate"] > SPIKE_RATE_ABORT:
            print(
                f"\nABORT: {train_stats['spike_rate'] * 100:.0f}% of steps were skipped this "
                f"epoch, so the optimiser barely moved.\n"
                f"The skip guard cannot rescue a run that is already diverging - fix the "
                f"cause instead.\n"
                f"Check the [spike] lines above for the dominant tensor, and verify "
                f"HU_RANGE_PRESET in config.py ('legacy' is the stable setting).\n"
                f"Best weights so far are kept at {paths['best_model']}."
            )
            break

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
    print("\nTraining complete!")
    print(f"Total time  : {total_time}")
    print(f"Best PSNR   : {best_psnr:.2f} dB")
    print(f"Best SSIM   : {best_ssim:.4f}")
    print(f"Model saved : {paths['best_model']}")

    writer.close()


if __name__ == "__main__":
    main()

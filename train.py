"""
LDCT Project - Training Script
=================================
Training loop, validation, checkpointing and TensorBoard logging.

Ablation axes are selected from the command line and each combination gets its
own run directory (runs/<input>_<mamba>/):

    python train.py --input-mode 2.5d --mamba-mode full
    python train.py --input-mode 2d   --mamba-mode basic

Mixed precision is enabled by default on CUDA (bfloat16 when supported,
otherwise float16 + GradScaler). Use --no-amp for pure FP32.
"""

import argparse
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from config import (
    TOTAL_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP_MAX_NORM, WARMUP_EPOCHS,
    LAMBDA_L1, LAMBDA_SSIM, LAMBDA_EDGE,
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
                   help="linear LR warmup length; raise it if training diverges early")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute the Mamba bottleneck in backward to save VRAM")
    p.add_argument("--output-root", default=cfg.OUTPUT_ROOT)
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


# ═══════════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════════
def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, total_epochs,
                    scaler=None, amp_dtype=None, use_amp=False):
    """One training epoch. Returns (avg_train_loss, skipped_steps, max_grad_norm)."""
    model.train()
    train_loss = 0.0
    skipped_steps = 0
    max_gnorm = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs} [Train]",
        unit="batch", leave=False, dynamic_ncols=True,
    )

    for batch in train_bar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mid_slice = extract_centre_slice(images)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred_res = model(images)
            # NOTE: the loss is computed on the UNCLAMPED prediction, exactly as
            # in validation. Clamping here would zero the gradient of every
            # saturated pixel. (MONAIHybridLoss clamps the SSIM term internally,
            # because SSIM is undefined outside [0, 1] - see losses.py.)
            pred_img = mid_slice + pred_res
            loss, loss_info = loss_fn(pred_img.float(), labels.float())

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            if not torch.isfinite(gnorm):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
            else:
                max_gnorm = max(max_gnorm, float(gnorm))
                scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            if not torch.isfinite(gnorm):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
            else:
                max_gnorm = max(max_gnorm, float(gnorm))
                optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            L1=f"{loss_info['L1']:.4f}",
            SSIM=f"{loss_info['SSIM']:.4f}",
            Edge=f"{loss_info['Edge']:.4f}",
            gn=f"{float(gnorm):.1f}",
            sk=skipped_steps,
        )

    return train_loss / max(1, len(train_loader)), skipped_steps, max_gnorm


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

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred_res = model(images)

        pred_img = (mid_slice + pred_res).float()
        loss, _ = loss_fn(pred_img, labels.float())          # unclamped, like training
        preds = pred_img.clamp(0.0, 1.0)                     # clamped, for metrics/viz

        val_loss += loss.item()
        body_types = batch.get("body_type", None)

        for b_idx in range(preds.shape[0]):
            pred_hu = denormalize_to_hu_offset(preds[b_idx:b_idx + 1], A_MIN, A_MAX).squeeze()
            lbl_hu = denormalize_to_hu_offset(labels[b_idx:b_idx + 1], A_MIN, A_MAX).squeeze()
            mid_hu = denormalize_to_hu_offset(mid_slice[b_idx:b_idx + 1], A_MIN, A_MAX).squeeze()

            bt = "Abdomen"
            if body_types is not None:
                bt_raw = body_types[b_idx] if isinstance(body_types, (list, tuple)) else body_types
                bt = "Chest" if str(bt_raw).lower().startswith("c") else "Abdomen"

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
def log_to_tensorboard(writer, epoch, avg_train, metrics, current_lr, epoch_time, max_gnorm=0.0):
    writer.add_scalars("Loss", {"Train": avg_train, "Val": metrics["avg_val"]}, epoch + 1)
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
    writer.add_scalar("Training/GradNormMax", max_gnorm, epoch + 1)

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
    warmup_epochs = max(0, min(args.warmup_epochs, total_epochs - 1))

    paths = cfg.run_paths(mamba_mode=mamba_mode, input_mode=input_mode, output_root=args.output_root)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)

    setup_reproducibility()
    device = get_device()

    print(f"Run: input_mode={input_mode} | mamba_mode={mamba_mode} | device={device}")
    print(f"Outputs -> {paths['run_dir']}")
    print(f"HU range: [{A_MIN}, {A_MAX}] (preset '{cfg.HU_RANGE_PRESET}')")
    print(f"LR: {args.lr:.2e} | warmup epochs: {warmup_epochs} | weight decay: {WEIGHT_DECAY}")

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
    ).to(device)

    optimizer = torch.optim.AdamW(build_param_groups(model, WEIGHT_DECAY), lr=args.lr)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=max(1, warmup_epochs),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=SCHEDULER_MIN_LR,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[max(1, warmup_epochs)],
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
            "hu_range": [A_MIN, A_MAX], "hu_preset": cfg.HU_RANGE_PRESET}

    training_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        avg_train, skipped_steps, max_gnorm = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, total_epochs,
            scaler=scaler, amp_dtype=amp_dtype, use_amp=use_amp,
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
            f"Train: {avg_train:.5f} | Val: {metrics['avg_val']:.5f} | "
            f"PSNR: {metrics['avg_psnr']:.3f} dB | dPSNR: {delta_psnr:+.3f} dB | "
            f"SSIM: {metrics['avg_ssim']:.5f} | RMSE: {metrics['avg_rmse']:.3f} | "
            f"|g|max: {max_gnorm:.1f} | "
            f"skipped: {skipped_steps}/{len(train_loader)} | "
            f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
        )

        log_to_tensorboard(writer, epoch, avg_train, metrics, current_lr, epoch_time, max_gnorm)

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

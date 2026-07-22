"""
LDCT Project — Training Script
=================================
Main entry point: training loop, validation, checkpointing, TensorBoard logging.
"""

import os
import time 
 
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import SSIMMetric
from tqdm import tqdm

from config import (
    MODEL_DIR, LOGS_DIR, CHECKPOINT_PATH, BEST_MODEL_PATH,
    TOTAL_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP_MAX_NORM,
    LAMBDA_L1, LAMBDA_SSIM, LAMBDA_PERC, LAMBDA_EDGE,
    SCHEDULER_MODE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_MIN_LR,
    A_MIN, A_MAX,
)
from utils import setup_reproducibility, get_device
from dataset import prepareCT2D
from model import build_model
from losses import MONAIHybridLoss
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed,
    compute_rmse_hu, compute_vif_hu,
    denormalize_to_hu_offset, psnr, rmse, VIFMetric
)


# ═══════════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════════
def train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch, total_epochs):
    """Run one training epoch. Returns average training loss."""
    model.train()
    train_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs} [Train]",
        unit="batch", leave=False, dynamic_ncols=True,
    )

    for batch in train_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        mid_slice = images[:, 1:2, :, :]

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pred_img = model(images)
            loss, loss_info = loss_fn(pred_img, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            L1=f"{loss_info['L1']:.4f}",
            SSIM=f"{loss_info['SSIM']:.4f}",
            Edge=f"{loss_info['Edge']:.4f}",
        )

    avg_train = train_loss / max(1, len(train_loader))
    return avg_train


# ═══════════════════════════════════════════
# VALIDATE ONE EPOCH
# ═══════════════════════════════════════════
@torch.no_grad()
def validate_one_epoch(model, val_loader, loss_fn, ssim_metric, vif_metric, device, epoch, total_epochs):
    """
    Run one validation epoch using exact ldct-benchmark physical HU metrics.
    Returns a dict containing all averaged metrics and visualization tensors.
    """
    model.eval()
    val_loss = 0.0
    val_psnr_sum = 0.0
    val_ssim_sum = 0.0
    val_rmse_sum = 0.0
    val_vif_sum = 0.0
    baseline_psnr_sum = 0.0
    total_samples = 0

    psnr_chest, psnr_abd = [], []
    ssim_chest, ssim_abd = [], []
    vif_chest, vif_abd = [], []

    viz_images = None

    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs} [Val]",
        unit="batch", leave=False, dynamic_ncols=True,
    )

    for i, batch in enumerate(val_bar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        mid_slice = images[:, 1:2, :, :]

        with autocast("cuda"):
            preds = model(images)
            loss, _ = loss_fn(preds, labels)

        val_loss += loss.item()
        body_types = batch.get("body_type", None)

        for b_idx in range(preds.shape[0]):
            pred_hu = denormalize_to_hu_offset(preds[b_idx:b_idx+1], A_MIN, A_MAX).squeeze()
            lbl_hu = denormalize_to_hu_offset(labels[b_idx:b_idx+1], A_MIN, A_MAX).squeeze()
            mid_hu = denormalize_to_hu_offset(mid_slice[b_idx:b_idx+1], A_MIN, A_MAX).squeeze()

            bt = "Abdomen"
            if body_types is not None:
                bt_raw = body_types[b_idx] if isinstance(body_types, (list, tuple)) else body_types
                bt = "Chest" if str(bt_raw).lower().startswith("c") else "Abdomen"

            p_val = compute_psnr_windowed(pred_hu, lbl_hu, bt)
            b_val = compute_psnr_windowed(mid_hu, lbl_hu, bt)
            s_val = compute_ssim_windowed(pred_hu, lbl_hu, bt)
            r_val = compute_rmse_hu(pred_hu, lbl_hu)
            v_val = compute_vif_hu(pred_hu, lbl_hu)

            val_psnr_sum += p_val
            baseline_psnr_sum += b_val
            val_ssim_sum += s_val
            val_rmse_sum += r_val
            val_vif_sum += v_val
            total_samples += 1

            if bt == "Chest":
                psnr_chest.append(p_val)
                ssim_chest.append(s_val)
                vif_chest.append(v_val)
            else:
                psnr_abd.append(p_val)
                ssim_abd.append(s_val)
                vif_abd.append(v_val)

        if i == 0:
            viz_images = (
                mid_slice.float().cpu(),
                labels.float().cpu(),
                preds.float().cpu(),
            )

        val_bar.set_postfix(loss=f"{loss.item():.4f}")

    n_val = max(1, len(val_loader))
    n_samples = max(1, total_samples)

    return {
        "avg_val": val_loss / n_val,
        "avg_psnr": val_psnr_sum / n_samples,
        "avg_rmse": val_rmse_sum / n_samples,
        "avg_baseline": baseline_psnr_sum / n_samples,
        "avg_ssim": val_ssim_sum / n_samples,
        "avg_vif": val_vif_sum / n_samples,
        "avg_psnr_chest": sum(psnr_chest) / max(1, len(psnr_chest)),
        "avg_psnr_abd": sum(psnr_abd) / max(1, len(psnr_abd)),
        "avg_ssim_chest": sum(ssim_chest) / max(1, len(ssim_chest)),
        "avg_ssim_abd": sum(ssim_abd) / max(1, len(ssim_abd)),
        "avg_vif_chest": sum(vif_chest) / max(1, len(vif_chest)),
        "avg_vif_abd": sum(vif_abd) / max(1, len(vif_abd)),
        "viz_images": viz_images,
    }


# ═══════════════════════════════════════════
# CHECKPOINT HELPERS
# ═══════════════════════════════════════════
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, best_ssim, best_psnr, best_vif, patience_counter):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_ssim": best_ssim,
        "best_psnr": best_psnr,
        "best_vif": best_vif,
        "patience_counter": patience_counter,
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer, scheduler, device):
    """Load checkpoint if it exists. Returns (start_epoch, best_val_loss, best_ssim, best_psnr, best_vif, patience_counter)."""
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_ssim = checkpoint.get("best_ssim", -float("inf"))
        best_psnr = checkpoint.get("best_psnr", -float("inf"))
        best_vif = checkpoint.get("best_vif", -float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        print(
            f"✅ Resumed from epoch {start_epoch} | "
            f"Best SSIM={best_ssim:.4f} | Best PSNR={best_psnr:.2f} dB"
        )
        return start_epoch, best_val_loss, best_ssim, best_psnr, best_vif, patience_counter

    return 0, float("inf"), -float("inf"), -float("inf"), -float("inf"), 0


# ═══════════════════════════════════════════
# TENSORBOARD LOGGING
# ═══════════════════════════════════════════
def log_to_tensorboard(writer, epoch, avg_train, metrics, current_lr, epoch_time):
    """Log training/validation metrics and images to TensorBoard."""
    writer.add_scalars("Loss", {"Train": avg_train, "Val": metrics["avg_val"]}, epoch + 1)
    writer.add_scalar("Metrics/PSNR", metrics["avg_psnr"], epoch + 1)
    writer.add_scalar("Metrics/DELTA_PSNR", metrics["avg_psnr"] - metrics["avg_baseline"], epoch + 1)
    writer.add_scalar("Metrics/SSIM", metrics["avg_ssim"], epoch + 1)
    writer.add_scalar("Metrics/RMSE", metrics["avg_rmse"], epoch + 1)
    writer.add_scalar("Metrics/VIF", metrics["avg_vif"], epoch + 1)
    writer.add_scalar("Chest/PSNR", metrics["avg_psnr_chest"], epoch + 1)
    writer.add_scalar("Chest/SSIM", metrics["avg_ssim_chest"], epoch + 1)
    writer.add_scalar("Chest/VIF", metrics["avg_vif_chest"], epoch + 1)
    writer.add_scalar("Abdomen/PSNR", metrics["avg_psnr_abd"], epoch + 1)
    writer.add_scalar("Abdomen/SSIM", metrics["avg_ssim_abd"], epoch + 1)
    writer.add_scalar("Abdomen/VIF", metrics["avg_vif_abd"], epoch + 1)
    writer.add_scalar("Training/LR", current_lr, epoch + 1)
    writer.add_scalar("Training/EpochTime", epoch_time, epoch + 1)

    if (epoch + 1) % 10 == 0 and metrics["viz_images"] is not None:
        inp, lbl, out = metrics["viz_images"]
        writer.add_images("Images/LDCT_Input", inp.repeat(1, 3, 1, 1), epoch + 1)
        writer.add_images("Images/NDCT_Target", lbl.repeat(1, 3, 1, 1), epoch + 1)
        writer.add_images("Images/Enhanced_Output", out.repeat(1, 3, 1, 1), epoch + 1)


# ═══════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════
def main():
    # ── Setup ──
    setup_reproducibility()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    device = get_device()

    # ── Model, Loss, Optimizer, Scheduler ──
    model = build_model(device)

    loss_fn = MONAIHybridLoss(
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_perc=LAMBDA_PERC,
        lambda_edge=LAMBDA_EDGE,
        spatial_dims=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    # ── Metrics ──
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0, reduction="mean")
    vif_metric = VIFMetric(device=device)

    # ── TensorBoard & Checkpoint ──
    writer = SummaryWriter(log_dir=LOGS_DIR)
    print(f"📊  TensorBoard logs → {LOGS_DIR}")

    start_epoch, best_val_loss, best_ssim, best_psnr, best_vif, patience_counter = \
        load_checkpoint(model, optimizer, scheduler, device)

    # ── Data ──
    train_loader, val_loader = prepareCT2D()

    # ── GradScaler ──
    scaler = GradScaler("cuda")

    training_start = time.time()

    # ── Training Loop ──
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start = time.time()

        # Train
        avg_train = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch, TOTAL_EPOCHS)

        # Validate
        metrics = validate_one_epoch(model, val_loader, loss_fn, ssim_metric, vif_metric, device, epoch, TOTAL_EPOCHS)

        delta_psnr = metrics["avg_psnr"] - metrics["avg_baseline"]
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
        eta = time.strftime("%H:%M:%S", time.gmtime(epoch_time * (TOTAL_EPOCHS - epoch - 1)))

        scheduler.step(metrics["avg_psnr"])

        # Print summary
        print(
            f"Epoch [{epoch + 1:03d}/{TOTAL_EPOCHS}] "
            f"Train: {avg_train:.4f}↓ | Val: {metrics['avg_val']:.4f}↓ | "
            f"PSNR: {metrics['avg_psnr']:.2f} dB↑ | ΔPSNR: +{delta_psnr:.2f} dB | "
            f"SSIM: {metrics['avg_ssim']:.4f}↑ | RMSE: {metrics['avg_rmse']:.4f}↓ | "
            f"VIF: {metrics['avg_vif']:.4f}↑ | "
            f"Chest PSNR: {metrics['avg_psnr_chest']:.2f} | Abd PSNR: {metrics['avg_psnr_abd']:.2f} | "
            f"LR: {current_lr:.2e} | ⏱️ {epoch_time:.1f}s | Elapsed: {elapsed} | ETA: {eta}"
        )

        # Log to TensorBoard
        log_to_tensorboard(writer, epoch, avg_train, metrics, current_lr, epoch_time)

        # Check for best model
        if metrics["avg_psnr"] > best_psnr:
            best_psnr = metrics["avg_psnr"]
            best_ssim = metrics["avg_ssim"]
            best_vif = metrics["avg_vif"]
            best_val_loss = metrics["avg_val"]
            patience_counter = 0

            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(
                f"  ✅ Best model saved! "
                f"PSNR={best_psnr:.2f} | "
                f"SSIM={best_ssim:.4f} | "
                f"VIF={best_vif:.4f}"
            )
        else:
            patience_counter += 1

        # Save checkpoint (every epoch)
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, best_ssim, best_psnr, best_vif, patience_counter)

        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch + 1}")
            break

    # ── Final Summary ──
    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
    print(f"\n🎉 Training complete!")
    print(f"⏱️ Total time  : {total_time}")
    print(f"📊 Best PSNR   : {best_psnr:.2f} dB")
    print(f"📊 Best SSIM   : {best_ssim:.4f}")
    print(f"📊 Best VIF    : {best_vif:.4f}")
    print(f"📂 Model saved : {BEST_MODEL_PATH}")

    writer.close()


if __name__ == "__main__":
    main()

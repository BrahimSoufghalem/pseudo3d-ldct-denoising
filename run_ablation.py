"""
LDCT Project - Parallel Ablation Runner
=======================================
Trains the whole ablation grid (input_mode x mamba_mode) inside ONE process,
with ONE in-RAM dataset cache and ONE data pass per epoch.

Why this script exists
----------------------
Launching eight `train.py` processes would build eight independent
`CacheDataset` copies. A single 2.5D cache already needs roughly 90 GB of RAM
(17,816 train slices x 3 channels + labels, float32 at 512x512), so even two
concurrent processes cannot fit in 100 GB. Here the cache is built once per
input mode and every model reads from it.

Schedule
--------
    phase 1 : input_mode=2d    -> basic, residual, multiscale, full  (concurrent)
    phase 2 : input_mode=2.5d  -> basic, residual, multiscale, full  (concurrent)
    phase 3 : evaluate.py on the test set for each finished run
    phase 4 : one comparison table (stdout + CSV + Markdown)

The 2D cache is released before the 2.5D cache is built, which is the only way
both phases fit in 100 GB of RAM.

Concurrency model
-----------------
One worker thread and one CUDA stream per model. The main thread reads the
batch once, copies it to the GPU once, and all four models consume the SAME
tensors. Consequences:

  * the dataset is cached once instead of four times (the RAM constraint),
  * the host->device copy and the augmentation happen once,
  * all models see identical batches in identical order, which turns the
    ablation into a PAIRED comparison instead of four independent runs. Any
    difference in the table is then attributable to the architecture and not to
    a different data order.

VRAM, not RAM, is the limit for this phase: four models at batch 16 in FP32
need roughly 40-80 GB. Lower --batch-size or --workers if you hit OOM; a trial
that OOMs is marked failed and the others keep going.

Honest note about speed
-----------------------
The guaranteed win is the shared data pipeline (cache built once, loader run
once). The additional win from overlapping kernels on separate streams depends
on whether one model already saturates the GPU. Four concurrent models will
therefore not be exactly 4x faster than one, but they are always much faster
than four sequential runs. Use --workers 1 to fall back to strictly sequential
per-batch execution while still keeping the single shared cache.

Precision
---------
FP32 is the default because bfloat16 was measured to cap dPSNR at about +3.7
and to collapse around epoch 4 (see the USE_AMP comment in config.py). Pass
--amp only if you want to re-test mixed precision.

Usage
-----
    python run_ablation.py --epochs 50
    python run_ablation.py --epochs 30 --val-every 2
    python run_ablation.py --input-modes 2d --modes basic full
    python run_ablation.py --skip-train          # only evaluate + compare
    python run_ablation.py --skip-eval           # train only, no test set
"""

import argparse
import contextlib
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from config import (
    WEIGHT_DECAY, PATIENCE, GRAD_CLIP_MAX_NORM, WARMUP_EPOCHS,
    LAMBDA_L1, LAMBDA_SSIM, LAMBDA_EDGE, SCHEDULER_MIN_LR,
    A_MIN, A_MAX, TEST_DIR, EVAL_OUTPUT_DIR,
)
from utils import (
    setup_reproducibility, get_device, extract_centre_slice, get_state_dict,
)
from dataset import prepareCT2D
from model import build_model
from losses import MONAIHybridLoss
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu,
    denormalize_to_hu_offset,
)
from train import (
    build_param_groups, top_grad_params, save_checkpoint, load_checkpoint,
    log_to_tensorboard, SPIKE_RATE_ABORT,
)

MAMBA_ORDER = ["basic", "residual", "multiscale", "full"]
INPUT_ORDER = ["2d", "2.5d"]
SUMMARY_FILE = "ablation_summary.json"


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Train the full ablation grid concurrently with one shared dataset cache",
    )
    p.add_argument("--input-modes", nargs="+", default=INPUT_ORDER,
                   help="phases to run, in order (default: 2d then 2.5d)")
    p.add_argument("--modes", nargs="+", default=MAMBA_ORDER,
                   help="mamba modes trained concurrently inside each phase")
    p.add_argument("--epochs", type=int, default=cfg.TOTAL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    p.add_argument("--grad-skip-norm", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=0,
                   help="concurrent models per phase; 0 = one per requested mode, "
                        "1 = strictly sequential steps (cache still shared)")
    p.add_argument("--val-every", type=int, default=1,
                   help="validate every N epochs; 2 roughly halves the metric cost")
    p.add_argument("--amp", action="store_true",
                   help="enable mixed precision (NOT recommended, see config.USE_AMP)")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--output-root", default="runs_ablation")
    p.add_argument("--cache-rate", type=float, default=1.0,
                   help="fraction of slices held in RAM for the 2D phase")
    p.add_argument("--cache-rate-25d", type=float, default=None,
                   help="same for the 2.5D phase; defaults to --cache-rate. "
                        "Lower this (e.g. 0.6) if the 3-channel cache exhausts RAM")
    p.add_argument("--resume", action="store_true",
                   help="resume each run from its checkpoint.pt if present")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--test-dir", default=TEST_DIR)
    p.add_argument("--eval-output", default=EVAL_OUTPUT_DIR)
    return p.parse_args()


def rss_gb():
    """Resident set size of this process, to watch the cache growing/shrinking."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except Exception:
        return float("nan")


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


# ═══════════════════════════════════════════
# ONE TRIAL = ONE (input_mode, mamba_mode) RUN
# ═══════════════════════════════════════════
class Trial:
    """All mutable state of a single run: model, optimiser, scheduler, stats.

    Every trial owns its own CUDA stream so the four models can overlap on the
    GPU. Nothing is shared between trials except the input tensors, which are
    read-only for them.
    """

    def __init__(self, input_mode, mamba_mode, device, args, total_epochs, use_amp, amp_dtype):
        self.input_mode = cfg.normalize_input_mode(input_mode)
        self.mamba_mode = cfg.normalize_mamba_mode(mamba_mode)
        self.name = cfg.run_name(self.mamba_mode, self.input_mode)
        self.label = f"{self.input_mode}/{self.mamba_mode}"
        self.paths = cfg.run_paths(
            mamba_mode=self.mamba_mode,
            input_mode=self.input_mode,
            output_root=args.output_root,
        )
        os.makedirs(self.paths["run_dir"], exist_ok=True)
        os.makedirs(self.paths["logs"], exist_ok=True)

        self.model = build_model(
            device,
            mamba_mode=self.mamba_mode,
            input_mode=self.input_mode,
            use_checkpoint=args.grad_checkpoint or cfg.USE_GRAD_CHECKPOINT,
            data_parallel=False,          # never DataParallel here: streams handle it
            verbose=False,
        )
        self.params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.loss_fn = MONAIHybridLoss(
            lambda_l1=LAMBDA_L1, lambda_ssim=LAMBDA_SSIM,
            lambda_edge=LAMBDA_EDGE, spatial_dims=2,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, WEIGHT_DECAY), lr=args.lr,
        )
        warmup_epochs = max(1, min(args.warmup_epochs, total_epochs - 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=SCHEDULER_MIN_LR,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
        self.scaler = (
            torch.amp.GradScaler(device.type, enabled=True)
            if use_amp and amp_dtype == torch.float16 else None
        )

        self.writer = SummaryWriter(log_dir=self.paths["logs"])
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

        self.meta = {
            "input_mode": self.input_mode, "mamba_mode": self.mamba_mode,
            "hu_range": [A_MIN, A_MAX], "hu_preset": cfg.HU_RANGE_PRESET,
            "params": self.params,
        }

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_ssim = -float("inf")
        self.best_psnr = -float("inf")
        self.best_dpsnr = -float("inf")
        self.best_rmse = float("nan")
        self.best_epoch = -1
        self.patience_counter = 0
        self.stopped = None            # reason string once the trial is done
        self.spike_budget = [4]
        self.reset_train_stats()
        self.reset_val_stats()

    # ---- lifecycle -------------------------------------------------
    def is_active(self, epoch):
        return self.stopped is None and epoch >= self.start_epoch

    def fail(self, reason):
        self.stopped = reason
        tqdm.write(f"  [{self.label}] STOPPED: {reason}")

    # ---- statistics ------------------------------------------------
    def reset_train_stats(self):
        self.t_loss = 0.0
        self.t_steps = 0
        self.t_spikes = 0
        self.t_nonfinite = 0
        self.t_max_gnorm = 0.0
        self.t_gnorms = []

    def train_stats(self, n_steps):
        n = max(1, n_steps)
        return {
            "avg_train": self.t_loss / max(1, self.t_steps),
            "skipped": self.t_nonfinite,
            "spikes": self.t_spikes,
            "spike_rate": (self.t_spikes + self.t_nonfinite) / n,
            "max_gnorm": self.t_max_gnorm,
            "median_gnorm": statistics.median(self.t_gnorms) if self.t_gnorms else 0.0,
        }

    def reset_val_stats(self):
        self.v = {
            "loss": 0.0, "batches": 0, "samples": 0,
            "psnr": 0.0, "baseline": 0.0, "ssim": 0.0, "rmse": 0.0,
            "psnr_chest": [], "psnr_abd": [], "ssim_chest": [], "ssim_abd": [],
        }

    def val_metrics(self):
        v = self.v
        n = max(1, v["samples"])
        mean = lambda lst: sum(lst) / max(1, len(lst))
        return {
            "avg_val": v["loss"] / max(1, v["batches"]),
            "avg_psnr": v["psnr"] / n,
            "avg_baseline": v["baseline"] / n,
            "avg_ssim": v["ssim"] / n,
            "avg_rmse": v["rmse"] / n,
            "avg_psnr_chest": mean(v["psnr_chest"]),
            "avg_psnr_abd": mean(v["psnr_abd"]),
            "avg_ssim_chest": mean(v["ssim_chest"]),
            "avg_ssim_abd": mean(v["ssim_abd"]),
            "viz_images": None,        # images are skipped in the grid runner
        }


# ═══════════════════════════════════════════
# PER-TRIAL STEPS (run inside worker threads)
# ═══════════════════════════════════════════
def _stream_ctx(trial, main_stream):
    """Run this trial's kernels on its own stream, ordered after the H2D copy."""
    if trial.stream is None:
        return contextlib.nullcontext()
    trial.stream.wait_stream(main_stream)
    return torch.cuda.stream(trial.stream)


def train_step(trial, images, labels, mid, device, epoch, args, use_amp, amp_dtype, main_stream):
    """One optimiser step for one model on the shared batch."""
    try:
        with _stream_ctx(trial, main_stream):
            trial.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_res = trial.model(images)
                # Loss on the UNCLAMPED prediction, exactly like train.py.
                pred_img = mid + pred_res
                loss, _ = trial.loss_fn(pred_img.float(), labels.float())

            if trial.scaler is not None and trial.scaler.is_enabled():
                trial.scaler.scale(loss).backward()
                trial.scaler.unscale_(trial.optimizer)
            else:
                loss.backward()

            gnorm = torch.nn.utils.clip_grad_norm_(
                trial.model.parameters(), max_norm=GRAD_CLIP_MAX_NORM,
            )
            finite = bool(torch.isfinite(gnorm))
            gnorm_val = float(gnorm) if finite else float("inf")
            spiking = args.grad_skip_norm > 0 and finite and gnorm_val > args.grad_skip_norm

            if spiking and trial.spike_budget[0] > 0:
                trial.spike_budget[0] -= 1
                offenders = ", ".join(f"{n}={v:.1f}" for v, n in top_grad_params(trial.model))
                tqdm.write(
                    f"  [spike] {trial.label} epoch {epoch + 1} | |g|={gnorm_val:.1f} "
                    f"> {args.grad_skip_norm:.0f} | step skipped | top grads: {offenders}"
                )

            if not finite:
                trial.t_nonfinite += 1
                trial.optimizer.zero_grad(set_to_none=True)
            elif spiking:
                trial.t_spikes += 1
                trial.optimizer.zero_grad(set_to_none=True)
            else:
                trial.t_max_gnorm = max(trial.t_max_gnorm, gnorm_val)
                trial.t_gnorms.append(gnorm_val)
                if trial.scaler is not None and trial.scaler.is_enabled():
                    trial.scaler.step(trial.optimizer)
                else:
                    trial.optimizer.step()

            if trial.scaler is not None and trial.scaler.is_enabled():
                trial.scaler.update()

            trial.t_loss += loss.item()
            trial.t_steps += 1
            return loss.item()

    except torch.cuda.OutOfMemoryError as err:
        trial.fail(f"CUDA OOM during training ({err.__class__.__name__}). "
                   f"Lower --batch-size or --workers.")
        trial.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None
    except Exception as err:                                  # keep the grid alive
        trial.fail(f"{err.__class__.__name__}: {err}")
        return None


@torch.no_grad()
def val_step(trial, images, labels, mid, body_types, device, use_amp, amp_dtype, main_stream):
    """Accumulate validation metrics for one model on the shared batch."""
    try:
        with _stream_ctx(trial, main_stream):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_res = trial.model(images)

            pred_img = (mid + pred_res).float()
            loss, _ = trial.loss_fn(pred_img, labels.float())
            preds = pred_img.clamp(0.0, 1.0)

            v = trial.v
            v["loss"] += loss.item()
            v["batches"] += 1

            for b in range(preds.shape[0]):
                pred_hu = denormalize_to_hu_offset(preds[b:b + 1], A_MIN, A_MAX).squeeze()
                lbl_hu = denormalize_to_hu_offset(labels[b:b + 1], A_MIN, A_MAX).squeeze()
                mid_hu = denormalize_to_hu_offset(mid[b:b + 1], A_MIN, A_MAX).squeeze()
                bt = body_types[b]

                p_val = compute_psnr_windowed(pred_hu, lbl_hu, bt)
                b_val = compute_psnr_windowed(mid_hu, lbl_hu, bt)
                s_val = compute_ssim_windowed(pred_hu, lbl_hu, bt)

                v["psnr"] += p_val
                v["baseline"] += b_val
                v["ssim"] += s_val
                v["rmse"] += compute_rmse_hu(pred_hu, lbl_hu)
                v["samples"] += 1

                if bt == "Chest":
                    v["psnr_chest"].append(p_val)
                    v["ssim_chest"].append(s_val)
                else:
                    v["psnr_abd"].append(p_val)
                    v["ssim_abd"].append(s_val)

    except torch.cuda.OutOfMemoryError as err:
        trial.fail(f"CUDA OOM during validation ({err.__class__.__name__}).")
        torch.cuda.empty_cache()
    except Exception as err:
        trial.fail(f"{err.__class__.__name__}: {err}")


def normalize_body_types(batch, size):
    """Return a plain list of 'Chest'/'Abdomen' strings for the batch."""
    raw = batch.get("body_type", None)
    if raw is None:
        return ["Abdomen"] * size
    if not isinstance(raw, (list, tuple)):
        raw = [raw] * size
    out = []
    for i in range(size):
        val = raw[i] if i < len(raw) else raw[-1]
        out.append("Chest" if str(val).lower().startswith("c") else "Abdomen")
    return out


def fan_out(executor, fn, trials, *fn_args):
    """Run `fn` for every trial concurrently and wait for all of them.

    Exceptions are already captured per trial, so a single failure cannot take
    down the phase.
    """
    if executor is None:
        for trial in trials:
            fn(trial, *fn_args)
        return
    futures = [executor.submit(fn, trial, *fn_args) for trial in trials]
    for fut in futures:
        fut.result()


# ═══════════════════════════════════════════
# ONE PHASE = ALL MAMBA MODES FOR ONE INPUT MODE
# ═══════════════════════════════════════════
def run_phase(input_mode, args, device, use_amp, amp_dtype, summary):
    """Train every requested mamba mode for one input mode, sharing one cache."""
    input_mode = cfg.normalize_input_mode(input_mode)
    total_epochs = args.epochs
    cache_rate = args.cache_rate
    if input_mode == "2.5d" and args.cache_rate_25d is not None:
        cache_rate = args.cache_rate_25d

    print("\n" + "=" * 78)
    print(f"PHASE  input_mode={input_mode}  modes={', '.join(args.modes)}")
    print(f"RAM before cache: {rss_gb():.1f} GB | cache_rate={cache_rate}")
    print("=" * 78)

    # ---- ONE cache for the whole phase ------------------------------
    train_loader, val_loader = prepareCT2D(
        input_mode=input_mode,
        train_batch_size=args.batch_size,
        cache_rate=cache_rate,
    )
    n_steps = len(train_loader)
    print(f"RAM after cache : {rss_gb():.1f} GB | {n_steps} train batches/epoch")

    # ---- One trial (model+optimiser+writer) per mamba mode ----------
    trials = [
        Trial(input_mode, mode, device, args, total_epochs, use_amp, amp_dtype)
        for mode in args.modes
    ]
    for trial in trials:
        print(f"  {trial.label:<16} params={trial.params:>11,}  ->  {trial.paths['run_dir']}")
        if args.resume:
            (trial.start_epoch, trial.best_val_loss, trial.best_ssim,
             trial.best_psnr, trial.patience_counter) = load_checkpoint(
                trial.paths["checkpoint"], trial.model, trial.optimizer,
                trial.scheduler, trial.scaler, device,
            )

    n_workers = len(trials) if args.workers <= 0 else min(args.workers, len(trials))
    executor = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    print(f"Concurrency: {n_workers} model(s) per data pass"
          f"{' (sequential steps, shared cache)' if executor is None else ''}")

    main_stream = torch.cuda.current_stream() if device.type == "cuda" else None
    phase_start = time.time()

    for epoch in range(total_epochs):
        active = [t for t in trials if t.is_active(epoch)]
        if not active:
            print("All trials in this phase have stopped.")
            break

        epoch_start = time.time()
        for trial in active:
            trial.model.train()
            trial.reset_train_stats()

        bar = tqdm(train_loader, desc=f"[{input_mode}] Epoch {epoch + 1:03d}/{total_epochs}",
                   unit="batch", leave=False, dynamic_ncols=True)
        for batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            mid = extract_centre_slice(images)

            running = [t for t in active if t.stopped is None]
            if not running:
                break
            fan_out(executor, train_step, running,
                    images, labels, mid, device, epoch, args, use_amp, amp_dtype, main_stream)

            if main_stream is not None:
                for trial in running:
                    if trial.stream is not None:
                        main_stream.wait_stream(trial.stream)

            bar.set_postfix({t.mamba_mode[:4]: f"{t.t_loss / max(1, t.t_steps):.4f}"
                             for t in running})

        # ---- validation: ONE pass over val_loader for all models ----
        do_val = ((epoch + 1) % max(1, args.val_every) == 0) or (epoch + 1 == total_epochs)
        running = [t for t in active if t.stopped is None]
        metrics_by_trial = {}

        if do_val and running:
            for trial in running:
                trial.model.eval()
                trial.reset_val_stats()
            vbar = tqdm(val_loader, desc=f"[{input_mode}] Epoch {epoch + 1:03d} [Val]",
                        unit="batch", leave=False, dynamic_ncols=True)
            for batch in vbar:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                mid = extract_centre_slice(images)
                body_types = normalize_body_types(batch, images.shape[0])
                alive = [t for t in running if t.stopped is None]
                if not alive:
                    break
                fan_out(executor, val_step, alive,
                        images, labels, mid, body_types, device, use_amp, amp_dtype, main_stream)
                if main_stream is not None:
                    for trial in alive:
                        if trial.stream is not None:
                            main_stream.wait_stream(trial.stream)
            metrics_by_trial = {t.name: t.val_metrics() for t in running if t.stopped is None}

        epoch_time = time.time() - epoch_start

        # ---- bookkeeping, logging, checkpointing --------------------
        print(f"\nEpoch [{epoch + 1:03d}/{total_epochs}] ({input_mode}) "
              f"{epoch_time:.1f}s | peak VRAM {vram_gb():.1f} GB | RAM {rss_gb():.1f} GB")

        for trial in trials:
            if trial.stopped is not None or epoch < trial.start_epoch:
                continue
            stats = trial.train_stats(n_steps)
            current_lr = trial.optimizer.param_groups[0]["lr"]
            trial.scheduler.step()
            metrics = metrics_by_trial.get(trial.name)

            if metrics is None:
                print(f"  {trial.label:<16} Train {stats['avg_train']:.5f} | "
                      f"|g|med {stats['median_gnorm']:.2f} max {stats['max_gnorm']:.1f} | "
                      f"sp {stats['spikes']}/{n_steps} | LR {current_lr:.2e} | (no val)")
            else:
                dpsnr = metrics["avg_psnr"] - metrics["avg_baseline"]
                print(f"  {trial.label:<16} Train {stats['avg_train']:.5f} | "
                      f"Val {metrics['avg_val']:.5f} | PSNR {metrics['avg_psnr']:.3f} | "
                      f"dPSNR {dpsnr:+.3f} | SSIM {metrics['avg_ssim']:.5f} | "
                      f"RMSE {metrics['avg_rmse']:.2f} | "
                      f"|g|med {stats['median_gnorm']:.2f} max {stats['max_gnorm']:.1f} | "
                      f"sp {stats['spikes']}/{n_steps} | LR {current_lr:.2e}")
                log_to_tensorboard(trial.writer, epoch, stats, metrics, current_lr, epoch_time)

                if metrics["avg_psnr"] > trial.best_psnr:
                    trial.best_psnr = metrics["avg_psnr"]
                    trial.best_ssim = metrics["avg_ssim"]
                    trial.best_dpsnr = dpsnr
                    trial.best_rmse = metrics["avg_rmse"]
                    trial.best_val_loss = metrics["avg_val"]
                    trial.best_epoch = epoch + 1
                    trial.patience_counter = 0
                    torch.save({"model_state_dict": get_state_dict(trial.model),
                                "meta": trial.meta, "psnr": trial.best_psnr,
                                "ssim": trial.best_ssim, "epoch": epoch},
                               trial.paths["best_model"])
                    print(f"    best saved: PSNR={trial.best_psnr:.2f} SSIM={trial.best_ssim:.4f}")
                else:
                    trial.patience_counter += 1

            save_checkpoint(trial.paths["checkpoint"], epoch, trial.model, trial.optimizer,
                            trial.scheduler, trial.scaler, trial.best_val_loss,
                            trial.best_ssim, trial.best_psnr, trial.patience_counter,
                            meta=trial.meta)

            if stats["spike_rate"] > SPIKE_RATE_ABORT:
                trial.fail(f"{stats['spike_rate'] * 100:.0f}% of steps skipped - diverging. "
                           f"Check HU_RANGE_PRESET and the [spike] lines above.")
            elif trial.patience_counter >= PATIENCE:
                trial.fail(f"early stopping at epoch {epoch + 1}")

    phase_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - phase_start))
    print(f"\nPhase {input_mode} done in {phase_time}")

    for trial in trials:
        summary[trial.name] = {
            "run": trial.name,
            "input_mode": trial.input_mode,
            "mamba_mode": trial.mamba_mode,
            "params": trial.params,
            "val_psnr": None if trial.best_psnr == -float("inf") else round(trial.best_psnr, 4),
            "val_dpsnr": None if trial.best_dpsnr == -float("inf") else round(trial.best_dpsnr, 4),
            "val_ssim": None if trial.best_ssim == -float("inf") else round(trial.best_ssim, 5),
            "val_rmse": None if trial.best_rmse != trial.best_rmse else round(trial.best_rmse, 3),
            "best_epoch": trial.best_epoch,
            "best_model": trial.paths["best_model"],
            "note": trial.stopped or "completed",
        }
        print(f"  {trial.label:<16} best PSNR {trial.best_psnr:.2f} | "
              f"SSIM {trial.best_ssim:.4f} | epoch {trial.best_epoch} | {trial.stopped or 'completed'}")
        trial.writer.close()

    # ---- release EVERYTHING before the next phase builds its cache --
    if executor is not None:
        executor.shutdown(wait=True)
    del trials, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print(f"RAM after cleanup: {rss_gb():.1f} GB")


# ═══════════════════════════════════════════
# EVALUATION (evaluate.py per run, on the test set)
# ═══════════════════════════════════════════
def run_evaluations(summary, args):
    """Call evaluate.py once per finished run and read back its CSV report.

    Runs are evaluated SEQUENTIALLY: evaluate.py works at full resolution on
    the test set and one model at a time keeps VRAM predictable.
    """
    test_path = Path(args.test_dir)
    if not test_path.exists():
        print(f"\nSkipping evaluation: test dir '{test_path}' does not exist. "
              f"The comparison table will use validation metrics only.")
        return

    print("\n" + "=" * 78)
    print("EVALUATION ON THE TEST SET (evaluate.py)")
    print("=" * 78)

    for name, row in summary.items():
        best_model = row.get("best_model")
        if not best_model or not os.path.exists(best_model):
            print(f"  {name:<16} no best_model.pt - skipped")
            continue

        cmd = [
            sys.executable, "evaluate.py",
            "--input-mode", row["input_mode"],
            "--mamba-mode", row["mamba_mode"],
            "--model", best_model,
            "--output", args.eval_output,
            "--test-dir", args.test_dir,
            "--no-amp",                 # metrics in FP32, like training
        ]
        print(f"\n$ {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"  {name:<16} evaluate.py exited with code {proc.returncode}")
            continue

        csv_path = Path(args.eval_output) / f"evaluation_report_{name}.csv"
        stats = read_eval_csv(csv_path)
        if stats:
            row.update(stats)


def read_eval_csv(csv_path):
    """Average the per-patient rows of an evaluate.py report."""
    if not csv_path.exists():
        return None
    cols = {"PSNR": "test_psnr", "Delta_PSNR": "test_dpsnr", "SSIM": "test_ssim",
            "RMSE_HU": "test_rmse", "VIF": "test_vif"}
    acc = {v: [] for v in cols.values()}
    with open(csv_path, newline="") as f:
        for rec in csv.DictReader(f):
            for src, dst in cols.items():
                try:
                    acc[dst].append(float(rec[src]))
                except (KeyError, TypeError, ValueError):
                    pass
    out = {}
    for dst, vals in acc.items():
        if vals:
            out[dst] = round(sum(vals) / len(vals), 4)
    out["test_report"] = str(csv_path)
    return out


# ═══════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════
COMPARE_COLUMNS = [
    ("input_mode", "input", 7, "s"),
    ("mamba_mode", "mamba", 11, "s"),
    ("params", "params", 11, ",d"),
    ("best_epoch", "ep*", 4, "d"),
    ("val_dpsnr", "val dPSNR", 10, "+.3f"),
    ("val_psnr", "val PSNR", 9, ".3f"),
    ("val_ssim", "val SSIM", 9, ".5f"),
    ("test_dpsnr", "test dPSNR", 11, "+.3f"),
    ("test_psnr", "test PSNR", 10, ".3f"),
    ("test_ssim", "test SSIM", 10, ".4f"),
    ("test_rmse", "RMSE HU", 9, ".2f"),
    ("test_vif", "VIF", 7, ".4f"),
]


def _fmt(value, width, spec):
    if value is None:
        return f"{'-':>{width}}"
    try:
        return f"{value:>{width}{spec}}"
    except (ValueError, TypeError):
        return f"{str(value):>{width}}"


def sort_key(row):
    """Rank by test dPSNR, then validation dPSNR; missing values go last."""
    for key in ("test_dpsnr", "val_dpsnr"):
        if row.get(key) is not None:
            return -float(row[key])
    return float("inf")


def print_comparison(summary, args):
    rows = sorted(summary.values(), key=sort_key)

    print("\n" + "=" * 118)
    print("ABLATION COMPARISON  (ranked by test dPSNR, falling back to validation dPSNR)")
    print("=" * 118)
    header = "".join(f"{title:>{width}}  " for _, title, width, _ in COMPARE_COLUMNS)
    print("  rank  " + header)
    print("-" * 118)
    for i, row in enumerate(rows, 1):
        line = "".join(_fmt(row.get(key), width, spec) + "  "
                       for key, _, width, spec in COMPARE_COLUMNS)
        print(f"  {i:>4}  " + line)
    print("=" * 118)
    print("  ep* = epoch of the best validation PSNR. All runs saw identical batches")
    print("        in identical order, so differences are architectural, not data noise.")

    notes = [(r["run"], r.get("note")) for r in rows
             if r.get("note") and r["note"] != "completed"]
    if notes:
        print("\n  Notes:")
        for name, note in notes:
            print(f"    {name:<16} {note}")

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = ["run"] + [key for key, _, _, _ in COMPARE_COLUMNS] + ["note", "best_model"]
    csv_path = out_dir / "ablation_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "ablation_comparison.md"
    titles = [title for _, title, _, _ in COMPARE_COLUMNS]
    with open(md_path, "w") as f:
        f.write("# Ablation comparison\n\n")
        f.write(f"HU preset: `{cfg.HU_RANGE_PRESET}` ({A_MIN} to {A_MAX}) | "
                f"precision: {'AMP' if args.amp else 'FP32'} | "
                f"epochs: {args.epochs} | batch: {args.batch_size} | lr: {args.lr}\n\n")
        f.write("| rank | " + " | ".join(titles) + " |\n")
        f.write("|" + "---|" * (len(titles) + 1) + "\n")
        for i, row in enumerate(rows, 1):
            cells = [_fmt(row.get(key), 0, spec).strip()
                     for key, _, _, spec in COMPARE_COLUMNS]
            f.write(f"| {i} | " + " | ".join(cells) + " |\n")

    json_path = out_dir / SUMMARY_FILE
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Comparison CSV : {csv_path}")
    print(f"  Comparison MD  : {md_path}")
    print(f"  Raw summary    : {json_path}")


def load_summary(args):
    """Reload a previous summary so --skip-train can still build the table."""
    path = Path(args.output_root) / SUMMARY_FILE
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    summary = {}
    for input_mode in args.input_modes:
        for mamba_mode in args.modes:
            im = cfg.normalize_input_mode(input_mode)
            mm = cfg.normalize_mamba_mode(mamba_mode)
            name = cfg.run_name(mm, im)
            paths = cfg.run_paths(mamba_mode=mm, input_mode=im, output_root=args.output_root)
            summary[name] = {"run": name, "input_mode": im, "mamba_mode": mm,
                             "best_model": paths["best_model"], "note": "not trained here"}
    return summary


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    args = parse_args()
    args.input_modes = [cfg.normalize_input_mode(m) for m in args.input_modes]
    args.modes = [cfg.normalize_mamba_mode(m) for m in args.modes]

    setup_reproducibility()
    device = get_device()

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("=" * 78)
    print("ABLATION RUNNER")
    print("=" * 78)
    print(f"Phases      : {' -> '.join(args.input_modes)}")
    print(f"Modes/phase : {', '.join(args.modes)}")
    print(f"Device      : {device}")
    print(f"HU range    : [{A_MIN}, {A_MAX}] (preset '{cfg.HU_RANGE_PRESET}')")
    print(f"Precision   : {amp_dtype if use_amp else 'FP32'}")
    print(f"Epochs      : {args.epochs} | batch {args.batch_size} | lr {args.lr:.2e} "
          f"| warmup {args.warmup_epochs} | val every {args.val_every}")
    print(f"Output root : {args.output_root}")

    if args.skip_train:
        summary = load_summary(args)
    else:
        summary = {}
        for input_mode in args.input_modes:
            run_phase(input_mode, args, device, use_amp, amp_dtype, summary)

    if not args.skip_eval:
        run_evaluations(summary, args)

    print_comparison(summary, args)


if __name__ == "__main__":
    main()

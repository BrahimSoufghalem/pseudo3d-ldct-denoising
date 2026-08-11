# Architecture review - applied fixes

This document lists every change made in the `fix/architecture-review` branch, in
the order of the original review. Nothing here has been executed on a GPU: the
changes are analytical, and `test_shapes.py` must be run locally before merging.

## Two new experiment axes

| Axis | Values | Where |
|---|---|---|
| `INPUT_MODE` | `2d`, `2.5d` | `config.py`, `--input-mode` on every script |
| `MAMBA_MODE` | `basic`, `residual`, `multiscale`, `full` | `config.py`, `--mamba-mode` |

Each combination writes to its own folder: `runs/<2d|25d>_<mamba_mode>/`
(`checkpoint.pt`, `best_model.pt`, `logs/`). Ablations can no longer overwrite
each other.

```bash
python train.py --input-mode 2d   --mamba-mode full   # pure 2D baseline
python train.py --input-mode 2.5d --mamba-mode full   # pseudo-3D
```

## Critical architecture fixes

1. **Attention gates were fed the wrong tensor.** `ag4..ag1` received `d1..d4`,
   i.e. a strided copy of the same encoder features they were gating, so they
   carried no additional context. They are now driven by the bottleneck output
   (`ag4`) and by the previous decoder stage (`ag3`, `ag2`, `ag1`), which is the
   standard Attention U-Net formulation. The existing `F_g` widths already
   matched this ordering.
2. **S6 recurrence used a Python `for t in range(L)` loop** over 4096 timesteps.
   It is now delegated to the official `mamba_ssm.selective_scan_fn` CUDA kernel
   when available, with a chunked pure-PyTorch fallback (`selective_scan_ref`)
   for CPU and debugging. Backend selection: `SCAN_BACKEND = auto | cuda | ref`.
3. **All four scan directions shared one S6 parameter set.** Each direction now
   owns its own `x_proj`, `dt_proj`, `A_log` and `D` (`[K, ...]` parameters,
   VMamba style), executed in a single fused scan through the kernel's group
   dimension.
4. **`dt` was clamped to `[1e-3, 0.1]`**, which zeroes the gradient outside the
   bounds. The clamp is gone; softplus plus the official `inv_dt` initialisation
   keeps the step size in range.
5. **Cross-scan / cross-merge were shape-inconsistent** with the fused kernel.
   Both now use the canonical `[B, K, C, L]` layout with a correct inverse
   transform per direction.
6. **The "pseudo-3D" claim was only three input channels.** In `2.5d` mode the
   stem is now `PseudoDepthStem`: a `Conv3d(1, w/2, (3,3,3))` over the real depth
   axis plus a `Conv2d` branch on the centre slice. In `2d` mode a plain
   `Conv2d(1, w, 3)` stem is used and the neighbour files are never even loaded.

## Real bugs

7. **Weight-format mismatch:** `save_checkpoint` stored `model.module.state_dict()`
   while the best model stored `model.state_dict()`, so under DataParallel the
   two files had incompatible key names. All saving/loading now goes through
   `utils.get_state_dict` / `utils.load_state_into`.
8. **Loss inconsistency:** training used the unclamped prediction, validation the
   clamped one. Both now compute the loss unclamped; clamping is applied only for
   metrics and visualisation.
9. **`LayerNorm2d` hardcoded `device_type='cuda'`**, breaking CPU runs. It now
   dispatches to a device-agnostic reference implementation off CUDA.
10. **`NAFBlock` `beta`/`gamma` were `ones`**, not the NAFNet-faithful `zeros`.
    Fixed, so every block (and `MultiScaleSpatialFusion`) starts as an identity.
11. **Deprecated `from torch.cuda.amp import autocast`** in `evaluate.py` and
    `inference_dicom.py` replaced by `torch.autocast(device_type=...)`.
12. **`utils.build_pseudo3d_input` had defaults `(-1024, 1600)`** that contradicted
    `config.A_MIN/A_MAX`. HU limits now always come from config. The function is
    kept as a deprecated alias of `build_model_input`.
13. **`_init_icnr` allocated `torch.randn` then overwrote it**, and used the
    leaky-ReLU gain in an activation-free network. Now `torch.empty` +
    `kaiming_normal_(nonlinearity='linear')`.

## Scientific / reviewer concerns

- **HU window.** `HU_RANGE_PRESET = "wide"` gives `[-1024, 3072]` HU, which fully
  contains both clinical evaluation windows. The old `[-1000, 600]` clipped all
  bone and part of the lung window, biasing windowed PSNR/SSIM. Set
  `HU_RANGE_PRESET = "legacy"` to reproduce the old numbers.
- **Metric clip bound** is derived from config (`A_MAX + 1024`) instead of a
  hardcoded 2924 that disagreed with the actual code (1600).
- **Loss naming**: the `lambda_l1` term is a Charbonnier loss; the logging dict
  now reports `Charbonnier` and keeps `L1` only as an alias.
- **Overclaiming docstrings** ("Authentic S6", "True selective scan", the fake
  "Residual Add" in `ResidualMambaBottleneck`, the "8MB instead of 4.3GB"
  comment) have been rewritten to describe what the code actually does.
- **Non-multiple-of-16 inputs** are reflection/replicate padded and cropped back,
  so full-resolution evaluation is safe.
- **Mixed precision** (bf16 when supported, else fp16 + GradScaler) with gradient
  clipping after `unscale_`, plus optional gradient checkpointing of the Mamba
  bottleneck. The SSM recurrence itself always runs in fp32.
- **DataParallel** is kept only as a convenience; DDP (`torchrun`) is recommended.

## Breaking changes

1. Output paths moved from `FinalCT_2.5D-UNET-DATASET/` to `runs/<run_name>/`.
2. `HU_RANGE_PRESET="wide"` changes normalisation: **previous checkpoints and
   reported metrics are not comparable** unless you switch back to `"legacy"`.
3. Checkpoints are saved unwrapped and best models are wrapped in a dict with
   `model_state_dict` + `meta`; old files still load through `load_state_into`,
   but architecture changes (per-direction S6 params, new stem, gate rewiring)
   mean **old weights are incompatible and retraining is required**.

## Before merging

```bash
python test_shapes.py           # CPU, all 8 mode combinations, seconds
pip install mamba-ssm --no-build-isolation   # fast CUDA scan (optional)
python train.py --input-mode 2.5d --mamba-mode full --epochs 1
```

# Training protocol fixes (code review, Aug 2026)

This document lists the concrete problems found in the `train_20p.py` pipeline
after the best 100-patient run
(`--use-multi-res --use-unet-decode --use-mu-mod --ssim-weight 0.5 --l1-weight 0.5`)
reached Chest PSNR 27.93 / SSIM 0.5977 / VIF 0.1771, short of the RED-CNN
reference (28.36 / 0.609 / 0.221). Every fix below is independently
switchable and fully backward compatible with existing checkpoints.

## 1. SSIM loss used a moving, per-batch data range (FIXED, always on)

`compute_loss` computed `data_range = target.max() - target.min()` per batch on
standardized z-values. The SSIM constants C1/C2 therefore changed scale on
every step, making the loss landscape batch-dependent and noisy.

**Fix:** the data range is now the fixed benchmark-derived constant
`EVAL_DATA_RANGE / BENCHMARK_PIXEL_STD` (= 2924 / 502.185 = 5.8226 in
standardized units).

## 2. MSE weight silently became zero (WARNING added)

With `--ssim-weight 0.5 --l1-weight 0.5` the residual MSE weight is
`1 - 0.5 - 0.5 = 0`. The benchmark protocol that reaches Chest VIF 0.221
(RED-CNN) trains with pure MSE. Dropping MSE entirely was an untested protocol
deviation. The trainer now prints a prominent warning when MSE weight is 0.

**Recommendation:** keep MSE weight >= 0.3, e.g.
`--ssim-weight 0.3 --l1-weight 0.3` (MSE gets 0.4), or run the pure-MSE
control first.

## 3. HU-bin loss bins were per-batch min/max (FIXED, always on)

`hu_bin_bias_loss` derived its bin edges from each batch's min/max, so the
loss compared different intensity ranges on every step. Bins are now the fixed
physical tissue boundaries also used by `physics_losses.HUBinBiasLoss`
(-1024 / -500 / -200 / 200 / 600 / 1900 HU), converted to the standardized
domain. `--hu-bin-bins` is kept for CLI compatibility but ignored.

## 4. Checkpoint selection ignored the target metric (NEW: `--select-by`)

`best_model.pt` was selected by overall validation SSIM, a mean dominated by
abdomen slices (SSIM ~0.91) while the target gap is Chest VIF. Validation now
reports per-region (Chest/Abdomen) PSNR and SSIM, and optionally VIF, and the
best checkpoint can be selected with:

```
--select-by {ssim,psnr,vif,chest_ssim,chest_vif}   # default: ssim (old behavior)
--val-vif                                           # force VIF computation
```

VIF on 128-px validation crops is indicative only, not benchmark-comparable;
full-resolution `evaluate_20p.py` remains the ground truth.

## 5. No learning-rate decay (NEW: `--lr-schedule cosine`)

The fixed lr=1e-4 matches the benchmark ResNet protocol, but with 30k
iterations and heavier architectures a cosine decay typically adds
+0.1-0.3 dB at no cost:

```
--lr-schedule cosine --min-lr 1e-6
```

Default remains `constant` (old behavior).

## 6. mu-mod global pooling: patch vs full-slice mismatch (NEW: `--mu-mod-mode local`)

`MuAwareModulation` used `AdaptiveAvgPool2d(1)`: during training the FiLM
parameters are computed from a 128x128 patch mean, at test time from a 512x512
full-slice mean - a different distribution. `--mu-mod-mode local`
(with `--mu-local-window 64`) computes the context over fixed-size 64-px
windows instead, so train and test see statistics of the same physical extent.
Default remains `global` (old behavior, old checkpoints load unchanged).

## 7. U-Net decode had zero full-resolution blocks after the decoder (NEW: `--unet-final-blocks`)

With `blocks=20` the allocation was enc 5+5+5 / dec 5 / **final 0**: 15 of 20
blocks operate on downsampled features that cannot represent high-frequency
noise - and chest (lung) noise is predominantly high-frequency, which is what
VIF is most sensitive to. `--unet-final-blocks 4` reallocates to
enc 4+4+4 / dec 4 / final 4 (same 20-block budget, 4 full-resolution blocks
after the decoder). Default `None` keeps the old allocation.

## Recommended next runs (in order)

```bash
# A. Protocol control: pure MSE, benchmark settings, sequential trunk
HU_RANGE_PRESET=benchmark python train_20p.py --arch local_residual \
  --data-dir dataset --output-root runs_ctrl_mse \
  --max-iterations 100000 --iterations-before-val 2500 \
  --batch-size 64 --patch-size 64 --lr 1e-4 \
  --lr-schedule cosine --min-lr 1e-6 --select-by chest_vif --val-vif

# B. U-Net decode with full-res depth restored + balanced loss
HU_RANGE_PRESET=benchmark python train_20p.py --arch local_residual \
  --data-dir dataset --output-root runs_unet_fixed \
  --max-iterations 100000 --iterations-before-val 2500 \
  --batch-size 16 --patch-size 128 \
  --use-multi-res --use-unet-decode --unet-final-blocks 4 \
  --use-mu-mod --mu-mod-mode local \
  --ssim-weight 0.3 --l1-weight 0.3 \
  --lr-schedule cosine --min-lr 1e-6 --select-by chest_vif --val-vif
```

Interpret against the pre-registered thresholds in
`LOCAL_RESIDUAL_BASELINE.md` (Chest VIF >= 0.221 = target achieved). Only
after the trunk reaches the RED-CNN regime should the physics losses
(`physics_losses.py`) be added, one at a time.

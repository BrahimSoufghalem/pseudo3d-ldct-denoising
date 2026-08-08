# Option 1: Physics additions on a benchmark-grade trunk (RED-CNN)

## Why earlier "physics didn't help" conclusions are invalid

1. **Wrong measuring stick.** The additions were judged by PSNR/SSIM/VIF only.
   NPS matching and HU-bin bias are not designed to raise those metrics --
   PSNR *rewards* over-smoothing while the NPS loss *penalizes* it, so a
   physics-faithful model can look "no better" or slightly worse on PSNR by
   construction. `evaluate_20p.py` contains no physics metrics at all, so the
   value of the additions was literally unmeasurable. Use the new
   `evaluate_physics.py`.
2. **Contaminated substrate.** All earlier physics ablations ran on the
   local_residual trunk with a broken protocol (per-batch SSIM data range,
   MSE weight 0, dynamic HU bins -- see TRAINING_FIXES.md). Any conclusion
   drawn from those runs is void.
3. **Weak baseline.** A trunk stuck at Chest VIF ~0.18 cannot demonstrate
   anything; the additions must be shown on a trunk that already matches the
   benchmark (RED-CNN, Chest VIF 0.221).

## The claim to defend

> Plug-in physics-informed losses (radial NPS matching + per-tissue HU-bin
> bias) improve the physical fidelity of the denoised image -- correct noise
> texture and per-tissue HU calibration -- **at equal PSNR/SSIM/VIF**, on top
> of a strong benchmark baseline (RED-CNN), and are architecture-independent.

This is the fair, reviewer-proof comparison: RED-CNN vs RED-CNN + physics,
same data, same protocol; every delta is attributable to the additions.

## Experiments

All runs: benchmark protocol (pure MSE base, Adam lr 1e-4, batch 64, patch 64,
100k iterations), `--select-by chest_vif --val-vif`.

### C0 -- baseline control
```bash
HU_RANGE_PRESET=benchmark python train_20p.py --arch redcnn \
    --data-dir dataset --output-root runs_100p_C0_redcnn_pure --split 100p \
    --max-iterations 100000 --iterations-before-val 2500 \
    --batch-size 64 --patch-size 64 --val-patch-size 128 \
    --lr 1e-4 --num-workers 8 --cache-rate 1 \
    --select-by chest_vif --val-vif
```

### C1 -- + NPS
Same command, `--output-root runs_100p_C1_redcnn_nps`, plus:
```
    --nps-weight 0.05
```

### C2 -- + NPS + HU-bin
Same command, `--output-root runs_100p_C2_redcnn_physics`, plus:
```
    --nps-weight 0.05 --hu-bin-loss 0.1
```

### Weight sweep (cheap, do FIRST on --split 20p)
Sweep `--nps-weight` in {0.02, 0.05, 0.1} on the 20p split (~few hours each).
Pick the largest weight whose PSNR/SSIM stay within noise of C0. Then run the
chosen weight on 100p. Never tune weights on the test set.

## Evaluation

```bash
# Standard metrics (parity check):
HU_RANGE_PRESET=benchmark python evaluate_20p.py \
    --runs-root runs_100p_C0_redcnn_pure --test-dir dataset --split 100p \
    --output eval_C0
# Physics metrics (the contribution):
HU_RANGE_PRESET=benchmark python evaluate_physics.py \
    --runs-root runs_100p_C0_redcnn_pure --test-dir dataset --split 100p \
    --output eval_physics_C0 --archs redcnn
```
Repeat for C1/C2 and compare tables.

## Success criteria (pre-registered)

Declare the contribution successful iff, comparing C1/C2 to C0:
1. **Parity**: Chest/Abdomen PSNR, SSIM, VIF within +/-0.05 dB / +/-0.005 /
   +/-0.005 of C0 (no meaningful degradation), and
2. **Physics gain**: NPS_LogL1 decreases AND NoiseSTD_Ratio moves toward 1.0
   (C1), plus mean |Bias_<tissue>| decreases across tissue bins (C2).

If (2) holds while (1) fails, report the trade-off honestly and sweep the
weight down. If neither holds, the additions are refuted on this trunk --
that, too, is a publishable negative ablation when reported with the physics
metrics.

## Optional second trunk

Repeat C0->C2 on local_residual (baseline config) to support the
"architecture-independent" part of the claim. The absolute Chest VIF of that
trunk is irrelevant to the physics claim; only the C-deltas matter.
""

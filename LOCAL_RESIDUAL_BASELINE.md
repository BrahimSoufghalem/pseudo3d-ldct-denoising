# Dense local residual mean/std control

## Purpose

This control tests the common pattern behind the two strongest 2D methods in
`eeulig/ldct-benchmark`: RED-CNN and the noise-subtraction ResNet. It is trained
from scratch and uses no pretrained weights, teacher, distillation, metadata or
another model's output. It is **not** presented as a novel architecture or as
the final medical-physics contribution.

The failed physics-spectral baseline reached chest/abdomen VIF 0.1576/0.4130
after its double residual attenuation was fixed. Its spatial-only ablation was
slightly worse. Therefore NPS and HU losses are not added to that weak trunk.
This control first asks whether the repository can reach the benchmark regime
with the successful local-CNN inductive bias and training protocol.

## Controlled design

```text
standardized LDCT
  -> 9x9 convolution, 128 channels
  -> 10 x [3x3 conv, BN, ReLU, grouped 3x3 conv, BN, ReLU, 1x1 conv + identity]
  -> 3x3 predicted noise
  -> standardized LDCT - predicted noise
```

There is no dilation, pooling, stride, upsampling, attention, Mamba, NAFNet,
spectral decomposition, residual scaling or auxiliary loss. Residual branches
are added at full strength. The receptive field is approximately 51x51.

## Exact mean/std convention

The benchmark stores CT values in the non-negative pixel domain `HU + 1024` and
publishes these global training-set constants:

```text
mean = 481.45419786099086
std  = 502.18507379395044
```

Training uses:

```text
z = (HU + 1024 - mean) / std
```

No input clipping is performed before the model. Evaluation reverses the
standardization and only then applies the existing benchmark physical range
`[0,2924]` for metrics. This distinction matters: converting the existing
clipped `[0,1]` tensors inside the model would permanently lose values above
1900 HU and would not reproduce benchmark preprocessing.

## Training protocol

Defaults reproduce the benchmark ResNet settings that matter for this control:

```text
MSE, Adam(beta1=0.9,beta2=0.999), lr=1e-4
batch=64, patch=64, validation patch=128
20,000 optimizer iterations, validation every 1,000 iterations
patient-balanced replacement sampling
FP32, fixed learning rate, best checkpoint selected by validation SSIM
```

Validation uses all validation slices with deterministic center crops. The
reference implementation also applies its weighted replacement sampler to
validation; the deterministic full validation here is intentionally more
stable and does not alter training exposure or final full-resolution testing.

## Commands

```bash
python test_local_residual.py

HU_RANGE_PRESET=benchmark python train_local_residual.py \
  --max-iterations 20000 \
  --iterations-before-val 1000 \
  --batch-size 64 \
  --patch-size 64 \
  --val-patch-size 128 \
  --lr 1e-4 \
  --output-root runs_local_residual

HU_RANGE_PRESET=benchmark python evaluate_local_residual.py \
  --model runs_local_residual/2d_local_residual_meanstd/best_model.pt \
  --output eval_local_residual
```

The pre-registered interpretation remains:

```text
Chest VIF <= 0.175  : failure
0.176 - 0.186       : inconclusive
>= 0.187            : preliminary success
>= 0.200            : strong success
>= 0.218            : near benchmark
>= 0.221            : target achieved
```

If this control reaches the strong-success region, physics components will be
introduced one at a time on top of a working local trunk. The first physical
addition should be calibrated from measured training-set spectra, not arbitrary
Gaussian cutoffs.

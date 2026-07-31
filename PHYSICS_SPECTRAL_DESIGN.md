# Physics-spectral 2D LDCT denoising research path

> Status: implementation-ready research hypothesis, **not** a novelty claim.
> Novelty must be established by a literature review and controlled ablations.

## Independence constraint

The model is trained from scratch and consumes one LDCT slice only. It uses no
pretrained model, teacher, distillation, patient identity, acquisition metadata,
Mamba, NAFNet, or another denoiser's prediction.

## Why a new branch exists

`feat/physics-spectral-denoiser` was forked from `fix/architecture-review`, not
from `main`, because the unsuccessful architecture study still produced the
infrastructure required for an honest publication comparison:

- explicit 2D versus 2.5D inputs;
- fixed train/validation/test patient lists;
- FP32 training after bf16 was shown to collapse silently;
- stable HU normalisation and a recorded train/eval preset invariant;
- full-resolution evaluation;
- the exact `ldct-benchmark` metric contract;
- per-patient and per-region reporting;
- LDCT baseline metrics and deltas for every metric;
- state-dict handling independent of DataParallel wrappers;
- deterministic seeds and diagnostic gradient logging.

Those are retained. The unsuccessful Mamba/NAF architecture and its checkpoints
remain available and are not overwritten by the new standalone scripts.

## The failed family: what was learned

For 2D/basic under the benchmark-compatible evaluation:

- the baseline family stopped near chest VIF 0.1627;
- pure Charbonnier reached 0.1671;
- 9.43 M deep and 11.37 M wide variants both stopped near 0.1667;
- 512x512 training did not break the plateau;
- changing SSIM/edge weights, MS-SSIM and clinical-window losses was negligible;
- reducing the pyramid to two stages worsened chest VIF to 0.1554;
- bypassing attention gates gave 0.1668;
- blending the output towards LDCT decreased every metric monotonically;
- 50 and then 90 epochs did not exceed about 0.1671, with all metrics already
  nearly flat after epoch 15.

A RED-CNN control reached the expected VIF 0.221 on the same data/evaluation.
Therefore the data and metric implementation can express the target, while the
Mamba/NAF encoder-decoder family has a reproducible architectural ceiling.

## Fair comparison with `eeulig/ldct-benchmark`

Publication-facing runs must use:

```bash
HU_RANGE_PRESET=benchmark
```

This gives `[-1024, 1900] HU` and `DATA_RANGE=2924`, matching the benchmark.
Metrics are computed exactly as follows:

- PSNR: clinical window, data range 1;
- SSIM: clinical window, data range 1;
- RMSE: unwindowed physical HU, clipped to the benchmark physical range;
- VIF: unwindowed physical HU, `sigma_n_sq=2.0`;
- chest window: center -600 HU, width 1500 HU;
- abdomen window: center 50 HU, width 400 HU.

The invariant is strict: evaluation must use the same HU preset as training.
`evaluate_physics.py` reads checkpoint metadata and refuses a mismatch by
default. This guard exists because a legacy-trained checkpoint was previously
evaluated under the benchmark preset, producing a convincing but false model
regression.

Reference input-only baseline rows under the benchmark preset:

```text
Chest    PSNR 18.09 | SSIM 0.3119 | RMSE 235.52 | VIF 0.0993
Abdomen  PSNR 29.01 | SSIM 0.8527 | RMSE  18.37 | VIF 0.3882
Overall  PSNR 23.55 | SSIM 0.5823 | RMSE 126.95 | VIF 0.2437
```

Any report with different baseline rows is not directly comparable.

## Architecture

The network remains at full spatial resolution from input to output.

1. A fixed, undecimated Gaussian decomposition forms low, mid and high bands:
   `low=G_coarse(x)`, `mid=G_fine(x)-G_coarse(x)`, `high=x-G_fine(x)`.
   Their sum is exactly the input.
2. Three independent small encoders embed the bands.
3. A 5x5 spatial stem embeds the unmodified LDCT slice.
4. Spectral features are projected to the trunk width and re-injected before
   every residual group through a learned scalar initially equal to 0.1.
5. Four groups each contain dilation 1/2/3/4 blocks. Every block combines a
   dilated 3x3 context convolution with a local 3x3 convolution. No normalisation
   is used.
6. A zero-initialised 5x5 head predicts noise. The repository-facing output is
   its negative, so existing code forms `denoised = LDCT + correction`.

There is no pooling, stride, bottleneck, upsampling, attention, LayerNorm,
BatchNorm or patient-specific conditioning.

The Gaussian cutoffs are initial engineering values. The medical-physics phase
must replace them with values calibrated from training-set signal/noise spectra;
this is explicitly not hidden as a learned or arbitrary claim.

## Loss ablations

The implementation makes every contribution independently switchable.

### A. Architecture baseline

```bash
HU_RANGE_PRESET=benchmark python train_physics.py \
  --lambda-nps 0 --lambda-hu 0 --output-root runs_physics
```

This is pure MSE. It answers whether the full-resolution architecture alone
breaks the 0.167 plateau.

### B. Spectral representation ablation

```bash
HU_RANGE_PRESET=benchmark python train_physics.py \
  --no-spectral --lambda-nps 0 --lambda-hu 0 \
  --output-root runs_physics
```

Comparing A and B isolates fixed low/mid/high representation and re-injection.

### C. NPS regularisation

```bash
HU_RANGE_PRESET=benchmark python train_physics.py \
  --lambda-nps 0.01 --lambda-hu 0 --output-root runs_physics
```

The loss compares the batch-mean radial NPS of removed noise `LDCT-prediction`
with the paired residual `LDCT-NDCT`. It removes the mean, applies a Hann taper,
and compares log spectra so high-power bins do not dominate. The coefficient is
only a starting point: print MSE and NPS contributions and calibrate their
weighted gradient magnitudes before a publication run.

### D. HU-bin preservation

```bash
HU_RANGE_PRESET=benchmark python train_physics.py \
  --lambda-nps 0.01 --lambda-hu 0.1 --output-root runs_physics
```

The term penalises mean HU bias in fixed air/lung/fat-soft-tissue/dense-tissue
intervals, without metadata or patient-specific parameters.

## Required physical validation

PSNR, SSIM and VIF remain necessary for benchmark comparability, but a medical-
physics contribution should also report:

- 2D and radial NPS magnitude, shape, peak and mean frequency;
- HU bias and limits of agreement by tissue interval;
- CNR for low-contrast tasks;
- TTF/MTF, preferably on an appropriate phantom;
- task detectability such as a channelised Hotelling observer when feasible.

A paired LDCT-NDCT difference is not pure noise if there is motion,
misregistration, or reconstruction mismatch. Before interpreting the NPS term
physically, verify alignment and repeat the analysis in homogeneous/low-gradient
regions. This limitation must be reported rather than hidden.

## Smoke test and evaluation

```bash
python test_physics.py

HU_RANGE_PRESET=benchmark python evaluate_physics.py \
  --model runs_physics/2d_physics_spectral_nps0_hu0/best_model.pt \
  --output eval_physics_mse
```

`best_model.pt`, `last_model.pt` and `checkpoint.pt` are separate. The last
model is retained because the old study showed that selecting only by PSNR can
hide a different metric's trajectory.

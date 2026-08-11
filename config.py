"""
LDCT Project - Centralized Configuration
==========================================
All paths, hyperparameters and constants in one place.

Two orthogonal experiment axes are supported:

  INPUT_MODE  : "2d"   -> single centre slice            (in_channels = 1)
                "2.5d" -> (prev, curr, next) pseudo-3D   (in_channels = 3)

  MAMBA_MODE  : "basic" | "residual" | "multiscale" | "full"

Every (INPUT_MODE, MAMBA_MODE) combination writes to its OWN run directory,
so ablation runs never overwrite each other.
"""

import os


def _env_flag(name, default):
    """Read a boolean switch from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name, default):
    """Read an integer from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got '{raw}'") from None


def _env_float(name, default):
    """Read a float from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be a float, got '{raw}'") from None


def _env_blocks(name, default):
    """Read a per-stage block count tuple, e.g. ENC_BLOCKS=2,2,4,8.

    Validated on purpose. A typo here does not crash - it silently builds a
    DIFFERENT architecture, and you only discover it when a checkpoint refuses
    to load hours later.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    parts = [p for p in raw.replace(" ", "").split(",") if p]
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"{name} must be comma-separated integers, got '{raw}'"
        ) from None
    if len(values) != len(default):
        raise ValueError(
            f"{name} needs exactly {len(default)} values (one per stage), got '{raw}'. "
            f"NUM_STAGES is currently {len(default)}."
        )
    if any(v < 1 for v in values):
        raise ValueError(f"{name} entries must all be >= 1, got '{raw}'")
    return values


# ═══════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════
DATA_DIR = "dataset"
TEST_DIR = "test"
EVAL_OUTPUT_DIR = "eval_results"

# Root folder that holds ONE sub-folder per experiment run.
# NOTE: previously this pointed at "FinalCT_2.5D-UNET-DATASET", which is the
# dataset folder. Mixing weights and data in one folder made every ablation
# mode overwrite the same checkpoint.pt. They are now separated.
OUTPUT_ROOT = "runs"
MODEL_DIR = OUTPUT_ROOT          # backward-compatible alias
LEGACY_MODEL_DIR = "FinalCT_2.5D-UNET-DATASET"


# ═══════════════════════════════════════════
# EXPERIMENT AXES
# ═══════════════════════════════════════════
INPUT_MODE = "2.5d"              # "2d" | "2.5d"
NUM_ADJACENT_SLICES = 3          # only used when INPUT_MODE == "2.5d"

# Ablation flag: "basic" | "residual" | "multiscale" | "full"
MAMBA_MODE = "full"

VALID_INPUT_MODES = ("2d", "2.5d")
VALID_MAMBA_MODES = ("basic", "residual", "multiscale", "full")


def normalize_input_mode(input_mode=None):
    """Canonicalize an input-mode string to "2d" or "2.5d"."""
    m = str(input_mode or INPUT_MODE).strip().lower()
    if m in ("2d", "2-d", "single", "slice"):
        return "2d"
    if m in ("2.5d", "25d", "2_5d", "pseudo3d", "pseudo-3d"):
        return "2.5d"
    raise ValueError(f"Unknown INPUT_MODE '{input_mode}'. Use one of {VALID_INPUT_MODES}.")


def normalize_mamba_mode(mamba_mode=None):
    """Validate and lowercase a mamba-mode string."""
    m = str(mamba_mode or MAMBA_MODE).strip().lower()
    if m not in VALID_MAMBA_MODES:
        raise ValueError(f"Unknown MAMBA_MODE '{mamba_mode}'. Use one of {VALID_MAMBA_MODES}.")
    return m


def in_channels_for(input_mode=None):
    """Number of model input channels implied by an input mode."""
    return 1 if normalize_input_mode(input_mode) == "2d" else NUM_ADJACENT_SLICES


def centre_channel_index(input_mode=None):
    """Index of the current (centre) slice inside the input tensor."""
    return in_channels_for(input_mode) // 2


IN_CHANNELS = in_channels_for(INPUT_MODE)
OUT_CHANNELS = 1


# ═══════════════════════════════════════════
# RUN DIRECTORIES (one per INPUT_MODE x MAMBA_MODE combination)
# ═══════════════════════════════════════════
def run_name(mamba_mode=None, input_mode=None):
    im = normalize_input_mode(input_mode).replace(".", "")   # "2d" | "25d"
    mm = normalize_mamba_mode(mamba_mode)
    return f"{im}_{mm}"


def run_paths(mamba_mode=None, input_mode=None, output_root=None):
    """Return all output paths for one experiment run."""
    root = output_root or OUTPUT_ROOT
    run_dir = os.path.join(root, run_name(mamba_mode, input_mode))
    return {
        "run_dir": run_dir,
        "checkpoint": os.path.join(run_dir, "checkpoint.pt"),
        "best_model": os.path.join(run_dir, "best_model.pt"),
        "logs": os.path.join(run_dir, "logs"),
    }


_DEFAULT_PATHS = run_paths()
RUN_DIR = _DEFAULT_PATHS["run_dir"]
CHECKPOINT_PATH = _DEFAULT_PATHS["checkpoint"]
BEST_MODEL_PATH = _DEFAULT_PATHS["best_model"]
LOGS_DIR = _DEFAULT_PATHS["logs"]


# ═══════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════
TOTAL_EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 32
NUM_WORKERS = 8
PATIENCE = 15
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_EPOCHS = 5

# Mixed precision. DISABLED BY DEFAULT - this model does not tolerate bfloat16.
#
# Controlled A/B, identical code/seed/data, 2d + basic + legacy HU, lr 2e-4,
# beta/gamma = 1, 15 epochs:
#
#   precision   ep1     ep2     ep3     outcome
#   bf16 AMP    +1.29   +3.59   +3.76   COLLAPSE at ep4 (-0.48), then stuck at
#                                       dPSNR ~0 for 4 epochs, train loss frozen
#   FP32        +1.29   +5.81   +6.24   monotone to ep15 +7.17,
#                                       PSNR 29.06 dB, SSIM 0.7152
#
# The bf16 failure is invisible to every guard we have: |g|max peaked at 9.7
# (threshold is 100) and there were zero non-finite values. It is not overflow,
# it is accumulated rounding error. bfloat16 has 8 mantissa bits, and with
# beta/gamma = 1 each of the eight NAF stages adds a full-magnitude branch to
# the residual stream, so relative error compounds with depth until the
# optimiser settles in a degenerate identity solution it cannot escape.
#
# The collapse fired exactly as warmup pushed lr past ~1.2e-4. That is also why
# the learning rate appeared to matter enormously here while barely mattering in
# the original FP32 code: lr sensitivity was a symptom of the precision choice,
# not a property of the architecture.
#
# Cost of FP32 on an RTX PRO 6000: 173s vs 157s per epoch, roughly 10%. Cheap.
#
# If you re-enable AMP for VRAM reasons, treat any non-monotone dPSNR in the
# first five epochs as a precision artefact before blaming the model.
USE_AMP = False

# Recompute the Mamba bottleneck during backward to trade compute for VRAM.
USE_GRAD_CHECKPOINT = False


# ═══════════════════════════════════════════
# DATA / PREPROCESSING
# ═══════════════════════════════════════════
SPATIAL_SIZE = (256, 256)
CACHE_DATA = True

# HU windowing preset. The file default is 'legacy'; 'benchmark' is opt-in via
# the environment and is what publication-facing runs should use.
#
#   "legacy"    : [-1000, 600] HU. Bone above 600 HU is clipped, which is a real
#                 limitation for bone-detail claims, but it keeps soft tissue
#                 spread across most of [0, 1].
#
#   "benchmark" : [-1024, 1900] HU. Reproduces the ldct-benchmark convention
#                 EXACTLY: A_MAX + HU_OFFSET = 2924, which is the DATA_RANGE
#                 constant in ldctbench/evaluate/utils.py. Verified stable.
#
#   "wide"      : [-1024, 3072] HU. Historical. It diverges, AND it never
#                 matched the reference either (4096 != 2924). Kept only so old
#                 run directories remain interpretable.
#
# INVARIANT: THE EVAL PRESET MUST EQUAL THE TRAINING PRESET
# ----------------------------------------------------------
# normalize_hu() uses A_MIN/A_MAX, so the preset defines the input scaling the
# network was fitted to. Evaluating a legacy-trained checkpoint under
# 'benchmark' feeds it a distribution it has never seen and reports numbers that
# look exactly like a model regression. The tell is the baseline row: the LDCT
# input metrics depend ONLY on the preset, never on the model, so if two reports
# disagree on the baseline they were produced under different conventions and
# their denoised columns cannot be compared. Checkpoints record their preset in
# meta["hu_preset"] - check it before trusting a comparison.
#
# Reference baseline rows under 'benchmark', for exactly this purpose:
#   Chest    LDCT input: PSNR 18.09 | SSIM 0.3119 | RMSE 235.52 | VIF 0.0993
#   Abdomen  LDCT input: PSNR 29.01 | SSIM 0.8527 | RMSE  18.37 | VIF 0.3882
#
# WHAT THE PRESET DOES *NOT* EXPLAIN (tested twice, do not re-run)
# -----------------------------------------------------------------
# An earlier version of this comment claimed the chest and VIF gaps against the
# published table were "substantially a measurement-convention artefact". That
# was wrong, and it was falsified by the obvious experiment: the SAME checkpoint
# evaluated under 'legacy' and under 'benchmark' produced no meaningful
# difference in any column.
#
# The real mechanism runs the other way. Clipping at 600 HU makes every bone
# voxel IDENTICAL in the prediction and the target, so those pixels contribute
# perfect agreement to the unwindowed VIF and RMSE. 'legacy' numbers are
# therefore slightly OPTIMISTIC, not understated. And the lung window's upper
# bound is only +150 HU while its nominal lower bound (-1350 HU) sits below the
# physical CT floor of -1024 that the reference cannot exceed either, so the
# real difference inside the diagnostic window is about 24 HU.
#
# Conclusion: 'benchmark' is the honest convention to publish under, and it is
# numerically inert. Switching it will not close any gap.
#
# Measured A/B for the legacy-vs-wide instability, identical code and seed,
# mamba-mode=basic, lr=1e-4:
#
#     preset    epoch 3            |g|max        best dPSNR
#     legacy    +3.65 dB, 0 spikes  2.5 - 5.9     +3.90 dB
#     wide      collapse, 498/1113  3652 - 9271   +1.55 dB
#
# Mechanism: a 4096 HU span squeezes soft tissue into a narrow band of [0, 1].
# Local variance then falls far below the SSIM stabilisation constants
# (C2 = 0.03^2), so the SSIM gradient becomes ill-conditioned and scales like
# 1/sigma^2. The per-spike diagnostic confirmed head.weight (the final conv,
# adjacent to the loss) dominated every spike - no SSM tensor was involved.
HU_RANGE_PRESET = os.environ.get("HU_RANGE_PRESET", "legacy").strip().lower()

if HU_RANGE_PRESET == "wide":
    A_MIN = -1024.0
    A_MAX = 3072.0
elif HU_RANGE_PRESET == "benchmark":
    # A_MAX + HU_OFFSET == 2924 == ldctbench DATA_RANGE. Do not "round" these.
    A_MIN = -1024.0
    A_MAX = 1900.0
elif HU_RANGE_PRESET == "legacy":
    A_MIN = -1000.0
    A_MAX = 600.0
else:
    raise ValueError(
        f"HU_RANGE_PRESET must be 'legacy', 'benchmark' or 'wide', got '{HU_RANGE_PRESET}'"
    )

B_MIN = 0.0
B_MAX = 1.0


# ═══════════════════════════════════════════
# EVALUATION & BENCHMARK METRICS CONFIG (ldct-benchmark standard)
# ═══════════════════════════════════════════
# EVAL_DATA_RANGE is DERIVED from HU_RANGE_PRESET, so absolute PSNR, RMSE and
# VIF are NOT comparable across presets. Compare dPSNR (prediction minus noisy
# baseline) when comparing our own runs - it is preset-independent.
#
# Preset -> EVAL_DATA_RANGE:  legacy 1624 | benchmark 2924 | wide 4096
# Only 2924 equals the reference DATA_RANGE in ldctbench/evaluate/utils.py.
HU_OFFSET = 1024.0                     # HU -> non-negative display domain
HU_OFFSET_MAX = A_MAX + HU_OFFSET      # upper clip bound in the offset domain
EVAL_DATA_RANGE = HU_OFFSET_MAX        # backward-compatible alias

# Clinical diagnostic windows, expressed as (center, width) in the HU+1024
# domain. These match CW["C"] and CW["L"] in ldctbench/evaluate/utils.py exactly.
# The reference also defines CW["N"] = (1024 + 40, 80), a neuro/head window; we
# omit it because this dataset contains only chest and abdomen exams. If the
# published row you compare against averages over all three exam types, only
# its PER-REGION numbers are comparable to ours, never the overall mean.
CLINICAL_WINDOWS = {
    "Chest": (HU_OFFSET - 600, 1500),   # Lung window:        C=-600 HU, W=1500 HU
    "Abdomen": (HU_OFFSET + 50, 400),   # Soft tissue window: C=  50 HU, W= 400 HU
}

BENCHMARK_MODELS_LIST = ["redcnn", "wganvgg", "dugan", "transct", "qae", "resnet", "cnn10"]


# ═══════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════
# Capacity is read from the environment so a scaling sweep needs no file edit
# and behaves identically under train.py and run_ablation.py:
#
#   MODEL_WIDTH=48 ENC_BLOCKS=2,2,4,8 DEC_BLOCKS=2,2,2,2 python train.py ...
#
# The SAME variables must be set when running evaluate.py, because the model is
# constructed from this file before the checkpoint is loaded.
#
# CAPACITY WAS TESTED AND IS NOT THE CONSTRAINT
# ----------------------------------------------
# 2d / basic, 20 epochs, FP32, benchmark preset, everything else identical:
#
#   config                     params    chest VIF  chest dVIF  abd dVIF  dPSNR
#   width 32, (1,1,1,1)         5.14 M    0.1627      +0.0634    +0.0581   +6.58
#   width 32, enc (2,2,4,8)     9.43 M    0.1667      +0.0674    +0.0639   +6.78
#   width 48, (1,1,1,1)        11.37 M    0.1666      +0.0673    +0.0637   +6.84
#   width 32, pure Charbonnier  5.14 M    0.1671      +0.0679    +0.0607   +6.83
#
# 5.1 M -> 9.4 M bought +0.0040 chest VIF. 9.4 M -> 11.4 M bought -0.0001. And
# deleting the SSIM and Sobel terms from the loss reached the SAME 0.167 with
# 5.1 M parameters. Four independent interventions, one value. The
# pre-registered success threshold was +0.02 and every one of them missed it by
# a factor of five. Neither parameters nor loss terms are the binding
# constraint. Do not spend more GPU time on width, depth or loss weights.
MODEL_WIDTH = _env_int("MODEL_WIDTH", 32)              # stages are w,2w,4w,8w,16w
D_STATE = _env_int("D_STATE", 16)                      # SSM state dimension N
N_SCAN_DIRECTIONS = 4                                  # SS2D cross-scan directions

# ---------------------------------------------------------------------------
# NUM_STAGES - the depth of the resolution pyramid, and the current experiment
# ---------------------------------------------------------------------------
# Number of encoder downsampling steps. The network downsamples by 2**NUM_STAGES
# and the Mamba bottleneck runs at that resolution. Channels are
# w, 2w, 4w, ... with the bottleneck at w * 2**NUM_STAGES.
#
#   NUM_STAGES  downsampling  bottleneck on a 256 crop  bottleneck channels (w=32)
#        4          16x              16 x 16                     512
#        3           8x              32 x 32                     256
#        2           4x              64 x 64                     128
#
# WHY THIS IS NOW THE EXPERIMENT
# -------------------------------
# Everything non-structural has been eliminated (see the capacity table above).
# The one remaining difference between this network and the reference models is
# that RED-CNN - which reaches VIF 0.221 trained on nothing but MSE - is a FLAT
# convolutional stack that never downsamples at all, while we compress 16x and
# rebuild through PixelShuffle.
#
# That single fact predicts the entire measured pattern:
#
#   * PSNR and SSIM weight low spatial frequencies heavily, and they sit at
#     89-92% of the required gain. Low frequencies survive a bottleneck.
#   * VIF distributes its weight over sub-bands including the highest, and sits
#     at 56%. The highest band cannot be reconstructed faithfully from a 16x16
#     representation, no matter how many channels or parameters it has.
#   * Adding width or depth widens channels, not resolution, so it changed
#     nothing. This is why capacity failed.
#   * No loss function can request information that was destroyed in the forward
#     pass. This is why the loss experiments failed.
#   * Delta_VIF is nearly constant at 0.048-0.068 across patients whose baseline
#     VIF varies fourfold, i.e. the model applies a fixed filter bandwidth
#     rather than an adaptive filtering strength. A fixed bandwidth is exactly
#     what a fixed resolution pyramid imposes.
#
# NUM_STAGES=4 is bit-identical to the previous hard-coded architecture,
# including every parameter name, so all existing checkpoints still load.
#
#   NUM_STAGES=2 python train.py --input-mode 2d --mamba-mode basic \
#     --lr 2e-4 --no-amp --epochs 20 --output-root runs_s2
#
# Expect FEWER parameters and MORE compute per step: the same work moves to
# higher resolution. If a smaller model wins on VIF, the diagnosis is confirmed
# and the resolution pyramid, not the parameter budget, is what to redesign.
NUM_STAGES = _env_int("NUM_STAGES", 4)
if not 1 <= NUM_STAGES <= 5:
    raise ValueError(f"NUM_STAGES must be between 1 and 5, got {NUM_STAGES}")

# One entry per stage. Defaults follow NUM_STAGES, so NUM_STAGES=2 expects two
# values here, not four. DEC_BLOCKS is ordered DEEPEST FIRST.
ENC_BLOCKS = _env_blocks("ENC_BLOCKS", (1,) * NUM_STAGES)
DEC_BLOCKS = _env_blocks("DEC_BLOCKS", (1,) * NUM_STAGES)

# Inputs are reflection-padded up to a multiple of this. Derived, not fixed:
# a 2-stage network only needs a multiple of 4.
SIZE_DIVISOR = 2 ** NUM_STAGES

# Selective-scan backend:
#   "auto" -> official mamba_ssm CUDA kernel when available, else PyTorch fallback
#   "cuda" -> require the official kernel (raises if unavailable)
#   "ref"  -> always use the chunked PyTorch fallback (slow, for CPU/debug)
SCAN_BACKEND = "auto"
SCAN_CHUNK_SIZE = 32                   # sequence chunk for the PyTorch fallback


# ═══════════════════════════════════════════
# LOSS WEIGHTS
# ═══════════════════════════════════════════
# The active objective:
#
#     1.0 * Charbonnier  +  0.6 * SSIM (single scale)  +  0.2 * Sobel edge
#
# Env-overridable so terms can be removed without editing this file:
#
#   LAMBDA_SSIM=0 LAMBDA_EDGE=0 python train.py ...    # pure Charbonnier
#   LAMBDA_SSIM=0.2 python train.py ...                # weight sweep
LAMBDA_L1 = _env_float("LAMBDA_L1", 1.0)      # Charbonnier weight (historical name)
LAMBDA_SSIM = _env_float("LAMBDA_SSIM", 0.6)
LAMBDA_EDGE = _env_float("LAMBDA_EDGE", 0.2)

# ---------------------------------------------------------------------------
# MEASURED EFFECT OF THE LOSS WEIGHTS (2d/basic, 5.14 M, 20 epochs, benchmark)
# ---------------------------------------------------------------------------
#   weights              PSNR chest  SSIM chest  VIF chest  VIF abdomen
#   0.6 / 0.2 (default)    27.21       0.5856     0.1627      0.4463
#   0.2 / 0.2              27.23       0.5848     0.1633      0.4479
#   0.0 / 0.0              27.59       0.5787     0.1671      0.4489
#
# The trade-off is real and in the predicted direction: removing SSIM buys
# +0.38 dB PSNR and +0.0044 VIF at a cost of -0.0069 SSIM. SSIM is a LOCAL
# CONTRAST criterion, well satisfied by output that is locally smooth with the
# right local mean and variance, which is close to the opposite of what VIF
# rewards.
#
# But note that 0.6 -> 0.2 did nothing and all of the effect appeared between
# 0.2 and 0. That is threshold behaviour, not a tunable axis, and the magnitude
# is the same +0.004 that width and depth produced. Keep the defaults: chest
# SSIM is already below target and cannot afford -0.0069 for +0.0044 of VIF.
# ---------------------------------------------------------------------------

# Multi-scale SSIM (5 levels) in place of the single-scale term.
# OFF by default: measured at +0.0084 chest Delta_SSIM and +0.0013 chest
# Delta_VIF, i.e. it did not do what it was built to do. Enable with
# USE_MS_SSIM=1, or disable per-run with train.py's --no-ms-ssim.
# Falls back to single-scale automatically if torchmetrics is unavailable or the
# crop is too small for 5 levels. Costs about 5% more time per step.
USE_MS_SSIM = _env_flag("USE_MS_SSIM", False)
MS_SSIM_BETAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang et al. 2003
MS_SSIM_KERNEL_SIZE = 11

# Window-aligned loss. OFF by default - no separable effect was measured.
#
# The idea: the loss runs on the full normalized [0, 1] range, but PSNR and SSIM
# are only ever measured inside a clinical window (lung 1500 HU, soft tissue
# 400 HU). Everything outside - bone, external air, the scanner table - consumes
# gradient that no metric ever reads.
#
#   "off"   : DEFAULT. Loss on [0, 1] only.
#   "extra" : keeps the global term AND adds a windowed one weighted by
#             LAMBDA_WINDOW. Safer than "only", because the global term still
#             supplies a gradient to pixels outside the window (the windowed
#             term clamps them, so their gradient there is exactly zero).
#   "only"  : windowed term alone. Nothing constrains the out-of-window pixels
#             any more; expect bone and air to drift.
#
# The window is selected per SAMPLE from the batch's body_type. NOTE:
# run_ablation.py does not pass body_type to the loss, so this term is silently
# disabled there (with a warning) - use train.py for window-loss experiments.
WINDOW_LOSS_MODE = os.environ.get("WINDOW_LOSS_MODE", "off").strip().lower()
VALID_WINDOW_LOSS_MODES = ("off", "extra", "only")
if WINDOW_LOSS_MODE not in VALID_WINDOW_LOSS_MODES:
    raise ValueError(
        f"WINDOW_LOSS_MODE must be one of {VALID_WINDOW_LOSS_MODES}, got '{WINDOW_LOSS_MODE}'"
    )

LAMBDA_WINDOW = _env_float("LAMBDA_WINDOW", 0.5)


# ═══════════════════════════════════════════
# SCHEDULER (Cosine Annealing with Linear Warmup)
# ═══════════════════════════════════════════
SCHEDULER_MIN_LR = 5e-5


# ═══════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════
SEED = 0
SPLIT_RANDOM_STATE = 42
SPLIT_TEST_SIZE = 0.2


# ═══════════════════════════════════════════
# EXPLICIT PATIENT SPLITS (100 Patients Total)
# ═══════════════════════════════════════════
EXPECTED_TEST = {
    'C121', 'C249', 'C170', 'C135', 'C280', 'L241', 'L107', 'L006', 'L221', 'L220'
}

EXPECTED_VAL = {
    'C202', 'C219', 'C227', 'C258', 'C067', 'C295', 'C190', 'C232', 'C052', 'C107',
    'L033', 'L187', 'L123', 'L058', 'L212', 'L077', 'L179', 'L014', 'L186', 'L193'
}

EXPECTED_TRAIN = {
    'C095', 'C261', 'C296', 'C218', 'C224', 'C267', 'C099', 'C030', 'C241', 'C162',
    'C268', 'C128', 'C252', 'C234', 'C130', 'C246', 'C124', 'C077', 'C002', 'C021',
    'C203', 'C111', 'C179', 'C012', 'C081', 'C004', 'C120', 'C193', 'C166', 'C257',
    'C160', 'C016', 'C027', 'C050', 'C158', 'L081', 'L248', 'L203', 'L219', 'L210',
    'L277', 'L057', 'L229', 'L131', 'L114', 'L004', 'L237', 'L148', 'L145', 'L116',
    'L150', 'L110', 'L232', 'L134', 'L056', 'L075', 'L209', 'L019', 'L064', 'L299',
    'L160', 'L049', 'L072', 'L071', 'L273', 'L175', 'L178', 'L125', 'L266', 'L170'
}


# ═══════════════════════════════════════════
# DOWNLOADER CONFIG
# ═══════════════════════════════════════════
DATASET_CHEST_LIMIT = len([p for p in (EXPECTED_TRAIN | EXPECTED_VAL) if p.startswith('C')])
DATASET_ABDO_LIMIT = len([p for p in (EXPECTED_TRAIN | EXPECTED_VAL) if p.startswith('L')])
TEST_CHEST_LIMIT = len([p for p in EXPECTED_TEST if p.startswith('C')])
TEST_ABDO_LIMIT = len([p for p in EXPECTED_TEST if p.startswith('L')])

CHEST_LIMIT = DATASET_CHEST_LIMIT + TEST_CHEST_LIMIT     # 50
ABDOMEN_LIMIT = DATASET_ABDO_LIMIT + TEST_ABDO_LIMIT     # 50

DOWNLOAD_WORKERS = 6
COLLECTION = "LDCT-and-projection-data"
DOWNLOAD_TIMEOUT = 300
CHUNK_SIZE = 1 * 1024 * 1024   # 1 MB
NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

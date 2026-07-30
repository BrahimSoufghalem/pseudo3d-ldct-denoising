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

# HU windowing preset. Read this whole block before changing it: the preset
# decides both what the network can represent AND what the metrics mean.
#
#   "legacy"    : [-1000, 600] HU. THE DEFAULT, and the preset every strong
#                 result so far was produced with. Bone above 600 HU is clipped,
#                 which is a real limitation for bone-detail claims, but it
#                 keeps soft tissue spread across most of [0, 1].
#
#   "benchmark" : [-1024, 1900] HU. Reproduces the ldct-benchmark convention
#                 EXACTLY: A_MAX + HU_OFFSET = 2924, which is the DATA_RANGE
#                 constant in ldctbench/evaluate/utils.py. VERIFIED STABLE on a
#                 15-epoch probe (no spikes, no collapse). Use this for any
#                 number you intend to publish next to the reference table.
#
#   "wide"      : [-1024, 3072] HU. Historical. It diverges, AND it never
#                 matched the reference either (4096 != 2924). Kept only so old
#                 run directories remain interpretable.
#
# WHAT THE PRESET DOES *NOT* EXPLAIN (tested, do not re-run)
# ----------------------------------------------------------
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
# numerically inert. Switching it will not close any gap. Do not spend GPU hours
# there again.
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
#
# Overridable from the environment so a probe needs no file edit.
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
MODEL_WIDTH = 32                       # channels of stage 1; stages are w,2w,4w,8w,16w
ENC_BLOCKS = (1, 1, 1, 1)              # NAF blocks per encoder stage
DEC_BLOCKS = (1, 1, 1, 1)              # NAF blocks per decoder stage
D_STATE = 16                           # SSM state dimension N
N_SCAN_DIRECTIONS = 4                  # SS2D cross-scan directions

# Selective-scan backend:
#   "auto" -> official mamba_ssm CUDA kernel when available, else PyTorch fallback
#   "cuda" -> require the official kernel (raises if unavailable)
#   "ref"  -> always use the chunked PyTorch fallback (slow, for CPU/debug)
SCAN_BACKEND = "auto"
SCAN_CHUNK_SIZE = 32                   # sequence chunk for the PyTorch fallback

# The network downsamples 4x, so inputs are reflection-padded to a multiple of 16
SIZE_DIVISOR = 16


# ═══════════════════════════════════════════
# LOSS WEIGHTS
# ═══════════════════════════════════════════
LAMBDA_L1 = 1.0                        # Charbonnier weight (historical name)
LAMBDA_SSIM = 0.6
LAMBDA_EDGE = 0.2

# ---------------------------------------------------------------------------
# WHY THE NEXT TWO SWITCHES EXIST
# ---------------------------------------------------------------------------
# evaluate.py now reports the metrics of the raw LDCT input as well as of the
# prediction, which lets us score the model on the FRACTION OF THE REQUIRED GAIN
# it actually achieves instead of on absolute numbers. Measured on the 10 test
# patients (2d / basic, 20 epochs, 512x512), against the published targets:
#
#   metric        baseline   ours     target    needed    got      share
#   PSNR chest     18.09     27.21    28.36     +10.27    +9.12     89%
#   PSNR abdomen   29.01     33.04    33.22      +4.21    +4.03     96%
#   SSIM chest      0.3119    0.5856   0.609     +0.2971  +0.2737   92%
#   SSIM abdomen    0.8527    0.9089   0.9028    +0.0501  +0.0563  112%
#   VIF  chest      0.0993    0.1627   0.221     +0.1217  +0.0634   52%
#   VIF  abdomen    0.3882    0.4463   0.491     +0.1028  +0.0581   57%
#
# Two things follow, and they are the whole justification for this section.
#
# 1. Delta_VIF is POSITIVE everywhere (+0.048 to +0.068 on all ten patients).
#    The model adds visual information, it does not destroy it. The "the loss
#    family has an unbeatable posterior-mean ceiling" argument is therefore
#    dead, and so is the case for going adversarial right now.
#
# 2. The shortfall is SPECIFIC to VIF: 89-112% of the required gain on PSNR and
#    SSIM, but only 52-57% on VIF. A generic capacity shortage would drag all
#    three down proportionally. Something is missing that VIF measures and the
#    other two do not - and that is information spread over MULTIPLE SCALES.
#    VIF decomposes the image into sub-bands and sums the information preserved
#    in each. Our SSIM term uses a single 11x11 window, i.e. one scale.
#    We train on one scale and get scored on several.
# ---------------------------------------------------------------------------

# Replace the single-scale SSIM term with multi-scale SSIM (5 levels).
# Falls back to single-scale automatically if torchmetrics is unavailable or the
# crop is too small for 5 levels. Costs about 5% more time per step.
# Requires min(H, W) > (kernel_size - 1) * 2**(levels - 1) = 160 for 5 levels,
# which SPATIAL_SIZE (256, 256) satisfies.
USE_MS_SSIM = _env_flag("USE_MS_SSIM", True)
MS_SSIM_BETAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang et al. 2003
MS_SSIM_KERNEL_SIZE = 11

# Window-aligned loss.
#
# The loss runs on the full normalized [0, 1] range, but PSNR and SSIM are only
# ever measured inside a clinical window: lung is 1500 HU wide, soft tissue only
# 400 HU wide, out of a 1624 HU (legacy) or 2924 HU (benchmark) span. Everything
# outside - bone, external air, the scanner table - consumes gradient that no
# metric ever reads. This term re-spends that capacity where it is scored.
#
#   "off"   : previous behaviour, loss on [0, 1] only.
#   "extra" : DEFAULT. Keeps the global term AND adds a windowed one weighted by
#             LAMBDA_WINDOW. Safer, because the global term still supplies a
#             gradient to pixels outside the window (the windowed term clamps
#             them, so their gradient there is exactly zero).
#   "only"  : windowed term alone. Strongest effect, but nothing constrains the
#             out-of-window pixels any more; expect bone and air to drift.
#
# The window is selected per SAMPLE from the batch's body_type, so a mixed
# chest/abdomen batch is handled correctly.
WINDOW_LOSS_MODE = os.environ.get("WINDOW_LOSS_MODE", "extra").strip().lower()
VALID_WINDOW_LOSS_MODES = ("off", "extra", "only")
if WINDOW_LOSS_MODE not in VALID_WINDOW_LOSS_MODES:
    raise ValueError(
        f"WINDOW_LOSS_MODE must be one of {VALID_WINDOW_LOSS_MODES}, got '{WINDOW_LOSS_MODE}'"
    )

LAMBDA_WINDOW = float(os.environ.get("LAMBDA_WINDOW", "0.5"))


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

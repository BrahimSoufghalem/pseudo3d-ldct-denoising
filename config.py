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
#   "legacy"    : [-1000, 600] HU. THE DEFAULT, and the only preset verified to
#                 train stably end to end. Bone above 600 HU is clipped, which
#                 is a real limitation for bone-detail claims, but it keeps soft
#                 tissue spread across most of [0, 1].
#
#   "benchmark" : [-1024, 1900] HU. Reproduces the ldct-benchmark convention
#                 EXACTLY: A_MAX + HU_OFFSET = 2924, which is the DATA_RANGE
#                 constant in ldctbench/evaluate/utils.py (max bone HU 1900 plus
#                 the 1024 offset). Use this for any number you intend to
#                 compare against the published table. NOT yet verified stable.
#
#   "wide"      : [-1024, 3072] HU. Historical. It diverges, AND it never
#                 matched the reference either (4096 != 2924). Kept only so old
#                 run directories remain interpretable.
#
# Why "benchmark" matters, verified by reading eeulig/ldct-benchmark @ 09b1011:
#
#   Our metric code is already identical to the reference - apply_center_width
#   line for line, the same two clinical windows, PSNR and SSIM on windowed
#   images with data_range 1.0, VIF via torchmetrics with sigma_n_sq=2.0 on
#   UNWINDOWED images clipped to [0, DATA_RANGE], RMSE likewise, all per slice.
#   The one and only mismatch is that clip bound. And because the data pipeline
#   clips HU at A_MAX before the model ever sees a voxel, the consequence is not
#   cosmetic:
#
#     Abdomen PSNR/SSIM  COMPARABLE. The soft-tissue window spans [-150, 250] HU,
#                        entirely inside [-1000, 600]. Empirically confirmed:
#                        0.9043 SSIM here vs 0.9028 +- 0.0007 published, after
#                        only two epochs.
#     Chest PSNR/SSIM    NOT comparable. The lung window spans [-1350, 150] HU.
#                        Clipping at -1000 flattens 350 HU inside the diagnostic
#                        window itself, which caps chest scores by construction.
#     VIF, RMSE          NOT comparable. Both are computed unwindowed over the
#                        full [0, DATA_RANGE], so they see the whole HU span,
#                        including the bone we discard.
#
#   So the "gap" against the published chest and VIF columns is substantially a
#   measurement-convention artefact, not a model deficit. Fix the convention
#   before spending GPU hours chasing it.
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
# "benchmark" spans 2924 HU, i.e. 1.83x legacy, against 2.56x for "wide". It is
# therefore a milder version of the same risk, and the two other regressions
# that were confounded with the original divergence (NAFBlock beta/gamma = 0 and
# bf16 AMP) are both fixed now. That makes it worth retesting, not safe to
# assume. Probe it on the cheap configuration first and watch epochs 2-4:
#
#   HU_RANGE_PRESET=benchmark python train.py --input-mode 2d --mamba-mode basic \
#     --lr 2e-4 --epochs 15 --output-root probe_benchmark
#
# Accept the preset only if epoch 2 dPSNR > +3, epoch 3 does not regress, and
# spikes stay at 0/1113. Also watch abdomen PSNR specifically: a wider span
# gives soft tissue a narrower slice of [0, 1] and hence a smaller Charbonnier
# gradient, so if abdomen PSNR drops below the legacy 32.58 dB this is a
# trade-off rather than an upgrade.
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
LAMBDA_L1 = 1.0
LAMBDA_SSIM = 0.6
LAMBDA_EDGE = 0.2


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

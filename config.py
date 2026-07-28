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

# Mixed precision. bfloat16 is preferred when the GPU supports it (no GradScaler
# needed); otherwise float16 + GradScaler is used automatically.
USE_AMP = True

# Recompute the Mamba bottleneck during backward to trade compute for VRAM.
USE_GRAD_CHECKPOINT = False


# ═══════════════════════════════════════════
# DATA / PREPROCESSING
# ═══════════════════════════════════════════
SPATIAL_SIZE = (256, 256)
CACHE_DATA = True

# HU windowing preset.
#   "wide"   : [-1024, 3072] HU -> matches the ldct-benchmark convention and
#              fully contains BOTH clinical evaluation windows below.
#   "legacy" : [-1000, 600] HU -> the original setting. It clipped all bone
#              (>600 HU) and part of the lung window (down to -1350 HU), which
#              biases the windowed PSNR/SSIM numbers.
# Switch back with a single line if you need to reproduce the old results.
HU_RANGE_PRESET = "wide"

if HU_RANGE_PRESET == "wide":
    A_MIN = -1024.0
    A_MAX = 3072.0
elif HU_RANGE_PRESET == "legacy":
    A_MIN = -1000.0
    A_MAX = 600.0
else:
    raise ValueError("HU_RANGE_PRESET must be 'wide' or 'legacy'")

B_MIN = 0.0
B_MAX = 1.0


# ═══════════════════════════════════════════
# EVALUATION & BENCHMARK METRICS CONFIG (ldct-benchmark standard)
# ═══════════════════════════════════════════
HU_OFFSET = 1024.0                     # HU -> non-negative display domain
HU_OFFSET_MAX = A_MAX + HU_OFFSET      # upper clip bound in the offset domain
EVAL_DATA_RANGE = HU_OFFSET_MAX        # backward-compatible alias

# Clinical diagnostic windows, expressed as (center, width) in the HU+1024 domain
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

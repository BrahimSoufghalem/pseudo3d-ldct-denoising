"""
LDCT Project — Centralized Configuration
==========================================
All paths, hyperparameters, and constants in one place.
"""

import os  
 
# ═══════════════════════════════════════════ 
# PATHS
# ═══════════════════════════════════════════
DATA_DIR = "dataset"
TEST_DIR = "test"
MODEL_DIR = "FinalCT_2.5D-UNET-DATASET"

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pt")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
LOGS_DIR = os.path.join(MODEL_DIR, "logs")

# ═══════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════
TOTAL_EPOCHS = 70
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-6
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
NUM_WORKERS = 8
PATIENCE = 10
GRAD_CLIP_MAX_NORM = 1.0

# ═══════════════════════════════════════════
# DATA / PREPROCESSING
# ═══════════════════════════════════════════
SPATIAL_SIZE = (256, 256)
A_MIN = -1024  # HU window min
A_MAX = 1600   # HU window max
B_MIN = 0.0
B_MAX = 1.0
CACHE_DATA = True

# ═══════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════
IN_CHANNELS = 3       # pseudo-3D: prev + curr + next
OUT_CHANNELS = 1
CHANNELS = (32, 64, 128, 256, 512)
STRIDES = (2, 2, 2, 2)
NUM_RES_UNITS = 2
DROPOUT = 0.1

# ═══════════════════════════════════════════
# LOSS WEIGHTS
# ═══════════════════════════════════════════
LAMBDA_L1 = 1.0
LAMBDA_SSIM = 0.5
LAMBDA_PERC = 0.2
LAMBDA_EDGE = 0.1

# ═══════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════
SCHEDULER_MODE = "max"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-7

# ═══════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════
SEED = 0
SPLIT_RANDOM_STATE = 42
SPLIT_TEST_SIZE = 0.2

# ═══════════════════════════════════════════
# DOWNLOADER CONFIG
# ═══════════════════════════════════════════
DATASET_CHEST_LIMIT = 42
DATASET_ABDO_LIMIT = 42  
TEST_CHEST_LIMIT = 8       
TEST_ABDO_LIMIT = 8

CHEST_LIMIT = DATASET_CHEST_LIMIT + TEST_CHEST_LIMIT     # 50
ABDOMEN_LIMIT = DATASET_ABDO_LIMIT + TEST_ABDO_LIMIT     # 50

DOWNLOAD_WORKERS = 6
COLLECTION = "LDCT-and-projection-data"
DOWNLOAD_TIMEOUT = 300
CHUNK_SIZE = 1 * 1024 * 1024   # 1 MB
NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

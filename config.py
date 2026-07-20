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
EVAL_OUTPUT_DIR = "eval_results"

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
# EVALUATION & BENCHMARK METRICS CONFIG (ldct-benchmark standard)
# ═══════════════════════════════════════════
EVAL_DATA_RANGE = 2924.0  # Maximum HU of bone (1900) + DICOM offset (1024) -> 2924

# Clinical Diagnostic Windows in (HU + 1024 offset) domain
CLINICAL_WINDOWS = {
    "Chest": (1024 - 600, 1500),    # Lung window: C=-600 HU, W=1500 HU
    "Abdomen": (1024 + 50, 400),    # Soft tissue window: C=50 HU, W=400 HU
}

# Clinical Diagnostic Windows for Multi-Window Preprocessing (in raw HU)
WINDOW_LUNG_CENTER = -600.0
WINDOW_LUNG_WIDTH = 1500.0
WINDOW_SOFT_CENTER = 50.0
WINDOW_SOFT_WIDTH = 400.0

BENCHMARK_MODELS_LIST = ["redcnn", "wganvgg", "dugan", "transct", "qae", "resnet", "cnn10"]

# ═══════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════
IN_CHANNELS = 9       # Multi-Window Pseudo-3D: (prev, curr, next) x (Full, Lung, Soft)
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

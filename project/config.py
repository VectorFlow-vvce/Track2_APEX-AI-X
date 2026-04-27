"""
config.py - Central configuration for the off-road segmentation system.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(PROJECT_DIR, "data")

TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR  = os.path.join(DATA_ROOT, "train", "Segmentation")
VAL_IMAGE_DIR   = os.path.join(DATA_ROOT, "val",   "Color_Images")
VAL_MASK_DIR    = os.path.join(DATA_ROOT, "val",   "Segmentation")

TRAIN_IMG  = TRAIN_IMAGE_DIR
TRAIN_MASK = TRAIN_MASK_DIR
VAL_IMG    = VAL_IMAGE_DIR
VAL_MASK   = VAL_MASK_DIR
TEST_IMG   = os.path.join(DATA_ROOT, "Offroad_Segmentation_testImages", "Color_Images")

OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")
CKPT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
VIZ_DIR     = os.path.join(OUTPUT_DIR, "visualizations")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")

for _d in (OUTPUT_DIR, CKPT_DIR, VIZ_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "trees", "lush_bushes", "dry_grass", "dry_bushes", "ground_clutter",
    "flowers", "logs", "rocks", "landscape", "sky"
]
NUM_CLASSES = len(CLASS_NAMES)

PALETTE = [
    (0,   0,   0),    (194, 178, 128), (128, 128, 128), (169, 169, 169),
    (34,  139, 34),   (135, 206, 235), (255, 0,   0),   (210, 180, 140),
]

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "segformer"
BACKBONE        = "b3"
PRETRAINED      = True
IMAGE_SIZE      = 256

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 4
NUM_WORKERS     = 2
NUM_EPOCHS      = 25
LEARNING_RATE   = 3e-5
WEIGHT_DECAY    = 0.01
LR_SCHEDULER    = "cosine"
WARMUP_EPOCHS   = 2

CE_WEIGHT       = 0.5
DICE_WEIGHT     = 0.5

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUG_BRIGHTNESS  = 0.25
AUG_CONTRAST    = 0.25
AUG_SATURATION  = 0.15
AUG_HUE         = 0.03
AUG_BLUR_PROB   = 0.2
AUG_FLIP_PROB   = 0.5
AUG_CROP_SCALE  = (0.75, 1.0)

# ─── Class Weights ────────────────────────────────────────────────────────────
CLASS_WEIGHTS = [1, 2, 2, 3, 5, 3, 4, 4, 1, 1]

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED            = 42
SAVE_EVERY      = 5
LOG_INTERVAL    = 10
IGNORE_INDEX    = 255

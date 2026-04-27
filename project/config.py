"""
config.py - Central configuration for the off-road segmentation system.
All hyperparameters, paths, and settings live here.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
# PROJECT_DIR = the folder that contains this config.py file (i.e. project/)
# This makes all paths relative to the project folder, so the code works
# regardless of where you run it from or which machine you're on.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(PROJECT_DIR, "data")

# Dataset directories — matched to actual on-disk structure:
#   project/data/train/Color_Images/
#   project/data/train/Segmentation/
#   project/data/val/Color_Images/
#   project/data/val/Segmentation/
TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR  = os.path.join(DATA_ROOT, "train", "Segmentation")
VAL_IMAGE_DIR   = os.path.join(DATA_ROOT, "val",   "Color_Images")
VAL_MASK_DIR    = os.path.join(DATA_ROOT, "val",   "Segmentation")

# Legacy aliases kept so other modules (test.py, analyze.py) don't break
TRAIN_IMG  = TRAIN_IMAGE_DIR
TRAIN_MASK = TRAIN_MASK_DIR
VAL_IMG    = VAL_IMAGE_DIR
VAL_MASK   = VAL_MASK_DIR
TEST_IMG   = os.path.join(DATA_ROOT, "testImages")  # optional; may not exist

OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")
CKPT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
VIZ_DIR     = os.path.join(OUTPUT_DIR, "visualizations")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")

for _d in (OUTPUT_DIR, CKPT_DIR, VIZ_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Classes ──────────────────────────────────────────────────────────────────
# Adjust class names and count to match your dataset labels.
# Index 0 is typically background.
CLASS_NAMES = [
    "trees",
    "lush_bushes",
    "dry_grass",
    "dry_bushes",
    "ground_clutter",
    "flowers",
    "logs",
    "rocks",
    "landscape",
    "sky"
]
NUM_CLASSES = len(CLASS_NAMES)

# Colour palette for visualisation (BGR-safe, one RGB tuple per class)
PALETTE = [
    (0,   0,   0),    # background  – black
    (194, 178, 128),  # sand        – sandy
    (128, 128, 128),  # rock        – grey
    (169, 169, 169),  # gravel      – light grey
    (34,  139, 34),   # vegetation  – green
    (135, 206, 235),  # sky         – sky-blue
    (255, 0,   0),    # obstacle    – red
    (210, 180, 140),  # trail       – tan
]

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "segformer"          # "segformer" | "deeplabv3plus"
BACKBONE        = "b2"                 # SegFormer variant: b0-b5
PRETRAINED      = True
IMAGE_SIZE      = 256                  # Single int for square images - optimized for fast recovery

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 8                    # Optimized for GPU memory and speed
NUM_WORKERS     = 2                    # Parallel data loading for performance
NUM_EPOCHS      = 10                   # Fast recovery training
LEARNING_RATE   = 5e-5                 # Optimized LR for fast convergence
WEIGHT_DECAY    = 0.01
LR_SCHEDULER    = "cosine"             # "cosine" | "poly" | "step"
WARMUP_EPOCHS   = 2

# Loss weights
CE_WEIGHT       = 0.5
DICE_WEIGHT     = 0.5

# ─── Augmentation ─────────────────────────────────────────────────────────────
# Simple augmentations for better IoU - horizontal flip and brightness/contrast only
AUG_BRIGHTNESS  = 0.2      # Light brightness adjustment
AUG_CONTRAST    = 0.2      # Light contrast adjustment
AUG_SATURATION  = 0.0      # Disabled
AUG_HUE         = 0.0      # Disabled
AUG_BLUR_PROB   = 0.0      # Disabled - heavy augmentation
AUG_FLIP_PROB   = 0.5      # Keep horizontal flip
AUG_CROP_SCALE  = (0.8, 1.0)  # Less aggressive cropping

# ─── Class Weights for Imbalanced Dataset ────────────────────────────────────
# Weights optimized for weak classes: ground_clutter(5), logs(4), rocks(4), dry_bushes(3)
# Class order: trees, lush_bushes, dry_grass, dry_bushes, ground_clutter, flowers, logs, rocks, landscape, sky
CLASS_WEIGHTS = [1, 2, 2, 3, 5, 3, 4, 4, 1, 1]  # Boosted weak classes for fast recovery

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED            = 42
SAVE_EVERY      = 5          # save checkpoint every N epochs
LOG_INTERVAL    = 10         # log every N batches
IGNORE_INDEX    = 255        # mask pixels to ignore during loss computation

"""
data_loader.py - Dataset and DataLoader utilities.

Dataset layout expected:
    data/train/Color_Images/   ← RGB images
    data/train/Segmentation/   ← single-channel class-ID masks
    data/val/Color_Images/
    data/val/Segmentation/

Images and masks are matched by filename stem (name without extension).
Masks are always loaded as single-channel (mode "L") — never normalised.
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import config
from augmentations import get_train_transforms, get_val_transforms


# ─── Helpers ──────────────────────────────────────────────────────────────────

# All extensions we search for (case-insensitive via both cases)
_IMG_EXTS  = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp", "*.BMP")
_MASK_EXTS = (".png", ".PNG", ".jpg", ".JPG", ".bmp", ".BMP")


def _collect_pairs(img_dir: str, mask_dir: str):
    """
    Scan img_dir for all images, then find the matching mask in mask_dir
    by stem name (filename without extension).

    Returns a sorted list of (image_path, mask_path) tuples.
    Raises FileNotFoundError if either directory is missing or no pairs found.
    """
    # ── Validate directories ──────────────────────────────────────────────────
    for label, d in (("image", img_dir), ("mask", mask_dir)):
        abs_d = os.path.abspath(d)
        if not os.path.isdir(abs_d):
            raise FileNotFoundError(
                f"\n[DataLoader] ✗ {label} directory not found!\n"
                f"  Expected : {abs_d}\n"
                f"  Check that the data folder exists at project/data/ and that\n"
                f"  TRAIN_IMAGE_DIR / VAL_IMAGE_DIR in config.py are correct."
            )

    # ── Collect all image paths ───────────────────────────────────────────────
    img_paths = []
    for ext in _IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(set(img_paths))

    if not img_paths:
        raise FileNotFoundError(
            f"[DataLoader] No images found in: {img_dir}\n"
            f"  Supported extensions: jpg, jpeg, png, bmp"
        )

    # ── Match each image to its mask ──────────────────────────────────────────
    pairs   = []
    missing = []

    for ip in img_paths:
        stem  = os.path.splitext(os.path.basename(ip))[0]
        found = None
        for mext in _MASK_EXTS:
            mp = os.path.join(mask_dir, stem + mext)
            if os.path.exists(mp):
                found = mp
                break
        if found:
            pairs.append((ip, found))
        else:
            missing.append(stem)

    if missing:
        print(
            f"[DataLoader] Warning: {len(missing)} image(s) have no matching mask "
            f"and will be skipped.\n"
            f"  First few: {missing[:5]}"
        )

    if not pairs:
        raise FileNotFoundError(
            f"[DataLoader] No image-mask pairs could be matched.\n"
            f"  img_dir : {img_dir}\n"
            f"  mask_dir: {mask_dir}\n"
            f"  Make sure filenames match between Color_Images/ and Segmentation/."
        )

    return pairs


def _collect_images(img_dir: str):
    """Collect image paths for test inference (no masks needed)."""
    if not os.path.isdir(img_dir):
        return []
    paths = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(set(paths))


# ─── Dataset ──────────────────────────────────────────────────────────────────

def remap_mask(mask):
    mapping = {
        100: 0,
        200: 1,
        300: 2,
        500: 3,
        550: 4,
        600: 5,
        700: 6,
        800: 7,
        7100: 8,
        10000: 9,
        255: 0
    }
    new_mask = np.zeros_like(mask)
    for k, v in mapping.items():
        new_mask[mask == k] = v
    return new_mask


class SegmentationDataset(Dataset):
    """
    Paired image + mask dataset.

    Images  → opened as RGB  → tensor [3, H, W] float32, normalised to [0, 1]
    Masks   → opened as "L"  → tensor [H, W] int64 with raw class IDs
                                (never normalised, never divided by 255)
    """

    def __init__(self, img_dir: str, mask_dir: str, transform=None, split: str = ""):
        self.pairs     = _collect_pairs(img_dir, mask_dir)
        self.transform = transform
        self.split     = split
        print(f"[Dataset] {split or 'split'}: {len(self.pairs)} image-mask pairs")
        print(f"          images : {os.path.abspath(img_dir)}")
        print(f"          masks  : {os.path.abspath(mask_dir)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # ── Load ──────────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")   # always 3-channel
        mask  = Image.open(mask_path)
        
        mask_np = np.array(mask, dtype=np.int32)
        mask_np = remap_mask(mask_np)
        mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")

        # ── Apply paired transforms ───────────────────────────────────────────
        # Transforms handle: resize, augment, ToTensor, Normalize (image only)
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Fallback: just resize and convert — no augmentation
            from augmentations import Resize, ToTensor, Normalize, Compose
            fallback = Compose([Resize(config.IMAGE_SIZE), ToTensor(), Normalize()])
            image, mask = fallback(image, mask)

        # ── Safety: mask must be int64, not float ─────────────────────────────
        if isinstance(mask, torch.Tensor):
            mask = mask.long()

        return image, mask, os.path.basename(img_path)


class TestDataset(Dataset):
    """Images-only dataset for test inference (no masks)."""

    def __init__(self, img_dir: str, transform=None):
        self.paths     = _collect_images(img_dir)
        self.transform = transform
        print(f"[TestDataset] {len(self.paths)} images in {img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image    = Image.open(img_path).convert("RGB")

        if self.transform:
            image, _ = self.transform(image, None)
        else:
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1)
            ).float() / 255.0

        return image, os.path.basename(img_path)


# ─── Debug helper ─────────────────────────────────────────────────────────────

def debug_dataset(ds: SegmentationDataset, name: str = "dataset"):
    """
    Print shape and value-range info for the first sample.
    Called automatically by get_train_loader() / get_val_loader().
    """
    if len(ds) == 0:
        print(f"[Debug] {name} is empty!")
        return

    image, mask, fname = ds[0]
    print(f"\n[Debug] {name} sample — '{fname}'")
    print(f"  image : shape={tuple(image.shape)}  dtype={image.dtype}  "
          f"min={image.min():.3f}  max={image.max():.3f}")
    print(f"  mask  : shape={tuple(mask.shape)}   dtype={mask.dtype}  "
          f"min={int(mask.min())}  max={int(mask.max())}  "
          f"unique classes={mask.unique().tolist()}")


# ─── DataLoader factories ─────────────────────────────────────────────────────

def get_train_loader(debug: bool = True):
    ds = SegmentationDataset(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        transform=get_train_transforms(),
        split="train",
        # Use full dataset - no sample limit for better IoU
    )
    if debug:
        debug_dataset(ds, "train")

    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=len(ds) >= config.BATCH_SIZE,  # only drop if enough samples
    )


def get_val_loader(debug: bool = True):
    ds = SegmentationDataset(
        config.VAL_IMAGE_DIR,
        config.VAL_MASK_DIR,
        transform=get_val_transforms(),
        split="val",
    )
    if debug:
        debug_dataset(ds, "val")

    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )


def get_test_loader():
    ds = TestDataset(
        config.TEST_IMG,
        transform=get_val_transforms(),
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

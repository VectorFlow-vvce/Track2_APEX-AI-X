"""
augmentations.py - Paired image+mask augmentations using torchvision v2 / albumentations.
Falls back to a manual torchvision pipeline if albumentations is not installed.
"""

import random
import numpy as np
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as TF

import config


# ─── Paired transform base ────────────────────────────────────────────────────

class Compose:
    """Apply a list of paired (image, mask) transforms sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size):
        # Handle both int and tuple formats
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (H, W)

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        return image, mask


class RandomResizedCrop:
    """Random crop then resize back to target size."""
    def __init__(self, size, scale=(0.5, 1.0)):
        # Handle both int and tuple formats
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale

    def __call__(self, image, mask):
        w, h   = image.size
        area   = w * h
        target = random.uniform(*self.scale) * area
        ratio  = random.uniform(3/4, 4/3)
        cw     = int(round((target * ratio) ** 0.5))
        ch     = int(round((target / ratio) ** 0.5))
        cw     = min(cw, w)
        ch     = min(ch, h)
        x      = random.randint(0, w - cw)
        y      = random.randint(0, h - ch)

        image = TF.crop(image, y, x, ch, cw)
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.crop(mask, y, x, ch, cw)
            mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class ColorJitter:
    """Brightness, contrast, saturation, hue jitter."""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05):
        self.brightness  = brightness
        self.contrast    = contrast
        self.saturation  = saturation
        self.hue         = hue

    def __call__(self, image, mask):
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast),   1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)

        image = TF.adjust_brightness(image, b)
        image = TF.adjust_contrast(image, c)
        image = TF.adjust_saturation(image, s)
        image = TF.adjust_hue(image, h)
        return image, mask


class GaussianBlur:
    """Simulate dust/haze with Gaussian blur."""
    def __init__(self, p=0.3, radius_range=(1, 3)):
        self.p            = p
        self.radius_range = radius_range

    def __call__(self, image, mask):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            image  = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, mask


class ToTensor:
    """Convert PIL images to tensors."""
    def __call__(self, image, mask):
        image = TF.to_tensor(image)                          # [3,H,W] float32 in [0,1]
        if mask is not None:
            mask = torch.from_numpy(np.array(mask)).long()  # [H,W] int64
        return image, mask


class Normalize:
    """ImageNet normalisation."""
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std =(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask):
        image = TF.normalize(image, self.mean, self.std)
        return image, mask


# ─── Public API ───────────────────────────────────────────────────────────────

def get_train_transforms():
    return Compose([
        RandomResizedCrop(config.IMAGE_SIZE, scale=config.AUG_CROP_SCALE),
        RandomHorizontalFlip(p=config.AUG_FLIP_PROB),
        ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
            saturation=config.AUG_SATURATION,
            hue=config.AUG_HUE,
        ),
        GaussianBlur(p=config.AUG_BLUR_PROB, radius_range=(0.5, 2.0)),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms():
    return Compose([
        Resize(config.IMAGE_SIZE),
        ToTensor(),
        Normalize(),
    ])

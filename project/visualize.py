"""
visualize.py - Visualisation utilities.
Produces side-by-side comparisons of input / ground-truth / prediction.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

import config
from utils import tensor_to_numpy_image


# ─── Colour helpers ───────────────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray) -> Image.Image:
    """
    Convert a single-channel class-ID mask (H, W) uint8 to an RGB PIL image
    using the palette defined in config.PALETTE.
    """
    h, w    = mask.shape
    rgb     = np.zeros((h, w, 3), dtype=np.uint8)
    palette = config.PALETTE

    for cls_id, colour in enumerate(palette):
        rgb[mask == cls_id] = colour

    return Image.fromarray(rgb)


def mask_to_pil(mask_tensor: torch.Tensor) -> Image.Image:
    """Convert a [H, W] int64 tensor to a colourised PIL image."""
    return colorize_mask(mask_tensor.cpu().numpy().astype(np.uint8))


# ─── Side-by-side comparison ──────────────────────────────────────────────────

def save_comparison(
    image_tensor:  torch.Tensor,
    pred_tensor:   torch.Tensor,
    gt_tensor:     torch.Tensor | None,
    filename:      str,
    out_dir:       str = config.VIZ_DIR,
    title:         str = "",
):
    """
    Save a side-by-side figure:
        [Input Image] | [Ground Truth] | [Prediction]

    Args:
        image_tensor : [3, H, W] normalised float tensor
        pred_tensor  : [H, W] int64 predicted class IDs
        gt_tensor    : [H, W] int64 ground-truth class IDs (or None for test)
        filename     : output filename (e.g. "sample_001.png")
        out_dir      : directory to save into
        title        : optional suptitle
    """
    os.makedirs(out_dir, exist_ok=True)

    img_np   = tensor_to_numpy_image(image_tensor)
    pred_rgb = np.array(colorize_mask(pred_tensor.cpu().numpy().astype(np.uint8)))

    n_cols = 3 if gt_tensor is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    if gt_tensor is not None:
        gt_rgb = np.array(colorize_mask(gt_tensor.cpu().numpy().astype(np.uint8)))
        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth", fontsize=12)
        axes[1].axis("off")
        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction", fontsize=12)
        axes[2].axis("off")
    else:
        axes[1].imshow(pred_rgb)
        axes[1].set_title("Prediction", fontsize=12)
        axes[1].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=[c / 255 for c in colour], label=name)
        for name, colour in zip(config.CLASS_NAMES, config.PALETTE)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 4),
               fontsize=8, framealpha=0.8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out_path


# ─── Batch visualisation ──────────────────────────────────────────────────────

@torch.no_grad()
def visualize_batch(model, loader, n_samples: int = 8, out_dir: str = config.VIZ_DIR):
    """
    Run model on the first `n_samples` from `loader` and save comparisons.
    """
    model.eval()
    saved = 0

    for images, masks, filenames in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1)

        for img, pred, gt, fname in zip(images, preds, masks, filenames):
            stem    = os.path.splitext(fname)[0]
            out_fn  = f"compare_{stem}.png"
            path    = save_comparison(img, pred, gt, out_fn, out_dir)
            print(f"  Saved: {path}")
            saved  += 1
            if saved >= n_samples:
                return

    print(f"[Visualize] Saved {saved} comparison images to {out_dir}")


# ─── Training curve ───────────────────────────────────────────────────────────

def plot_training_curves(log_path: str, out_dir: str = config.VIZ_DIR):
    """
    Read a .jsonl training log and plot loss + mIoU curves.
    """
    import json

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[Visualize] Empty log file — nothing to plot.")
        return

    epochs     = [r["epoch"]      for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss   = [r["val_loss"]   for r in records]
    val_miou   = [r["val_miou"]   for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, val_miou, color="green", label="Val mIoU")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
    ax2.set_title("Validation mIoU"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visualize] Training curves saved to {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data_loader import get_val_loader
    from model       import build_model
    from utils       import find_best_checkpoint, load_checkpoint

    parser = argparse.ArgumentParser(description="Visualise model predictions")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n",          type=int, default=8, help="Number of samples")
    parser.add_argument("--log",        type=str, default=None, help="Path to .jsonl log")
    args = parser.parse_args()

    if args.log:
        plot_training_curves(args.log)

    model = build_model()
    ckpt  = args.checkpoint or find_best_checkpoint()
    if ckpt:
        load_checkpoint(model, ckpt)
        loader = get_val_loader()
        visualize_batch(model, loader, n_samples=args.n)
    else:
        print("No checkpoint found. Train first with: python train.py")

"""
test.py - Run inference on test images and save predicted masks.

Run:
    python test.py
    python test.py --checkpoint outputs/checkpoints/best_model.pth
    python test.py --checkpoint outputs/checkpoints/best_model.pth --save_masks
"""

import argparse
import os
import numpy as np
from PIL import Image

import torch

import config
from data_loader import get_test_loader, get_val_loader
from model       import build_model
from metrics     import SegmentationMetrics
from utils       import load_checkpoint, find_best_checkpoint, autocast_ctx
from visualize   import colorize_mask, save_comparison


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, save_masks: bool = True, out_dir: str = config.VIZ_DIR):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, "predicted_masks")
    os.makedirs(pred_dir, exist_ok=True)

    all_preds = []

    for images, filenames in loader:
        images = images.to(config.DEVICE, non_blocking=True)

        with autocast_ctx():
            logits = model(images)

        preds = logits.argmax(dim=1)  # [B, H, W]

        for pred, fname in zip(preds, filenames):
            pred_np = pred.cpu().numpy().astype(np.uint8)
            all_preds.append((pred_np, fname))

            if save_masks:
                mask_img = Image.fromarray(pred_np)
                mask_img.save(os.path.join(pred_dir, fname))

                # Also save colourised version
                colour = colorize_mask(pred_np)
                colour.save(os.path.join(pred_dir, "colour_" + fname))

    print(f"[Test] Saved {len(all_preds)} predictions to {pred_dir}")
    return all_preds


# ─── Evaluate on val set ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_val(model):
    """Run full evaluation on the validation set and print metrics."""
    loader  = get_val_loader()
    metrics = SegmentationMetrics()

    for images, masks, _ in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        masks  = masks.to(config.DEVICE,  non_blocking=True)

        with autocast_ctx():
            logits = model(images)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    results = metrics.compute()
    metrics.print_results(results)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model()

    ckpt_path = args.checkpoint or find_best_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Train the model first with: python train.py"
        )

    load_checkpoint(model, ckpt_path)
    model.eval()

    # ── Evaluate on validation set ────────────────────────────────────────────
    print("\n[Test] Evaluating on validation set …")
    results = evaluate_val(model)

    # ── Inference on test images ──────────────────────────────────────────────
    if os.path.isdir(config.TEST_IMG):
        print("\n[Test] Running inference on test images …")
        test_loader = get_test_loader()
        run_inference(model, test_loader, save_masks=args.save_masks)
    else:
        print(f"[Test] Test image directory not found: {config.TEST_IMG} — skipping.")

    print("\n✓ Testing complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test / evaluate segmentation model")
    parser.add_argument("--checkpoint",  type=str,  default=None,
                        help="Path to model checkpoint (auto-detects best if omitted)")
    parser.add_argument("--save_masks",  action="store_true", default=True,
                        help="Save predicted mask images")
    args = parser.parse_args()
    main(args)

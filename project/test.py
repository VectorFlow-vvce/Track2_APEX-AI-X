"""
test.py - Run inference on test images and save predicted masks.

Run:
    py test.py
    py test.py --checkpoint outputs/checkpoints/best_model.pth
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


@torch.no_grad()
def run_inference(model, loader, save_masks=True, out_dir=config.VIZ_DIR):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, "predicted_masks")
    os.makedirs(pred_dir, exist_ok=True)

    all_preds = []
    for images, filenames in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        with autocast_ctx():
            logits = model(images)
        preds = logits.argmax(dim=1)
        for pred, fname in zip(preds, filenames):
            pred_np = pred.cpu().numpy().astype(np.uint8)
            all_preds.append((pred_np, fname))
            if save_masks:
                mask_img = Image.fromarray(pred_np)
                mask_img.save(os.path.join(pred_dir, fname))
                colour = colorize_mask(pred_np)
                colour.save(os.path.join(pred_dir, "colour_" + fname))

    print(f"[Test] Saved {len(all_preds)} predictions to {pred_dir}")
    return all_preds


@torch.no_grad()
def evaluate_val(model):
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


def main(args):
    model = build_model()

    ckpt_path = args.checkpoint or find_best_checkpoint()
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"[Test] Checkpoint not found: {ckpt_path}")
        print(f"[Test] Train the model first: py train.py")
        return

    print(f"\n[Test] Using checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path)
    model.eval()

    print("\n[Test] Evaluating on validation set ...")
    results = evaluate_val(model)

    miou      = results["miou"]
    pixel_acc = results["pixel_acc"]
    map50     = results["map50"]

    print("\n" + "=" * 50)
    print(f"  FINAL RESULTS")
    print("=" * 50)
    print(f"  mIoU      : {miou:.4f}")
    print(f"  Pixel Acc : {pixel_acc:.4f}")
    print(f"  mAP50     : {map50:.4f}")
    print("=" * 50)

    if os.path.isdir(config.TEST_IMG):
        print(f"\n[Test] Running inference on test images ...")
        test_loader = get_test_loader()
        run_inference(model, test_loader, save_masks=args.save_masks)
    else:
        print(f"[Test] Test image directory not found: {config.TEST_IMG} - skipping.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test / evaluate segmentation model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-detects best if omitted)")
    parser.add_argument("--save_masks", action="store_true", default=True,
                        help="Save predicted mask images")
    args = parser.parse_args()
    main(args)

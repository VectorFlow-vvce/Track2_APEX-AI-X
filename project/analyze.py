"""
analyze.py - Failure analysis and self-improvement suggestions.

Evaluates the model on the validation set, identifies weak classes,
prints text insights, and suggests targeted augmentation improvements.

Run:
    python analyze.py
    python analyze.py --checkpoint outputs/checkpoints/best_model.pth
    python analyze.py --retrain          # analyze then immediately retrain
"""

import argparse
import os
import numpy as np
import torch

import config
from data_loader import get_val_loader
from model       import build_model, CombinedLoss
from metrics     import SegmentationMetrics
from utils       import load_checkpoint, find_best_checkpoint, autocast_ctx
from visualize   import save_comparison


# ─── Thresholds ───────────────────────────────────────────────────────────────

WEAK_CLASS_THRESHOLD = 0.40   # IoU below this → "weak"
POOR_CLASS_THRESHOLD = 0.20   # IoU below this → "poor"


# ─── Augmentation suggestion map ─────────────────────────────────────────────

AUGMENTATION_SUGGESTIONS = {
    "sand":       [
        "Increase brightness/contrast jitter — sand appearance varies widely with sun angle.",
        "Add random hue shift to simulate wet vs dry sand.",
        "Use CutMix with rock/gravel patches to help boundary discrimination.",
    ],
    "rock":       [
        "Add random rotation (±15°) — rocks appear at arbitrary orientations.",
        "Increase Gaussian blur probability to simulate dust-covered rocks.",
        "Try elastic deformation to simulate irregular rock surfaces.",
    ],
    "gravel":     [
        "Add fine-grained texture noise (Gaussian noise) to simulate gravel texture.",
        "Increase random crop scale range to capture gravel at multiple scales.",
    ],
    "vegetation": [
        "Add random green-channel boost to help distinguish sparse desert vegetation.",
        "Use RandomShadow to simulate shadows cast by vegetation.",
    ],
    "sky":        [
        "Add random cloud/haze overlay augmentation.",
        "Increase saturation jitter — sky colour varies with time of day.",
    ],
    "obstacle":   [
        "Apply class-frequency-weighted loss — obstacles are likely rare.",
        "Use copy-paste augmentation to synthetically increase obstacle frequency.",
        "Add random occlusion patches to improve partial-obstacle detection.",
    ],
    "trail":      [
        "Add perspective transform to simulate different camera angles on trails.",
        "Increase horizontal flip probability — trails are symmetric.",
    ],
    "background": [
        "Background is often over-represented; consider down-weighting in loss.",
    ],
}

GENERIC_SUGGESTIONS = [
    "Consider using class-frequency-weighted CrossEntropy to handle class imbalance.",
    "Try test-time augmentation (TTA): average predictions over flipped/scaled inputs.",
    "Increase image resolution to 640×640 if GPU memory allows.",
    "Use a larger SegFormer backbone (b3 or b4) for more capacity.",
    "Add MixUp or CutMix augmentation for better generalisation.",
]


# ─── Analysis ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_model(model, loader) -> dict:
    """Full evaluation + per-class analysis."""
    metrics = SegmentationMetrics()

    # Also collect worst-performing samples
    sample_ious = []

    for images, masks, filenames in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        masks  = masks.to(config.DEVICE,  non_blocking=True)

        with autocast_ctx():
            logits = model(images)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        # Per-sample mIoU for worst-sample identification
        for img, pred, gt, fname in zip(images, preds, masks, filenames):
            m = SegmentationMetrics()
            m.update(pred.unsqueeze(0), gt.unsqueeze(0))
            r = m.compute()
            sample_ious.append({
                "filename": fname,
                "miou":     r["miou"],
                "image":    img.cpu(),
                "pred":     pred.cpu(),
                "gt":       gt.cpu(),
            })

    results = metrics.compute()
    results["sample_ious"] = sorted(sample_ious, key=lambda x: x["miou"])
    return results


def print_analysis(results: dict):
    """Print a structured failure analysis report."""
    iou_per_class = results["iou_per_class"]
    class_names   = results["class_names"]
    miou          = results["miou"]

    print("\n" + "=" * 60)
    print("  FAILURE ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n  Overall mIoU : {miou:.4f}")
    print(f"  Pixel Acc    : {results['pixel_acc']:.4f}")

    # ── Per-class breakdown ────────────────────────────────────────────────
    print("\n  Per-Class IoU:")
    print(f"  {'Class':<20} {'IoU':>8}  Status")
    print("  " + "-" * 40)

    weak_classes = []
    poor_classes = []

    for name, iou in zip(class_names, iou_per_class):
        if np.isnan(iou):
            status = "N/A (absent)"
        elif iou < POOR_CLASS_THRESHOLD:
            status = "⚠ POOR"
            poor_classes.append(name)
        elif iou < WEAK_CLASS_THRESHOLD:
            status = "△ WEAK"
            weak_classes.append(name)
        else:
            status = "✓ OK"
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A "
        print(f"  {name:<20} {iou_str:>8}  {status}")

    # ── Worst samples ─────────────────────────────────────────────────────
    worst = results["sample_ious"][:5]
    if worst:
        print("\n  Worst-performing samples:")
        for s in worst:
            print(f"    {s['filename']:<40} mIoU={s['miou']:.4f}")

    # ── Insights ──────────────────────────────────────────────────────────
    print("\n  Root-cause insights:")
    all_problem_classes = poor_classes + weak_classes

    if not all_problem_classes:
        print("  → All classes are performing well (IoU ≥ 0.40). Consider fine-tuning.")
    else:
        for cls in all_problem_classes:
            label = "POOR" if cls in poor_classes else "WEAK"
            print(f"\n  [{label}] '{cls}':")
            suggestions = AUGMENTATION_SUGGESTIONS.get(cls, [])
            if suggestions:
                for s in suggestions:
                    print(f"    • {s}")
            else:
                print(f"    • Collect more labelled examples of '{cls}'.")
                print(f"    • Check for label noise in '{cls}' annotations.")

    # ── Generic suggestions ───────────────────────────────────────────────
    print("\n  General improvement suggestions:")
    for s in GENERIC_SUGGESTIONS:
        print(f"    • {s}")

    print("\n" + "=" * 60)
    return weak_classes, poor_classes


def save_worst_samples(results: dict, n: int = 5, out_dir: str = None):
    """Save side-by-side comparisons for the worst-performing samples."""
    if out_dir is None:
        out_dir = os.path.join(config.VIZ_DIR, "worst_samples")
    os.makedirs(out_dir, exist_ok=True)

    worst = results["sample_ious"][:n]
    for i, s in enumerate(worst):
        fname = f"worst_{i+1:02d}_{s['filename']}"
        title = f"{s['filename']} | mIoU={s['miou']:.4f}"
        save_comparison(s["image"], s["pred"], s["gt"], fname, out_dir, title)
        print(f"  Saved worst sample: {os.path.join(out_dir, fname)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    model = build_model()

    ckpt_path = args.checkpoint or find_best_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Train first: python train.py")

    load_checkpoint(model, ckpt_path)
    model.eval()

    loader  = get_val_loader()
    results = analyze_model(model, loader)
    weak, poor = print_analysis(results)

    save_worst_samples(results)

    # ── Self-improvement loop ─────────────────────────────────────────────────
    if args.retrain:
        print("\n[Analyze] Triggering retraining with current config …")
        print("  (Modify config.py augmentation parameters based on suggestions above,")
        print("   then retraining will apply the updated settings.)\n")
        import subprocess, sys
        subprocess.run([sys.executable, "train.py"], check=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model failures and suggest improvements")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--retrain",    action="store_true",
                        help="Retrain after analysis")
    args = parser.parse_args()
    main(args)

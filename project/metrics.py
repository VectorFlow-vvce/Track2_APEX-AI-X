"""
metrics.py - IoU metrics for semantic segmentation.
Implements per-class IoU, mean IoU (mIoU), and mAP50.
"""

import numpy as np
import torch

import config


class SegmentationMetrics:
    def __init__(self, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]
        preds   = np.clip(preds,   0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)
        indices = self.num_classes * targets + preds
        counts  = np.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.conf_matrix.astype(np.float64)
        tp  = np.diag(cm)
        fp  = cm.sum(axis=0) - tp
        fn  = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        iou   = np.where(denom > 0, tp / denom, np.nan)

        valid_ious = [v for v in iou if not np.isnan(v)]
        miou       = float(np.nanmean(iou))
        pixel_acc  = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
        map50      = sum(1 for v in valid_ious if v >= 0.5) / len(valid_ious) if valid_ious else 0.0

        return {
            "iou_per_class": iou,
            "miou":          miou,
            "pixel_acc":     pixel_acc,
            "map50":         map50,
            "class_names":   config.CLASS_NAMES,
        }

    def print_results(self, results=None):
        if results is None:
            results = self.compute()

        print("\n" + "=" * 50)
        print(f"  mIoU      : {results['miou']:.4f}")
        print(f"  Pixel Acc : {results['pixel_acc']:.4f}")
        print(f"  mAP50     : {results['map50']:.4f}")
        print("=" * 50)
        print(f"  {'Class':<20} {'IoU':>8}  {'>=0.5?':>6}")
        print("-" * 50)
        for name, iou in zip(results["class_names"], results["iou_per_class"]):
            if np.isnan(iou):
                print(f"  {name:<20} {'N/A':>8}  {'  -':>6}")
            else:
                flag = "  Y" if iou >= 0.5 else "  N"
                print(f"  {name:<20} {iou:>8.4f}  {flag:>6}")
        print("=" * 50 + "\n")


def compute_batch_iou(logits, targets, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX):
    preds = logits.argmax(dim=1)
    m = SegmentationMetrics(num_classes, ignore_index)
    m.update(preds, targets)
    return m.compute()["miou"]

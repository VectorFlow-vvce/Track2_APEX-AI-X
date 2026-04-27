"""
metrics.py - IoU metrics for semantic segmentation.
Implements per-class IoU and mean IoU (mIoU).
"""

import numpy as np
import torch

import config


class SegmentationMetrics:
    """
    Accumulates confusion matrix over batches, then computes IoU.

    Usage:
        metrics = SegmentationMetrics()
        for batch in loader:
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self,
                 num_classes:  int = config.NUM_CLASSES,
                 ignore_index: int = config.IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    # ── Accumulation ──────────────────────────────────────────────────────────

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : [B, H, W] int64 predicted class IDs
            targets : [B, H, W] int64 ground-truth class IDs
        """
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Remove ignored pixels
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Clip to valid range (safety)
        preds   = np.clip(preds,   0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)

        # Accumulate confusion matrix
        indices = self.num_classes * targets + preds
        counts  = np.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += counts.reshape(self.num_classes, self.num_classes)

    # ── Computation ───────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Returns a dict with:
            iou_per_class : np.ndarray [num_classes]  (NaN for absent classes)
            miou          : float
            pixel_acc     : float
            class_names   : list[str]
        """
        cm = self.conf_matrix.astype(np.float64)

        # IoU = TP / (TP + FP + FN)
        tp  = np.diag(cm)
        fp  = cm.sum(axis=0) - tp
        fn  = cm.sum(axis=1) - tp

        denom = tp + fp + fn
        iou   = np.where(denom > 0, tp / denom, np.nan)

        miou      = float(np.nanmean(iou))
        pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

        return {
            "iou_per_class": iou,
            "miou":          miou,
            "pixel_acc":     pixel_acc,
            "class_names":   config.CLASS_NAMES,
        }

    # ── Pretty print ──────────────────────────────────────────────────────────

    def print_results(self, results: dict | None = None):
        if results is None:
            results = self.compute()

        print("\n" + "=" * 50)
        print(f"  mIoU      : {results['miou']:.4f}")
        print(f"  Pixel Acc : {results['pixel_acc']:.4f}")
        print("-" * 50)
        print(f"  {'Class':<20} {'IoU':>8}")
        print("-" * 50)
        for name, iou in zip(results["class_names"], results["iou_per_class"]):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A "
            print(f"  {name:<20} {iou_str:>8}")
        print("=" * 50 + "\n")


# ─── Convenience function ─────────────────────────────────────────────────────

def compute_batch_iou(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
    ignore_index: int = config.IGNORE_INDEX,
) -> float:
    """Quick mIoU for a single batch (no accumulation)."""
    preds = logits.argmax(dim=1)
    m = SegmentationMetrics(num_classes, ignore_index)
    m.update(preds, targets)
    return m.compute()["miou"]

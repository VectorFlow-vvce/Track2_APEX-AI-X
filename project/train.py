"""
train.py - Main training loop.

Run:
    py train.py
    py train.py --resume outputs/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import torch

import config
from data_loader import get_train_loader, get_val_loader
from model       import build_model, CombinedLoss
from metrics     import SegmentationMetrics
from utils       import (
    set_seed, save_checkpoint, load_checkpoint, build_scheduler,
    TrainingLogger, get_scaler, autocast_ctx,
)


# ─── Debug / preflight check ──────────────────────────────────────────────────

def preflight_check(train_loader, val_loader):
    print("\n" + "─" * 60)
    print("  PRE-FLIGHT CHECK")
    print("─" * 60)
    print(f"  Training   batches : {len(train_loader)}")
    print(f"  Training   samples : {len(train_loader.dataset)}")
    print(f"  Validation batches : {len(val_loader)}")
    print(f"  Validation samples : {len(val_loader.dataset)}")

    try:
        images, masks, fnames = next(iter(train_loader))
        print(f"\n  Sample batch (train):")
        print(f"    image tensor : {tuple(images.shape)}  dtype={images.dtype}")
        print(f"    mask  tensor : {tuple(masks.shape)}   dtype={masks.dtype}")
        print(f"    mask  values : min={int(masks.min())}  max={int(masks.max())}  "
              f"unique={masks.unique().tolist()}")
        print(f"    first file   : {fnames[0]}")

        if int(masks.max()) >= config.NUM_CLASSES and int(masks.max()) != config.IGNORE_INDEX:
            print(
                f"\n  ⚠ WARNING: mask contains class ID {int(masks.max())} but "
                f"NUM_CLASSES={config.NUM_CLASSES}."
            )
    except Exception as e:
        print(f"\n  ✗ Could not load a sample batch: {e}")
        sys.exit(1)

    print("─" * 60 + "\n")


# ─── One training epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, (images, masks, _) in enumerate(loader):
        images = images.to(config.DEVICE, non_blocking=True)
        masks  = masks.to(config.DEVICE,  non_blocking=True)

        optimizer.zero_grad()

        with autocast_ctx():
            logits = model(images)
            loss   = criterion(logits, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        interval = min(config.LOG_INTERVAL, max(1, n_batches // 4))
        if (i + 1) % interval == 0 or (i + 1) == n_batches:
            pct = 100 * (i + 1) / n_batches
            print(f"  [{pct:5.1f}%  {i+1:>4}/{n_batches}]  loss={loss.item():.4f}")

    return total_loss / n_batches


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    metrics    = SegmentationMetrics()

    for images, masks, _ in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        masks  = masks.to(config.DEVICE,  non_blocking=True)

        with autocast_ctx():
            logits = model(images)
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    results = metrics.compute()
    return total_loss / len(loader), results["miou"], results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(config.SEED)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Off-Road Semantic Segmentation — Training")
    print(f"{'='*60}")
    print(f"  Device      : {config.DEVICE}")
    print(f"  Model       : {config.MODEL_NAME} ({config.BACKBONE})")
    print(f"  Classes     : {config.NUM_CLASSES}  →  {config.CLASS_NAMES}")
    print(f"  Image size  : {config.IMAGE_SIZE}")
    print(f"  Epochs      : {config.NUM_EPOCHS}")
    print(f"  Batch size  : {config.BATCH_SIZE}")
    print(f"  LR          : {config.LEARNING_RATE}")
    print(f"  Scheduler   : {config.LR_SCHEDULER}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = get_train_loader(debug=True)
    val_loader   = get_val_loader(debug=True)
    preflight_check(train_loader, val_loader)

    # ── Model / Loss / Optimizer ──────────────────────────────────────────────
    model     = build_model()
    criterion = CombinedLoss(class_weights=getattr(config, 'CLASS_WEIGHTS', None))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer)
    scaler    = get_scaler()
    logger    = TrainingLogger()

    # ── Resume or fresh start ─────────────────────────────────────────────────
    start_epoch = 1
    best_miou   = 0.0
    best_epoch  = 0
    best_ckpt   = os.path.join(config.CKPT_DIR, "best_model.pth")

    if args.resume and os.path.isfile(args.resume):
        print(f"\n[Training] Resuming from checkpoint: {args.resume}")
        loaded_epoch, loaded_miou = load_checkpoint(model, args.resume, optimizer)
        start_epoch = loaded_epoch + 1
        best_miou   = loaded_miou
        best_epoch  = loaded_epoch
        print(f"[Training] Resuming at epoch {start_epoch}, best mIoU so far: {best_miou:.4f}")
    else:
        print(f"\n[Training] Starting fresh from pretrained SegFormer weights")

    # Early stopping: patience only, no target cap
    patience = 15
    no_improvement_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'─'*60}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        val_loss, val_miou, val_results = validate(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.print_epoch(epoch, train_loss, val_loss, val_miou, current_lr)
        scheduler.step()

        # Save checkpoint every epoch (never overwrite previous)
        epoch_ckpt = os.path.join(config.CKPT_DIR, f"epoch_{epoch}_miou{val_miou:.4f}.pth")
        save_checkpoint(model, optimizer, epoch, val_miou, epoch_ckpt)

        # Save best model when mIoU improves
        if val_miou > best_miou:
            best_miou  = val_miou
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_miou, best_ckpt)
            print(f"  ★★★ NEW BEST mIoU: {best_miou:.4f} ★★★")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"  → No improvement ({no_improvement_count}/{patience})")

        # Patience-based early stopping only
        if no_improvement_count >= patience:
            print(f"\n  🛑 Early stopping: No improvement for {patience} consecutive epochs")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🏁 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best mIoU achieved : {best_miou:.4f}")
    print(f"  Best epoch         : {best_epoch}")
    print(f"  Best checkpoint    : {best_ckpt}")
    print(f"{'='*60}")

    # ── Per-class IoU analysis ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  FINAL PER-CLASS IoU ANALYSIS")
    print(f"{'─'*60}")

    _, _, final_results = validate(model, val_loader, criterion)
    iou_per_class = final_results.get("iou_per_class", [])
    weak_classes  = ["ground_clutter", "logs", "rocks", "dry_bushes"]

    print(f"  {'Class':<15} {'IoU':<8} {'Weight':<8} {'Status'}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10}")

    for i, (class_name, weight) in enumerate(zip(config.CLASS_NAMES, config.CLASS_WEIGHTS)):
        if i < len(iou_per_class):
            iou = iou_per_class[i]
            status = "🎯 WEAK" if class_name in weak_classes else "✓"
            print(f"  {class_name:<15} {iou:<8.4f} {weight:<8} {status}")

    print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)

"""
utils.py - Shared utilities: seeding, checkpointing, logging, LR scheduling.
"""

import os
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn

import config


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, optimizer, epoch: int, miou: float,
                    path: str | None = None):
    if path is None:
        path = os.path.join(config.CKPT_DIR, f"epoch_{epoch:03d}_miou{miou:.4f}.pth")
    torch.save({
        "epoch":      epoch,
        "miou":       miou,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    print(f"[Checkpoint] Saved → {path}")
    return path


def load_checkpoint(model: nn.Module, path: str, optimizer=None, device=config.DEVICE):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    miou  = ckpt.get("miou",  0.0)
    print(f"[Checkpoint] Loaded epoch {epoch}, mIoU={miou:.4f} from {path}")
    return epoch, miou


def find_best_checkpoint(ckpt_dir: str = config.CKPT_DIR) -> str | None:
    """Return path of checkpoint with highest mIoU in filename."""
    paths = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not paths:
        return None
    best = max(paths, key=lambda f: float(f.split("miou")[-1].replace(".pth", "")))
    return os.path.join(ckpt_dir, best)


# ─── LR Scheduler ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, scheduler_name: str = config.LR_SCHEDULER,
                    num_epochs: int = config.NUM_EPOCHS,
                    warmup_epochs: int = config.WARMUP_EPOCHS):
    """
    Returns a (scheduler, warmup_scheduler) tuple.
    warmup_scheduler is None if warmup_epochs == 0.
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, PolynomialLR, StepLR, LinearLR, SequentialLR
    )

    main_epochs = num_epochs - warmup_epochs

    if scheduler_name == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-7)
    elif scheduler_name == "poly":
        main_sched = PolynomialLR(optimizer, total_iters=main_epochs, power=0.9)
    elif scheduler_name == "step":
        main_sched = StepLR(optimizer, step_size=max(1, main_epochs // 3), gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched],
                                 milestones=[warmup_epochs])
    else:
        scheduler = main_sched

    return scheduler


# ─── Logger ───────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Logs metrics to a JSON-lines file and prints to stdout."""

    def __init__(self, log_path: str | None = None):
        if log_path is None:
            ts       = time.strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(config.LOG_DIR, f"train_{ts}.jsonl")
        self.log_path = log_path
        self.history  = []
        print(f"[Logger] Logging to {log_path}")

    def log(self, record: dict):
        self.history.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_epoch(self, epoch: int, train_loss: float, val_loss: float,
                    val_miou: float, lr: float):
        print(
            f"  Epoch [{epoch:03d}/{config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | LR: {lr:.2e}"
        )
        self.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_miou":   val_miou,
            "lr":         lr,
        })


# ─── AMP helper ───────────────────────────────────────────────────────────────

def get_scaler():
    """Return a GradScaler if CUDA is available, else a no-op."""
    if config.DEVICE == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def autocast_ctx():
    """Return the appropriate autocast context manager."""
    if config.DEVICE == "cuda":
        return torch.cuda.amp.autocast()
    import contextlib
    return contextlib.nullcontext()


# ─── Misc ─────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised [3,H,W] float tensor back to a uint8 HWC numpy array.
    Reverses ImageNet normalisation.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)

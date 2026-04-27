"""
model.py - Model factory supporting SegFormer and DeepLabV3+.

SegFormer  → uses HuggingFace transformers (pip install transformers)
DeepLabV3+ → uses segmentation_models_pytorch (pip install segmentation-models-pytorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ─── SegFormer wrapper ────────────────────────────────────────────────────────

class SegFormerModel(nn.Module):
    """
    Thin wrapper around HuggingFace SegformerForSemanticSegmentation.
    Upsamples logits to full resolution before returning.
    """

    VARIANT_MAP = {
        "b0": "nvidia/mit-b0",
        "b1": "nvidia/mit-b1",
        "b2": "nvidia/mit-b2",
        "b3": "nvidia/mit-b3",
        "b4": "nvidia/mit-b4",
        "b5": "nvidia/mit-b5",
    }

    def __init__(self, num_classes: int, backbone: str = "b2", pretrained: bool = True):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError(
                "transformers is required for SegFormer.\n"
                "Install with: pip install transformers"
            )

        hf_name = self.VARIANT_MAP.get(backbone, "nvidia/mit-b2")

        if pretrained:
            print(f"[Model] Loading pretrained SegFormer {backbone} from HuggingFace …")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                hf_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(hf_name, num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, x):
        h, w    = x.shape[-2], x.shape[-1]
        outputs = self.model(pixel_values=x)
        logits  = outputs.logits                          # [B, C, H/4, W/4]
        logits  = F.interpolate(logits, size=(h, w),
                                mode="bilinear", align_corners=False)
        return logits


# ─── DeepLabV3+ wrapper ───────────────────────────────────────────────────────

class DeepLabV3PlusModel(nn.Module):
    """
    Wrapper around segmentation_models_pytorch DeepLabV3+.
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required for DeepLabV3+.\n"
                "Install with: pip install segmentation-models-pytorch"
            )

        encoder_weights = "imagenet" if pretrained else None
        print(f"[Model] Building DeepLabV3+ with {backbone} backbone …")
        self.model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(
    model_name: str  = config.MODEL_NAME,
    backbone:   str  = config.BACKBONE,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
) -> nn.Module:
    """
    Build and return the segmentation model.

    Args:
        model_name  : "segformer" or "deeplabv3plus"
        backbone    : backbone variant (e.g. "b2" for SegFormer, "resnet50" for DeepLab)
        num_classes : number of segmentation classes
        pretrained  : whether to load ImageNet / HuggingFace pretrained weights

    Returns:
        nn.Module on config.DEVICE
    """
    name = model_name.lower().replace("-", "").replace("_", "")

    if name == "segformer":
        model = SegFormerModel(num_classes, backbone, pretrained)
    elif name in ("deeplabv3plus", "deeplabv3"):
        model = DeepLabV3PlusModel(num_classes, backbone, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'segformer' or 'deeplabv3plus'.")

    model = model.to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} | trainable params: {n_params:,} | device: {config.DEVICE}")
    return model


# ─── Loss ─────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.smooth        = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                   # [B, C, H, W]
        B, C, H, W = probs.shape

        # Build valid pixel mask
        valid = (targets != self.ignore_index)             # [B, H, W]
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0

        # One-hot encode targets
        one_hot = F.one_hot(targets_clamped, C).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        valid_4d = valid.unsqueeze(1).float()

        probs   = probs   * valid_4d
        one_hot = one_hot * valid_4d

        dims  = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of CrossEntropy and Dice losses with class weights for imbalanced data."""

    def __init__(self,
                 num_classes:  int   = config.NUM_CLASSES,
                 ignore_index: int   = config.IGNORE_INDEX,
                 ce_weight:    float = config.CE_WEIGHT,
                 dice_weight:  float = config.DICE_WEIGHT,
                 class_weights: list = None):
        super().__init__()
        
        # Store class weights to move to device later
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            print(f"[Loss] Using class weights: {class_weights}")
            
        self.dice = DiceLoss(num_classes, ignore_index)
        self.ce_w = ce_weight
        self.di_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Create CE loss with weights on the same device as logits
        if self.class_weights is not None:
            # Ensure weights are on the same device as logits
            weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32).to(logits.device)
            ce_loss = nn.functional.cross_entropy(logits, targets, 
                                                weight=weights_tensor, 
                                                ignore_index=self.ignore_index)
        else:
            ce_loss = nn.functional.cross_entropy(logits, targets, ignore_index=self.ignore_index)
            
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.di_w * dice_loss

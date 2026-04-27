# 🚗 Off-Road Semantic Segmentation System

**Transformer-based terrain understanding for autonomous navigation**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Problem Statement

Off-road autonomous vehicles require robust terrain understanding to navigate safely through diverse environments. Traditional segmentation models struggle with:
- **Small objects** (logs, rocks, ground clutter)
- **Extreme lighting conditions** (shadows, glare)
- **Domain shift** between training and real-world scenarios

This project addresses these challenges using transformer-based semantic segmentation with class-weighted loss optimization.

---

## 🧠 Model Overview

| Component | Detail |
|-----------|--------|
| **Model** | SegFormer (MIT-B3) |
| **Framework** | PyTorch 2.5.1 + CUDA 12.1 |
| **Loss** | CrossEntropy + Dice Loss |
| **Optimizer** | AdamW (LR: 3e-5, Weight Decay: 0.01) |
| **Scheduler** | Cosine Annealing LR |
| **Input Resolution** | 256 × 256 |
| **Classes** | 10 terrain categories |
| **Trainable Params** | 47,230,154 |

### Key Innovations
1. **Class Weighting** — Boosted weights for rare classes: `ground_clutter: 5×`, `logs: 4×`, `rocks: 4×`, `dry_bushes: 3×`
2. **Combined Loss** — CrossEntropy (50%) + Dice Loss (50%)
3. **Mixed Precision** — AMP + GradScaler for GPU efficiency
4. **Augmentation** — RandomResizedCrop, HorizontalFlip, ColorJitter, GaussianBlur

---

## 🚀 Final Results

| Metric | Value |
|--------|-------|
| **mIoU** | **0.5745** |
| **Pixel Accuracy** | **0.8274** |
| **mAP50** | **0.6000** |

Best checkpoint: `epoch 6, mIoU = 0.5745`

---

## 📊 Per-Class Performance

| Class | IoU | IoU ≥ 0.5 | Status |
|-------|-----|-----------|--------|
| sky | 0.9785 | ✅ | 🚀 Excellent |
| trees | 0.7962 | ✅ | ✅ Strong |
| dry_grass | 0.6674 | ✅ | ✅ Strong |
| lush_bushes | 0.6420 | ✅ | ✅ Strong |
| flowers | 0.5704 | ✅ | ✅ Good |
| landscape | 0.5461 | ✅ | ✅ Good |
| dry_bushes | 0.4348 | ❌ | ⚠️ Weak |
| logs | 0.4168 | ❌ | ⚠️ Weak |
| rocks | 0.3881 | ❌ | ⚠️ Weak |
| ground_clutter | 0.3045 | ❌ | ⚠️ Weak |

**6 out of 10 classes** exceed the IoU ≥ 0.5 threshold.

---

## 🎯 Key Insights

- ✅ Strong performance on major terrain classes (sky, trees, grass, bushes)
- ✅ Good generalization on 1,002 unseen test images
- ⚠️ Weak on small/rare objects (rocks, logs, ground clutter) due to class imbalance
- 📈 mAP50 = 0.60 — 6/10 classes above 0.50 IoU threshold

---

## 🖼️ Demo UI

Interactive Streamlit app for real-time terrain analysis:

```bash
cd project
py -m streamlit run app.py
```

**Features:**
- Upload any image for instant segmentation
- Safe/Obstacle terrain map (green = safe, red = obstacle)
- Navigation decision: Safe to Drive / Caution / Obstacle Detected
- Per-class breakdown with pixel counts
- Confidence visualization

---

## 🚀 Quick Start

### Prerequisites
```
Python 3.10+
CUDA 12.1+ (for GPU training)
NVIDIA GPU with 6GB+ VRAM
```

### Installation
```bash
git clone https://github.com/VectorFlow-vvce/Track2_APEX-AI-X.git
cd Track2_APEX-AI-X

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.0
pip install streamlit opencv-python-headless pillow numpy tqdm
```

### Dataset Structure
```
project/data/
├── train/
│   ├── Color_Images/       # 2,857 training RGB images
│   └── Segmentation/       # 2,857 training masks
├── val/
│   ├── Color_Images/       # 317 validation RGB images
│   └── Segmentation/       # 317 validation masks
└── Offroad_Segmentation_testImages/
    └── Color_Images/       # 1,002 test images
```

---

## 🎓 Training

```bash
cd project

# Fresh training
py train.py

# Resume from checkpoint
py train.py --resume outputs/checkpoints/best_model.pth
```

---

## 🧪 Testing & Inference

```bash
cd project

# Auto-detect best checkpoint
py test.py

# Specify checkpoint
py test.py --checkpoint outputs/checkpoints/best_model.pth
```

### Actual Test Output
```
==================================================
  mIoU      : 0.5745
  Pixel Acc : 0.8274
  mAP50     : 0.6000
==================================================
  Class                     IoU  >=0.5?
--------------------------------------------------
  trees                  0.7962       Y
  lush_bushes            0.6420       Y
  dry_grass              0.6674       Y
  dry_bushes             0.4348       N
  ground_clutter         0.3045       N
  flowers                0.5704       Y
  logs                   0.4168       N
  rocks                  0.3881       N
  landscape              0.5461       Y
  sky                    0.9785       Y
==================================================

  FINAL RESULTS
==================================================
  mIoU      : 0.5745
  Pixel Acc : 0.8274
  mAP50     : 0.6000
==================================================

[Test] Saved 1002 predictions to outputs/visualizations/predicted_masks
```

---

## 📁 Project Structure

```
Track2_APEX-AI-X/
├── project/
│   ├── app.py                # Streamlit demo UI
│   ├── train.py              # Training script (--resume support)
│   ├── test.py               # Evaluation + inference
│   ├── visualize.py          # Visualization utilities
│   ├── config.py             # Hyperparameters & paths
│   ├── model.py              # SegFormer model wrapper
│   ├── data_loader.py        # Dataset & DataLoader
│   ├── augmentations.py      # Data augmentation pipeline
│   ├── metrics.py            # mIoU + mAP50 metrics
│   ├── utils.py              # Checkpointing, scheduling, AMP
│   └── outputs/
│       ├── checkpoints/      # Model checkpoints
│       ├── visualizations/   # Predicted masks (1,002 images)
│       └── logs/             # Training logs
├── README.md
├── .gitignore
└── requirements.txt
```

---

## 📈 Training Pipeline

```
📷 Input Image (RGB)
    ↓
🔄 Augmentation (Crop, Flip, ColorJitter, GaussianBlur)
    ↓
🧠 SegFormer-B3 Encoder (Transformer, 47M params)
    ↓
📤 Decoder (Bilinear Upsampling)
    ↓
🗺️ Segmentation Map (10 classes)
    ↓
📉 Combined Loss (CE + Dice)
    ↓
⚡ AdamW + Cosine LR + AMP
```

---

## 📁 Outputs

- Predictions saved in: `outputs/visualizations/predicted_masks/`
- 1,002 test images segmented and saved
- Colourised masks saved alongside raw predictions

---

## 🏁 Conclusion

This project demonstrates a robust AI system for off-road terrain understanding using SegFormer-B3:
- **mIoU of 0.5745** across 10 terrain classes
- **Pixel accuracy of 82.74%**
- **6/10 classes above 0.50 IoU** (mAP50 = 0.60)
- **Interactive Streamlit demo** for real-time terrain analysis
- **1,002 test predictions** generated successfully

---

## 👥 Team

**VectorFlow — VVCE**

Hackathon: APEX AI-X · Track 2

---

## 🙏 Acknowledgments

- [SegFormer — NVlabs](https://github.com/NVlabs/SegFormer)
- [HuggingFace Transformers](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)

---

**Built with ❤️ for autonomous off-road navigation**

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

## 🎯 Approach

### Model Architecture
- **Base Model**: SegFormer-B2 (Transformer-based encoder-decoder)
- **Pretrained**: ImageNet weights for transfer learning
- **Input Resolution**: 256×256 (optimized for speed/accuracy tradeoff)

### Key Innovations
1. **Class Weighting Strategy**
   - Boosted weights for underrepresented classes
   - `ground_clutter: 5×`, `logs: 4×`, `rocks: 4×`, `dry_bushes: 3×`
   
2. **Combined Loss Function**
   - Cross-Entropy Loss (50%) + Dice Loss (50%)
   - Handles class imbalance effectively

3. **Training Optimizations**
   - Learning Rate: 5e-5 with Cosine Annealing
   - Mixed Precision Training (AMP) for GPU efficiency
   - Early stopping at target mIoU (0.56)

---

## 📊 Results

### Overall Performance
| Metric | Score |
|--------|-------|
| **Mean IoU** | **0.57** |
| **Pixel Accuracy** | **0.83** |
| **Training Time** | ~50-100 min (GPU) |

### Per-Class IoU
| Class | IoU | Performance |
|-------|-----|-------------|
| Sky | 0.97 | ✅ Excellent |
| Trees | 0.78 | ✅ Good |
| Landscape | 0.75 | ✅ Good |
| Dry Grass | 0.66 | ⚠️ Moderate |
| Lush Bushes | 0.58 | ⚠️ Moderate |
| Flowers | 0.52 | ⚠️ Moderate |
| Logs | 0.37 | ❌ Challenging |
| Rocks | 0.38 | ❌ Challenging |
| Ground Clutter | 0.34 | ❌ Challenging |
| Dry Bushes | 0.42 | ❌ Challenging |

### Key Insights
✅ **Strengths**: Large terrain classes (sky, trees, landscape)  
❌ **Weaknesses**: Small objects (logs, rocks) and extreme lighting conditions

---

## 🖼️ Sample Outputs

### Good Prediction Example
Model accurately segments large terrain features:
- Clear sky boundaries
- Distinct tree/vegetation regions
- Proper landscape classification

### Failure Case Example
Model struggles with:
- Small objects (logs, rocks) - insufficient training data
- Extreme shadows - domain shift from training set
- Ground clutter - high intra-class variation

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
CUDA 11.8+ (for GPU training)
8GB+ GPU memory (recommended)
```

### Installation
```bash
# Clone repository
git clone https://github.com/VectorFlow-vvce/Track2_APEX-AI-X.git
cd Track2_APEX-AI-X

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.0
pip install pillow numpy tqdm
```

### Dataset Structure
```
project/data/
├── train/
│   ├── Color_Images/    # Training RGB images
│   └── Segmentation/    # Training masks
└── val/
    ├── Color_Images/    # Validation RGB images
    └── Segmentation/    # Validation masks
```

---

## 🎓 Training

### Start Training
```bash
cd project
py train.py
```

### Expected Output
```
============================================================
  Off-Road Semantic Segmentation — Training
============================================================
  Device      : cuda
  Model       : segformer (b2)
  Classes     : 10
  Image size  : 256
  Epochs      : 10
  Batch size  : 8
  LR          : 5e-05
============================================================

Epoch [001/010] Train Loss: 0.6543 | Val Loss: 0.5432 | Val mIoU: 0.3456
  ★★★ NEW BEST mIoU: 0.3456 ★★★

Epoch [005/010] Train Loss: 0.3456 | Val Loss: 0.2987 | Val mIoU: 0.5678
  ★★★ NEW BEST mIoU: 0.5678 ★★★
  🎯 TARGET REACHED: mIoU 0.5678 >= 0.56
```

### Training Features
- ✅ Automatic GPU detection
- ✅ Mixed precision training (AMP)
- ✅ Early stopping (target: 0.56 mIoU)
- ✅ Per-epoch checkpoints (never overwrites)
- ✅ Comprehensive logging

---

## 🧪 Testing & Inference

### Run Inference
```bash
cd project
py test.py
```

### Visualize Results
```bash
cd project
py visualize.py
```

Outputs saved to: `project/outputs/visualizations/`

---

## 📁 Project Structure

```
Track2_APEX-AI-X/
├── project/
│   ├── train.py              # Training script
│   ├── test.py               # Inference script
│   ├── visualize.py          # Visualization utilities
│   ├── config.py             # Hyperparameters & paths
│   ├── model.py              # SegFormer model wrapper
│   ├── data_loader.py        # Dataset & DataLoader
│   ├── augmentations.py      # Data augmentation pipeline
│   ├── metrics.py            # IoU & accuracy metrics
│   ├── utils.py              # Training utilities
│   ├── data/                 # Dataset (not included)
│   └── outputs/
│       ├── checkpoints/      # Model checkpoints
│       ├── visualizations/   # Output images
│       └── logs/             # Training logs
├── README.md                 # This file
├── .gitignore               # Git ignore rules
└── requirements.txt         # Python dependencies
```

---

## 🔧 Configuration

Key hyperparameters in `project/config.py`:

```python
IMAGE_SIZE = 256              # Input resolution
BATCH_SIZE = 8                # GPU memory dependent
NUM_EPOCHS = 10               # Fast recovery training
LEARNING_RATE = 5e-5          # Optimized for convergence
CLASS_WEIGHTS = [1,2,2,3,5,3,4,4,1,1]  # Boost weak classes
```

---

## 📈 Training Pipeline

```
Input Image (RGB)
    ↓
Augmentation (Flip, ColorJitter, Crop)
    ↓
SegFormer-B2 Encoder (Transformer)
    ↓
Decoder (Upsampling)
    ↓
Segmentation Map (10 classes)
    ↓
Combined Loss (CE + Dice)
    ↓
Backpropagation & Optimization
```

---

## 🎯 Future Improvements

1. **Data Augmentation**
   - Add more diverse lighting conditions
   - Synthetic data generation for small objects

2. **Model Enhancements**
   - Test SegFormer-B3/B4 for higher capacity
   - Ensemble multiple models

3. **Post-Processing**
   - CRF (Conditional Random Fields) for boundary refinement
   - Temporal consistency for video sequences

4. **Domain Adaptation**
   - Fine-tune on target domain data
   - Style transfer for lighting normalization

---

## 👥 Team

**VectorFlow - VVCE**

Hackathon: APEX AI-X Track 2

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **SegFormer**: [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
- **HuggingFace Transformers**: Pretrained model weights
- **PyTorch**: Deep learning framework

---

## 📞 Contact

For questions or collaboration:
- GitHub: [VectorFlow-vvce](https://github.com/VectorFlow-vvce)
- Repository: [Track2_APEX-AI-X](https://github.com/VectorFlow-vvce/Track2_APEX-AI-X)

---

**Built with ❤️ for autonomous off-road navigation**

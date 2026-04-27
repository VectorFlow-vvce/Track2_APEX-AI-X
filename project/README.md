# Off-Road Desert Semantic Segmentation

A production-ready semantic segmentation system for off-road desert environments,
built around **SegFormer** (HuggingFace) with a combined CrossEntropy + Dice loss,
AdamW optimiser, and a cosine LR schedule.

---

## Folder Structure

```
project/
├── config.py          # All hyperparameters and paths
├── data_loader.py     # Dataset + DataLoader factories
├── model.py           # SegFormer / DeepLabV3+ + CombinedLoss
├── augmentations.py   # Paired image+mask augmentations
├── train.py           # Training loop
├── test.py            # Inference + evaluation
├── metrics.py         # IoU / mIoU computation
├── analyze.py         # Failure analysis + improvement suggestions
├── visualize.py       # Side-by-side comparison plots
├── utils.py           # Seeding, checkpointing, logging, schedulers
├── requirements.txt
└── README.md

outputs/
├── checkpoints/       # Saved model weights
├── visualizations/    # Comparison images
└── logs/              # JSONL training logs
```

---

## Dataset Layout

Place your dataset so it matches:

```
C:\Users\kusha\Downloads\Offroad_Segmentation_Training_Dataset\
└── data\
    ├── train\
    │   ├── images\   ← RGB images (.jpg / .png)
    │   └── masks\    ← Single-channel class-ID masks (.png)
    ├── val\
    │   ├── images\
    │   └── masks\
    └── testImages\   ← Images only (no masks)
```

Mask pixel values must be integer class IDs (0, 1, 2 …).  
Edit `CLASS_NAMES` in `config.py` to match your actual classes.

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Configure paths and classes

Open `config.py` and verify:
- `DATA_ROOT` points to your dataset
- `CLASS_NAMES` matches your mask labels
- `NUM_CLASSES` equals `len(CLASS_NAMES)`

---

## Training

```bash
# Default settings (50 epochs, SegFormer-B2)
python train.py

# Custom settings
python train.py --epochs 100 --lr 3e-5 --batch 8

# Resume from checkpoint
python train.py --resume outputs/checkpoints/best_model.pth
```

Training logs are saved to `outputs/logs/train_<timestamp>.jsonl`.  
The best checkpoint (by val mIoU) is saved to `outputs/checkpoints/best_model.pth`.

---

## Testing / Evaluation

```bash
# Auto-detect best checkpoint, evaluate on val, run inference on test images
python test.py

# Specify checkpoint explicitly
python test.py --checkpoint outputs/checkpoints/best_model.pth
```

Predicted masks are saved to `outputs/visualizations/predicted_masks/`.

---

## Failure Analysis

```bash
# Analyse weak classes and print improvement suggestions
python analyze.py

# Analyse then immediately retrain
python analyze.py --retrain
```

The analyser:
1. Evaluates the model on the validation set
2. Identifies classes with IoU < 0.40 (weak) or < 0.20 (poor)
3. Prints targeted augmentation suggestions per class
4. Saves side-by-side comparisons of the 5 worst-performing images

---

## Visualisation

```bash
# Save 8 comparison images (input | ground truth | prediction)
python visualize.py

# Plot training loss + mIoU curves
python visualize.py --log outputs/logs/train_<timestamp>.jsonl
```

---

## Key Configuration Options (`config.py`)

| Parameter        | Default       | Description                          |
|------------------|---------------|--------------------------------------|
| `MODEL_NAME`     | `"segformer"` | `"segformer"` or `"deeplabv3plus"`   |
| `BACKBONE`       | `"b2"`        | SegFormer variant b0–b5              |
| `NUM_CLASSES`    | 8             | Must match your mask labels          |
| `IMAGE_SIZE`     | `(512, 512)`  | Input resolution                     |
| `BATCH_SIZE`     | 4             | Increase if GPU memory allows        |
| `NUM_EPOCHS`     | 50            | Training epochs                      |
| `LEARNING_RATE`  | `6e-5`        | AdamW learning rate                  |
| `LR_SCHEDULER`   | `"cosine"`    | `"cosine"`, `"poly"`, or `"step"`    |

---

## Example Outputs

After training you should see output like:

```
Epoch [050/050] train_loss=0.1823  val_loss=0.2041  val_mIoU=0.6714  lr=1.00e-07

==================================================
  mIoU      : 0.6714
  Pixel Acc : 0.8932
--------------------------------------------------
  Class                    IoU
--------------------------------------------------
  background            0.9102
  sand                  0.7841
  rock                  0.6523
  gravel                0.5910
  vegetation            0.4872
  sky                   0.8801
  obstacle              0.3214
  trail                 0.6201
==================================================
```

---

## Self-Improvement Loop

```
Train → Analyze → Adjust config.py → Retrain
```

1. Run `python train.py`
2. Run `python analyze.py` — read the suggestions
3. Adjust augmentation parameters in `config.py`
4. Run `python train.py --resume outputs/checkpoints/best_model.pth`
5. Repeat until mIoU plateaus

---

## License

MIT — free to use and modify.

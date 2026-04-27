# 🚀 Quick Start - Training Recovery

## ⚡ Fast Track (3 Steps)

### 1. Verify Setup
```bash
cd project
python verify_changes.py
```

### 2. Start Training
```bash
python train.py
```

### 3. Monitor Progress
Watch for:
- ✅ "★★★ NEW BEST mIoU" messages
- ✅ mIoU increasing each epoch
- ✅ Target reached (≥0.56) or early stopping

---

## 📊 What to Expect

### Training Output Example:
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

[Training] Starting fresh from pretrained SegFormer weights
[Training] NO checkpoint resume - clean start for recovery

────────────────────────────────────────────────────────────
  Epoch 1/10
────────────────────────────────────────────────────────────
  [100.0%   357/357]  loss=0.6543
  Epoch [001/010] Train Loss: 0.6543 | Val Loss: 0.5432 | Val mIoU: 0.3456 | LR: 5.00e-05
  ★★★ NEW BEST mIoU: 0.3456 (improvement: +0.3456) ★★★

────────────────────────────────────────────────────────────
  Epoch 2/10
────────────────────────────────────────────────────────────
  [100.0%   357/357]  loss=0.5234
  Epoch [002/010] Train Loss: 0.5234 | Val Loss: 0.4321 | Val mIoU: 0.4567 | LR: 4.95e-05
  ★★★ NEW BEST mIoU: 0.4567 (improvement: +0.1111) ★★★

... (continues for more epochs)

────────────────────────────────────────────────────────────
  Epoch 5/10
────────────────────────────────────────────────────────────
  [100.0%   357/357]  loss=0.3456
  Epoch [005/010] Train Loss: 0.3456 | Val Loss: 0.2987 | Val mIoU: 0.5678 | LR: 4.50e-05
  ★★★ NEW BEST mIoU: 0.5678 (improvement: +0.0234) ★★★

  🎯 TARGET REACHED: mIoU 0.5678 >= 0.56
  Stopping early - goal achieved!

============================================================
  🏁 TRAINING COMPLETE!
============================================================
  Best mIoU achieved : 0.5678
  Best epoch         : 5
  Best checkpoint    : outputs/checkpoints/best_model.pth
============================================================
```

---

## 📁 Output Files

After training, you'll find:

```
project/outputs/
├── checkpoints/
│   ├── epoch_1_miou0.3456.pth
│   ├── epoch_2_miou0.4567.pth
│   ├── epoch_3_miou0.5123.pth
│   ├── epoch_4_miou0.5456.pth
│   ├── epoch_5_miou0.5678.pth
│   └── best_model.pth  ← Use this for inference
├── logs/
│   └── train_20260427_120000.jsonl
└── visualizations/
```

---

## ⏱️ Time Estimates

- **With GPU (CUDA)**: ~5-10 minutes per epoch
- **With CPU**: ~30-60 minutes per epoch
- **Total (10 epochs max)**: 50-100 minutes with GPU

**Early stopping** may finish in 4-6 epochs if target reached!

---

## 🎯 Success Criteria

Training is successful if:
- ✅ mIoU reaches ≥0.56 (target)
- ✅ mIoU improves from initial ~0.20 to 0.55-0.60 range
- ✅ No errors or crashes
- ✅ Checkpoints saved successfully

---

## 🔧 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce `BATCH_SIZE` in `config.py` from 8 to 4 or 2

### Issue: Training too slow
**Solution**: 
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Increase `NUM_WORKERS` in `config.py` to 4 (if you have more CPU cores)

### Issue: mIoU not improving
**Solution**: 
- Check data quality (verify images and masks match)
- Let it train for at least 3-4 epochs before judging
- Review per-class IoU in final output to identify problem classes

### Issue: Import errors
**Solution**: Install missing packages:
```bash
pip install torch torchvision transformers pillow numpy
```

---

## 📈 After Training

### Use the best model for inference:
```bash
python test.py  # or your inference script
```

### Analyze results:
```bash
python analyze.py  # if available
```

### Check per-class performance:
Look at the "FINAL PER-CLASS IoU ANALYSIS" section in training output to see which classes improved most.

---

## 🎓 Key Changes Made

1. ✅ **No checkpoint resume** - Fresh start from pretrained weights
2. ✅ **Optimized hyperparameters** - Fast convergence settings
3. ✅ **Class weights** - Boosted weak classes (ground_clutter, logs, rocks)
4. ✅ **Early stopping** - Stops at target (0.56) or no improvement
5. ✅ **Per-epoch checkpoints** - Never overwrites previous epochs
6. ✅ **GPU optimization** - AMP, GradScaler, pin_memory enabled

---

## 📞 Need Help?

Check the detailed documentation:
- `TRAINING_RECOVERY_SUMMARY.md` - Complete change log
- `config.py` - All hyperparameters
- `train.py` - Training logic

---

**Ready to train? Run `python verify_changes.py` first, then `python train.py`!** 🚀

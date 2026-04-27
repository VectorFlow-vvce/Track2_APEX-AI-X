# 🎮 GPU Setup Instructions for NVIDIA GPU

## 🔍 Problem Detected
Your system has:
- ✓ NVIDIA GPU detected
- ✗ PyTorch **CPU-only** version installed (2.11.0+cpu)

**Result**: Training will run on CPU (very slow) instead of GPU

---

## ✅ Solution: Install PyTorch with CUDA Support

### Option 1: Automatic Installation (Recommended)

**Run this command in your terminal:**

```bash
cd project
install_pytorch_gpu.bat
```

This will:
1. Uninstall CPU-only PyTorch
2. Install PyTorch with CUDA 11.8 support (~2-3 GB download)
3. Verify GPU is working

---

### Option 2: Manual Installation

**Step 1: Uninstall CPU-only PyTorch**
```bash
pip uninstall torch torchvision torchaudio
```

**Step 2: Install PyTorch with CUDA 11.8**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 3: Verify GPU is working**
```bash
cd project
py setup_gpu.py
```

---

## 🔧 Alternative CUDA Versions

If CUDA 11.8 doesn't work, try CUDA 12.1:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ✓ Verification

After installation, run:
```bash
cd project
py setup_gpu.py
```

**Expected output:**
```
✓ PyTorch installed: version 2.x.x+cu118
✓ CUDA is available in PyTorch
✓ NVIDIA GPU detected
✓ GPU tensor operations working
STATUS: ✓ READY FOR GPU TRAINING
```

---

## 🚀 Start Training on GPU

Once GPU is verified:

```bash
cd project
py train.py
```

**You should see:**
```
Device      : cuda
Model       : segformer (b2)
...
[Model] segformer | trainable params: 27,000,000 | device: cuda
```

---

## ⚡ Performance Difference

- **CPU Training**: ~30-60 minutes per epoch ❌
- **GPU Training**: ~5-10 minutes per epoch ✅

**GPU is 6-10x faster!**

---

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 4  # or 2
```

### Issue: "nvidia-smi not found"
**Solution**: Install NVIDIA drivers
- Download: https://www.nvidia.com/Download/index.aspx
- Restart computer after installation

### Issue: Still shows CPU after installation
**Solution**: 
1. Close all terminals and IDE
2. Reopen terminal
3. Run `py setup_gpu.py` again

---

## 📦 What Gets Installed

- **torch**: PyTorch with CUDA support (~2 GB)
- **torchvision**: Vision utilities with CUDA
- **torchaudio**: Audio utilities with CUDA

**Total download**: ~2-3 GB
**Installation time**: 5-10 minutes

---

## ✨ Ready to Install?

**Run this now:**
```bash
cd project
install_pytorch_gpu.bat
```

Or manually:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then verify and train:
```bash
py setup_gpu.py
py train.py
```

---

**Your NVIDIA GPU will make training 6-10x faster!** 🚀

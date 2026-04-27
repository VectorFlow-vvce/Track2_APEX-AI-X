"""
GPU Setup Check - Diagnose and fix GPU/CUDA issues for NVIDIA GPUs
"""

import sys
import subprocess

print("="*70)
print("NVIDIA GPU SETUP DIAGNOSTIC")
print("="*70)

# Step 1: Check if torch is installed
print("\n[1/5] Checking PyTorch installation...")
try:
    import torch
    print(f"✓ PyTorch installed: version {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed!")
    print("\nInstall PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Step 2: Check CUDA availability in PyTorch
print("\n[2/5] Checking CUDA availability in PyTorch...")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✓ CUDA is available in PyTorch")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("✗ CUDA not available in PyTorch")
    print("\nYour PyTorch is CPU-only version!")
    print("\nTo fix, reinstall PyTorch with CUDA support:")
    print("\n  For CUDA 11.8:")
    print("    pip uninstall torch torchvision torchaudio")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n  For CUDA 12.1:")
    print("    pip uninstall torch torchvision torchaudio")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n  Visit: https://pytorch.org/get-started/locally/")

# Step 3: Check NVIDIA GPU
print("\n[3/5] Checking NVIDIA GPU...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ NVIDIA GPU detected:")
        # Parse nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line or 'GeForce' or 'RTX' in line or 'GTX' in line:
                print(f"  {line.strip()}")
                break
    else:
        print("✗ nvidia-smi command failed")
        print("  NVIDIA drivers may not be installed")
except FileNotFoundError:
    print("✗ nvidia-smi not found")
    print("  NVIDIA drivers not installed or not in PATH")
    print("\n  Download drivers from: https://www.nvidia.com/Download/index.aspx")
except Exception as e:
    print(f"✗ Error checking GPU: {e}")

# Step 4: Test GPU if available
if cuda_available:
    print("\n[4/5] Testing GPU operations...")
    try:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ GPU Details:")
        print(f"  Device Name    : {device_name}")
        print(f"  Device Count   : {device_count}")
        print(f"  Total Memory   : {total_memory:.2f} GB")
        
        # Test tensor operation
        x = torch.randn(1000, 1000).cuda()
        y = x @ x
        print(f"✓ GPU tensor operations working")
        del x, y
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("\n[4/5] Skipping GPU test (CUDA not available)")

# Step 5: Recommendations
print("\n[5/5] Recommendations...")
if cuda_available:
    print("✓ Your system is ready for GPU training!")
    print("\n  Run training with:")
    print("    cd project")
    print("    python train.py")
    print("\n  Training will automatically use your NVIDIA GPU")
else:
    print("✗ GPU training not available")
    print("\n  ACTION REQUIRED:")
    print("  1. Install NVIDIA drivers (if not installed)")
    print("     Download: https://www.nvidia.com/Download/index.aspx")
    print("\n  2. Reinstall PyTorch with CUDA support:")
    print("     pip uninstall torch torchvision torchaudio")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n  3. Restart your terminal/IDE after installation")
    print("\n  4. Run this script again to verify")

print("\n" + "="*70)

# Summary
print("\nSUMMARY:")
print("-" * 70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"STATUS: ✓ READY FOR GPU TRAINING")
        print(f"DEVICE: {torch.cuda.get_device_name(0)}")
        print(f"MEMORY: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"STATUS: ✗ CPU ONLY (GPU not available)")
        print(f"ACTION: Reinstall PyTorch with CUDA support")
except:
    print(f"STATUS: ✗ PyTorch not installed")
    print(f"ACTION: Install PyTorch with CUDA support")

print("="*70 + "\n")

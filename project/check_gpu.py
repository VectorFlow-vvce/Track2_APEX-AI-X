"""
Quick GPU check - Run this to verify CUDA is available and will be used for training.
"""

import torch

print("="*60)
print("GPU/CUDA VERIFICATION")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"✓ GPU will be used for training")
    print(f"\nGPU Details:")
    print(f"  Device Name    : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version   : {torch.version.cuda}")
    print(f"  Device Count   : {torch.cuda.device_count()}")
    print(f"  Current Device : {torch.cuda.current_device()}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Total Memory   : {total_memory:.2f} GB")
    
    # Check if enough memory for batch_size=8
    if total_memory >= 6:
        print(f"\n✓ Sufficient GPU memory for BATCH_SIZE=8")
    elif total_memory >= 4:
        print(f"\n⚠ GPU memory may be tight - consider BATCH_SIZE=4 if OOM errors occur")
    else:
        print(f"\n⚠ Limited GPU memory - recommend BATCH_SIZE=2")
    
    # Test a simple operation
    print(f"\nTesting GPU operation...")
    try:
        x = torch.randn(1, 3, 256, 256).cuda()
        print(f"✓ GPU tensor creation successful: {x.device}")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"READY TO TRAIN ON GPU!")
    print(f"{'='*60}")
    print(f"\nRun: python train.py")
    print(f"Training will automatically use: {torch.cuda.get_device_name(0)}")
    
else:
    print(f"✗ CUDA not available - would use CPU (slow)")
    print(f"\nPossible reasons:")
    print(f"  1. PyTorch CPU-only version installed")
    print(f"  2. CUDA drivers not installed")
    print(f"  3. GPU not detected by system")
    print(f"\nTo fix:")
    print(f"  - Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
    print(f"  - Install NVIDIA CUDA drivers")

print(f"\n{'='*60}\n")

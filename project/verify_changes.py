"""
Quick verification script to check if all changes are correctly applied.
Run this before starting training to verify the setup.
"""

import os
import sys

def check_file_exists(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False

def check_config():
    """Verify config.py settings."""
    print("\n" + "="*60)
    print("CHECKING CONFIG.PY SETTINGS")
    print("="*60)
    
    try:
        import config
        
        checks = [
            ("IMAGE_SIZE", config.IMAGE_SIZE, 256, "Should be 256 for fast recovery"),
            ("BATCH_SIZE", config.BATCH_SIZE, 8, "Should be 8"),
            ("NUM_EPOCHS", config.NUM_EPOCHS, 10, "Should be 10"),
            ("LEARNING_RATE", config.LEARNING_RATE, 5e-5, "Should be 5e-5"),
            ("NUM_WORKERS", config.NUM_WORKERS, 2, "Should be >= 2"),
        ]
        
        all_good = True
        for name, actual, expected, desc in checks:
            if actual == expected:
                print(f"✓ {name} = {actual} ({desc})")
            else:
                print(f"⚠ {name} = {actual}, expected {expected} ({desc})")
                all_good = False
        
        # Check CLASS_WEIGHTS
        expected_weights = [1, 2, 2, 3, 5, 3, 4, 4, 1, 1]
        if hasattr(config, 'CLASS_WEIGHTS') and config.CLASS_WEIGHTS == expected_weights:
            print(f"✓ CLASS_WEIGHTS = {config.CLASS_WEIGHTS}")
        else:
            print(f"⚠ CLASS_WEIGHTS mismatch")
            all_good = False
        
        # Check DEVICE
        print(f"✓ DEVICE = {config.DEVICE}")
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def check_data_directories():
    """Verify data directories exist."""
    print("\n" + "="*60)
    print("CHECKING DATA DIRECTORIES")
    print("="*60)
    
    try:
        import config
        
        dirs = [
            (config.TRAIN_IMAGE_DIR, "Training images"),
            (config.TRAIN_MASK_DIR, "Training masks"),
            (config.VAL_IMAGE_DIR, "Validation images"),
            (config.VAL_MASK_DIR, "Validation masks"),
        ]
        
        all_good = True
        for path, desc in dirs:
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"✓ {desc}: {path} ({count} files)")
            else:
                print(f"✗ {desc} NOT FOUND: {path}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error checking directories: {e}")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\n" + "="*60)
    print("CHECKING CUDA/GPU")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available - training will use CPU (slower)")
            return False
            
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_dependencies():
    """Check required packages."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers (for SegFormer)"),
        ("PIL", "Pillow (for image loading)"),
        ("numpy", "NumPy"),
    ]
    
    all_good = True
    for package, desc in packages:
        try:
            __import__(package)
            print(f"✓ {desc} installed")
        except ImportError:
            print(f"✗ {desc} NOT installed - run: pip install {package}")
            all_good = False
    
    return all_good

def main():
    print("\n" + "="*60)
    print("TRAINING SETUP VERIFICATION")
    print("="*60)
    
    # Check files exist
    print("\n" + "="*60)
    print("CHECKING FILES")
    print("="*60)
    files = [
        ("project/train.py", "Training script"),
        ("project/config.py", "Configuration"),
        ("project/model.py", "Model definition"),
        ("project/data_loader.py", "Data loader"),
        ("project/utils.py", "Utilities"),
        ("project/augmentations.py", "Augmentations"),
    ]
    
    files_ok = all(check_file_exists(f, desc) for f, desc in files)
    
    # Run checks
    config_ok = check_config()
    data_ok = check_data_directories()
    cuda_ok = check_cuda()
    deps_ok = check_dependencies()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if files_ok and config_ok and data_ok and deps_ok:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nRun training with:")
        print("  cd project")
        print("  python train.py")
        return 0
    else:
        print("⚠ Some checks failed - please fix issues before training")
        if not cuda_ok:
            print("\nNote: CUDA not available, but training can still run on CPU (slower)")
        return 1

if __name__ == "__main__":
    sys.exit(main())

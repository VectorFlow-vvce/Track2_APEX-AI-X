#!/bin/bash
# Training script with GPU verification

echo "=========================================="
echo "GPU Training - Verification & Start"
echo "=========================================="
echo ""

# Check GPU first
echo "Step 1: Checking GPU availability..."
python check_gpu.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2: Starting training on GPU..."
    echo ""
    python train.py
else
    echo ""
    echo "ERROR: GPU check failed. Please fix GPU issues before training."
    exit 1
fi

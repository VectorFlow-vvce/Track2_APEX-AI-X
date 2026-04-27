@echo off
REM Training script with GPU verification for Windows

echo ==========================================
echo GPU Training - Verification and Start
echo ==========================================
echo.

echo Step 1: Checking GPU availability...
python check_gpu.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Step 2: Starting training on GPU...
    echo.
    python train.py
) else (
    echo.
    echo ERROR: GPU check failed. Please fix GPU issues before training.
    exit /b 1
)

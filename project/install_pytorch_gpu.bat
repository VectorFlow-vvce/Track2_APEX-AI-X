@echo off
REM Install PyTorch with CUDA support for NVIDIA GPU

echo ======================================================================
echo Installing PyTorch with CUDA Support for NVIDIA GPU
echo ======================================================================
echo.

echo Step 1: Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch with CUDA 11.8 support...
echo (This will download ~2-3 GB, please wait...)
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo ======================================================================
echo Installation Complete!
echo ======================================================================
echo.
echo Verifying GPU setup...
py setup_gpu.py

echo.
echo ======================================================================
echo Next Steps:
echo   1. Close and reopen your terminal/IDE
echo   2. Run: py train.py
echo ======================================================================
pause

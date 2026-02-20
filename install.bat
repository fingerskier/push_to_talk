@echo off
echo Installing push-to-talk dependencies...
pip install -r requirements.txt
echo.
echo Installing CUDA libraries for GPU support (optional, skip if CPU-only)...
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 2>nul
echo.
echo Done! Run with:  python main.py
echo.
echo Recommended: Run your terminal as Administrator for global hotkey capture.
pause

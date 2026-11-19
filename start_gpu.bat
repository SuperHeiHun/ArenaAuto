@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-gpu\Scripts\python.exe" (
    echo GPU environment not found. Run:
    echo python -m venv --system-site-packages .venv-gpu
    echo .venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
    pause
    exit /b 1
)
".venv-gpu\Scripts\python.exe" -c "import rapidocr, onnxruntime, PySide6" >nul 2>&1
if errorlevel 1 (
    echo Required packages are missing from .venv-gpu. See README GPU OCR setup.
    pause
    exit /b 1
)
".venv-gpu\Scripts\python.exe" scripts\run_gpu.py
if errorlevel 1 pause

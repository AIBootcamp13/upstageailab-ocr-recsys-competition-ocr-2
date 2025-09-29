# Simple in-container smoke tests for project environment
import sys

print("PYTHON:", sys.executable, sys.version.split()[0])
try:
    import torch

    print("TORCH_IMPORTED:", True)
    try:
        print("TORCH_VERSION:", torch.__version__)
    except Exception:
        pass
    try:
        print("TORCH_CUDA_AVAILABLE:", torch.cuda.is_available())
    except Exception as e:
        print("TORCH_CUDA_ERROR:", e)
except Exception as e:
    print("TORCH_IMPORT_ERROR:", e)

try:
    import subprocess

    res = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    if res.returncode == 0:
        print("UV_VERSION:", res.stdout.strip())
    else:
        print("UV_NOT_AVAILABLE")
except Exception as e:
    print("UV_CHECK_ERROR:", e)

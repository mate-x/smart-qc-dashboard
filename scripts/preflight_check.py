"""
scripts/preflight_check.py — 첫 실행 전 전체 환경 검증 (09_Infrastructure I.3)
사용법: python scripts/preflight_check.py
"""
import sys
from pathlib import Path

CHECKS: list[tuple[str, bool]] = []

# Python 버전
major, minor = sys.version_info[:2]
CHECKS.append(("Python 3.12", (major, minor) == (3, 12)))

# PyTorch
try:
    import torch

    ver_parts = torch.__version__.split("+")[0].split(".")[:2]
    CHECKS.append(("PyTorch >= 2.1", tuple(int(x) for x in ver_parts) >= (2, 1)))
    CHECKS.append(("CUDA available", torch.cuda.is_available()))
except ImportError:
    CHECKS.append(("PyTorch", False))
    CHECKS.append(("CUDA available", False))

# Anomalib
try:
    import anomalib

    CHECKS.append(("Anomalib >= 1.0", anomalib.__version__ >= "1.0"))
except ImportError:
    CHECKS.append(("Anomalib", False))

# Streamlit
try:
    import streamlit

    CHECKS.append(("Streamlit >= 1.32", streamlit.__version__ >= "1.32"))
except ImportError:
    CHECKS.append(("Streamlit", False))

# OpenCV
try:
    import cv2

    CHECKS.append(("OpenCV installed", True))
except ImportError:
    CHECKS.append(("OpenCV installed", False))

# Required dirs
for d in ["experiments", "models", "logs", "results", "dataset/imagenet_penalty"]:
    CHECKS.append((f"Dir: {d}", Path(f"./{d}").exists()))

# Report
print("\n=== Preflight Check ===")
all_ok = True
for name, result in CHECKS:
    status = "OK" if result else "FAIL"
    print(f"  [{status}]  {name}")
    if not result:
        all_ok = False

print()
if all_ok:
    print("All checks passed.")
else:
    print("Some checks FAILED — fix before running.")
sys.exit(0 if all_ok else 1)

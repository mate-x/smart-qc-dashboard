# 09. Infrastructure and Environment

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [04_System_Architecture.md](./04_System_Architecture.md), [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)
> **버전**: v1.0
> **작성일**: 2026-05-09
> **중요**: 이 문서는 실행 환경의 Single Source of Truth다. 하드웨어 요구사항, Python 환경, 디렉터리 초기화, 선택적 Docker 패키징 방법을 확정한다.

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Hardware Requirements](#b-hardware-requirements)
- [C. Software Requirements](#c-software-requirements)
- [D. Python Environment Setup](#d-python-environment-setup)
- [E. Directory Structure Initialization](#e-directory-structure-initialization)
- [F. Dataset Preparation](#f-dataset-preparation)
- [G. Application Launch](#g-application-launch)
- [H. Optional Docker Packaging](#h-optional-docker-packaging)
- [I. Environment Validation Checklist](#i-environment-validation-checklist)

---

## A. Objective & Scope

### A.1 이 문서의 목적

이 프로젝트는 **로컬 실행 기반** Streamlit 대시보드다. 클라우드 배포 없이 연구자·엔지니어의 단일 워크스테이션에서 실행된다.  
이 문서는 다음을 확정한다:

1. 실행에 필요한 최소/권장 하드웨어 사양
2. OS·Python·CUDA 버전 요구사항
3. 가상환경 구성 및 의존성 설치 순서
4. 첫 실행 전 필수 디렉터리 및 파일 초기화 절차
5. ImageNet penalty 데이터셋 배치 방법
6. 선택적 Docker 컨테이너 패키징 (재현성 목적)

### A.2 스코프 외 사항

| 항목 | 이유 |
|------|------|
| 클라우드 서버 배포 (AWS, GCP, Azure) | MVP 스코프 아님 |
| CI/CD 파이프라인 | 14_Deployment 문서에서 다룸 |
| 분산 학습 (multi-GPU, DDP) | 단일 GPU 환경 대상 |
| 데이터베이스 서버 | 파일시스템 기반 설계 |

---

## B. Hardware Requirements

### B.1 최소 사양 (CPU 전용)

| 항목 | 최소 | 비고 |
|------|------|------|
| CPU | 8코어 이상 | PatchCore 코어셋 연산 CPU-bound |
| RAM | 16 GB | 이미지 배치 로딩 + anomaly map 캐시 |
| 디스크 | 50 GB 이상 여유 | 모델 가중치 + 로그 + 결과 이미지 |
| OS | Windows 10+ / Ubuntu 20.04+ / macOS 12+ | |

CPU 전용 실행 시 EfficientAD 학습(`train_steps=70000` 기본값)은 수시간 소요될 수 있다.

### B.2 권장 사양 (GPU 사용)

| 항목 | 권장 | 비고 |
|------|------|------|
| GPU | NVIDIA RTX 3070 이상 (VRAM 8 GB 이상) | VRAM < 8 GB 시 batch_size 축소 필요 |
| CUDA | 11.8 또는 12.1 | PyTorch 공식 지원 버전 |
| RAM | 32 GB | GPU 학습 중 시스템 RAM 별도 사용 |
| 디스크 | SSD 권장, 100 GB 여유 | |

### B.3 VRAM 요구사항 추정

| 모델 | image_size | batch_size | 추정 VRAM |
|------|-----------|-----------|----------|
| EfficientAD small | 256 | 1 | ~3 GB |
| EfficientAD medium | 256 | 1 | ~5 GB |
| PatchCore wide_resnet50_2 | 256 | 16 | ~6 GB |
| PatchCore wide_resnet50_2 | 512 | 16 | ~10 GB |

> VRAM 부족 시 `batch_size=1`로 설정하거나 `image_size`를 줄인다.  
> 추론 시에는 학습 모델을 메모리에 로드하므로 학습 완료 후 `del model; torch.cuda.empty_cache()` 호출 (07절 §C.3).

---

## C. Software Requirements

### C.1 OS별 지원 상태

| OS | 지원 | 비고 |
|----|------|------|
| Windows 10/11 (x64) | ✅ 공식 지원 | PowerShell 7+ 권장 |
| Ubuntu 20.04 / 22.04 | ✅ 공식 지원 | |
| macOS 12+ (Apple Silicon) | ⚠️ 부분 지원 | GPU: MPS (Metal) / anomalib 일부 제한 가능 |
| macOS Intel | ⚠️ CPU 전용 | |

### C.2 Python 버전

```
Python 3.12 (확정 — 00절 §10.2 기준)
```

- Python 3.9 이하: `match-case` 구문 미지원 — 사용 금지
- Python 3.10 / 3.11: 호환 가능하나 공식 환경은 3.12

### C.3 CUDA / cuDNN 버전

| CUDA | cuDNN | PyTorch 버전 | 비고 |
|------|-------|-------------|------|
| **12.4** | **9.1** | ≥ 2.3.x | **확정 — 00절 §10.2 기준** |

NVIDIA 드라이버 버전: CUDA 12.4 기준 **550.xx 이상**.  
베이스 Docker 이미지: `nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04` + `libcudnn9-cuda-12` (apt 설치, 00절 §10.2).

> GPU 없는 환경에서는 CUDA 설치 불필요. `device_info.device == "cpu"` 자동 선택.

---

## D. Python Environment Setup

### D.1 Conda 환경 (권장)

```bash
# 1. conda 환경 생성
conda create -n smart-qc python=3.12 -y
conda activate smart-qc

# 2. PyTorch 설치 (CUDA 12.4 기준 — 00절 §10.2)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
    -c pytorch -c nvidia -y

# CPU 전용:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. 나머지 의존성 설치
pip install -r requirements.txt
```

### D.2 venv 환경 (대안)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# PyTorch 설치 (CUDA 12.4 — 00절 §10.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 나머지 의존성
pip install -r requirements.txt
```

### D.3 requirements.txt 핵심 의존성

```
# requirements.txt

# Core ML
torch>=2.1.0
torchvision>=0.16.0
anomalib>=1.0.0

# Dashboard
streamlit>=1.32.0

# Image processing
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# Data / visualization
pandas>=2.0.0
plotly>=5.18.0
matplotlib>=3.7.0
scikit-learn>=1.3.0

# Utilities
PyYAML>=6.0.1
tqdm>=4.66.0
```

> `anomalib>=1.0.0` 설치 시 `EfficientAd`(소문자 d)와 `Patchcore`(소문자 c) 클래스명을 사용한다 (08절 §B.1 기준).

### D.4 설치 검증

```python
# verify_env.py — 환경 설치 후 실행
import torch
import anomalib
import streamlit
import cv2

print(f"Python:     {__import__('sys').version}")
print(f"PyTorch:    {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Anomalib:   {anomalib.__version__}")
print(f"Streamlit:  {streamlit.__version__}")
print(f"OpenCV:     {cv2.__version__}")
```

기대 출력 (GPU 환경 예시):
```
Python:     3.12.x (...)
PyTorch:    2.3.x+cu124
CUDA avail: True
GPU:        NVIDIA GeForce RTX 3080
VRAM:       10.0 GB
Anomalib:   1.0.1
Streamlit:  1.32.0
OpenCV:     4.8.1
```

---

## E. Directory Structure Initialization

### E.1 필수 디렉터리

최초 실행 전 아래 디렉터리가 존재해야 한다. 없으면 `app.py` 시작 시 자동 생성한다.

```
smart-qc-dashboard/
├── experiments/            # history.json 저장 위치 (05절 §2.1)
├── models/                 # {exp_id}/ 하위 디렉터리 (05절 §2.2)
├── logs/                   # {exp_id}.log 저장 위치 (05절 §2.3)
├── results/                # {exp_id}/{stem}_triplet.png (07절 §C.4)
└── dataset/
    └── imagenet_penalty/   # EfficientAD ImageNet penalty 이미지 (05절 §2.4)
```

### E.2 자동 초기화 코드

```python
# utils/env_init.py — app.py 시작 시 1회 호출
from pathlib import Path

REQUIRED_DIRS = [
    Path("./experiments"),
    Path("./models"),
    Path("./logs"),
    Path("./results"),
    Path("./dataset/imagenet_penalty"),
]

def ensure_required_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
```

```python
# app.py 상단
from utils.env_init import ensure_required_dirs
ensure_required_dirs()
```

### E.3 experiments/history.json 초기화

`history.json`이 존재하지 않을 경우 빈 배열로 초기화한다:

```python
# utils/storage.py — load_history() 내부 (05절 §3.1 참조)
HISTORY_FILE = Path("./experiments/history.json")

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
```

---

## F. Dataset Preparation

### F.1 MVTec AD 형식 데이터셋 구조

이 시스템은 MVTec AD 형식 디렉터리 구조를 요구한다 (01절 §B.2 기준):

```
{dataset_root}/
├── train/
│   └── good/
│       ├── 000.png
│       └── ...
└── test/
    ├── good/
    │   ├── 000.png
    │   └── ...
    └── {defect_class}/
        ├── 000.png
        └── ...
```

Ground truth 마스크를 포함하는 경우 (선택적):

```
{dataset_root}/
└── ground_truth/
    └── {defect_class}/
        ├── 000_mask.png    # {stem}_mask{ext} 형식 (07절 §C.3 확정)
        └── ...
```

### F.2 지원 이미지 포맷

| 포맷 | 확장자 | 비고 |
|------|-------|------|
| PNG | `.png` | 권장 (무손실) |
| JPEG | `.jpg`, `.jpeg` | 허용 |
| BMP | `.bmp` | 허용 |
| TIFF | `.tiff`, `.tif` | 허용 |

지원 포맷 외 파일이 발견되면 `dataset_meta.has_invalid_files = True`로 표시하고 해당 파일을 학습에서 제외한다.

### F.3 ImageNet Penalty 데이터셋 (EfficientAD 전용)

EfficientAD 학습 시 `./dataset/imagenet_penalty/` 에 ImageNet 이미지가 필요하다 (05절 §2.4, 08절 Z.1).

```
dataset/
└── imagenet_penalty/
    ├── n01440764/          # ImageNet synset 디렉터리 (권장 구조)
    │   ├── img_0001.JPEG
    │   └── ...
    └── ...
```

**최소 이미지 수**: 1,000장 이상 권장 (학습 중 penalty 샘플링에 사용).

**경로 검증**: `validate_imagenet_penalty_dir()`가 아래 조건을 검사한다 (05절 §2.4):
- 디렉터리 존재 여부
- 내부 이미지 파일 1개 이상 존재 여부

조건 미충족 시 `ValueError`를 raise하고, EfficientAD 학습 시작이 차단된다.

```bash
# imagenet_penalty 디렉터리 최소 검증 (bash)
find ./dataset/imagenet_penalty -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l
# 출력이 0이면 EfficientAD 실행 불가
```

---

## G. Application Launch

### G.1 기본 실행

```bash
# 프로젝트 루트에서 실행
conda activate smart-qc   # 또는 source .venv/bin/activate

streamlit run app.py
```

브라우저가 자동으로 `http://localhost:8501` 을 연다.

### G.2 포트 및 설정 변경

```bash
# 포트 변경
streamlit run app.py --server.port 8502

# 브라우저 자동 열기 비활성화
streamlit run app.py --server.headless true

# 파일 변경 감지 비활성화 (배포 시 권장)
streamlit run app.py --server.fileWatcherType none
```

### G.3 `.streamlit/config.toml` 권장 설정

```toml
# .streamlit/config.toml

[server]
maxUploadSize = 200        # MB, 대용량 이미지 업로드 허용
enableXsrfProtection = true
headless = false

[theme]
base = "light"

[runner]
fastReruns = true          # 빠른 rerun 활성화 (tab4 폴링 최적화)
```

### G.4 실행 전 체크리스트

```
□ conda activate smart-qc (또는 venv 활성화)
□ python verify_env.py — 모든 항목 정상 출력 확인
□ ./dataset/imagenet_penalty/ — EfficientAD 사용 시 이미지 존재 확인
□ 데이터셋 디렉터리 MVTec AD 형식 확인 (train/good/, test/{class}/)
□ streamlit run app.py
```

---

## H. Optional Docker Packaging

> Docker는 필수가 아닌 선택적 재현성 수단이다. 개발·실험 환경에서는 직접 실행을 권장한다.

### H.1 Dockerfile

```dockerfile
FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Python 3.12 설치
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 필수 디렉터리 생성
RUN mkdir -p experiments models logs results dataset/imagenet_penalty

# Streamlit 포트
EXPOSE 8501

# 실행
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.headless", "true", \
            "--server.port", "8501", \
            "--server.address", "0.0.0.0"]
```

### H.2 docker-compose.yml (GPU 사용 시)

```yaml
version: "3.8"

services:
  smart-qc:
    build: .
    ports:
      - "8501:8501"
    volumes:
      # 데이터·모델·결과는 호스트와 마운트 — 컨테이너 재시작 후 데이터 유지
      - ./dataset:/app/dataset
      - ./experiments:/app/experiments
      - ./models:/app/models
      - ./logs:/app/logs
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
```

### H.3 Docker 실행

```bash
# 빌드
docker compose build

# 실행
docker compose up

# GPU 없는 환경 (deploy 섹션 제거 후)
docker compose -f docker-compose.cpu.yml up
```

### H.4 Docker 사용 시 제약

| 항목 | 내용 |
|------|------|
| 데이터셋 경로 | 컨테이너 내부 `/app/dataset/` 로 마운트 필요 |
| ImageNet penalty | 호스트 `./dataset/imagenet_penalty/` → 컨테이너 마운트 |
| 브라우저 접속 | `http://localhost:8501` (호스트에서) |
| Windows Docker Desktop | WSL2 백엔드 + NVIDIA Container Toolkit 필요 |

---

## I. Environment Validation Checklist

### I.1 설치 완료 기준

| 검증 항목 | 명령 | 기대 결과 |
|-----------|------|----------|
| Python 버전 | `python --version` | `3.12.x` |
| PyTorch 설치 | `python -c "import torch; print(torch.__version__)"` | `2.3.x+cu124` 등 |
| CUDA 사용 가능 | `python -c "import torch; print(torch.cuda.is_available())"` | `True` (GPU 환경) |
| Anomalib 설치 | `python -c "import anomalib; print(anomalib.__version__)"` | `1.0.x` |
| Streamlit 설치 | `streamlit --version` | `1.32.x` |
| 디렉터리 존재 | `ls experiments/ models/ logs/ results/` | 오류 없음 |

### I.2 EfficientAD 실행 가능 기준 추가 검증

```bash
# ImageNet penalty 이미지 존재 확인
python -c "
from utils.storage import validate_imagenet_penalty_dir
try:
    validate_imagenet_penalty_dir()
    print('OK: ImageNet penalty dir is valid')
except ValueError as e:
    print(f'FAIL: {e}')
"
```

### I.3 첫 실행 전 전체 검증 스크립트

```python
# scripts/preflight_check.py
import sys
import torch
from pathlib import Path

CHECKS = []

# Python 버전
major, minor = sys.version_info[:2]
CHECKS.append(("Python 3.12", (major, minor) == (3, 12)))

# PyTorch
try:
    import torch
    CHECKS.append(("PyTorch >= 2.1", tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 1)))
    CHECKS.append(("CUDA available", torch.cuda.is_available()))
except ImportError:
    CHECKS.append(("PyTorch", False))

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

# Required dirs
for d in ["experiments", "models", "logs", "results", "dataset/imagenet_penalty"]:
    CHECKS.append((f"Dir: {d}", Path(f"./{d}").exists()))

# Report
print("\n=== Preflight Check ===")
all_ok = True
for name, result in CHECKS:
    status = "✅" if result else "❌"
    print(f"  {status}  {name}")
    if not result:
        all_ok = False

print("\n" + ("All checks passed." if all_ok else "Some checks FAILED — fix before running."))
sys.exit(0 if all_ok else 1)
```

---

*다음 문서*: [11_Non_Functional_Requirements.md](./11_Non_Functional_Requirements.md)

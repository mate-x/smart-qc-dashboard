# Smart QC Dashboard

제조산업 품질검사를 위한 딥러닝 기반 비전검사 최적 모델 탐색 대시보드 v1.0 (MVP)

EfficientAD · PatchCore 두 모델을 전처리 파라미터와 함께 실험하고,
결과를 비교·시각화할 수 있는 Streamlit 기반 단일-사용자 대시보드입니다.

---

## 주요 기능

| 탭 | 기능 |
|---|---|
| 탭1. 데이터 폴더 | MVTec AD 형식 데이터셋 경로 검증 및 구조 확인 |
| 탭2. 전처리 설정 | None / Homomorphic / HE / CLAHE 전처리 파라미터 설정 및 미리보기 |
| 탭3. 모델 파라미터 | EfficientAD · PatchCore 하이퍼파라미터 설정, 디바이스 자동 감지 |
| 탭4. 학습 | 백그라운드 학습 실행, 실시간 Progress Bar · Loss 곡선 · 로그 |
| 탭5. 실험 히스토리 | 실험 목록 비교, ROC Curve, Confusion Matrix, 모델 저장 |
| 탭6. 이상 영역 시각화 | Anomaly Map 히트맵, Threshold 슬라이더, 3분할(원본/GT/히트맵) 저장 |

---

## 기술 스택

- **UI**: Streamlit ≥ 1.32
- **ML**: PyTorch ≥ 2.1 · Anomalib ≥ 1.0
- **이미지 처리**: OpenCV · Pillow
- **시각화**: Plotly · Matplotlib
- **실행 환경**: Python 3.12 · CUDA 12.4 · cuDNN 9.1

---

## 환경 설정

### 1. Conda (권장)

```bash
conda create -n smart-qc python=3.12 -y
conda activate smart-qc

# CUDA 12.4 기준
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -r requirements.txt
```

### 2. venv (대안)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Docker (선택)

> **Conda/venv를 권장하는 이유**: 이 프로젝트는 실험·반복 작업이 중심입니다.
> Docker는 코드 변경 시마다 이미지 재빌드가 필요해 개발 속도가 느립니다.
> GPU 사용 시 호스트에 NVIDIA Container Toolkit 추가 설치도 필요합니다.
> 재현 가능한 배포 환경이 필요할 때 Docker를 선택하세요.

GPU 사용 (NVIDIA Container Toolkit 필요):

```bash
docker compose build
docker compose up
# 브라우저: http://localhost:8501
```

CPU 전용 환경:

```bash
docker compose -f docker-compose.cpu.yml build
docker compose -f docker-compose.cpu.yml up
```

> 데이터셋과 학습 결과는 호스트 디렉터리와 마운트됩니다.
> EfficientAD 사용 시 `dataset/imagenet_penalty/` 에 이미지를 배치한 뒤 빌드하세요.

---

## 실행 전 준비

### 데이터셋 (MVTec AD 형식)

```
{dataset_root}/
├── train/
│   └── good/
└── test/
    ├── good/
    └── {defect_class}/
```

### EfficientAD 전용 — ImageNet Penalty 이미지

```bash
# 1,000장 이상의 ImageNet 이미지를 아래 경로에 배치
dataset/imagenet_penalty/
```

### 환경 검증

```bash
python scripts/preflight_check.py
```

---

## 실행

```bash
streamlit run app.py
# 브라우저: http://localhost:8501
```

---

## 테스트

### 빠른 실행 (단위 + 통합, ML 제외)

```bash
pytest tests/unit/ tests/integration/ -m "not slow" -v
```

### 커버리지 포함

```bash
pytest tests/unit/ tests/integration/ -m "not slow" --cov=utils --cov-report=term-missing
```

### 전체 실행 (E2E 포함, ML 환경 필요)

```bash
pytest -v
```

### 테스트 구조

```
tests/
├── conftest.py             # 공유 픽스처 (mvtec_dataset 등)
├── unit/                   # 단위 테스트 — utils 함수별
├── integration/            # 통합 테스트 — 모듈 간 연동
└── e2e/                    # E2E 테스트 (@pytest.mark.slow, ML 환경 필요)
```

> `slow` 마커가 붙은 테스트는 ML 라이브러리(anomalib 등)가 설치된 환경에서만 실행하세요.

---

## 프로젝트 구조

```
smart-qc-dashboard/
├── app.py                  # Streamlit 진입점
├── configs.yaml            # 공유 설정 파일 (탭2/3 Write)
├── requirements.txt
│
├── tabs/                   # 탭별 UI 모듈
├── utils/                  # 비즈니스 로직 · ML 래퍼
├── components/             # 사이드바 공통 컴포넌트
├── scripts/                # 환경 검증 스크립트
├── tests/                  # pytest 테스트 (unit / integration / e2e)
│
├── experiments/            # history.json (자동 생성)
├── models/                 # 학습된 모델 가중치 (gitignore)
├── logs/                   # 학습 로그 (gitignore)
├── results/                # Anomaly Map 저장 이미지 (gitignore)
└── dataset/                # 데이터셋 (gitignore)
    └── imagenet_penalty/
```

---

## 라이선스

이 프로젝트는 학습·연구 목적으로 제작되었습니다.

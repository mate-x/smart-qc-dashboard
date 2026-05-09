# 00. Global Context Document — Single Source of Truth

> **프로젝트명**: 제조산업 품질검사를 위한 딥러닝 기반 비전검사 최적 모델 탐색 대시보드 v1.0 (MVP)
> **버전**: v1.0
> **작성일**: 2026-05-08
> **목적**: 이 문서는 이후 생성되는 01~14번 PRD 파일 전체의 유일한 기준점이다. 모든 파일은 이 문서에 정의된 스키마, 계약, 용어, 규칙을 그대로 따른다. 변경은 이 문서를 먼저 수정한 후 전파한다.

---

## 목차

1. [Core Data Model](#1-core-data-model)
2. [Entity Relationship](#2-entity-relationship)
3. [Global State Contract Standard](#3-global-state-contract-standard)
4. [Universal Terminology Dictionary](#4-universal-terminology-dictionary)
5. [System Architecture](#5-system-architecture)
6. [Global Non-Functional Requirements](#6-global-non-functional-requirements)
7. [Observability Standards](#7-observability-standards)
8. [Deterministic Design Rules](#8-deterministic-design-rules)
9. [Made Assumptions](#9-made-assumptions)
10. [Global Technology Stack](#10-global-technology-stack)

---

## 1. Core Data Model

> **설계 원칙**: 이 시스템은 전통적인 RDBMS를 사용하지 않는다. 모든 영속 데이터는 파일시스템(JSON, YAML, `.pth`)에 저장된다. 아래 스키마는 Python dataclass 및 파일 직렬화 구조를 SQL 형식으로 표현한 논리 모델이다. 실제 구현 시 아래 필드명, 타입, 제약조건을 그대로 사용한다.

---

### 1.1 experiment (실험 레코드)

```
파일 위치: ./experiments/history.json  (배열 형태로 저장)

필드명                   타입        Nullable   제약조건
─────────────────────────────────────────────────────────────────
experiment_id           STRING      NOT NULL   PK, 형식: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_난수}
                                               예: efficientad_20260508_140023_7f3a
name                    STRING      NOT NULL   사용자 입력 또는 자동 생성, 최대 64자
status                  ENUM        NOT NULL   값: "completed" | "중단"
created_at              STRING      NOT NULL   ISO 8601, 예: "2026-05-08T14:00:23+09:00"
model_type              ENUM        NOT NULL   값: "efficientad" | "patchcore"
preprocessing_method    ENUM        NOT NULL   값: "none" | "homomorphic" | "he" | "clahe"
preprocessing_params    OBJECT      NULLABLE   NULL if preprocessing_method == "none"
model_params            OBJECT      NOT NULL   모델별 파라미터 전체 (1.4절 참조)
metrics                 OBJECT      NULLABLE   NULL if status == "중단"
threshold_method        ENUM        NOT NULL   값: "percentile" | "absolute"
threshold_value         FLOAT       NOT NULL   percentile: 0.0~100.0 / absolute: 0.0~1.0
model_path              STRING      NULLABLE   NULL if status == "중단", 예: "./models/efficientad_20260508_140023_7f3a/"
configs_path            STRING      NULLABLE   NULL if status == "중단", 예: "./models/efficientad_20260508_140023_7f3a/configs.yaml"
duration_seconds        INTEGER     NULLABLE   NULL if status == "중단", 학습 소요 시간(초)
dataset_path            STRING      NOT NULL   학습 시 사용한 데이터셋 경로
image_size              INTEGER     NOT NULL   학습 시 사용한 이미지 크기(픽셀)
```

---

### 1.2 metrics (성능 지표 오브젝트)

```
experiment.metrics 필드의 중첩 오브젝트 구조

필드명          타입      Nullable   제약조건
──────────────────────────────────────────────────────
accuracy        FLOAT     NOT NULL   0.0 ~ 1.0
precision       FLOAT     NOT NULL   0.0 ~ 1.0
recall          FLOAT     NOT NULL   0.0 ~ 1.0
f1_score        FLOAT     NOT NULL   0.0 ~ 1.0
f2_score        FLOAT     NOT NULL   0.0 ~ 1.0
auc             FLOAT     NOT NULL   0.0 ~ 1.0
confusion_matrix  OBJECT  NOT NULL   {"tp": int, "fp": int, "tn": int, "fn": int}
anomaly_scores  ARRAY     NOT NULL   테스트 이미지별 Float 배열, 길이 == 테스트 이미지 수
image_labels    ARRAY     NOT NULL   테스트 이미지별 정답 레이블 배열 (0=정상, 1=결함), 길이 == anomaly_scores 길이
```

---

### 1.3 preprocessing_params (전처리 파라미터 오브젝트)

```
experiment.preprocessing_params 필드의 중첩 오브젝트 구조
preprocessing_method에 따라 포함되는 필드가 다르다.

[method == "homomorphic"]
필드명       타입     Nullable   범위/기본값
─────────────────────────────────────────────
sigma        FLOAT    NOT NULL   0.1 ~ 50.0 / 기본값: 10.0
gamma_H      FLOAT    NOT NULL   1.0 ~ 3.0  / 기본값: 1.5
gamma_L      FLOAT    NOT NULL   0.1 ~ 1.0  / 기본값: 0.5
cutoff       FLOAT    NOT NULL   1.0 ~ 100.0 / 기본값: 30.0
normalize    BOOLEAN  NOT NULL   기본값: true

[method == "clahe"]
필드명       타입     Nullable   범위/기본값
─────────────────────────────────────────────
clip_limit   FLOAT    NOT NULL   0.1 ~ 40.0 / 기본값: 2.0

[method == "he"]
(파라미터 없음 — 빈 오브젝트 {} 허용)

[method == "none"]
(전체 필드 NULL 처리)
```

---

### 1.4 model_params (모델 파라미터 오브젝트)

```
[model_type == "efficientad"]
필드명                    타입      Nullable   범위/기본값
──────────────────────────────────────────────────────────────────
model_size               ENUM      NOT NULL   "small" | "medium" / 기본값: "medium"
train_steps              INTEGER   NOT NULL   1000 ~ 200000 / 기본값: 70000
optimizer                ENUM      NOT NULL   "adam" | "adamw" | "sgd" / 기본값: "adam"
learning_rate            FLOAT     NOT NULL   1e-6 ~ 1e-1 / 기본값: 0.0001
weight_decay             FLOAT     NOT NULL   0.0 ~ 0.1 / 기본값: 0.0001
out_channels             INTEGER   NOT NULL   128 | 256 | 384 | 512 / 기본값: 384
padding                  BOOLEAN   NOT NULL   기본값: false
ae_loss_weight           FLOAT     NOT NULL   0.0 ~ 1.0, st_loss_weight와 합산 1.0 / 기본값: 0.5
st_loss_weight           FLOAT     NOT NULL   0.0 ~ 1.0, ae_loss_weight와 합산 1.0 / 기본값: 0.5
autoencoder_lr           FLOAT     NOT NULL   1e-6 ~ 1e-1 / 기본값: 0.0001
autoencoder_weight_decay FLOAT     NOT NULL   0.0 ~ 0.1 / 기본값: 0.00001
lr_decay_epochs          INTEGER   NOT NULL   1000 ~ 200000 / 기본값: 50000
lr_decay_factor          FLOAT     NOT NULL   0.01 ~ 1.0 / 기본값: 0.1
scheduler                ENUM      NOT NULL   "StepLR" | "CosineAnnealingLR" / 기본값: "StepLR"
imagenet_penalty_weight  FLOAT     NOT NULL   0.0 ~ 10.0 / 기본값: 1.0
penalty_batch_size       INTEGER   NOT NULL   1 ~ 64 / 기본값: 8

[model_type == "patchcore"]
필드명                    타입      Nullable   범위/기본값
──────────────────────────────────────────────────────────────────
backbone                 ENUM      NOT NULL   "wide_resnet50_2" | "resnet18" | "resnet50" / 기본값: "wide_resnet50_2"
pretrained_source        ENUM      NOT NULL   "torchvision" | "local" / 기본값: "torchvision"
pretrained_path          STRING    NULLABLE   NULL if pretrained_source == "torchvision"
coreset_sampling_ratio   FLOAT     NOT NULL   0.01 ~ 1.0 / 기본값: 0.1
neighbourhood_kernel_size INTEGER  NOT NULL   1 ~ 9 (홀수만) / 기본값: 3
max_train                INTEGER   NOT NULL   100 ~ 10000 / 기본값: 1000
knn                      INTEGER   NOT NULL   1 ~ 50 / 기본값: 9
top_k_ratio              FLOAT     NOT NULL   0.0 ~ 1.0 / 기본값: 0.1
```

---

### 1.5 dataset_meta (데이터셋 메타데이터 오브젝트)

```
session_state.dataset_meta 필드 구조

필드명            타입      Nullable   제약조건
─────────────────────────────────────────────────────
dataset_path      STRING    NOT NULL   검증된 절대경로
train_good_count  INTEGER   NOT NULL   train/good/ 이미지 수
test_counts       OBJECT    NOT NULL   {"good": int, "<defect_class>": int, ...}
gt_counts         OBJECT    NOT NULL   {"<defect_class>": int, ...}
total_test_count  INTEGER   NOT NULL   test_counts 합계
channels          INTEGER   NOT NULL   1 (Grayscale) | 3 (RGB)
defect_classes    ARRAY     NOT NULL   결함 클래스 이름 문자열 배열, test/ 하위 디렉토리명 기준
supported_formats ARRAY     NOT NULL   실제 발견된 포맷 배열, 예: [".jpg", ".png"]
has_invalid_files BOOLEAN   NOT NULL   지원 포맷 외 파일 존재 여부
```

---

### 1.6 preprocessing_config (전처리 설정 오브젝트)

```
session_state.preprocessing_config 및 configs.yaml preprocessing 섹션

필드명           타입      Nullable   제약조건
──────────────────────────────────────────────────────
method           ENUM      NOT NULL   "none" | "homomorphic" | "he" | "clahe"
resize_mode      STRING    NOT NULL   고정값: "padding" (변경 불가)
image_size       INTEGER   NOT NULL   32 ~ 1024 (32의 배수) / 기본값: 256
normalization    ENUM      NOT NULL   "imagenet" | "custom"
mean             ARRAY     NOT NULL   [float, float, float] / imagenet: [0.485, 0.456, 0.406]
std              ARRAY     NOT NULL   [float, float, float] / imagenet: [0.229, 0.224, 0.225]
params           OBJECT    NULLABLE   1.3절 preprocessing_params 구조 참조
```

---

### 1.7 model_config (모델 설정 오브젝트)

```
session_state.model_config 및 configs.yaml model 섹션

필드명            타입      Nullable   제약조건
──────────────────────────────────────────────────
model_type        ENUM      NOT NULL   "efficientad" | "patchcore"
image_size        INTEGER   NOT NULL   preprocessing_config.image_size 와 동일
batch_size        INTEGER   NOT NULL   1 ~ 128 / 기본값: 16
random_seed       INTEGER   NOT NULL   0 ~ 2147483647 / 기본값: 42
threshold_method  ENUM      NOT NULL   "percentile" | "absolute"
threshold_value   FLOAT     NOT NULL   percentile: 0.0~100.0, absolute: 0.0~1.0
params            OBJECT    NOT NULL   1.4절 model_params 구조 참조
```

---

### 1.8 device_info (디바이스 정보 오브젝트)

```
session_state.device_info 필드 구조

필드명      타입      Nullable   제약조건
───────────────────────────────────────────────────
device      ENUM      NOT NULL   "cuda" | "cpu"
gpu_name    STRING    NULLABLE   NULL if device == "cpu" / 예: "Tesla T4"
vram_gb     FLOAT     NULLABLE   NULL if device == "cpu" / GPU VRAM 크기(GB)
```

---

### 1.9 configs.yaml (전체 파일 구조)

```yaml
# configs.yaml — 전처리 + 모델 파라미터 통합 파일
# 각 탭에서 해당 섹션만 업데이트한다.

experiment:
  name: STRING              # 실험명
  created_at: STRING        # ISO 8601

preprocessing:              # preprocessing_config 1.6절 스키마 그대로
  method: STRING
  resize_mode: "padding"
  image_size: INTEGER
  normalization: STRING
  mean: [FLOAT, FLOAT, FLOAT]
  std: [FLOAT, FLOAT, FLOAT]
  params:
    # method별 하위 필드 (1.3절)

model:                      # model_config 1.7절 스키마 그대로
  model_type: STRING
  image_size: INTEGER
  batch_size: INTEGER
  random_seed: INTEGER
  threshold_method: STRING
  threshold_value: FLOAT
  params:
    # model_type별 하위 필드 (1.4절)
```

---

## 2. Entity Relationship

> 이 시스템은 RDBMS가 없으므로 파일 간 참조 관계를 ERD로 표현한다.

```
[session_state] ─── (1:1) ─── [dataset_meta]
     │
     ├── (1:1) ──── [preprocessing_config]
     │
     ├── (1:1) ──── [model_config]
     │
     ├── (1:1) ──── [device_info]
     │
     ├── (1:N) ──── [experiments]  ←→  history.json (영속)
     │                   │
     │                   └── (1:1) ── [metrics]
     │                   └── (1:1) ── [preprocessing_params]
     │                   └── (1:1) ── [model_params]
     │
     ├── (1:1) ──── [selected_experiment_id]  ──→  experiments[exp_id] 참조
     │
     └── (1:1) ──── [anomaly_map_threshold]

[experiments[exp_id]] ─── (1:1) ─── [./models/{exp_id}/model_state_dict.pth]
                      ─── (1:1) ─── [./models/{exp_id}/configs.yaml]

[history.json] ─── (1:N) ─── [experiment 레코드]

[configs.yaml] ─── (1:1) ─── [preprocessing 섹션]  (탭2 Write)
               ─── (1:1) ─── [model 섹션]           (탭3 Write)
               ─── (1:1) ─── [experiment 섹션]      (탭4 Write)
```

### 참조 무결성 규칙

| 규칙 | 설명 |
|------|------|
| R-01 | `selected_experiment_id`는 반드시 `experiments` 배열 내 존재하는 `experiment_id`를 참조한다. |
| R-02 | `model_config.image_size`는 항상 `preprocessing_config.image_size`와 동일한 값으로 자동 동기화된다. |
| R-03 | `ae_loss_weight + st_loss_weight == 1.0` 불변 조건. 어느 한쪽 변경 시 다른 쪽을 자동 보정한다. |
| R-04 | `experiment.model_path`가 NOT NULL인 경우, 해당 경로에 `model_state_dict.pth`와 `configs.yaml`이 반드시 존재한다. |
| R-05 | `status == "중단"`인 레코드의 `metrics`, `model_path`, `configs_path`는 반드시 NULL이다. |

---

## 3. Global State Contract Standard

> 이 시스템은 REST API가 아닌 Streamlit 기반 단일 프로세스 애플리케이션이다. "API 계약"에 해당하는 것은 **session_state 인터페이스 계약**과 **파일 I/O 계약**이다.

---

### 3.1 session_state 초기화 명세

```python
# utils/session_state_init.py — 반드시 app.py 시작 시 1회 실행

SESSION_STATE_SCHEMA = {
    # 탭1 Write
    "dataset_path": None,           # str | None
    "dataset_meta": None,           # dict | None (1.5절 스키마)

    # 탭2 Write
    "preprocessing_config": None,   # dict | None (1.6절 스키마)

    # 탭3 Write
    "model_config": None,           # dict | None (1.7절 스키마)
    "device_info": None,            # dict | None (1.8절 스키마)

    # 탭4 Write
    "experiments": {},              # dict[str, dict] — key: experiment_id
    "current_run_status": "idle",   # "idle" | "running" | "stopped" | "completed"
    "current_exp_id": None,         # str | None — 현재 실행 중인 실험 ID

    # 탭5 Write
    "selected_experiment_id": None, # str | None

    # 탭6 Write
    "anomaly_map_threshold": None,  # float | None — 초기값: model_config.threshold_value
}

def init_session_state():
    for key, default in SESSION_STATE_SCHEMA.items():
        if key not in st.session_state:
            st.session_state[key] = default
```

---

### 3.2 탭 간 데이터 흐름 계약

| 키 | Write 탭 | Read 탭 | 타입 | NULL 처리 |
|----|----------|---------|------|-----------|
| `dataset_path` | 탭1 | 탭2, 탭4 | `str \| None` | None이면 해당 탭 진입 차단 + 안내 메시지 |
| `dataset_meta` | 탭1 | 탭2 | `dict \| None` | None이면 탭2 전처리 미리보기 비활성화 |
| `preprocessing_config` | 탭2 | 탭3(image_size), 탭4 | `dict \| None` | None이면 탭3/탭4 진입 차단 + 안내 메시지 |
| `model_config` | 탭3 | 탭4 | `dict \| None` | None이면 탭4 진입 차단 + 안내 메시지 |
| `device_info` | 탭3 | 탭4 | `dict \| None` | None이면 CPU fallback |
| `experiments` | 탭4 | 탭5, 탭6 | `dict` | 빈 dict이면 탭5/탭6 안내 메시지 |
| `current_run_status` | 탭4 | 탭4 UI | `str` | 항상 유효한 ENUM 값 |
| `selected_experiment_id` | 탭5 | 탭6 | `str \| None` | None이면 탭6 안내 메시지 |
| `anomaly_map_threshold` | 탭6 | 탭6 내부 | `float \| None` | None이면 model_config.threshold_value 사용 |

---

### 3.3 파일 I/O 계약

#### history.json 읽기/쓰기

```python
# 읽기: 파일 없으면 빈 배열 반환 (예외 발생 금지)
def load_history() -> list[dict]:
    path = Path("./experiments/history.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 쓰기: 원자적 쓰기 (tmpfile → rename)
def save_history(records: list[dict]) -> None:
    path = Path("./experiments/history.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
```

#### configs.yaml 읽기/쓰기

```python
# utils/config_manager.py

def load_config(path: str = "./configs.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config_section(section: str, data: dict, path: str = "./configs.yaml") -> None:
    # 기존 파일 로드 후 해당 섹션만 업데이트 (다른 섹션 보존)
    config = load_config(path)
    config[section] = data
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
```

---

### 3.4 표준 안내 메시지 상수

```python
# utils/messages.py — 변경 시 이 파일만 수정

MSG = {
    "NO_DATASET":       "먼저 탭1에서 데이터 폴더를 설정해 주세요.",
    "NO_PREPROCESSING": "먼저 탭2에서 전처리 설정을 완료해 주세요.",
    "NO_MODEL_CONFIG":  "먼저 탭3에서 모델 파라미터를 설정해 주세요.",
    "NO_EXPERIMENTS":   "아직 실행된 실험이 없습니다. 탭4에서 학습을 먼저 실행해 주세요.",
    "NO_SELECTED_EXP":  "탭5에서 분석할 실험을 먼저 선택해 주세요.",
    "GRAYSCALE_DETECT": "Grayscale 이미지가 감지되었습니다. 모델 입력을 위해 RGB 3채널로 자동 변환됩니다.",
    "INVALID_FOLDER":   "MVTec AD 형식의 폴더 구조가 아닙니다. (필수: train/good/, test/, ground_truth/)",
    "TRAIN_STOPPED":    "학습이 중단되었습니다. 해당 실험은 '중단' 상태로 히스토리에 기록되었습니다.",
}
```

---

### 3.5 오류 코드 레지스트리

| 코드 | 설명 | 발생 조건 |
|------|------|-----------|
| `ERR_DATASET_NOT_FOUND` | 지정 경로가 존재하지 않음 | 탭1 경로 검증 실패 |
| `ERR_INVALID_FOLDER_STRUCTURE` | MVTec AD 구조 미충족 | `train/good/` 또는 `test/` 미존재 |
| `ERR_NO_VALID_IMAGES` | 지원 포맷 이미지 없음 | `.jpg/.png/.bmp` 외 파일만 존재 |
| `ERR_PREPROCESSING_CONFIG_MISSING` | 전처리 설정 없음 | 탭2 미완료 상태에서 탭3/탭4 접근 |
| `ERR_MODEL_CONFIG_MISSING` | 모델 설정 없음 | 탭3 미완료 상태에서 탭4 접근 |
| `ERR_MODEL_INIT_FAILED` | 모델 초기화 실패 | CUDA OOM 또는 Anomalib 오류 |
| `ERR_TRAINING_INTERRUPTED` | 학습 강제 중단 | 사용자 [학습 중지] 클릭 |
| `ERR_CONFIG_LOAD_FAILED` | YAML 파싱 실패 | 잘못된 configs.yaml 형식 |
| `ERR_MODEL_SAVE_FAILED` | 모델 저장 실패 | 디스크 공간 부족 또는 권한 없음 |
| `ERR_EXPERIMENT_NOT_FOUND` | 실험 ID 미존재 | 삭제된 실험에 접근 시 |
| `ERR_INVALID_PARAM_RANGE` | 파라미터 범위 초과 | 1.4절 범위 벗어난 값 입력 |
| `ERR_WEIGHT_SUM_INVALID` | ae+st weight 합 != 1.0 | 자동 보정 불가 상태 (edge case) |

---

## 4. Universal Terminology Dictionary

| 용어 | 정의 |
|------|------|
| **실험 (Experiment)** | 단일 학습 사이클 1회 실행 결과. `experiment_id`로 식별. |
| **experiment_id** | `{model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_난수}` 형식 문자열. 예: `efficientad_20260508_140023_7f3a`. UUID 미사용. |
| **전처리 파이프라인** | None/Homomorphic/HE/CLAHE 필터 적용 → Resize+Padding → 정규화 순서로 고정된 이미지 변환 과정. |
| **Anomaly Score** | 모델이 각 테스트 이미지에 대해 출력하는 이상 정도 수치. 0.0(정상)에 가까울수록 정상. |
| **Anomaly Map** | 각 픽셀의 Anomaly Score를 히트맵으로 시각화한 2D 배열. 원본 이미지와 동일한 해상도. |
| **Threshold** | Anomaly Score 이진화 기준값. `threshold_method`가 "percentile"이면 훈련 데이터 Score 분포의 N% 백분위수, "absolute"이면 직접 지정 값. |
| **GT 마스크 (Ground Truth Mask)** | MVTec AD `ground_truth/` 하위의 결함 위치 이진 마스크 이미지. 0=정상, 255=결함. |
| **MVTec AD 폴더 구조** | `{dataset_root}/train/good/`, `{dataset_root}/test/{class}/`, `{dataset_root}/ground_truth/{class}/` 구조. |
| **coreset** | PatchCore에서 메모리 뱅크의 대표 특징 벡터 서브셋. `coreset_sampling_ratio`로 비율 지정. |
| **session_state** | Streamlit 재실행 간 데이터를 보존하는 딕셔너리. 페이지 새로고침 시 초기화. |
| **history.json** | 모든 실험 레코드를 배열로 저장하는 영속 파일. `./experiments/history.json`. |
| **configs.yaml** | 전처리+모델 파라미터 통합 설정 파일. 탭별 해당 섹션만 업데이트. |
| **model_state_dict** | PyTorch `torch.save(model.state_dict(), ...)` 형식으로 저장된 모델 가중치 파일. |
| **완료 상태** | `status == "completed"`. 학습이 정상 종료되고 metrics, model_path, configs_path 모두 존재하는 상태. |
| **중단 상태** | `status == "중단"`. 사용자가 [학습 중지]를 클릭하여 중단된 상태. metrics, model_path, configs_path 모두 NULL. |
| **결함 클래스** | MVTec AD `test/` 하위 디렉토리명. `good`은 결함 없음. 그 외(crack, scratch 등)는 결함. |
| **지원 이미지 포맷** | `.jpg`, `.png`, `.bmp` 세 가지. 확장자 대소문자 구분 없이 처리. |
| **Resize+Padding** | 원본 비율 유지하여 `image_size×image_size`로 리사이즈 후 부족한 영역을 검정(0)으로 패딩. `resize_mode`는 항상 "padding"으로 고정. |

---

## 5. System Architecture

### 5.1 컴포넌트 구성

```
┌─────────────────────────────────────────────────────────────┐
│                   브라우저 (localhost:8501)                   │
│              Streamlit Web UI (6-Tab Layout)                 │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP (Streamlit WebSocket)
┌─────────────────────────────▼───────────────────────────────┐
│                  app.py (Streamlit 진입점)                    │
│  ┌────────────┐  ┌──────────────────────────────────────┐   │
│  │  sidebar   │  │           st.tabs([탭1~탭6])          │   │
│  │ component  │  │  tabs/tab1.py ~ tabs/tab6.py         │   │
│  └────────────┘  └──────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    utils/ 레이어                       │   │
│  │  session_state_init.py  │  config_manager.py         │   │
│  │  messages.py            │  metrics.py                │   │
│  │  model_factory.py       │  image_utils.py            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ Python function call
┌─────────────────────────────▼───────────────────────────────┐
│               ML 레이어 (Anomalib / PyTorch)                  │
│  EfficientAD Engine  │  PatchCore Engine  │  Evaluator      │
└──────────┬───────────┴───────────┬────────┴────────┬────────┘
           │                       │                 │
┌──────────▼───────────────────────▼─────────────────▼────────┐
│                    파일시스템 레이어                            │
│  ./dataset/          데이터셋 (읽기 전용)                      │
│  ./experiments/      history.json (읽기/쓰기)                 │
│  ./models/{exp_id}/  model_state_dict.pth, configs.yaml      │
│  ./logs/{exp_id}.log 학습 로그 (쓰기 전용)                     │
│  ./configs.yaml      공유 설정 파일 (읽기/쓰기)                │
└──────────────────────────────────────────────────────────────┘
```

---

### 5.2 디렉토리 구조

```
smart-qc-dashboard/
├── app.py                          # Streamlit 진입점
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
├── configs.yaml                    # 공유 설정 파일 (탭2/3 Write)
│
├── tabs/
│   ├── tab1_data_folder.py
│   ├── tab2_preprocessing.py
│   ├── tab3_model_params.py
│   ├── tab4_training.py
│   ├── tab5_history.py
│   └── tab6_anomaly_map.py
│
├── utils/
│   ├── session_state_init.py       # SESSION_STATE_SCHEMA 정의 (3.1절)
│   ├── config_manager.py           # YAML 읽기/쓰기 (3.3절)
│   ├── messages.py                 # MSG 상수 (3.4절)
│   ├── metrics.py                  # Accuracy/Precision/Recall/F1/F2/AUC 계산
│   ├── model_factory.py            # EfficientAD/PatchCore Anomalib 래퍼
│   └── image_utils.py              # 전처리, Resize+Padding, 채널 변환
│
├── components/
│   └── sidebar.py                  # 사이드바 공통 컴포넌트
│
├── experiments/
│   └── history.json                # 실험 히스토리 (자동 생성)
│
├── models/
│   └── {experiment_id}/
│       ├── model_state_dict.pth
│       └── configs.yaml
│
├── logs/
│   └── {experiment_id}.log
│
└── docs/
    └── *.md
```

---

### 5.3 탭별 데이터 흐름 단계

```
[탭1: 데이터 폴더 구조]
  Step 1. 사용자 경로 입력 (st.text_input)
  Step 2. os.walk()로 폴더 구조 스캔
  Step 3. MVTec AD 검증 (train/good/, test/, ground_truth/ 존재 확인)
  Step 4. 이미지 파일 수 카운트 (지원 포맷만)
  Step 5. 채널 수 감지 (PIL.Image → mode 확인)
  Step 6. 대표 썸네일 추출 (각 클래스 첫 번째 이미지)
  Step 7. session_state.dataset_path, dataset_meta Write
  ↓
[탭2: 전처리 파라미터 설정]
  Step 1. dataset_path None 체크 → 차단 또는 통과
  Step 2. 전처리 라디오 선택
  Step 3. 선택된 전처리 파라미터 UI 렌더링 (비선택 숨김)
  Step 4. 샘플 이미지 로드 → 전처리 적용 → 미리보기
  Step 5. Resize+Padding 적용 (image_size 기준)
  Step 6. 정규화 설정
  Step 7. session_state.preprocessing_config Write
  Step 8. configs.yaml preprocessing 섹션 Save (선택적)
  ↓
[탭3: 모델 파라미터 설정]
  Step 1. preprocessing_config None 체크 → 차단 또는 통과
  Step 2. torch.cuda.is_available() → device_info Write
  Step 3. 모델 라디오 선택 (EfficientAD / PatchCore)
  Step 4. image_size = preprocessing_config.image_size 자동 반영
  Step 5. 모델별 파라미터 UI 렌더링
  Step 6. ae/st weight 합산 1.0 자동 보정
  Step 7. session_state.model_config Write
  Step 8. configs.yaml model 섹션 Save (선택적)
  ↓
[탭4: 학습 시작 + 학습 로그]
  Step 1. dataset_path, preprocessing_config, model_config None 체크 → 차단
  Step 2. 실험명 입력 (자동 생성 또는 사용자 입력)
  Step 3. [학습 시작] 클릭 → current_run_status = "running"
  Step 4. 백그라운드 스레드 + Queue로 학습 루프 실행
  Step 5. 메인 스레드: st.empty() + 주기적 rerun으로 Progress Bar, Loss 곡선 갱신
  Step 6. [학습 중지] 클릭 → 중단 신호 → current_run_status = "stopped"
  Step 7a. 완료: metrics 계산 → experiments[exp_id] Write → history.json Append → model 저장
  Step 7b. 중단: experiments[exp_id] Write (status="중단") → history.json Append
  Step 8. 완료/중단 알림 표시
  ↓
[탭5: 실험 히스토리 + 결과 상세 + 모델 저장]
  Step 1. history.json 로드 → 실험 목록 테이블 렌더링
  Step 2. 실험 선택 → selected_experiment_id Write
  Step 3. 선택 실험 상세: Confusion Matrix, ROC Curve, Anomaly Score 분포
  Step 4. 다중 선택 → 비교 차트 렌더링 (막대 또는 레이더)
  Step 5. [모델 저장] → state_dict + configs.yaml 저장
  Step 6. 저장 완료 메시지 (경로·파일명·용량)
  Step 7. [실험 삭제] → history.json에서 제거
  ↓
[탭6: 이상 영역 시각화]
  Step 1. selected_experiment_id None 체크 → 차단 또는 통과
  Step 2. 실험의 metrics.anomaly_scores, image_labels 로드
  Step 3. 테스트 이미지 목록 테이블 렌더링 (결함 유형 필터 적용)
  Step 4. 이미지 선택
  Step 5. 모델 로드 → 선택 이미지 추론 → Anomaly Map 생성
  Step 6. Threshold 슬라이더 → 실시간 이진화 갱신
  Step 7. 3분할 시각화: 원본 / GT 마스크 / Heatmap
  Step 8. PNG 저장 버튼
```

---

## 6. Global Non-Functional Requirements

> 이후 모든 파일에서 이 절을 참조하며, 강화가 필요한 경우 "6.X절 기준 강화: ..." 형식으로 명시한다.

| 항목 | 요구사항 | 측정 방법 |
|------|----------|-----------|
| **UI 응답성** | 학습 중 UI 블로킹 없음. 탭 전환 응답 < 1초 | 수동 테스트 |
| **학습 성능 — EfficientAD** | g4dn.xlarge에서 EfficientAD-medium 70,000 steps **20분 이내** | 실측 측정 |
| **학습 성능 — PatchCore** | g4dn.xlarge에서 PatchCore (coreset 10%) **10분 이내** | 실측 측정 |
| **모델 정확도** | MVTec AD Screw 데이터셋 기준 AUC ≥ 0.95 | history.json metrics.auc |
| **디바이스 자동 감지** | `torch.cuda.is_available()` 결과에 따라 자동 전환, UI 명시 | 자동 감지 로직 |
| **재현성** | 동일 `random_seed` + 동일 하이퍼파라미터 시 동일 결과 | 2회 실행 결과 비교 |
| **데이터 무결성** | 폴더 구조 검증 실패 시 탭4 진입 완전 차단 | E2E 테스트 |
| **이미지 처리** | Resize+Padding 적용 후 가로·세로 비율 유지 오차 < 1픽셀 | 픽셀 단위 검증 |
| **채널 처리** | Grayscale 이미지 자동 RGB 변환, 3채널 보장 | 단위 테스트 |
| **디스크 용량** | 모델 저장 전 여유 공간 < 500MB 시 경고 메시지 표시 | `shutil.disk_usage()` |
| **Streamlit 버전** | ≥ 1.30 (st.tabs 지원 보장) | requirements.txt |

---

## 7. Observability Standards

### 7.1 로그 포맷 (JSON)

```python
# 모든 로그 엔트리는 아래 필드를 포함한다
LOG_FORMAT = {
    "timestamp": "ISO 8601",        # 예: "2026-05-08T14:00:23.456+09:00"
    "level": "INFO|WARNING|ERROR",
    "experiment_id": "str|null",    # 실험 컨텍스트가 없으면 null
    "tab": "tab1|tab2|...|tab6|null",
    "event": "이벤트명 (snake_case)",
    "message": "사람이 읽을 수 있는 설명",
    "data": {}                      # 이벤트별 추가 데이터 (선택)
}
```

### 7.2 필수 로그 이벤트

| 이벤트명 | 탭 | 레벨 | 설명 |
|---------|-----|------|------|
| `dataset_validated` | tab1 | INFO | 폴더 구조 검증 완료 |
| `dataset_validation_failed` | tab1 | ERROR | 폴더 구조 검증 실패 |
| `preprocessing_config_saved` | tab2 | INFO | 전처리 설정 저장 |
| `model_config_saved` | tab3 | INFO | 모델 설정 저장 |
| `training_started` | tab4 | INFO | 학습 시작 |
| `training_step` | tab4 | INFO | 매 N step마다 (N=1000) |
| `training_completed` | tab4 | INFO | 학습 완료 + 소요 시간 |
| `training_stopped` | tab4 | WARNING | 사용자 중단 |
| `training_failed` | tab4 | ERROR | 예외 발생 + traceback |
| `model_saved` | tab5 | INFO | 모델 파일 저장 완료 |
| `experiment_deleted` | tab5 | WARNING | 실험 삭제 |

### 7.3 학습 로그 파일 형식

```
# ./logs/{experiment_id}.log
# 각 줄: ISO 8601 타임스탬프 + 탭 구분자 + 메시지

2026-05-08T14:00:23+09:00	[시작] 실험: efficientad_20260508_140023_7f3a
2026-05-08T14:00:25+09:00	[Step 1000/70000] Loss: 0.0521 | 경과: 2.1s
2026-05-08T14:01:05+09:00	[Step 2000/70000] Loss: 0.0412 | 경과: 62.3s
...
2026-05-08T14:20:00+09:00	[완료] 총 소요: 1177s | AUC: 0.971
```

### 7.4 알림 조건

| 조건 | 알림 방법 | 메시지 |
|------|-----------|--------|
| 학습 완료 | `st.success()` | "학습이 완료되었습니다. 소요 시간: {N}분 {M}초" |
| 학습 중단 | `st.warning()` | MSG["TRAIN_STOPPED"] |
| 학습 실패 | `st.error()` | "학습 중 오류가 발생했습니다. 로그를 확인해 주세요." |
| 모델 저장 완료 | `st.success()` | "저장 완료: {path} ({size_mb:.1f} MB)" |
| 폴더 구조 오류 | `st.error()` | MSG["INVALID_FOLDER"] |
| Grayscale 감지 | `st.info()` | MSG["GRAYSCALE_DETECT"] |
| 디스크 공간 부족 | `st.warning()` | "디스크 여유 공간이 {size_mb:.0f} MB로 부족합니다. 500 MB 이상 확보 후 저장해 주세요." |

---

## 8. Deterministic Design Rules

> 이후 모든 파일의 코드 및 명세에서 아래 규칙을 강제 적용한다. 예외는 허용되지 않는다.

| 규칙 | 적용 범위 | 세부 규칙 |
|------|-----------|-----------|
| **R-NAMING-01** | 모든 Python 변수·함수·파일명 | `snake_case` 고정. 클래스명만 `PascalCase`. |
| **R-NAMING-02** | session_state 키명 | `snake_case`. 예: `dataset_path`, `model_config`. |
| **R-NAMING-03** | experiment_id 형식 | `{model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_소문자_16진수}`. 예: `efficientad_20260508_140023_7f3a`. |
| **R-ID-01** | experiment_id 생성 | `uuid.uuid4().hex[:4]`로 4자리 난수 생성. |
| **R-TIME-01** | 모든 시간 표현 | ISO 8601. 한국시간(KST, UTC+9). 예: `"2026-05-08T14:00:23+09:00"`. |
| **R-FLOAT-01** | 지표 수치 저장 | `round(value, 6)`. 표시는 소수점 4자리. |
| **R-BOOL-01** | YAML boolean | Python `True/False`, YAML `true/false`. |
| **R-PATH-01** | 모든 파일 경로 | `pathlib.Path` 사용. 문자열 연산 금지. |
| **R-ENCODE-01** | 파일 읽기/쓰기 | 항상 `encoding="utf-8"` 명시. |
| **R-ENUM-01** | ENUM 값 비교 | 소문자 문자열 비교. 예: `if model_type == "efficientad":`. |
| **R-NULL-01** | 없는 값 | Python `None`. JSON `null`. YAML `null`. |
| **R-SEED-01** | 재현성 | `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)` 동시 설정. |
| **R-ATOMIC-01** | 파일 쓰기 | 임시 파일(.tmp) 생성 후 rename. 부분 쓰기 방지. |
| **R-UI-01** | UI 언어 | 모든 라벨·버튼·안내 메시지 한국어. 기술 용어는 한국어+영문 병기. |
| **R-UI-02** | 숨김 처리 | 비선택 파라미터 UI는 DOM 미렌더링 (`if` 분기). `disabled=True` 금지. |

---

## 9. Made Assumptions

> PRD에 명시되지 않은 항목에 대한 합리적 가정. 이후 모든 설계는 이 가정을 기준으로 한다.

| # | 가정 항목 | 가정 내용 | 근거 |
|---|-----------|-----------|------|
| A-01 | 동시 사용자 수 | 1명 (단일 사용자). Streamlit 기본 단일 세션 모델 적용. | PRD에 다중 사용자 언급 없음 |
| A-02 | 학습 비동기 처리 | Python `threading.Thread` + `queue.Queue`로 백그라운드 학습, 메인 스레드는 UI 갱신 전담. | PRD 5절 "UI 블로킹 없음" |
| A-03 | Anomalib 버전 | `anomalib >= 1.0.0` (v1 API 기준). | PRD 6절 기술 스택 |
| A-04 | 모델 저장 크기 | EfficientAD-medium ≈ 200~400 MB, PatchCore (WideResNet50) ≈ 400~800 MB. | 일반적인 모델 크기 |
| A-05 | 테스트 이미지 추론 | 탭6에서 모델 재로드 후 전체 테스트셋 일괄 추론. 실험 완료 시 `metrics.anomaly_scores` 이미 계산·저장. | 탭6 응답성 확보 |
| A-06 | configs.yaml 위치 | 작업 디렉토리 루트 `./configs.yaml`. Docker 실행 시 `/app/configs.yaml`. | 11.1절 Dockerfile WORKDIR /app |
| A-07 | 데이터셋 마운트 | Docker 실행 시 `-v /path/to/dataset:/app/dataset`. 탭1에서 `/app/dataset` 하위 경로 입력. | PRD 11.3절 docker run |
| A-08 | Loss 곡선 갱신 주기 | EfficientAD: 매 500 step. PatchCore: 에포크 단위. | UI 성능 vs 정보 밀도 균형 |
| A-09 | 학습 로그 버퍼 | UI에 표시되는 로그 텍스트 박스는 최신 100줄만 유지. 파일에는 전량 저장. | Streamlit 메모리 제한 |
| A-10 | GT 마스크 없는 결함 클래스 | `ground_truth/{class}/` 디렉토리가 없으면 해당 이미지의 GT는 빈 마스크(전체 0)로 처리. | MVTec AD 일부 클래스 |
| A-11 | image_size 기본값 | 256. 변경 시 preprocessing_config와 model_config 동시 업데이트. | PRD 9.2절 YAML 예시 |
| A-12 | 한국 시간대 | 모든 시간은 KST (UTC+9) 기준. `datetime.now(tz=timezone(timedelta(hours=9)))`. | PRD 한국어 작성 기준 |
| A-13 | 비교 차트 최대 실험 수 | 한 번에 최대 10개 실험 비교. 초과 시 `st.warning()` 안내. | Plotly 레이더 차트 가독성 |
| A-14 | 폴더 구조 검증 깊이 | `train/good/` 디렉토리 존재 + 이미지 최소 1개 이상. `test/good/` 디렉토리 선택적. | MVTec AD 표준 |

---

## 10. Global Technology Stack

### 10.1 핵심 스택

| 카테고리 | 기술 | 버전 기준 |
|----------|------|-----------|
| **UI** | Streamlit | ≥ 1.30 |
| **ML 프레임워크** | PyTorch | ≥ 2.1 |
| **ML 프레임워크** | torchvision | PyTorch와 호환 버전 |
| **이상 탐지 모델** | Anomalib | ≥ 1.0.0 (v1 API) |
| **이미지 처리** | OpenCV | `opencv-python-headless` |
| **이미지 처리** | Pillow (PIL) | ≥ 10.0 |
| **평가 지표** | scikit-learn | ≥ 1.3 |
| **시각화** | Matplotlib | ≥ 3.7 |
| **시각화** | Plotly | ≥ 5.15 |
| **설정 관리** | PyYAML | ≥ 6.0 |
| **컨테이너** | Docker | ≥ 24.0 |
| **GPU 컨테이너** | NVIDIA Container Toolkit | 최신 안정 버전 |
| **클라우드** | AWS EC2 | g4dn.xlarge |

### 10.2 환경 명세

| 항목 | 값 |
|------|----|
| **Python** | 3.12 |
| **CUDA** | 12.4 |
| **cuDNN** | 9.1 |
| **베이스 Docker 이미지** | `nvcr.io/nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04` |

### 10.3 지원 이미지 포맷

`.jpg` (`.jpeg`), `.png`, `.bmp`
확장자 매칭: `{".jpg", ".jpeg", ".png", ".bmp"}` (소문자 변환 후 비교)

### 10.4 requirements.txt 기준 패키지 목록

```
streamlit>=1.30.0
torch>=2.1.0
torchvision
anomalib>=1.0.0
opencv-python-headless
Pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
PyYAML>=6.0
numpy>=1.24.0
```

### 10.5 하드웨어 기준

| 항목 | 개발/테스트 | 프로덕션 (AWS) |
|------|------------|----------------|
| **인스턴스** | 로컬 GPU 머신 | AWS EC2 g4dn.xlarge |
| **GPU** | CUDA 지원 GPU | NVIDIA Tesla T4 (16GB VRAM) |
| **vCPU** | ≥ 4 | 4 |
| **RAM** | ≥ 16GB | 16GB |
| **Storage** | ≥ 50GB | 100GB gp3 EBS |
| **Port** | 8501/tcp | 8501/tcp, 22/tcp |

---

*문서 끝. Phase 2 파일 생성 전 이 문서의 모든 절이 기준으로 사용된다.*

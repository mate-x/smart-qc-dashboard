# 00. Global Context Document — Single Source of Truth

> **프로젝트명**: 제조산업 품질검사를 위한 딥러닝 기반 비전검사 시스템 v2.0
> **버전**: v2.0
> **작성일**: 2026-05-08
> **최종수정**: 2026-06-11
> **목적**: 이 문서는 이후 생성되는 01~15번 PRD 파일 전체의 유일한 기준점이다. 모든 파일은 이 문서에 정의된 스키마, 계약, 용어, 규칙을 그대로 따른다. 변경은 이 문서를 먼저 수정한 후 전파한다.

<!-- v2.0: 버전 히스토리 추가 — 3개 레포 구조 확정 및 FastAPI 공식화 반영 -->
## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-08 | 최초 작성 — Streamlit 단독 앱, 모델 탐색 대시보드 전용 |
| v1.1 | 2026-05-26 | 이중 대시보드 구조 확장 — 비전검사 대시보드(탭1~3) 추가; 사이드바 전환 버튼 도입 |
| v2.0 | 2026-06-11 | ADR-01 폐기 및 FastAPI 공식화; React UI를 공식 UI로 확정; 3개 레포 구조(Explorer/Vision/Dashboard) 반영; [3.2] session_state 계약 → Zustand + FastAPI 계약으로 교체; [5.1] 컴포넌트 구성 전면 교체; ADR 섹션 신규 추가 |

---

> **v1.1 변경 요약 (참고)**: 단일 대시보드(모델 탐색 전용)에서 **이중 대시보드 구조**로 확장.
> - **모델 탐색 대시보드** (기존 유지): AI/ML 엔지니어용, 탭1~5, 학습·실험·비교
> - **비전검사 대시보드** (신규): 현장 작업자/관리자용, 탭1~3, 추론·검사 이력·모델 교체
> - 사이드바: 데이터셋·디바이스 표시 제거 → 대시보드 전환 버튼 2개로 교체

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
11. [Architecture Decision Records (ADR)](#11-architecture-decision-records-adr)

> **문서 범위 안내 (v2.0)**: 이 문서는 smart-qc-explorer / smart-qc-vision / smart-qc-dashboard 3개 레포의 공통 단일 기준점이다.
> 1.1~1.9절 및 3.x절은 **모델 탐색(Explorer)** 전용 스키마다.
> 1.10~1.11절은 **비전검사(Vision)** 전용 스키마다.

---

## 1. Core Data Model

> **설계 원칙**: 이 시스템은 MySQL 8.0과 파일시스템을 병행한다. 모델 가중치(`.pth`), 학습 로그(`.log`), 공유 설정(`configs.yaml`)은 파일시스템에 저장하고, MySQL은 인프라 레이어에 포함되어 향후 구조화 데이터 저장에 활용된다. 아래 스키마는 Python dataclass 및 파일 직렬화 구조를 SQL 형식으로 표현한 논리 모델이다. 실제 구현 시 아래 필드명, 타입, 제약조건을 그대로 사용한다.

---

### 1.1 experiment (실험 레코드)

<!-- v2.0: TypeScript types/experiments.ts 와 대조하여 불일치 필드에 [확인 필요] 표기 -->

```
파일 위치: ./experiments/history.json  (배열 형태로 저장)

필드명                   타입        Nullable   제약조건
─────────────────────────────────────────────────────────────────
experiment_id           STRING      NOT NULL   PK, 형식: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_난수}
                                               예: efficientad_20260508_140023_7f3a
name                    STRING      NOT NULL   사용자 입력 또는 자동 생성, 최대 64자
status                  ENUM        NOT NULL   값: "completed" | "중단"
                                               [확인 필요: Explorer types/experiments.ts에
                                               "실패" | "건너뜀" 추가됨 — 공식 ENUM 확정 필요]
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
product_name            STRING      NULLABLE   [확인 필요: Explorer types/experiments.ts에 존재 — PRD 미정의 필드]
set_id                  STRING      NULLABLE   [확인 필요: Explorer types/experiments.ts에 존재 — 배치 비교용 그룹 ID]
background_method       ENUM        NULLABLE   [확인 필요: Explorer types/experiments.ts에 "none" | "sam2" 존재 — PRD 미정의 필드]
early_stopped           BOOLEAN     NULLABLE   [확인 필요: Explorer types/experiments.ts에 존재 — PRD 미정의 필드]
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
train_steps              INTEGER   NOT NULL   1 ~ 200000 / 기본값: 70000
optimizer                ENUM      NOT NULL   "adam" | "adamw" | "sgd" / 기본값: "adam"
learning_rate            FLOAT     NOT NULL   1e-6 ~ 1e-1 / 기본값: 0.0001
weight_decay             FLOAT     NOT NULL   0.0 ~ 0.1 / 기본값: 0.0001
out_channels             INTEGER   NOT NULL   128 | 256 | 384 | 512 / 기본값: 384
padding                  BOOLEAN   NOT NULL   기본값: false
ae_loss_weight           FLOAT     NOT NULL   0.0 ~ 1.0 (α, ST 비중은 학습 루프에서 1-α 자동 적용) / 기본값: 0.5
autoencoder_lr           FLOAT     NOT NULL   1e-6 ~ 1e-1 / 기본값: 0.0001
autoencoder_weight_decay FLOAT     NOT NULL   0.0 ~ 0.1 / 기본값: 0.00001
lr_decay_epochs          INTEGER   NOT NULL   1000 ~ 200000 / 기본값: 50000
lr_decay_factor          FLOAT     NOT NULL   0.01 ~ 1.0 / 기본값: 0.1
scheduler                ENUM      NOT NULL   "StepLR" | "CosineAnnealingLR" / 기본값: "StepLR"
use_imagenet_penalty     BOOLEAN   NOT NULL   기본값: false
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

<!-- v2.0: Explorer types/dataset.ts (DatasetValidateResponse)와 대조하여 불일치 필드에 [확인 필요] 표기 -->

```
session_state.dataset_meta 필드 구조 (v1.x 참고)
Explorer v2.0 기준: FastAPI POST /api/dataset/validate 응답 스키마 (DatasetValidateResponse)

필드명              타입      Nullable   제약조건
─────────────────────────────────────────────────────────────────────
dataset_path        STRING    NOT NULL   검증된 절대경로
dataset_format      ENUM      NOT NULL   "mvtec" | "oking"
train_good_count    INTEGER   NOT NULL   학습에 사용되는 정상 이미지 수
                                          MVTec: train/good/ 이미지 수
                                          OK/NG: OK 이미지의 앞 80% 수
test_counts         OBJECT    NOT NULL   {"good": int, "<defect_class>": int, ...}
                                          OK/NG: {"good": OK 뒤 20%, "<ng_key>": NG 전체}
gt_counts           OBJECT    NOT NULL   {"<defect_class>": int, ...}
                                          OK/NG 형식은 항상 {} (GT 마스크 없음)
total_test_count    INTEGER   NOT NULL   test_counts 합계
channels            INTEGER   NOT NULL   1 (Grayscale) | 3 (RGB)
defect_classes      ARRAY     NOT NULL   결함 클래스 이름 문자열 배열
supported_formats   ARRAY     NOT NULL   실제 발견된 포맷 배열, 예: [".jpg", ".png"]
has_invalid_files   BOOLEAN   NOT NULL   지원 포맷 외 파일 존재 여부
invalid_file_count  INTEGER   NOT NULL   [확인 필요: Explorer dataset.ts에 추가됨 — PRD 미정의]
folder_tree         STRING    NOT NULL   [확인 필요: Explorer dataset.ts에 추가됨 — 폴더 트리 문자열]
has_background_clean BOOLEAN  NOT NULL   [확인 필요: Explorer dataset.ts에 추가됨 — background_clean/ 존재 여부]

─── OK/NG 형식 전용 필드 (dataset_format == "oking" 일 때만 존재) ───
oking_ok_dir        STRING    NULLABLE   OK 계열 폴더명 (예: "OK", "good")
                                          [확인 필요: Explorer에서 _oking_ok_dir → oking_ok_dir 로 접두사 변경됨]
oking_ng_dir        STRING    NULLABLE   NG 계열 폴더명. NG 없으면 None
oking_ok_count      INTEGER   NULLABLE   OK 폴더 전체 이미지 수
oking_ng_count      INTEGER   NULLABLE   NG 폴더 이미지 수 (없으면 0)
train_ratio         FLOAT     NULLABLE   학습/테스트 분할 비율 (기본값: 0.8)
```

**OK 계열 폴더 인식 별칭**: `ok`, `good`, `normal`, `pass`, `neg` (대소문자 무관)
**NG 계열 폴더 인식 별칭**: `ng`, `bad`, `defect`, `fail`, `abnormal`, `anomaly`, `pos` (대소문자 무관)

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

### 1.10 inspection_record (검사 레코드)

<!-- v2.0: Vision types/inspection.ts (InspectionResult / InspectionRecord)와 대조하여 불일치 필드에 [확인 필요] 표기 -->

```
저장 위치 (v1.x): session_state.insp_records (세션 기반)
저장 위치 (v2.0): FastAPI 서버 메모리 (GET /api/inspection/records 로 조회)
                  모델 교체 시 초기화 (DELETE /api/inspection/records)

필드명            타입       Nullable   제약조건
─────────────────────────────────────────────────────────────────
seq               INTEGER    NOT NULL   PK, 세션 내 순번 (1부터 자동 증가)
inspected_at      STRING     NOT NULL   ISO 8601, KST. 예: "2026-05-26T14:02:31+09:00"
image_name        STRING     NOT NULL   이미지 파일명 (경로 제외). 예: "crack_001.png"
image_path        STRING     NOT NULL   절대경로. 추론·Anomaly Map 재생성에 사용
                                         [확인 필요: InspectionRecord에는 image_path 없음 — InspectionResult에만 존재]
verdict           ENUM       NOT NULL   값: "양품" | "불량"
anomaly_score     FLOAT      NOT NULL   round(value, 4). 모델 출력 raw score
anomaly_map_cache STRING     NULLABLE   캐시된 Anomaly Map numpy 배열 (v1.x session_state 내 별도 관리)
                                         [확인 필요: Vision에서 이미지 서빙은 /api/inspection/anomaly-map/last 엔드포인트로 대체됨]
was_reshuffled    BOOLEAN    NOT NULL   [확인 필요: Vision types/inspection.ts에 추가됨 — test pool 재셔플 여부]
```

---

### 1.11 INSPECTION_SESSION_SCHEMA (비전검사 세션 스키마 — v1.x 참고)

> **v2.0 안내**: 아래는 v1.x Streamlit 기반 세션 스키마다. v2.0에서는 Vision React가 FastAPI 서버 상태를 API로 조회한다. Streamlit 내부 구현 참고용으로 유지한다.

```python
# inspection/utils/insp_session_init.py

INSPECTION_SESSION_SCHEMA = {
    # 대시보드 전환
    "active_dashboard":      "explorer",   # "explorer" | "inspection"

    # 비전검사 대시보드 — 적용 모델
    "insp_active_model":     None,         # dict | None — history.json의 experiment 레코드 전체
                                           # None이면 탭1 검사 버튼 비활성화

    # 비전검사 대시보드 — 검사 이력
    "insp_records":          [],           # list[dict] — inspection_record 배열 (1.10절)
    "insp_seq_counter":      0,            # int — 다음 seq 값

    # 비전검사 대시보드 — 자동 검사 상태
    "insp_auto_active":      False,        # bool — 자동 검사(3초마다) 실행 중 여부

    # 비전검사 대시보드 — 마지막 추론 결과 (탭1 화면 유지용)
    "insp_last_result":      None,         # dict | None — 직전 추론 결과 전체

    # 비전검사 대시보드 — 팝업 제어
    "insp_defect_popup":     False,        # bool — 불량 감지 팝업 표시 여부

    # 비전검사 대시보드 — 테스트 이미지 풀
    "insp_test_pool":        [],           # list[tuple[str, str]]
    "insp_pool_index":       0,            # int — 현재 샘플링 위치
}
```

---

## 2. Entity Relationship

> 이 시스템은 MySQL 8.0을 인프라 레이어에 포함하며, 현재 앱 데이터는 파일시스템에 저장된다. 아래는 파일 간 참조 관계를 ERD로 표현한 것이다.

### 2.1 모델 탐색 대시보드 ERD (Explorer → FastAPI)

```
[Explorer Zustand stores]
     │
     ├── datasetStore  ──→  POST /api/dataset/validate  ──→  [dataset_meta]
     │
     ├── configStore   ──→  POST /api/config            ──→  [preprocessing_config / model_config]
     │
     ├── trainingStore ──→  POST /api/training/start    ──→  [experiments] ←→ history.json (영속)
     │                   WS  /ws/training (수신)               └── metrics / model_params
     │
     ├── experimentsStore ──→  GET /api/experiments     ──→  [selected_experiment_id]
     │
     └── anomalyMapStore  ──→  GET /api/anomaly-map/... ──→  [anomaly_map_threshold]

[experiments[exp_id]] ─── (1:1) ─── [./models/{exp_id}/model_state_dict.pth]
                      ─── (1:1) ─── [./models/{exp_id}/configs.yaml]
[history.json] ─── (1:N) ─── [experiment 레코드]
```

### 2.2 비전검사 대시보드 ERD (Vision → FastAPI)

```
[Vision Zustand store: inspectionStore]
     │
     ├── activeModel  ──→  GET  /api/inspection/model   ──→  history.json[experiment_id] 읽기 전용 참조
     │                     POST /api/inspection/model   (모델 교체)
     │                          └── experiment.model_path ──→ ./models/{exp_id}/model_state_dict.pth
     │
     ├── records      ──→  GET  /api/inspection/records
     │                     DELETE /api/inspection/records (이력 초기화)
     │
     ├── lastResult   ──→  POST /api/inspection/run     (수동 검사)
     │                     GET  /api/inspection/image/last
     │                     GET  /api/inspection/anomaly-map/last
     │                     GET  /api/inspection/overlay/last
     │
     └── autoActive   ──→  WS   /ws/inspection/auto     (자동 검사)
```

### 참조 무결성 규칙

| 규칙 | 설명 |
|------|------|
| R-01 | `selected_experiment_id`는 반드시 `experiments` 배열 내 존재하는 `experiment_id`를 참조한다. |
| R-02 | `model_config.image_size`는 항상 `preprocessing_config.image_size`와 동일한 값으로 자동 동기화된다. |
| R-04 | `experiment.model_path`가 NOT NULL인 경우, 해당 경로에 `model_state_dict.pth`와 `configs.yaml`이 반드시 존재한다. |
| R-05 | `status == "중단"`인 레코드의 `metrics`, `model_path`, `configs_path`는 반드시 NULL이다. |
| R-06 | `activeModel`은 반드시 `history.json` 내 `status == "completed"`인 레코드를 참조한다. |
| R-07 | `activeModel`이 None인 경우, `records`, `lastResult`는 모두 빈값/None이어야 한다. |
| R-08 | 모델 교체 시 Vision 서버는 records, lastResult 를 즉시 초기화한다. |

---

## 3. Global State Contract Standard

<!-- v2.0: v2.0 기준 설명으로 교체 — FastAPI + React 아키텍처 반영 -->
> **v2.0 기준**: 이 시스템은 FastAPI REST API + WebSocket을 통해 React 프론트엔드와 통신한다. "상태 계약"에 해당하는 것은 **FastAPI API 계약**, **Zustand store 인터페이스**, **파일 I/O 계약**이다.

---

### 3.1 Explorer Zustand Store 초기화 명세

```typescript
// Explorer: src/store/ 하위 각 store의 초기 상태

// datasetStore.ts
{ datasetPath: null, productName: '', datasetMeta: null }

// configStore.ts
{ preprocessingConfig: null, modelConfig: null, deviceInfo: null, queueItems: [] }

// trainingStore.ts
{ status: 'idle', progress: null, lossHistory: [], logs: [] }

// experimentsStore.ts
{ selectedExperimentId: null }

// anomalyMapStore.ts
{ threshold: null }
```

---

<!-- v2.0: [3.2] session_state 기반 탭 간 데이터 흐름 계약을 Zustand + FastAPI 계약으로 교체 -->
### 3.2 화면 간 데이터 흐름 계약 (v2.0 — Zustand + FastAPI)

#### Explorer 화면 간 데이터 흐름

| Store 키 | Write 화면 | Read 화면 | 타입 | NULL 처리 |
|----------|-----------|-----------|------|-----------|
| `datasetPath` | Dataset 화면 | Config 화면, Training 화면 | `string \| null` | null이면 Config/Training 화면 진입 차단 + 안내 메시지 |
| `datasetMeta` | Dataset 화면 (API 응답) | Config 화면 | `DatasetValidateResponse \| null` | null이면 Config 전처리 미리보기 비활성화 |
| `preprocessingConfig` | Config 화면 | Training 화면 | `PreprocessingConfig \| null` | null이면 Training 화면 진입 차단 |
| `modelConfig` | Config 화면 | Training 화면 | `ModelConfig \| null` | null이면 Training 화면 진입 차단 |
| `deviceInfo` | Config 화면 (API 응답) | Training 화면 | `DeviceInfo \| null` | null이면 CPU fallback |
| `selectedExperimentId` | Experiments 화면 | AnomalyMap 화면 | `string \| null` | null이면 AnomalyMap 화면 안내 메시지 |
| `threshold` | AnomalyMap 화면 | AnomalyMap 화면 내부 | `number \| null` | null이면 experiment.threshold_value 사용 |

**Explorer 탭 간 흐름 요약**:
```
Dataset 화면 → datasetStore 업데이트 → POST /api/dataset/validate
  ↓ datasetPath, datasetMeta 확정
Config 화면 → configStore 업데이트 → POST /api/config
  ↓ preprocessingConfig, modelConfig 확정
Training 화면 → trainingStore 업데이트 → POST /api/training/start
  ↓ WS /ws/training 실시간 수신 → status, progress, lossHistory, logs 갱신
Experiments 화면 → experimentsStore 업데이트 → GET /api/experiments
  ↓ selectedExperimentId 확정
AnomalyMap 화면 → anomalyMapStore 업데이트 → POST /api/anomaly-map/{expId}/build
```

#### Vision 화면 간 데이터 흐름

| Store 키 | Write 화면 | Read 화면 | 타입 | NULL 처리 |
|----------|-----------|-----------|------|-----------|
| `activeModel` | Model Settings 화면 (POST) | Realtime Inspection 화면 | `ActiveModel \| null` | null이면 검사 버튼 비활성화 |
| `records` | Realtime Inspection 화면 (추론 시) | History 화면 | `InspectionRecord[]` | 빈 배열이면 "검사 이력이 없습니다." |
| `lastResult` | Realtime Inspection 화면 (추론 시) | Realtime Inspection 화면 | `InspectionResult \| null` | null이면 결과 영역 미렌더링 |
| `autoActive` | Realtime Inspection 화면 | Realtime Inspection 화면 | `boolean` | false이면 자동 검사 버튼 활성 상태 |

**Vision 화면 간 흐름 요약**:
```
Model Settings 화면 → GET /api/models (30초 폴링) → 완료 실험 목록
  → POST /api/inspection/model → activeModel 확정 → 이력 자동 초기화
Realtime Inspection 화면
  수동 검사: POST /api/inspection/run → lastResult 업데이트
  자동 검사: WS /ws/inspection/auto → result/defect_stopped/stopped 수신
History 화면 → GET /api/inspection/records → records 표시
```

---

### 3.2.x v1.x 구현 참고 — session_state 탭 간 데이터 흐름 계약

> **v1.x 참고 전용**: 아래는 v1.x Streamlit 기반 session_state 계약이다. v2.0 React/FastAPI 계약은 3.2절을 참조한다. 삭제하지 않고 Streamlit 내부 구현 참고용으로 보존한다.

| 키 | Write 탭 | Read 탭 | 타입 | NULL 처리 |
|----|----------|---------|------|-----------|
| `dataset_path` | 탭1 | 탭2, 탭3 | `str \| None` | None이면 해당 탭 진입 차단 + 안내 메시지 |
| `dataset_meta` | 탭1 | 탭2 | `dict \| None` | None이면 탭2 전처리 미리보기 비활성화 |
| `preprocessing_config` | 탭2(image_size) | 탭3 | `dict \| None` | None이면 탭3 진입 차단 + 안내 메시지 |
| `model_config` | 탭2 | 탭3 | `dict \| None` | None이면 탭3 진입 차단 + 안내 메시지 |
| `device_info` | 탭2 | 탭3 | `dict \| None` | None이면 CPU fallback |
| `experiments` | 탭3 | 탭4, 탭5 | `dict` | 빈 dict이면 탭4/탭5 안내 메시지 |
| `current_run_status` | 탭3 | 탭3 UI | `str` | 항상 유효한 ENUM 값 (`"idle"` \| `"running"` \| `"paused"` \| `"stopped"` \| `"completed"`) |
| `selected_experiment_id` | 탭4 | 탭5 | `str \| None` | None이면 탭5 안내 메시지 |
| `anomaly_map_threshold` | 탭5 | 탭5 내부 | `float \| None` | None이면 model_config.threshold_value 사용 |

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

**history.json 접근 권한 (v2.0 확정)**:
- Explorer (smart-qc-explorer): `save_history()` 호출 가능 — **쓰기 전용**
- Vision (smart-qc-vision): `load_history()` 호출만 가능 — **읽기 전용**
- R-INSP-04 유지: Vision은 `save_history()` 호출 금지

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

# 모델 탐색 대시보드 메시지 (v1.x Streamlit 기준 — React에서는 컴포넌트 내 직접 정의)
MSG = {
    "NO_DATASET":       "먼저 Dataset 화면에서 데이터 폴더를 설정해 주세요.",
    "NO_PREPROCESSING": "먼저 Config 화면에서 전처리 설정을 완료해 주세요.",
    "NO_MODEL_CONFIG":  "먼저 Config 화면에서 모델 파라미터를 설정해 주세요.",
    "NO_EXPERIMENTS":   "아직 실행된 실험이 없습니다. Training 화면에서 학습을 먼저 실행해 주세요.",
    "NO_SELECTED_EXP":  "Experiments 화면에서 분석할 실험을 먼저 선택해 주세요.",
    "GRAYSCALE_DETECT": "Grayscale 이미지가 감지되었습니다. 모델 입력을 위해 RGB 3채널로 자동 변환됩니다.",
    "INVALID_FOLDER":   "MVTec AD 형식의 폴더 구조가 아닙니다. (필수: train/good/, test/, ground_truth/)",
    "TRAIN_STOPPED":    "학습이 중단되었습니다. 해당 실험은 '중단' 상태로 히스토리에 기록되었습니다.",
}

# 비전검사 대시보드 메시지 (v1.x Streamlit 기준 — React에서는 컴포넌트 내 직접 정의)
INSP_MSG = {
    "NO_MODEL":         "선택된 모델이 없습니다. Model Settings 화면에서 모델을 선택한 후 검사를 시작해 주세요.",
    "NO_COMPLETED_EXP": "적용 가능한 완료 실험이 없습니다. Explorer에서 학습을 완료해 주세요.",
    "DEFECT_DETECTED":  "불량이 감지되었습니다. 해당 부품을 라인에서 제거하고 확인해 주세요.",
    "MODEL_REPLACED":   "모델이 교체되었습니다. 검사 이력이 초기화되었습니다.",
    "HISTORY_CLEARED":  "검사 이력이 초기화되었습니다.",
    "AUTO_STOPPED":     "불량 감지로 자동 검사가 중지되었습니다. 확인 후 검사를 재시작해 주세요.",
    "POOL_RESHUFFLED":  "테스트 이미지 풀을 모두 소진하여 재구성했습니다.",
}
```

---

### 3.5 오류 코드 레지스트리

**모델 탐색 (Explorer / FastAPI)**

| 코드 | 설명 | 발생 조건 |
|------|------|-----------|
| `ERR_DATASET_NOT_FOUND` | 지정 경로가 존재하지 않음 | Dataset 화면 경로 검증 실패 |
| `ERR_INVALID_FOLDER_STRUCTURE` | MVTec AD 구조 미충족 | `train/good/` 또는 `test/` 미존재 |
| `ERR_NO_VALID_IMAGES` | 지원 포맷 이미지 없음 | `.jpg/.png/.bmp` 외 파일만 존재 |
| `ERR_PREPROCESSING_CONFIG_MISSING` | 전처리 설정 없음 | Config 미완료 상태에서 Training 접근 |
| `ERR_MODEL_CONFIG_MISSING` | 모델 설정 없음 | Config 미완료 상태에서 Training 접근 |
| `ERR_MODEL_INIT_FAILED` | 모델 초기화 실패 | CUDA OOM 또는 Anomalib 오류 |
| `ERR_TRAINING_INTERRUPTED` | 학습 강제 중단 | 사용자 [학습 중지] 클릭 |
| `ERR_CHECKPOINT_SAVE_FAILED` | 체크포인트 저장 실패 | 디스크 공간 부족 또는 권한 없음 |
| `ERR_CHECKPOINT_LOAD_FAILED` | 체크포인트 로드 실패 | 파일 손상 또는 호환 불가 |
| `ERR_CONFIG_LOAD_FAILED` | YAML 파싱 실패 | 잘못된 configs.yaml 형식 |
| `ERR_MODEL_SAVE_FAILED` | 모델 저장 실패 | 디스크 공간 부족 또는 권한 없음 |
| `ERR_EXPERIMENT_NOT_FOUND` | 실험 ID 미존재 | 삭제된 실험에 접근 시 |
| `ERR_INVALID_PARAM_RANGE` | 파라미터 범위 초과 | 1.4절 범위 벗어난 값 입력 |

**비전검사 (Vision / FastAPI)**

| 코드 | 설명 | 발생 조건 |
|------|------|-----------|
| `ERR_INSP_NO_MODEL` | 적용 모델 미선택 | Realtime Inspection 화면에서 검사 버튼 클릭 시 activeModel == null |
| `ERR_INSP_MODEL_LOAD_FAILED` | 모델 로드 실패 | model_path 파일 없거나 손상 |
| `ERR_INSP_TEST_POOL_EMPTY` | 테스트 이미지 없음 | dataset_path/test/ 하위 이미지 0개 |
| `ERR_INSP_INFERENCE_FAILED` | 추론 실패 | 모델 오류 또는 이미지 손상 |
| `ERR_INSP_MODEL_NOT_COMPLETED` | 완료되지 않은 실험 | status != "completed" 실험 교체 시도 |

---

### 3.6 비전검사 세션 상태 초기화 명세 (v1.x 참고 — Streamlit 내부 한정)

> **v1.x 참고 전용**: 아래는 Streamlit 내부 session_state 초기화 코드다. v2.0에서는 FastAPI 서버가 상태를 관리하며 Vision React가 API로 조회한다.

```python
# inspection/utils/insp_session_init.py

def init_inspection_session():
    """비전검사 대시보드 전용 session_state 키 초기화. app.py 시작 시 1회 실행. (Streamlit 내부 한정)"""
    defaults = {k: v for k, v in INSPECTION_SESSION_SCHEMA.items()}
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def reset_inspection_state():
    """모델 교체 또는 이력 초기화 시 호출. insp_active_model은 유지. (Streamlit 내부 한정)"""
    st.session_state.insp_records       = []
    st.session_state.insp_seq_counter   = 0
    st.session_state.insp_auto_active   = False
    st.session_state.insp_last_result   = None
    st.session_state.insp_defect_popup  = False
    st.session_state.insp_test_pool     = []
    st.session_state.insp_pool_index    = 0
```

---

### 3.7 비전검사 탭 간 데이터 흐름 계약 (v1.x 참고 — Streamlit 내부 한정)

> **v1.x 참고 전용**: v2.0 Vision 화면 간 데이터 흐름은 3.2절을 참조한다.

| 키 | Write 탭 | Read 탭 | 타입 | NULL/빈값 처리 |
|----|----------|---------|------|----------------|
| `insp_active_model` | 탭3 | 탭1, 탭2 | `dict \| None` | None이면 탭1 검사 버튼 비활성화 + INSP_MSG["NO_MODEL"] |
| `insp_records` | 탭1 (추론 시) | 탭2 | `list[dict]` | 빈 list이면 탭2에 "아직 검사 이력이 없습니다." |
| `insp_auto_active` | 탭1 | 탭1 | `bool` | False이면 자동 검사 버튼 활성 상태 |
| `insp_last_result` | 탭1 (추론 시) | 탭1 | `dict \| None` | None이면 탭1에 결과 영역 미렌더링 |
| `insp_defect_popup` | 탭1 (불량 감지 시) | 탭1 | `bool` | True이면 st.dialog 팝업 렌더링 |
| `insp_test_pool` | 탭1 (모델 선택 시) | 탭1 | `list[tuple]` | 빈 list이면 ERR_INSP_TEST_POOL_EMPTY |
| `insp_pool_index` | 탭1 (매 검사 시) | 탭1 | `int` | len(pool) 도달 시 재셔플 후 0 리셋 |

---

### 3.8 비전검사 대시보드 — 검사 실행 흐름 계약 (v1.x 참고 — Streamlit 내부 한정)

> **v1.x 참고 전용**: v2.0 Vision 검사 흐름은 `POST /api/inspection/run` 및 `WS /ws/inspection/auto` 기준으로 동작한다.

```
[수동 검사 (1개 검사) 클릭] — v1.x Streamlit 흐름
  1. insp_active_model None 체크 → None이면 st.warning + 중단
  2. insp_test_pool[insp_pool_index] → (image_path, label) 샘플링
  3. insp_pool_index += 1. len(pool) 초과 시 random.shuffle(pool) + index = 0
  4. image_utils.apply_preprocessing(image_path, preprocessing_params) → tensor
  5. model_factory.run_inference(model, tensor) → {anomaly_score, anomaly_map}
  6. score > threshold → verdict = "불량", else → verdict = "양품"
  7. inspection_record 생성 → insp_records.append + insp_seq_counter += 1
  8. insp_last_result 갱신
  9. verdict == "불량" → insp_defect_popup = True + insp_auto_active = False → st.rerun()

[자동 검사 (3초마다 1개) 클릭] — v1.x Streamlit 흐름
  1. insp_auto_active = True → st.rerun()
  2. 탭1 렌더링 시 insp_auto_active == True 감지
  3. 수동 검사 흐름(1~8) 실행
  4. verdict == "불량" → insp_defect_popup = True + insp_auto_active = False → st.rerun()
  5. verdict == "양품" → time.sleep(3) → st.rerun()
     [확인 필요: Vision WebSocket 대체 여부 — v2.0에서 WS /ws/inspection/auto로 대체됐는지 확인]
```

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
| **session_state** | Streamlit 재실행 간 데이터를 보존하는 딕셔너리. **v2.0: Streamlit 내부 구현 한정**. |
| **Zustand store** | Explorer/Vision React 앱의 전역 상태 관리. v2.0 기준 상태 계약의 핵심. |
| **history.json** | 모든 실험 레코드를 배열로 저장하는 영속 파일. `./experiments/history.json`. Explorer 쓰기 전용 / Vision 읽기 전용. |
| **configs.yaml** | 전처리+모델 파라미터 통합 설정 파일. Explorer Config 화면에서 해당 섹션 업데이트. |
| **model_state_dict** | PyTorch `torch.save(model.state_dict(), ...)` 형식으로 저장된 모델 가중치 파일. |
| **완료 상태** | `status == "completed"`. 학습이 정상 종료되고 metrics, model_path, configs_path 모두 존재하는 상태. |
| **중단 상태** | `status == "중단"`. 사용자가 [학습 중지]를 클릭하여 중단된 상태. metrics, model_path, configs_path 모두 NULL. |
| **일시정지 상태** | `current_run_status == "paused"`. 사용자가 [⏸ 일시정지]를 클릭하여 체크포인트 저장 후 학습 스레드가 대기 중인 상태. history.json에는 기록되지 않으며 [▶ 재시작]으로 학습을 재개할 수 있다. |
| **체크포인트** | 일시정지 시 저장되는 중간 학습 상태 파일. `./models/checkpoints/{exp_id}_step{N}.ckpt`. |
| **결함 클래스** | MVTec AD `test/` 하위 디렉토리명. `good`은 결함 없음. 그 외(crack, scratch 등)는 결함. OK/NG 형식에서는 NG 폴더명이 유일한 결함 클래스. |
| **OK/NG 형식** | `OK/`(또는 `good/`, `normal/` 등) + `NG/`(또는 `bad/`, `defect/` 등) 구조의 단순 이진 분류 데이터셋. |
| **dataset_format** | `dataset_meta`의 필드. `"mvtec"` (MVTec AD 형식) 또는 `"oking"` (OK/NG 폴더 형식). |
| **OK/NG 자동 분할** | OK 이미지를 고정 시드(random_seed)로 셔플 후 80%는 학습, 20%는 테스트(정상)로 사용. NG 이미지는 전부 테스트(결함). |
| **지원 이미지 포맷** | `.jpg`, `.png`, `.bmp` 세 가지. 확장자 대소문자 구분 없이 처리. |
| **Resize+Padding** | 원본 비율 유지하여 `image_size×image_size`로 리사이즈 후 부족한 영역을 검정(0)으로 패딩. `resize_mode`는 항상 "padding"으로 고정. |
| **Explorer (smart-qc-explorer)** | AI/ML 엔지니어용 React 프론트엔드. EfficientAD/PatchCore 학습·비교·자산화 전용. v2.0 공식 모델 탐색 UI. |
| **Vision (smart-qc-vision)** | 현장 작업자/관리자용 React 프론트엔드. 검증된 모델로 검사·판정·이력 관리. v2.0 공식 비전검사 UI. |
| **Dashboard (smart-qc-dashboard)** | FastAPI 백엔드 + Anomalib ML 레이어. REST API + WebSocket 제공. Streamlit은 개발자 디버그용 보조 도구. |
| **inspection_record** | 비전검사 1회 추론 결과 레코드. v2.0: FastAPI 서버 메모리에 저장. 1.10절 스키마. |
| **test_pool** | 비전검사에서 추론에 사용하는 이미지 목록. 적용 모델의 `dataset_path/test/` 스캔 결과. |
| **verdict (판정)** | 비전검사 추론 결과 이진 분류. `anomaly_score > threshold_value` → "불량", 이하 → "양품". |
| **수동 검사** | 사용자가 버튼을 클릭할 때마다 `POST /api/inspection/run` 으로 추론 1회 실행. |
| **자동 검사** | `WS /ws/inspection/auto` 를 통해 서버 주도로 자동 추론. 불량 감지 시 `defect_stopped` 이벤트 전송. |
| **불량 팝업** | 불량 판정 시 자동 표시되는 모달 알림. 확인 버튼 클릭 전까지 검사 재개 불가. |

---

## 5. System Architecture

<!-- v2.0: [5.1 컴포넌트 구성] Streamlit 단독 구조 → 3개 레포 구조로 전면 교체 -->
### 5.1 컴포넌트 구성 (v2.0 — 3개 레포 구조)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         사용자 브라우저                                        │
│                                                                               │
│  ┌──────────────────────────────┐   ┌──────────────────────────────────────┐  │
│  │  smart-qc-explorer           │   │  smart-qc-vision                     │  │
│  │  React 19 + Vite + TS        │   │  React + Vite + TS                   │  │
│  │  localhost:5173               │   │  localhost:5173                      │  │
│  │                               │   │                                      │  │
│  │  Dataset 화면     (/          │   │  Realtime Inspection  (/             │  │
│  │  Config 화면      (/config)   │   │  History              (/history)     │  │
│  │  Training 화면    (/training) │   │  Model Settings       (/models)      │  │
│  │  Experiments      (/experiments)  │                                      │  │
│  │  AnomalyMap       (/anomaly-map)  │                                      │  │
│  │                               │   │                                      │  │
│  │  Zustand v5 스토어            │   │  Zustand 스토어                      │  │
│  │  (dataset/config/training/    │   │  (inspectionStore)                   │  │
│  │   experiments/anomalyMap)     │   │                                      │  │
│  └──────────┬───────────────────┘   └──────────────┬───────────────────────┘  │
│             │ HTTP REST / WebSocket                  │ HTTP REST / WebSocket    │
└─────────────┼──────────────────────────────────────┼─────────────────────────┘
              │                                        │
┌─────────────▼────────────────────────────────────────▼─────────────────────┐
│                    smart-qc-dashboard (FastAPI :8000)                        │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  api/ FastAPI 라우터 레이어                                              │  │
│  │                                                                          │  │
│  │  /api/dataset/*    /api/config/*    /api/training/*                     │  │
│  │  /api/experiments/*  /api/anomaly-map/*                                 │  │
│  │  /api/models  /api/inspection/*                                         │  │
│  │  /ws/training   /ws/inspection/auto  (WebSocket)                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  ML 레이어 (Anomalib / PyTorch)                                         │  │
│  │  EfficientAD Engine  │  PatchCore Engine  │  Evaluator                  │  │
│  │  run_inference()  ← Vision 자동/수동 검사에서 호출                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  [보조] Streamlit UI (개발자 디버그용 — 공식 UI 아님)                    │  │
│  │  streamlit run app.py  →  localhost:8501                                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└────────────────────────────┬──────────────────────────────────────────────────┘
                              │ 파일시스템 접근
┌─────────────────────────────▼──────────────────────────────────────────────┐
│                          파일시스템 레이어                                    │
│                                                                               │
│  ./dataset/            데이터셋 (읽기 전용 — Explorer/Vision 공유)           │
│  ./experiments/        history.json (Explorer 쓰기 / Vision 읽기 전용)      │
│  ./models/{exp_id}/    model_state_dict.pth, configs.yaml                   │
│  ./logs/{exp_id}.log   학습 로그 (Explorer 전용)                             │
│  ./configs.yaml        공유 설정 파일 (Explorer Config 화면 R/W)             │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### 5.1.x v1.x 참고 — Streamlit 단독 컴포넌트 구성

> **v1.x 참고 전용**: 아래는 v1.x Streamlit 단독 구조다. v2.0 공식 구조는 5.1절을 참조한다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      브라우저 (localhost:8501)                        │
│                   Streamlit Web UI (Dual Dashboard)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (Streamlit WebSocket)
┌──────────────────────────────▼──────────────────────────────────────┐
│                     app.py (Streamlit 진입점 · 라우터)                │
│  ┌────────────────┐                                                   │
│  │  sidebar.py    │  [모델 탐색 대시보드] [비전검사 대시보드]          │
│  └────────────────┘  → active_dashboard 값에 따라 렌더링 분기         │
│                                                                       │
│  ┌──────────────────────────┐  ┌─────────────────────────────────┐  │
│  │  모델 탐색 대시보드       │  │  비전검사 대시보드                │  │
│  │  st.tabs([탭1~탭5])      │  │  inspection_app.py              │  │
│  │  tabs/tab1.py ~ tab5.py  │  │  st.tabs([탭1~탭3])             │  │
│  └──────────────────────────┘  └─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 5.2 디렉토리 구조

#### smart-qc-dashboard (백엔드)

```
smart-qc-dashboard/
├── api/                            # FastAPI 라우터 (v2.0 추가)
│   └── main.py                     # FastAPI 진입점 (uvicorn api.main:app --port 8000)
│
├── app.py                          # Streamlit 진입점 (개발자 보조 도구)
├── requirements.txt
├── Dockerfile
├── docker-compose.base.yml
├── docker-compose.yml
├── docker-compose.cpu.yml
├── .env
├── configs.yaml                    # 공유 설정 파일
│
├── tabs/                           # Streamlit 모델 탐색 탭 (보조 도구)
│   ├── tab1_data_folder.py
│   ├── tab2_config.py
│   ├── tab3_training.py
│   ├── tab4_history.py
│   └── tab5_anomaly_map.py
│
├── inspection/                     # Streamlit 비전검사 탭 (보조 도구)
│   ├── inspection_app.py
│   └── tabs/
│
├── utils/                          # 공유 유틸리티
│   ├── session_state_init.py
│   ├── config_manager.py
│   ├── messages.py
│   ├── metrics.py
│   ├── model_factory.py
│   ├── storage.py
│   ├── cache_manager.py
│   ├── checkpoint_manager.py
│   ├── dataset_converter.py
│   └── image_utils.py
│
├── experiments/
│   └── history.json
│
├── models/
│   └── {experiment_id}/
│       ├── model_state_dict.pth
│       └── configs.yaml
│
└── logs/
    └── {experiment_id}.log
```

#### smart-qc-explorer (Explorer 프론트엔드)

```
smart-qc-explorer/
├── src/
│   ├── main.tsx                    # React 진입점
│   ├── App.tsx                     # TabBar + Route
│   ├── pages/                      # Tab1Dataset / Tab2Config / Tab3Training / Tab4Experiments / Tab5AnomalyMap
│   ├── components/                 # layout / tab1 / config / training / tab4 / tab5
│   ├── hooks/
│   │   └── useTrainingWs.ts        # WebSocket 연결
│   ├── api/                        # client / datasetApi / configApi / trainingApi / experimentsApi / anomalyMapApi
│   ├── store/                      # datasetStore / configStore / trainingStore / experimentsStore / anomalyMapStore
│   └── types/                      # dataset / config / training / experiments / anomalyMap
├── package.json
└── vite.config.ts
```

#### smart-qc-vision (Vision 프론트엔드)

```
smart-qc-vision/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── pages/                      # Tab1Realtime / Tab2History / Tab3Setting
│   ├── components/                 # layout / tab1 / tab2 / tab3
│   ├── hooks/                      # useActiveModel / useModels / useApplyModel / useManualInspection / useAutoInspection / useInspectionImages / useInspectionRecords / useStatCharts
│   ├── api/                        # client / modelsApi / inspectionApi / recordsApi
│   ├── store/
│   │   └── inspectionStore.ts
│   └── types/                      # model / inspection / api
├── package.json
└── vite.config.ts
```

---

### 5.3 Explorer 화면별 데이터 흐름 단계

```
[Dataset 화면: 데이터 폴더]
  Step 1. 사용자 경로 입력
  Step 2. POST /api/dataset/validate → DatasetValidateResponse 수신
  Step 3. datasetStore.datasetPath, datasetMeta 업데이트
  Step 4. 클래스별 이미지 수 테이블, 썸네일 그리드 렌더링
  ↓
[Config 화면: 전처리 및 모델 설정]
  Step 1. datasetPath null 체크 → 차단 또는 통과
  Step 2. 전처리 방법 선택 (none/homomorphic/he/clahe)
  Step 3. 배경 제거 방법 선택 (none/sam2)
  Step 4. 모델 타입 선택 (EfficientAD/PatchCore)
  Step 5. 모델별 파라미터 UI 렌더링
  Step 6. POST /api/config → configStore 업데이트
  Step 7. 배치 학습 큐 관리 (POST/GET/DELETE /api/queue)
  ↓
[Training 화면: 학습]
  Step 1. preprocessingConfig, modelConfig null 체크 → 차단
  Step 2. POST /api/training/start (또는 /api/training/batch/start)
  Step 3. WS /ws/training 연결 → 실시간 진행상황 수신
  Step 4. trainingStore 업데이트 (status, progress, lossHistory, logs)
  Step 5. 학습 제어: pause/unpause/stop → POST /api/training/{action}
  ↓
[Experiments 화면: 실험 히스토리]
  Step 1. GET /api/experiments → 실험 목록 렌더링
  Step 2. 실험 선택 → experimentsStore.selectedExperimentId 업데이트
  Step 3. 상세 결과: Confusion Matrix, ROC Curve, Score 분포
  Step 4. POST /api/experiments/{id}/save → 모델 저장
  Step 5. DELETE /api/experiments/{id} → 실험 삭제
  ↓
[AnomalyMap 화면: 이상 영역 시각화]
  Step 1. selectedExperimentId null 체크 → 차단
  Step 2. GET /api/anomaly-map/{expId}/status → 캐시 여부 확인
  Step 3. POST /api/anomaly-map/{expId}/build → job 시작 (캐시 없을 때)
  Step 4. GET /api/anomaly-map/job/{jobId} 폴링 → 완료 대기
  Step 5. GET /api/anomaly-map/{expId}/images → 이미지 목록
  Step 6. Threshold 슬라이더 조정 → anomalyMapStore 업데이트
  Step 7. GET /api/anomaly-map/{expId}/image/{path}/triplet → Triplet 이미지
```

---

### 5.4 Vision 화면별 데이터 흐름

```
[Realtime Inspection 화면]
  진입 시:
    - GET /api/inspection/model → activeModel 확인
    - activeModel == null → "모델 없음" 안내, 검사 버튼 비활성

  [수동 검사] 클릭:
    → POST /api/inspection/run
    → GET /api/inspection/image/last, /anomaly-map/last, /overlay/last
    → lastResult 업데이트, 이미지 패널 갱신
    → verdict == "불량" → 불량 팝업 표시

  [자동 검사 시작] 클릭:
    → WS /ws/inspection/auto 연결
    → type=="result" 수신 → lastResult 갱신, 이미지 갱신
    → type=="defect_stopped" 수신 → 자동 중지 + 불량 팝업
    → type=="stopped" 수신 → 자동 중지

[History 화면]
  → GET /api/inspection/records → records 테이블 렌더링
  → KPI 카드: 총/양품/불량 수, 불량률
  → GET /api/inspection/records/csv → CSV 다운로드
  → DELETE /api/inspection/records → 이력 초기화

[Model Settings 화면]
  → GET /api/models (30초 폴링) → 완료 실험 목록
  → POST /api/inspection/model (모델 선택) → activeModel 갱신 + 이력 초기화
```

---

## 6. Global Non-Functional Requirements

> 이후 모든 파일에서 이 절을 참조하며, 강화가 필요한 경우 "6.X절 기준 강화: ..." 형식으로 명시한다.

| 항목 | 요구사항 | 측정 방법 |
|------|----------|-----------|
| **UI 응답성 (Explorer)** | 화면 전환 응답 < 1초 | 수동 테스트 |
| **UI 응답성 (Vision)** | 수동 검사 결과 표시 < 3초 | 수동 테스트 |
| **학습 성능 — EfficientAD** | g4dn.xlarge에서 EfficientAD-medium 70,000 steps **20분 이내** | 실측 측정 |
| **학습 성능 — PatchCore** | g4dn.xlarge에서 PatchCore (coreset 10%) **10분 이내** | 실측 측정 |
| **모델 정확도** | MVTec AD Screw 데이터셋 기준 AUC ≥ 0.95 | history.json metrics.auc |
| **디바이스 자동 감지** | `torch.cuda.is_available()` 결과에 따라 자동 전환, UI 명시 | 자동 감지 로직 |
| **재현성** | 동일 `random_seed` + 동일 하이퍼파라미터 시 동일 결과 | 2회 실행 결과 비교 |
| **데이터 무결성** | 폴더 구조 검증 실패 시 Training 화면 진입 완전 차단 | E2E 테스트 |
| **이미지 처리** | Resize+Padding 적용 후 가로·세로 비율 유지 오차 < 1픽셀 | 픽셀 단위 검증 |
| **채널 처리** | Grayscale 이미지 자동 RGB 변환, 3채널 보장 | 단위 테스트 |
| **디스크 용량** | 모델 저장 전 여유 공간 < 500MB 시 경고 메시지 표시 | `shutil.disk_usage()` |
| **WebSocket 학습 지연** | 학습 step 이벤트 → Explorer UI 갱신까지 **1초 이내** | 수동 테스트 |
| **추론 지연 (비전검사)** | 이미지 1장 추론 → 판정 결과 표시까지 **3초 이내** | 수동 테스트 |
| **자동 검사 타이밍 (비전검사)** | 서버 주도 3초 간격 오차 **0.5초 이내** | 수동 측정 |
| **불량 팝업 표시 지연 (비전검사)** | 불량 판정 후 팝업 표시까지 **0.5초 이내** | 수동 테스트 |

---

## 7. Observability Standards

### 7.1 로그 포맷 (JSON)

```python
# 모든 로그 엔트리는 아래 필드를 포함한다
LOG_FORMAT = {
    "timestamp": "ISO 8601",        # 예: "2026-05-08T14:00:23.456+09:00"
    "level": "INFO|WARNING|ERROR",
    "experiment_id": "str|null",    # 실험 컨텍스트가 없으면 null
    "screen": "dataset|config|training|experiments|anomaly-map|realtime|history|model-settings|null",
    "event": "이벤트명 (snake_case)",
    "message": "사람이 읽을 수 있는 설명",
    "data": {}                      # 이벤트별 추가 데이터 (선택)
}
```

### 7.2 필수 로그 이벤트

| 이벤트명 | 화면 | 레벨 | 설명 |
|---------|------|------|------|
| `dataset_validated` | dataset | INFO | 폴더 구조 검증 완료 |
| `dataset_validation_failed` | dataset | ERROR | 폴더 구조 검증 실패 |
| `preprocessing_config_saved` | config | INFO | 전처리 설정 저장 |
| `model_config_saved` | config | INFO | 모델 설정 저장 |
| `training_started` | training | INFO | 학습 시작 |
| `training_step` | training | INFO | 매 N step마다 (N=1000) → WS 전송 |
| `training_completed` | training | INFO | 학습 완료 + 소요 시간 |
| `training_stopped` | training | WARNING | 사용자 중단 |
| `training_failed` | training | ERROR | 예외 발생 + traceback |
| `model_saved` | experiments | INFO | 모델 파일 저장 완료 |
| `experiment_deleted` | experiments | WARNING | 실험 삭제 |
| `insp_model_applied` | model-settings | INFO | 비전검사 모델 교체 완료 |
| `insp_inspection_started_manual` | realtime | INFO | 수동 검사 1회 실행 |
| `insp_inspection_started_auto` | realtime | INFO | 자동 검사 WebSocket 연결 |
| `insp_inspection_stopped_auto` | realtime | INFO | 자동 검사 중지 (수동 또는 불량 감지) |
| `insp_defect_detected` | realtime | WARNING | 불량 판정 발생 + score |
| `insp_history_cleared` | history | WARNING | 검사 이력 초기화 |

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
| 학습 완료 | Explorer UI 토스트 | "학습이 완료되었습니다. 소요 시간: {N}분 {M}초" |
| 학습 중단 | Explorer UI 경고 | MSG["TRAIN_STOPPED"] |
| 학습 실패 | Explorer UI 오류 | "학습 중 오류가 발생했습니다. 로그를 확인해 주세요." |
| 모델 저장 완료 | Explorer UI 토스트 | "저장 완료: {path} ({size_mb:.1f} MB)" |
| 폴더 구조 오류 | Explorer UI 오류 | MSG["INVALID_FOLDER"] |
| Grayscale 감지 | Explorer UI 정보 | MSG["GRAYSCALE_DETECT"] |
| 디스크 공간 부족 | Explorer UI 경고 | "디스크 여유 공간이 {size_mb:.0f} MB로 부족합니다. 500 MB 이상 확보 후 저장해 주세요." |

---

## 8. Deterministic Design Rules

> 이후 모든 파일의 코드 및 명세에서 아래 규칙을 강제 적용한다. 예외는 허용되지 않는다.

| 규칙 | 적용 범위 | 세부 규칙 |
|------|-----------|-----------|
| **R-NAMING-01** | 모든 Python 변수·함수·파일명 | `snake_case` 고정. 클래스명만 `PascalCase`. |
| **R-NAMING-02** | Zustand store 키명 (v2.0) | `camelCase`. 예: `datasetPath`, `modelConfig`. |
| **R-NAMING-03** | experiment_id 형식 | `{model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_소문자_16진수}`. 예: `efficientad_20260508_140023_7f3a`. |
| **R-ID-01** | experiment_id 생성 | `uuid.uuid4().hex[:4]`로 4자리 난수 생성. |
| **R-TIME-01** | 모든 시간 표현 | ISO 8601. 한국시간(KST, UTC+9). 예: `"2026-05-08T14:00:23+09:00"`. |
| **R-FLOAT-01** | 지표 수치 저장 | `round(value, 6)`. 표시는 소수점 4자리. |
| **R-BOOL-01** | YAML boolean | Python `True/False`, YAML `true/false`. |
| **R-PATH-01** | 모든 파일 경로 | `pathlib.Path` 사용. 문자열 연산 금지. |
| **R-ENCODE-01** | 파일 읽기/쓰기 | 항상 `encoding="utf-8"` 명시. |
| **R-ENUM-01** | ENUM 값 비교 | 소문자 문자열 비교. 예: `if model_type == "efficientad":`. |
| **R-NULL-01** | 없는 값 | Python `None`. JSON `null`. YAML `null`. TypeScript `null`. |
| **R-SEED-01** | 재현성 | `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)` 동시 설정. |
| **R-ATOMIC-01** | 파일 쓰기 | 임시 파일(.tmp) 생성 후 rename. 부분 쓰기 방지. |
| **R-UI-01** | UI 언어 | 모든 라벨·버튼·안내 메시지 한국어. 기술 용어는 한국어+영문 병기. |
| **R-UI-02** | 숨김 처리 | 비선택 파라미터 UI는 DOM 미렌더링 (`if` 분기 또는 조건부 렌더링). |
| **R-INSP-01** | history.json 쓰기 금지 | Vision(smart-qc-vision)은 `history.json`을 읽기 전용으로만 사용. |
| **R-INSP-02** | 비전검사 파일 위치 | FastAPI 비전검사 라우터는 `api/inspection/` 하위에만 위치. |
| **R-INSP-03** | 비전검사 이력 | `inspection_record`는 FastAPI 서버 메모리에 저장. 파일·DB 쓰기는 선택적 (v2.0 미확정). |
| **R-INSP-04** | 모델 교체 시 이력 초기화 | `POST /api/inspection/model` 호출 시 서버에서 records 자동 초기화. |
| **R-INSP-05** | 완료 실험만 교체 가능 | `status == "completed"`가 아닌 실험의 [적용] 버튼은 렌더링하지 않음. |

---

## 9. Made Assumptions

> PRD에 명시되지 않은 항목에 대한 합리적 가정. 이후 모든 설계는 이 가정을 기준으로 한다.

| # | 가정 항목 | 가정 내용 | 근거 |
|---|-----------|-----------|------|
| A-01 | 동시 사용자 수 | 1명 (단일 사용자). FastAPI 단일 인스턴스 모델 적용. | PRD에 다중 사용자 언급 없음 |
| A-02 | 학습 비동기 처리 | Python `threading.Thread` + `queue.Queue`로 백그라운드 학습, FastAPI가 WebSocket으로 진행상황 Push. | PRD 6절 "UI 블로킹 없음" |
| A-03 | Anomalib 버전 | `anomalib >= 1.0.0` (v1 API 기준). | PRD 6절 기술 스택 |
| A-04 | 모델 저장 크기 | EfficientAD-medium ≈ 200~400 MB, PatchCore (WideResNet50) ≈ 600~1000 MB. | P1 fix 반영 |
| A-05 | 테스트 이미지 추론 | AnomalyMap 화면에서 모델 재로드 후 전체 테스트셋 일괄 추론. 실험 완료 시 `metrics.anomaly_scores` 이미 계산·저장. | 화면 응답성 확보 |
| A-06 | configs.yaml 위치 | 작업 디렉토리 루트 `./configs.yaml`. Docker 실행 시 `/app/configs.yaml`. | 11.1절 Dockerfile WORKDIR /app |
| A-07 | 데이터셋 마운트 | Docker 실행 시 `-v /path/to/dataset:/app/dataset`. | PRD 14절 docker run |
| A-08 | Loss 곡선 갱신 주기 | EfficientAD: 매 100 step. PatchCore: 에포크 단위. WebSocket으로 Push. | UI 성능 vs 정보 밀도 균형 |
| A-09 | 학습 로그 버퍼 | Explorer UI에 표시되는 로그는 최신 100줄만 유지. 파일에는 전량 저장. | React 메모리 제한 |
| A-10 | GT 마스크 없는 결함 클래스 | `ground_truth/{class}/` 디렉토리가 없으면 해당 이미지의 GT는 빈 마스크(전체 0)로 처리. | MVTec AD 일부 클래스 |
| A-11 | image_size 기본값 | 256. 변경 시 preprocessing_config와 model_config 동시 업데이트. | PRD 9.2절 YAML 예시 |
| A-12 | 한국 시간대 | 모든 시간은 KST (UTC+9) 기준. `datetime.now(tz=timezone(timedelta(hours=9)))`. | PRD 한국어 작성 기준 |
| A-13 | 비교 차트 최대 실험 수 | 한 번에 최대 10개 실험 비교. 초과 시 안내 메시지. | Recharts 레이더 차트 가독성 |
| A-14 | 폴더 구조 검증 깊이 | `train/good/` 디렉토리 존재 + 이미지 최소 1개 이상. | MVTec AD 표준 |
| A-15 | 비전검사 접근 권한 | 접근 권한 분리 없음. 모든 화면에 누구나 접근 가능. | 단일 로컬 사용자 환경 |
| A-16 | test_pool 구성 시점 | 모델 교체 직후 서버에서 1회 구성. 소진 시 재셔플. | 일관성 유지 |
| A-17 | test_pool 레이블 기준 | `test/good/` 하위 → "양품". `test/{그 외}/` 하위 → "불량". 레이블 미표시(현장 정답 미공개 원칙). | 현장 실제 사용 시나리오 모사 |
| A-18 | 자동 검사 구현 | v2.0: `WS /ws/inspection/auto` 서버 주도 Push 방식. | Vision README WebSocket 확인 |
| A-19 | 비전검사 모델 로드 캐싱 | activeModel 변경 시에만 모델 재로드. 동일 모델 연속 추론 시 서버 캐시 활용. | 추론 지연 최소화 |
| A-20 | KPI 계산 분모 | 총 검사 수 == 0이면 불량률 표시를 "-"로. ZeroDivisionError 방지. | 세션 초기 상태 처리 |

---

## 10. Global Technology Stack

### 10.1 핵심 스택

<!-- v2.0: React/TypeScript/FastAPI 스택 추가, Streamlit은 보조 도구로 격하 -->

| 카테고리 | 기술 | 버전 기준 | 비고 |
|----------|------|-----------|------|
| **Explorer UI** | React | 19 | smart-qc-explorer 공식 UI |
| **Explorer UI** | Vite | 8 | 빌드 도구 |
| **Explorer UI** | TypeScript | — | |
| **Explorer UI** | Zustand | v5 | 상태 관리 |
| **Explorer UI** | Recharts | v2 | 차트 |
| **Explorer UI** | Tailwind CSS | v4 | 스타일 |
| **Explorer UI** | react-router-dom | v7 | 라우팅 |
| **Vision UI** | React | — | smart-qc-vision 공식 UI |
| **Vision UI** | Vite | — | 빌드 도구 |
| **Vision UI** | TypeScript | — | |
| **Vision UI** | Zustand | — | 상태 관리 |
| **Vision UI** | Recharts | — | 차트 |
| **Vision UI** | react-router-dom | v6 | 라우팅 |
| **백엔드 API** | FastAPI | — | REST + WebSocket |
| **[보조] 개발자 UI** | Streamlit | ≥ 1.30 | 개발자 디버그 전용 — 공식 UI 아님 |
| **ML 프레임워크** | PyTorch | ≥ 2.1 | |
| **ML 프레임워크** | torchvision | PyTorch 호환 | |
| **이상 탐지 모델** | Anomalib | ≥ 1.0.0 (v1 API) | |
| **이미지 처리** | OpenCV | `opencv-python-headless` | |
| **이미지 처리** | Pillow (PIL) | ≥ 10.0 | |
| **평가 지표** | scikit-learn | ≥ 1.3 | |
| **시각화** | Matplotlib | ≥ 3.7 | 서버 사이드 |
| **시각화** | Plotly | ≥ 5.15 | 서버 사이드 |
| **설정 관리** | PyYAML | ≥ 6.0 | |
| **컨테이너** | Docker | ≥ 24.0 | |
| **GPU 컨테이너** | NVIDIA Container Toolkit | 최신 안정 버전 | |
| **클라우드** | AWS EC2 | g4dn.xlarge | |

### 10.2 환경 명세

| 항목 | 값 |
|------|----|
| **Python** | 3.12 |
| **Node.js** | ≥ 18 |
| **CUDA** | 12.4 |
| **cuDNN** | 9.1 |
| **베이스 Docker 이미지** | `nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04` + `libcudnn9-cuda-12` (apt) |

### 10.3 지원 이미지 포맷

`.jpg` (`.jpeg`), `.png`, `.bmp`
확장자 매칭: `{".jpg", ".jpeg", ".png", ".bmp"}` (소문자 변환 후 비교)

### 10.4 실행 명령 요약

```bash
# 백엔드 (smart-qc-dashboard)
uvicorn api.main:app --reload --port 8000

# Explorer 프론트엔드 (smart-qc-explorer)
npm install && npm run dev   # http://localhost:5173

# Vision 프론트엔드 (smart-qc-vision)
npm install && npm run dev   # http://localhost:5173

# [보조] Streamlit 개발자 UI
streamlit run app.py         # http://localhost:8501
```

### 10.5 하드웨어 기준

| 항목 | 개발/테스트 | 프로덕션 (AWS) |
|------|------------|----------------|
| **인스턴스** | 로컬 GPU 머신 | AWS EC2 g4dn.xlarge |
| **GPU** | CUDA 지원 GPU | NVIDIA Tesla T4 (16GB VRAM) |
| **vCPU** | ≥ 4 | 4 |
| **RAM** | ≥ 16GB | 16GB |
| **Storage** | ≥ 50GB | 100GB gp3 EBS |
| **Port** | 8000/tcp, 5173/tcp | 8000/tcp, 5173/tcp, 22/tcp |

---

<!-- v2.0: ADR 섹션 신규 추가 — 프로젝트 주요 아키텍처 결정 기록 -->
## 11. Architecture Decision Records (ADR)

> 이 시스템의 주요 아키텍처 결정과 그 근거를 기록한다. 모든 하위 PRD 파일은 여기서 확정된 결정을 기준으로 한다.

---

### ~~ADR-01 (v1.x): REST API 서버 없음~~ — **폐기**

> ~~**결정**: smart-qc-dashboard는 REST API 서버를 포함하지 않는다.~~
> ~~**근거**: Streamlit 단독 앱으로 설계. 외부 클라이언트 없음.~~
> ~~**결과**: 모든 상태는 session_state로 관리. 파일 I/O로 영속.~~
>
> **v2.0 폐기 이유**: React 기반 프론트엔드(Explorer/Vision) 분리 및 실제 FastAPI 구현으로 이 결정은 무효다. 아래 ADR-01 (v2.0)으로 교체한다.

---

### ADR-01 (v2.0): FastAPI 서버 도입

**상태**: 확정 (2026-06-11)

**결정**: smart-qc-dashboard에 FastAPI 서버를 포함한다.

**근거**: React 기반 프론트엔드(Explorer/Vision) 분리를 위해 REST API 및 WebSocket 인터페이스가 필요하다. Streamlit은 단일 프로세스 모델로 외부 클라이언트를 지원할 수 없다.

**결과**:
- Explorer(React :5173) ↔ HTTP/WebSocket ↔ FastAPI(:8000)
- Vision(React :5173) ↔ HTTP/WebSocket ↔ FastAPI(:8000)
- Streamlit UI는 개발자용 보조 도구로 유지하되 공식 UI가 아니다.
- 두 프론트엔드는 `http://localhost:8000`을 baseURL로 사용한다.

---

### ADR-02: 학습 백그라운드 스레드

**상태**: 유효 (유지)

**결정**: 모델 학습은 Python `threading.Thread` + `queue.Queue`로 백그라운드에서 실행한다.

**근거**: 학습 중 UI 블로킹 없음 요구사항(NFR 6절). Python GIL 하에서 I/O 바운드 작업 및 ML 라이브러리의 C 확장 코드는 별도 스레드에서 실행 가능하다.

**결과 (v2.0 업데이트)**: 학습 스레드 → `queue.Queue` → FastAPI WebSocket(`/ws/training`) → Explorer React UI로 진행상황 Push.

---

### ADR-03: 파일시스템 전용 영속

**상태**: 유효 (유지)

**결정**: 실험 레코드는 `history.json`, 모델 가중치는 `./models/{exp_id}/`에 저장한다. MySQL 8.0은 인프라 레이어에 포함되나 현재 앱은 파일시스템만 사용한다.

**근거**: 단일 사용자 환경. 단순성 우선. DB 스키마 마이그레이션 없이 빠른 프로토타이핑 가능.

**결과**: `save_history()` / `load_history()` 원자적 파일 I/O (tmpfile → rename). Explorer 쓰기 전용 / Vision 읽기 전용.

---

### ADR-04: session_state 메인 스레드만 (Streamlit 내부 한정)

**상태**: 범위 축소 (Streamlit 보조 도구 내부에만 적용)

**결정**: Streamlit `session_state`는 메인 스레드에서만 접근한다.

**근거**: Streamlit의 단일 스레드 렌더링 모델. 백그라운드 스레드에서 `session_state` 직접 접근 시 race condition 발생.

**v2.0 범위 축소**: 이 규칙은 Streamlit 보조 도구 내부에만 적용된다. v2.0 공식 아키텍처(FastAPI + React)에서는 해당 없음. FastAPI는 thread-safe하게 설계한다.

---

### ADR-INSP-01: 비전검사 이력 영속 없음 (세션 한정)

**상태**: 유효 (v2.0에서 서버 메모리로 위치 변경)

**결정**: 검사 이력(`inspection_record`)은 영속 저장하지 않는다.

**근거**: 현장 운영 시나리오에서 이력은 현재 세션 기준이며, 재시작 시 초기화가 적절하다.

**v2.0 업데이트**: v1.x에서는 `session_state.insp_records`에 저장. v2.0에서는 FastAPI 서버 메모리에 저장. 서버 재시작 시 초기화.

---

### ADR-INSP-02: 비전검사는 history.json 읽기 전용

**상태**: 유효 (유지)

**결정**: Vision(smart-qc-vision) 및 비전검사 관련 FastAPI 라우터는 `history.json`에 쓰지 않는다.

**근거**: Explorer가 실험 데이터의 단일 소유자여야 한다. Vision이 실험 레코드를 수정하면 Explorer의 데이터 무결성이 깨진다.

**결과**: R-INSP-04 규칙으로 강제.

---

### ADR-INSP-03: 자동 검사 타이머 구현

**상태**: 변경 필요 — [확인 필요: Vision WebSocket 대체 여부]

**v1.x 결정**: `time.sleep(3)` + `st.rerun()` 조합. 별도 스레드 없음.

**근거 (v1.x)**: Streamlit 단일 스레드 모델. `time.sleep(3)`이 가장 단순한 구현.

**v2.0 현황**: Vision README에 `WS /ws/inspection/auto`가 존재한다. 서버가 자동 검사를 주도하고 결과를 WebSocket으로 Push하는 방식으로 변경됐을 가능성이 높다.

**[확인 필요]**: `time.sleep(3)` 로직이 FastAPI 서버 내부에서 유지되는지, 또는 Vision WebSocket 클라이언트 측 타이머로 대체됐는지 확인 후 이 ADR을 업데이트한다.

---

*문서 끝. v2.0 기준: smart-qc-explorer / smart-qc-vision / smart-qc-dashboard 3개 레포는 이 문서의 스키마·계약·규칙·ADR을 공통 기준으로 한다. 모든 하위 PRD 파일(01~15번)은 이 문서를 먼저 참조한다.*

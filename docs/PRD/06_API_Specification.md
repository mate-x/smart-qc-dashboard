# 06. API Specification

> **참조 문서**: [04_System_Architecture.md](./04_System_Architecture.md) §B.3, [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **최종수정**: 2026-06-11
> **변경 요약**: v2.0 — 전면 재작성. FastAPI 엔드포인트 기준. v1.x Streamlit utils 인터페이스 명세는 §10(v1.x 참고)으로 이동.

---

## 목차

1. [기본 설정](#1-기본-설정)
2. [데이터셋 API](#2-데이터셋-api)
3. [설정·큐 API](#3-설정큐-api)
4. [학습 제어 API](#4-학습-제어-api)
5. [실험 히스토리 API](#5-실험-히스토리-api)
6. [Anomaly Map API](#6-anomaly-map-api)
7. [비전검사 API](#7-비전검사-api)
8. [WebSocket 프로토콜 명세](#8-websocket-프로토콜-명세)
9. [공통 스키마 참조](#9-공통-스키마-참조)
10. [v1.x 참고 — Streamlit utils 인터페이스](#10-v1x-참고--streamlit-utils-인터페이스)

---

## 1. 기본 설정

### 1.1 서버 정보

| 항목 | 값 |
|------|----|
| **진입점** | `uvicorn api.main:app --reload --port 8000` |
| **baseURL** | `http://localhost:8000` |
| **Content-Type** | `application/json` |
| **OpenAPI 문서** | `http://localhost:8000/docs` |
| **CORS 허용 Origin** | `http://localhost:5173`, `http://localhost:5174` |
| **CORS 허용 Methods** | `*` (GET, POST, PUT, PATCH, DELETE, OPTIONS) |

### 1.2 공통 에러 코드

| HTTP 코드 | 발생 조건 | 응답 형식 |
|-----------|----------|-----------|
| `400 Bad Request` | 요청 파라미터 오류, 경로 없음, 잘못된 값 | `{"detail": "..."}` |
| `404 Not Found` | exp_id·job_id·checkpoint 없음 | `{"detail": "..."}` |
| `409 Conflict` | 학습 중 재시작, 상태 충돌 | `{"detail": "..."}` |
| `422 Unprocessable Entity` | Pydantic 유효성 검사 실패 | `{"detail": [...]}` |
| `500 Internal Server Error` | 추론 실패, 파일시스템 오류 | `{"detail": "..."}` |

### 1.3 이미지 응답 규칙

이미지를 반환하는 엔드포인트는 `image/png` 또는 원본 파일의 MIME 타입 바이너리로 응답한다. JSON 응답이 아님에 주의.

---

## 2. 데이터셋 API

> **태그**: `탭1 · 데이터셋` | **prefix**: `/api/dataset`

---

### POST /api/dataset/validate

데이터셋 경로를 검증하고 메타 정보를 반환한다. MVTec AD 형식과 OK/NG(oking) 이진 형식을 자동 감지한다.

**Request Body**

```json
{
  "path": "C:/datasets/bottle",
  "product_name": "bottle"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `path` | `string` | ✅ | 데이터셋 루트 경로 (빈 문자열 금지) |
| `product_name` | `string` | ❌ | 실험 기록용 제품명 (기본값 `""`) |

**Response 200**

```json
{
  "dataset_format": "mvtec",
  "channels": 3,
  "train_good_count": 209,
  "test_counts": {"good": 42, "broken_large": 12, "broken_small": 10},
  "gt_counts": {"broken_large": 12, "broken_small": 10},
  "total_test_count": 64,
  "defect_classes": ["broken_large", "broken_small"],
  "supported_formats": ["png", "jpg"],
  "has_invalid_files": false,
  "invalid_file_count": 0,
  "folder_tree": "bottle/\n  train/\n    good/ (209)\n  test/\n    ...",
  "has_background_clean": false,
  "oking_ok_dir": null,
  "oking_ng_dir": null,
  "oking_ok_count": null,
  "oking_ng_count": null,
  "train_ratio": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `dataset_format` | `"mvtec" \| "oking"` | 감지된 데이터셋 형식 |
| `channels` | `int` | 이미지 채널 수 (1=그레이스케일, 3=RGB) |
| `train_good_count` | `int` | 학습 정상 이미지 수 |
| `test_counts` | `dict[str, int]` | 테스트 클래스별 이미지 수 |
| `gt_counts` | `dict[str, int]` | GT 마스크 클래스별 수 (MVTec 전용) |
| `total_test_count` | `int` | 전체 테스트 이미지 수 |
| `defect_classes` | `list[str]` | 결함 클래스 목록 |
| `has_invalid_files` | `bool` | 지원되지 않는 형식 파일 존재 여부 |
| `invalid_file_count` | `int` | 지원되지 않는 파일 수 |
| `folder_tree` | `string` | 폴더 구조 텍스트 |
| `has_background_clean` | `bool` | `{dataset_path}/background_clean/` 폴더 존재 여부 |
| `oking_ok_dir` | `string \| null` | OK 폴더 경로 (oking 형식 전용) |
| `oking_ng_dir` | `string \| null` | NG 폴더 경로 (oking 형식 전용) |
| `oking_ok_count` | `int \| null` | OK 이미지 수 (oking 형식 전용) |
| `oking_ng_count` | `int \| null` | NG 이미지 수 (oking 형식 전용) |
| `train_ratio` | `float \| null` | 학습/전체 비율 (oking 형식 전용) |

**에러**

| 코드 | 조건 |
|------|------|
| `400` | `path`가 비어있거나 존재하지 않는 경로, 지원되지 않는 데이터셋 형식 |

---

### GET /api/dataset/thumbnail/{class_name}

특정 클래스의 대표 썸네일 이미지를 반환한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `class_name` | `string` | 클래스 이름 (`good`, `broken_large` 등) |

**Query Parameter**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `dataset_path` | `string` | ✅ | 데이터셋 루트 경로 |

**Response 200**: `image/png` 바이너리

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 클래스의 이미지를 찾을 수 없음 |

---

## 3. 설정·큐 API

> **태그**: `탭2 · 설정`, `탭2 · 큐`

---

### GET /api/config

현재 저장된 설정과 device 정보를 반환한다.

**Response 200**

```json
{
  "preprocessing_config": {
    "method": "none",
    "background_method": "none",
    "image_size": 256,
    "params": {}
  },
  "model_config": {
    "model_type": "EfficientAD",
    "model_size": "S",
    "train_steps": 70000,
    "batch_size": 1,
    "threshold_method": "percentile",
    "threshold_value": 95.0
  },
  "device_info": {
    "available": true,
    "free_mb": 4096,
    "total_mb": 8192,
    "used_mb": 4096
  }
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `preprocessing_config` | `dict \| null` | 저장된 전처리 설정 (미설정 시 `null`) |
| `model_config` | `dict \| null` | 저장된 모델 설정 (미설정 시 `null`) |
| `device_info` | `dict` | GPU 정보. `available: false`이면 CPU |

---

### POST /api/config

전처리·모델 설정을 서버 메모리에 저장하고 저장된 설정을 반환한다.

**Request Body**

```json
{
  "preprocessing_config": {
    "method": "none",
    "background_method": "none",
    "image_size": 256,
    "params": {}
  },
  "model_config": {
    "model_type": "EfficientAD",
    "model_size": "S",
    "train_steps": 70000,
    "batch_size": 1,
    "threshold_method": "percentile",
    "threshold_value": 95.0
  }
}
```

> **주의**: 요청 JSON 키는 `model_config`이다 (`model_cfg` 아님). 서버 내부에서 Pydantic alias를 사용하나 클라이언트는 `model_config` 키를 사용.

**Response 200**: `GET /api/config` 응답과 동일

---

### POST /api/config/preview

현재 학습 이력의 Anomaly Score 기준으로 threshold 값이 정상/결함 비율을 어떻게 나누는지 미리보기한다.

**Request Body**

```json
{
  "threshold_method": "percentile",
  "threshold_value": 95.0
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `threshold_method` | `"percentile" \| "absolute"` | Threshold 결정 방법 |
| `threshold_value` | `float` | 백분위수 (percentile) 또는 절댓값 (absolute) |

**Response 200**

```json
{
  "normal_ratio": 0.95,
  "defect_ratio": 0.05
}
```

비교 대상 데이터가 없으면 두 필드 모두 `null`.

---

### POST /api/config/preview-image

전처리 설정 적용 전/후 이미지를 Base64로 반환한다.

**Request Body**

```json
{
  "dataset_path": "C:/datasets/bottle",
  "background_method": "none",
  "method": "clahe",
  "params": {"clip_limit": 2.0, "tile_grid_size": 8},
  "image_size": 256
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `dataset_path` | `string` | ✅ | 샘플 이미지를 가져올 데이터셋 경로 |
| `background_method` | `string` | ❌ | 배경 제거 방법 (`"none"` \| `"sam2"`) |
| `method` | `string` | ✅ | 전처리 방법 (`"none"` \| `"homomorphic"` \| `"he"` \| `"clahe"`) |
| `params` | `dict \| null` | ❌ | 방법별 파라미터 |
| `image_size` | `int` | ✅ | 리사이즈 목표 크기 (픽셀) |

**Response 200**

```json
{
  "original_b64": "iVBORw0KGgo...",
  "processed_b64": "iVBORw0KGgo...",
  "warning": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `original_b64` | `string` | 원본 이미지 Base64 PNG |
| `processed_b64` | `string` | 전처리 후 이미지 Base64 PNG |
| `warning` | `string \| null` | 경고 메시지 (GPU 부족 등) |

**에러**

| 코드 | 조건 |
|------|------|
| `400` | 경로 없음, 이미지 없음, 잘못된 파라미터 |

---

### POST /api/config/yaml/save

현재 서버 메모리의 설정을 `configs.yaml` 파일로 저장한다.

**Request Body**: 없음

**Response 200**

```json
{"success": true}
```

**에러**

| 코드 | 조건 |
|------|------|
| `400` | 저장할 설정이 없음 |
| `500` | 파일시스템 쓰기 실패 |

---

### POST /api/config/yaml/load

`configs.yaml` 파일을 읽어 서버 메모리에 반영하고 내용을 반환한다.

**Request Body**: 없음

**Response 200**

```json
{
  "preprocessing_config": {...},
  "model_config": {...}
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `400` | `configs.yaml` 파일 없음 또는 파싱 실패 |

---

### GET /api/queue

배치 학습 대기열 전체를 반환한다.

**Response 200**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "EfficientAD-S #1",
    "preprocessing_config": {...},
    "model_config": {...},
    "status": "대기중",
    "set_id": "set-2026-06-11"
  }
]
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | `string` (UUID) | 큐 항목 ID |
| `name` | `string` | 자동 생성 이름 (`{모델타입} #순번`) |
| `preprocessing_config` | `dict` | 전처리 설정 |
| `model_config` | `dict` | 모델 설정 |
| `status` | `"대기중" \| "실행중" \| "완료" \| "실패" \| "건너뜀"` | 항목 상태 |
| `set_id` | `string \| null` | 배치 실험 세트 ID (그룹화용) |

---

### POST /api/queue

현재 설정을 배치 대기열에 추가한다.

**Request Body**

```json
{
  "preprocessing_config": {...},
  "model_config": {...},
  "set_id": "set-2026-06-11"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `preprocessing_config` | `dict` | ✅ | 전처리 설정 |
| `model_config` | `dict` | ✅ | 모델 설정 |
| `set_id` | `string \| null` | ❌ | 실험 세트 식별자 |

**Response 200**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "EfficientAD-S #1"
}
```

---

### DELETE /api/queue/{item_id}

대기열 항목을 삭제한다. `"대기중"` 상태인 항목만 삭제 가능.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `item_id` | `string` | 큐 항목 ID |

**Response 200**

```json
{"success": true}
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 ID 없음 |
| `400` | `"대기중"` 이외 상태 항목 삭제 시도 |

---

### PATCH /api/queue/reorder

대기열 항목 순서를 변경한다. `"대기중"` 상태인 항목만 이동 가능.

**Request Body**

```json
{
  "item_id": "550e8400-e29b-41d4-a716-446655440000",
  "direction": "up"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `item_id` | `string` | 이동할 항목 ID |
| `direction` | `"up" \| "down"` | 이동 방향 |

**Response 200**

```json
{"success": true}
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 ID 없음 |
| `400` | `"대기중"` 이외 항목, 이미 처음/마지막 위치 |

---

## 4. 학습 제어 API

> **태그**: `탭3 · 학습` | **prefix**: `/api/training`

---

### POST /api/training/start

단일 설정으로 학습을 시작한다. 학습 스레드를 백그라운드에서 시작하고 즉시 반환.

**Request Body**

```json
{
  "experiment_name": "bottle-efficientad-v1"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `experiment_name` | `string` | ❌ | 실험 이름 (기본값 `""`, 빈 문자열이면 자동 생성) |

**Response 201**

```json
{
  "exp_id": "bottle-20260611-143022-a1b2c3"
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `400` | 설정(`preprocessing_config` 또는 `model_config`)이 서버 메모리에 없음 |
| `409` | 이미 학습 실행 중 (status != "idle") |

---

### POST /api/training/resume

체크포인트에서 학습을 재개한다.

**Request Body**

```json
{
  "checkpoint_name": "ckpt_step_5000_20260611_143022.pkl"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `checkpoint_name` | `string` | ✅ | 재개할 체크포인트 파일명 |

**Response 201**

```json
{
  "exp_id": "bottle-20260611-143022-a1b2c3"
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 체크포인트 파일 없음 |
| `409` | 이미 학습 실행 중 |

---

### POST /api/training/pause

실행 중인 학습에 일시정지 신호를 전송한다. 실제 일시정지는 현재 스텝 완료 후 WS `paused` 메시지로 확인.

**Request Body**: 없음

**Response 200**

```json
{
  "success": true,
  "message": "일시정지 신호를 전송했습니다."
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `409` | 학습 실행 중이 아님 |

---

### POST /api/training/unpause

일시정지된 학습을 재개한다.

**Request Body**: 없음

**Response 200**

```json
{
  "success": true,
  "message": "학습을 재개합니다."
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `409` | 일시정지 상태가 아님 |

---

### POST /api/training/stop

실행 중인 학습에 중단 신호를 전송한다. 실제 중단은 현재 스텝 완료 후 WS `stopped` 메시지로 확인.

**Request Body**: 없음

**Response 200**

```json
{
  "success": true,
  "message": "중지 신호를 전송했습니다."
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `409` | 학습 실행 중이 아님 |

---

### GET /api/training/status

현재 학습 상태를 반환한다. WS 재연결 복구 또는 화면 복귀 시 사용.

**Response 200**

```json
{
  "status": "running",
  "exp_id": "bottle-20260611-143022-a1b2c3",
  "batch_mode": false,
  "batch_total": 0,
  "progress": {
    "step": 12000,
    "total": 70000,
    "loss": 0.003141,
    "elapsed": 342.5
  },
  "current_stage_idx": 1,
  "current_stage_name": "Feature Extractor 학습",
  "log_lines": ["[14:30:22] 학습 시작", "..."],
  "loss_history": [
    {"step": 500, "loss": 0.025},
    {"step": 1000, "loss": 0.018}
  ],
  "last_ckpt_path": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `"idle" \| "running" \| "paused"` | 현재 학습 상태 |
| `exp_id` | `string \| null` | 실행 중인 실험 ID |
| `batch_mode` | `bool` | 배치 학습 실행 중 여부 |
| `batch_total` | `int` | 배치 항목 총 수 |
| `progress` | `dict \| null` | 진행상황 (`{step, total, loss, elapsed}`) |
| `current_stage_idx` | `int \| null` | 현재 학습 단계 인덱스 |
| `current_stage_name` | `string \| null` | 현재 학습 단계 이름 |
| `log_lines` | `list[string]` | 최근 로그 줄 (최대 100줄) |
| `loss_history` | `list[dict]` | Loss 히스토리 (`[{step, loss}, ...]`) |
| `last_ckpt_path` | `string \| null` | 마지막 체크포인트 경로 (일시정지 시) |

---

### GET /api/training/checkpoints

저장된 체크포인트 목록을 반환한다.

**Response 200**

```json
{
  "checkpoints": [
    {
      "name": "ckpt_step_5000_20260611_143022.pkl",
      "model_type": "EfficientAD",
      "created_at": "2026-06-11 14:30:22",
      "step": 5000,
      "total_steps": 70000,
      "batch_idx": null,
      "total_batches": null,
      "n_patches": null
    }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | `string` | 체크포인트 파일명 |
| `model_type` | `string` | 모델 타입 (`"EfficientAD"` \| `"PatchCore"`) |
| `created_at` | `string` | 생성 시각 (`"YYYY-MM-DD HH:MM:SS"`) |
| `step` | `int \| null` | 중단 스텝 (EfficientAD 전용) |
| `total_steps` | `int \| null` | 전체 스텝 수 (EfficientAD 전용) |
| `batch_idx` | `int \| null` | 배치 인덱스 (PatchCore 전용) |
| `total_batches` | `int \| null` | 전체 배치 수 (PatchCore 전용) |
| `n_patches` | `int \| null` | 누적 패치 수 (PatchCore 전용) |

---

### DELETE /api/training/checkpoints/{name}

체크포인트 파일을 삭제한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `name` | `string` | 삭제할 체크포인트 파일명 |

**Response 200**

```json
{"success": true}
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 체크포인트 없음 |

---

### POST /api/training/batch/start

대기열에 있는 설정들을 순차적으로 학습하는 배치를 시작한다.

**Request Body**: 없음

**Response 201**

```json
{
  "exp_id": "bottle-20260611-143022-a1b2c3",
  "batch_total": 3
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `exp_id` | `string` | 첫 번째 학습 실험 ID |
| `batch_total` | `int` | 배치 항목 총 수 |

**에러**

| 코드 | 조건 |
|------|------|
| `400` | 대기열이 비어있음 |
| `409` | 이미 학습 실행 중 |

---

### POST /api/training/batch/skip

배치 학습에서 현재 항목을 건너뛰고 다음 항목으로 이동한다.

**Request Body**: 없음

**Response 200**

```json
{
  "success": true,
  "message": "건너뜀 신호를 전송했습니다."
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `409` | 배치 학습 실행 중이 아님 |

---

### POST /api/training/batch/stop

배치 학습 전체를 중단한다.

**Request Body**: 없음

**Response 200**

```json
{
  "success": true,
  "message": "전체 배치 중단 신호를 전송했습니다."
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `409` | 배치 학습 실행 중이 아님 |

---

## 5. 실험 히스토리 API

> **태그**: `탭4 · 실험 히스토리` | **prefix**: `/api/experiments`

---

### GET /api/experiments

`history.json`에 저장된 모든 실험 레코드를 반환한다. `created_at` 역순 정렬.

**Response 200**

```json
[
  {
    "experiment_id": "bottle-20260611-143022-a1b2c3",
    "name": "bottle-efficientad-v1",
    "status": "completed",
    "model_type": "EfficientAD",
    "product_name": "bottle",
    "created_at": "2026-06-11T14:30:22+09:00",
    "duration_seconds": 1823,
    "model_path": "./models/bottle-20260611-143022-a1b2c3",
    "metrics": {
      "accuracy": 0.984,
      "precision": 0.976,
      "recall": 0.991,
      "f1_score": 0.983,
      "f2_score": 0.988,
      "auc": 0.996,
      "threshold": 0.312
    }
  }
]
```

응답 레코드의 전체 필드 구조는 [00_Global_Context_Document.md §1.5](./00_Global_Context_Document.md) `ExperimentRecord` 참조.

---

### DELETE /api/experiments/{exp_id}

실험 레코드를 삭제한다. `history.json`에서 제거하고 연관 모델 파일도 삭제.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `exp_id` | `string` | 삭제할 실험 ID |

**Response 200**

```json
{"success": true}
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 실험 없음 |

---

### POST /api/experiments/{exp_id}/save

실험의 모델 파일을 지정한 경로에 복사 저장한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `exp_id` | `string` | 저장할 실험 ID |

**Request Body**

```json
{
  "save_path": "C:/exports/bottle_model.pth"
}
```

**Response 200**

```json
{
  "success": true,
  "saved_path": "C:/exports/bottle_model.pth",
  "size_mb": 42.3,
  "warning": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | `bool` | 저장 성공 여부 |
| `saved_path` | `string` | 실제 저장된 경로 |
| `size_mb` | `float` | 저장된 파일 크기 (MB) |
| `warning` | `string \| null` | 경고 메시지 (디스크 공간 부족 등) |

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 실험 없음 |
| `400` | `save_path`가 비어있거나 `completed` 상태 아님 |
| `500` | 파일 복사 실패 |

---

## 6. Anomaly Map API

> **태그**: `탭5 · Anomaly Map` | **prefix**: `/api/anomaly-map`
>
> **라우터 선언 순서 주의**: 정적 prefix(`/job/`, `/zip/`)가 동적 `/{exp_id}/` 보다 먼저 선언되어야 FastAPI가 정적 경로를 우선 매칭한다.

---

### GET /api/anomaly-map/{exp_id}/status

실험의 Anomaly Map 캐시 빌드 상태를 반환한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `exp_id` | `string` | 실험 ID |

**Response 200**

```json
{
  "built": true,
  "image_count": 64
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `built` | `bool` | 캐시 빌드 완료 여부 |
| `image_count` | `int` | 캐시된 이미지 수 (미빌드 시 0) |

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 실험 없음 |

---

### POST /api/anomaly-map/{exp_id}/build

Anomaly Map 생성 Job을 시작한다. 이미 캐시가 존재하면 즉시 완료 상태의 job을 반환한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `exp_id` | `string` | 실험 ID |

**Request Body**: 없음

**Response 200**

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 실험 없음 또는 모델 파일 없음 |

---

### GET /api/anomaly-map/job/{job_id}

Build 또는 ZIP 생성 Job의 상태를 조회한다.

> **라우터 선언 순서**: 이 엔드포인트는 `/{exp_id}/...` 경로보다 먼저 선언되어야 한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `job_id` | `string` | Job ID |

**Response 200**

```json
{
  "status": "completed",
  "error": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `"pending" \| "running" \| "completed" \| "failed"` | Job 상태 |
| `error` | `string \| null` | 실패 시 오류 메시지 |

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 job_id 없음 |

---

### GET /api/anomaly-map/{exp_id}/images

Anomaly Map 이미지 목록과 TP/FP/TN/FN 통계를 반환한다. 캐시가 없으면 404.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `exp_id` | `string` | 실험 ID |

**Query Parameters**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `threshold` | `float` | ✅ | 정규화된 threshold 값 (0~1.2 범위) |
| `defect_class` | `string` | ❌ | 결함 유형 필터 (`"전체"` 또는 클래스명, 기본값 `"전체"`) |

**Response 200**

```json
{
  "images": [
    {
      "image_name": "000.png",
      "defect_class": "broken_large",
      "anomaly_score": 0.843,
      "verdict": "NG",
      "gt_match": true,
      "classification": "TP",
      "image_path": "broken_large/000.png"
    }
  ],
  "score_max": 0.843,
  "score_avg": 0.412,
  "tp": 18,
  "fp": 2,
  "tn": 38,
  "fn": 6
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `images` | `list[ImageRow]` | 이미지 목록 |
| `score_max` | `float` | 최대 Anomaly Score |
| `score_avg` | `float` | 평균 Anomaly Score |
| `tp` / `fp` / `tn` / `fn` | `int` | 분류 통계 |

**ImageRow 필드**

| 필드 | 타입 | 설명 |
|------|------|------|
| `image_name` | `string` | 파일명 |
| `defect_class` | `string` | 결함 클래스명 (`"good"` 포함) |
| `anomaly_score` | `float` | Min-Max 정규화된 Anomaly Score |
| `verdict` | `"OK" \| "NG"` | 판정 결과 |
| `gt_match` | `bool` | GT 레이블과 판정 일치 여부 |
| `classification` | `"TP" \| "FP" \| "TN" \| "FN"` | 분류 결과 |
| `image_path` | `string` | triplet 등 이미지 API의 `{path:path}` 파라미터에 사용할 값 (`"{class}/{filename}"`) |

**에러**

| 코드 | 조건 |
|------|------|
| `400` | threshold 범위 오류 |
| `404` | 실험 없음 또는 캐시 미빌드 |

---

### GET /api/anomaly-map/{exp_id}/image/{image_path:path}/triplet

원본/GT마스크/히트맵 3분할 합성 이미지를 PNG로 반환한다.

**Path Parameters**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `exp_id` | `string` | 실험 ID |
| `image_path` | `string` | `"{class_name}/{image_name}"` 형식 (`ImageRow.image_path` 값) |

**Response 200**: `image/png` 바이너리 (3분할 합성)

**에러**

| 코드 | 조건 |
|------|------|
| `400` | `image_path` 형식 오류 (`"/"` 포함 필요) |
| `404` | 실험 없음, 캐시 없음, 이미지 없음 |

---

### GET /api/anomaly-map/{exp_id}/image/{image_path:path}/original

원본 이미지를 PNG로 반환한다.

> 요청/에러 구조: `/triplet`과 동일

---

### GET /api/anomaly-map/{exp_id}/image/{image_path:path}/gt_mask

GT 마스크 이미지를 PNG로 반환한다. MVTec AD 결함 클래스에만 존재.

> 요청 구조: `/triplet`과 동일
>
> **에러 추가**: GT 마스크 파일이 없으면 `404` 반환

---

### GET /api/anomaly-map/{exp_id}/image/{image_path:path}/heatmap

Anomaly Map 히트맵 이미지를 PNG로 반환한다.

> 요청/에러 구조: `/triplet`과 동일

---

### GET /api/anomaly-map/{exp_id}/export/csv

Anomaly Map 결과를 CSV로 다운로드한다.

**Path Parameter**: `exp_id`

**Query Parameters**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `threshold` | `float` | ✅ | 정규화된 threshold |
| `defect_class` | `string` | ❌ | 필터 클래스 (기본값 `"전체"`) |

**Response 200**: `text/csv; charset=utf-8-sig`

```
Content-Disposition: attachment; filename={exp_id}_results.csv
```

CSV 컬럼: `image_name`, `defect_class`, `anomaly_score`, `verdict`, `classification`

**에러**

| 코드 | 조건 |
|------|------|
| `400` | threshold 오류 |
| `404` | 실험 없음 또는 캐시 없음 |

---

### POST /api/anomaly-map/{exp_id}/export/zip

Triplet 이미지 전체를 ZIP으로 묶는 Job을 시작한다.

**Path Parameter**: `exp_id`

**Request Body**

```json
{
  "threshold": 0.45,
  "defect_class": "전체"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `threshold` | `float` | ✅ | 정규화된 threshold |
| `defect_class` | `string` | ❌ | 필터 클래스 (기본값 `"전체"`) |

**Response 200**

```json
{
  "job_id": "a3b8c2d1-f4e7-4b9a-a123-456789abcdef"
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `400` | threshold 오류 |
| `404` | 실험 없음 또는 캐시 없음 |

---

### GET /api/anomaly-map/zip/{job_id}

ZIP Job 완료 후 파일을 다운로드한다.

> **라우터 선언 순서**: 이 엔드포인트는 `/{exp_id}/...` 경로보다 먼저 선언되어야 한다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `job_id` | `string` | ZIP Job ID |

**Response 200**: `application/zip`

```
Content-Disposition: attachment; filename={job_id}_anomaly_maps.zip
```

**에러**

| 코드 | 조건 |
|------|------|
| `404` | Job 없음 |
| `400` | Job 미완료 (pending/running/failed) |
| `500` | ZIP 파일 생성 실패 |

---

## 7. 비전검사 API

> **태그**: `탭3 · 모델 교체`, `탭1 · 실시간 검사`, `탭2 · 검사 이력`

---

### GET /api/models

`history.json`에서 `status == "completed"` 인 실험 목록을 반환한다. F1 내림차순 정렬.

**Response 200**

```json
[
  {
    "experiment_id": "bottle-20260611-143022-a1b2c3",
    "name": "bottle-efficientad-v1",
    "status": "completed",
    "model_type": "EfficientAD",
    "product_name": "bottle",
    "created_at": "2026-06-11 14:30:22",
    "metrics": {
      "f1_score": 0.983,
      "auc": 0.996,
      ...
    },
    ...
  }
]
```

> `metrics.anomaly_scores` 등 히스토리 데이터도 포함 — `POST /api/inspection/model`의 threshold 재계산에 사용.

---

### POST /api/inspection/model

실험 모델을 비전검사에 적용한다. 모델 로드, 검사 이력 초기화, test pool 구성을 수행한다.

**Request Body**

```json
{
  "experiment_id": "bottle-20260611-143022-a1b2c3",
  "source_path": "C:/new-dataset/bottle"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `experiment_id` | `string` | ✅ | 적용할 실험 ID |
| `source_path` | `string \| null` | ❌ | 검사 이미지 소스 경로. `null`이면 실험의 `dataset_path` 사용 |

**Response 200**

```json
{
  "success": true,
  "active_model": {
    "experiment_id": "bottle-20260611-143022-a1b2c3",
    "name": "bottle-efficientad-v1",
    "model_path": "./models/bottle-20260611-143022-a1b2c3",
    "model_type": "EfficientAD",
    "threshold": 0.472,
    "dataset_path": "C:/datasets/bottle",
    "preprocessing_config": {
      "method": "none",
      "background_method": "none",
      "image_size": 256,
      "params": {}
    },
    "score_min": 0.012,
    "score_max": 0.891,
    "device": "cuda",
    "background_method": "none",
    "product_name": "bottle"
  },
  "gpu_warning": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | `bool` | 적용 성공 여부 |
| `active_model` | `dict` | 적용된 모델 정보 (§9.1 `ActiveModel` 참조) |
| `gpu_warning` | `string \| null` | GPU 메모리 1GB 미만 시 경고 메시지 |

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 실험 없음 |
| `400` | `completed` 상태 아닌 실험, 빈 test pool, 데이터셋 경로 없음 |
| `500` | 모델 로드 실패 |

---

### GET /api/inspection/model

현재 적용된 모델 정보를 반환한다.

**Response 200**

```json
{
  "active_model": { ... }
}
```

모델 미선택 시: `{"active_model": null}`

---

### PATCH /api/inspection/source-path

모델 재로드 없이 검사 이미지 소스 경로만 변경한다.

**Request Body**

```json
{
  "source_path": "C:/production/bottle"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `source_path` | `string \| null` | ❌ | 새 소스 경로. `null` 또는 빈 문자열이면 실험의 원래 `dataset_path`로 초기화 |

**Response 200**

```json
{
  "success": true,
  "source_path": "C:/production/bottle"
}
```

**에러**

| 코드 | 조건 |
|------|------|
| `400` | 모델 미선택, 경로 없음, 빈 pool, 유효성 검사 실패 |

---

### POST /api/inspection/run

수동 검사를 1회 비동기로 실행한다. 즉시 `job_id`를 반환하고 추론은 백그라운드에서 실행.

**Request Body**

```json
{
  "defect_only": false
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `defect_only` | `bool` | ❌ | `true`이면 불량 이미지(`gt_label == "불량"`)만 대상으로 샘플링 (기본값 `false`) |

**Response 200**

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

> 추론 완료 확인: `GET /api/inspection/job/{job_id}` 폴링

---

### GET /api/inspection/job/{job_id}

수동 검사 Job 상태를 조회한다. 완료(`completed` 또는 `failed`) 상태이면 데이터를 반환 후 서버에서 제거된다.

**Path Parameter**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `job_id` | `string` | Job ID |

**Response 200 — pending/running**

```json
{
  "status": "running",
  "result": null,
  "error": null
}
```

**Response 200 — completed**

```json
{
  "status": "completed",
  "result": {
    "seq": 42,
    "inspected_at": "2026-06-11 14:35:22",
    "image_name": "000.png",
    "image_path": "C:/datasets/bottle/test/broken_large/000.png",
    "verdict": "불량",
    "anomaly_score": 0.843621,
    "was_reshuffled": false
  },
  "error": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `"pending" \| "running" \| "completed" \| "failed"` | Job 상태 |
| `result` | `InspectionResult \| null` | 완료 시 검사 결과 (§9.2 참조) |
| `error` | `string \| null` | 실패 시 오류 메시지 |

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 해당 job_id 없음 (이미 소비됨 포함) |

---

### GET /api/inspection/image/last

마지막 검사 원본 이미지를 반환한다.

**Response 200**: 원본 파일 포맷 바이너리 (`image/jpeg`, `image/png`, `image/bmp` 중 하나)

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 검사 이력 없음, 이미지 파일 없음 |

---

### GET /api/inspection/anomaly-map/last

마지막 검사의 Anomaly Map 히트맵 이미지를 PNG로 반환한다.

**Response 200**: `image/png` 바이너리 (히트맵 컬러맵 적용)

**에러**

| 코드 | 조건 |
|------|------|
| `404` | 검사 이력 없음 또는 Anomaly Map 없음 |

---

### GET /api/inspection/overlay/last

마지막 검사의 이상 영역 오버레이 이미지를 PNG로 반환한다.

**Response 200**: `image/png` 바이너리 (원본 위에 threshold 초과 영역 마스크 오버레이)

**에러**

| 코드 | 조건 |
|------|------|
| `400` | 모델 미선택 |
| `404` | 검사 이력 없음 또는 Anomaly Map 없음 |

---

### GET /api/inspection/records

검사 이력 목록을 반환한다. `seq` 역순 정렬. `image_path` 필드는 제외되어 반환.

**Query Parameters**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `verdict` | `"양품" \| "불량" \| "전체"` | ❌ | 필터 (기본값 `"전체"`) |

**Response 200**

```json
[
  {
    "seq": 42,
    "inspected_at": "2026-06-11 14:35:22",
    "image_name": "000.png",
    "verdict": "불량",
    "anomaly_score": 0.843621
  }
]
```

> `image_path` 필드는 보안 상 응답에서 제외됨. 이미지 조회는 `/api/inspection/image/last` 사용.

---

### GET /api/inspection/records/csv

검사 이력 전체를 CSV로 다운로드한다. 필터 없이 전체 반환.

**Response 200**: `text/csv; charset=utf-8-sig`

```
Content-Disposition: attachment; filename=inspection_history_20260611_143522.csv
```

CSV 컬럼: `번호`, `시각`, `이미지명`, `판정결과`, `Anomaly Score`

---

### DELETE /api/inspection/records

검사 이력을 초기화한다. test pool과 active model은 유지.

**Response 200**

```json
{"success": true}
```

---

## 8. WebSocket 프로토콜 명세

### 8.1 WS /ws/training — 학습 진행상황 Push

Explorer 학습 화면과 연결. 서버 → 클라이언트 단방향 Push (push-only).

**동작 방식**
1. 연결 즉시 현재 상태 스냅샷 전송 (재연결 복구 지원)
2. 이후 학습 이벤트 실시간 Push
3. 단일 사용자 전제: 새 연결이 들어오면 이전 연결의 브로드캐스트 큐가 교체됨

#### 연결 직후 서버 → 클라이언트

```json
{
  "type": "snapshot",
  "status": "running",
  "exp_id": "bottle-20260611-143022-a1b2c3",
  "progress": {"step": 12000, "total": 70000, "loss": 0.003141, "elapsed": 342.5},
  "current_stage_idx": 1,
  "current_stage_name": "Feature Extractor 학습",
  "log_lines": ["[14:30:22] 학습 시작", "..."],
  "loss_history": [{"step": 500, "loss": 0.025}],
  "last_ckpt_path": null
}
```

#### 학습 진행 중 메시지 타입

| `type` | 필드 | 설명 |
|--------|------|------|
| `"progress"` | `step: int`, `total: int`, `loss: float`, `elapsed: float` | 학습 진행상황 |
| `"log"` | `message: string` | 학습 로그 한 줄 |
| `"stage"` | `stage_idx: int`, `stage_name: string` | 학습 단계 전환 |
| `"paused"` | `step: int`, `ckpt_path: string` | 일시정지 완료, 체크포인트 경로 |
| `"completed"` | `exp_id: string`, `auc: float`, `duration_seconds: int`, `message: string` | 학습 완료 |
| `"stopped"` | `step: int` | 사용자 중단 완료 |
| `"error"` | `message: string`, `traceback: string` | 학습 중 예외 |

#### 배치 학습 메시지 타입

| `type` | 필드 | 설명 |
|--------|------|------|
| `"batch_item_started"` | `exp_id: string`, `queue_idx: int` | 배치 항목 학습 시작 |
| `"batch_item_skipped"` | — | 현재 항목 건너뜀 완료 |
| `"batch_item_error"` | `traceback: string` | 배치 항목 오류 (다음 항목 계속 진행) |
| `"batch_stopped"` | `step: int` | 전체 배치 중단 |
| `"batch_completed"` | `completed: int`, `failed: int`, `skipped: int` | 전체 배치 완료 |

#### 메시지 발생 순서 불변 조건

- `completed`, `stopped`, `error` 중 정확히 하나가 단일 학습의 종료를 의미한다.
- `paused`는 종료 메시지가 아니다. 이후 `unpause → progress` 또는 `stop → stopped`로 이어진다.

---

### 8.2 WS /ws/inspection/auto — 자동 검사 Push

Vision 실시간 검사 화면과 연결. 양방향 프로토콜.

**동작 방식**
1. 클라이언트가 `"start"` 전송 → 서버 자동 검사 루프 시작
2. 3초 간격으로 추론 실행 후 `type: "result"` Push
3. 불량 감지 시 `type: "defect_stopped"` Push 후 루프 자동 중지
4. 클라이언트가 `"stop"` 전송 → `type: "stopped"` Push 후 루프 중지

#### 클라이언트 → 서버 (텍스트 메시지)

| 메시지 | 설명 |
|--------|------|
| `"start"` | 자동 검사 루프 시작 |
| `"stop"` | 자동 검사 루프 중지 요청 |

#### 서버 → 클라이언트 (JSON)

| `type` | 필드 | 설명 |
|--------|------|------|
| `"result"` | `seq`, `inspected_at`, `image_name`, `verdict`, `anomaly_score`, `was_reshuffled` | 정상 추론 결과 (§9.2 `InspectionResult` 참조) |
| `"defect_stopped"` | — | 불량 감지, 루프 자동 중지 |
| `"stopped"` | — | `"stop"` 메시지 수신 후 확인 |
| `"error"` | `message: string` | 추론 중 RuntimeError |

#### 자동 검사 타이밍 규칙

- 검사 간격: 3초 (`_INSPECTION_INTERVAL = 3.0`)
- 검사 완료 후 3초 내 `"stop"` 수신 시 즉시 중지
- 3초 타임아웃 → 다음 검사 자동 실행
- 연결 해제 시 `insp_auto_active = False` 자동 설정

---

## 9. 공통 스키마 참조

### 9.1 ActiveModel 구조

`GET /api/inspection/model` 및 `POST /api/inspection/model` 응답의 `active_model` 필드.

```json
{
  "experiment_id": "bottle-20260611-143022-a1b2c3",
  "name": "bottle-efficientad-v1",
  "model_path": "./models/bottle-20260611-143022-a1b2c3",
  "model_type": "EfficientAD",
  "threshold": 0.472,
  "dataset_path": "C:/datasets/bottle",
  "preprocessing_config": {
    "method": "none",
    "background_method": "none",
    "image_size": 256,
    "params": {}
  },
  "score_min": 0.012,
  "score_max": 0.891,
  "device": "cuda",
  "background_method": "none",
  "product_name": "bottle"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `experiment_id` | `string` | 실험 ID |
| `name` | `string` | 실험 이름 |
| `model_path` | `string` | 모델 디렉토리 경로 |
| `model_type` | `string` | `"EfficientAD"` \| `"PatchCore"` |
| `threshold` | `float` | 정규화된 판정 threshold [0, 1] |
| `dataset_path` | `string` | 검사 이미지 소스 경로 (현재 적용 중인 경로) |
| `preprocessing_config` | `dict` | 전처리 설정 |
| `score_min` | `float` | 학습 테스트셋 최솟값 (정규화 기준) |
| `score_max` | `float` | 학습 테스트셋 최댓값 (정규화 기준) |
| `device` | `"cuda" \| "cpu"` | 추론 디바이스 |
| `background_method` | `string` | 배경 제거 방법 |
| `product_name` | `string` | 제품명 |

---

### 9.2 InspectionResult 구조

`GET /api/inspection/job/{job_id}` 응답의 `result` 필드 및 `WS /ws/inspection/auto` `type: "result"` 메시지.

```json
{
  "seq": 42,
  "inspected_at": "2026-06-11 14:35:22",
  "image_name": "000.png",
  "image_path": "C:/datasets/bottle/test/broken_large/000.png",
  "verdict": "불량",
  "anomaly_score": 0.843621,
  "was_reshuffled": false
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `seq` | `int` | 검사 순번 (1-based, 서버 메모리 기준) |
| `inspected_at` | `string` | 검사 시각 (`"YYYY-MM-DD HH:MM:SS"` KST) |
| `image_name` | `string` | 파일명만 |
| `image_path` | `string` | 절대 경로 (이력 GET에서는 제외됨) |
| `verdict` | `"양품" \| "불량"` | 판정 결과 |
| `anomaly_score` | `float` | 정규화된 Anomaly Score [0, 1] |
| `was_reshuffled` | `bool` | 이 샘플에서 test pool이 재셔플됐는지 여부 |

---

### 9.3 서버 인메모리 상태 지속성

FastAPI 서버는 in-memory 상태를 사용한다. **서버 재시작 시 아래 상태는 초기화**된다.

| 상태 | 유지 여부 | 영속 저장소 |
|------|----------|------------|
| 전처리/모델 설정 | ❌ 초기화 | `configs.yaml` (명시적 저장 시) |
| 실험 히스토리 | ✅ 유지 | `history.json` |
| 체크포인트 | ✅ 유지 | 파일시스템 |
| 비전검사 이력 | ❌ 초기화 | 없음 (세션 한정) |
| 적용 중인 모델 | ❌ 초기화 | 없음 (재적용 필요) |
| Anomaly Map 캐시 | ❌ 초기화 | 없음 (재빌드 필요) |
| 학습 상태 | ❌ 초기화 | 체크포인트로 재개 가능 |

---

## 10. v1.x 참고 — Streamlit utils 인터페이스

> **v1.x 참고 전용**: 아래 내용은 v1.x Streamlit 단독 아키텍처의 `utils/` 레이어 인터페이스 명세다. v2.0에서 REST API로 대체됐으나 `utils/` 모듈은 FastAPI Service Layer와 공유하므로 구현 참조용으로 보존.

### 10.1 utils/storage.py

```python
def load_history() -> list[dict]:
    """파일 미존재·파싱 실패 시 [] 반환. 예외 없음."""

def append_experiment(record: dict) -> None:
    """원자적 쓰기. Raises: RuntimeError("ERR_HISTORY_WRITE_FAILED")"""

def delete_experiment_from_history(experiment_id: str) -> bool:
    """제거 성공 시 True, ID 없음 시 False."""

def save_completed_experiment(experiment_id, model, experiment_record) -> None:
    """3단계 원자성 프로토콜. Raises: RuntimeError (단계별 에러코드)"""

def delete_experiment(experiment_id: str, model_path: str | None = None) -> None:
    """예외 없음 (ignore_errors=True)"""

def check_disk_space(required_mb: float = 500.0, path: str = ".") -> tuple[bool, float]:
    """Returns: (충분 여부, 여유 공간 MB)"""

def get_log_writer(experiment_id: str): ...
def read_log_tail(experiment_id: str, n_lines: int = 100) -> str: ...
```

### 10.2 utils/cache_manager.py

```python
MAX_ANOMALY_MAP_CACHE: int = 3

def set_anomaly_map_cache(experiment_id: str, data: dict) -> None:
    """캐시 저장. MAX 초과 시 가장 오래된 항목 자동 제거."""

def get_anomaly_map_cache(experiment_id: str) -> dict | None:
    """반환: {"anomaly_maps": np.ndarray, "image_paths": list[str], "cached_at": float} | None"""

def invalidate_anomaly_map_cache(experiment_id: str) -> None:
    """특정 실험 캐시 제거. no-op if not exists."""
```

### 10.3 v1.x Queue 메시지 프로토콜

`TrainingWorker` → `result_queue` 메시지 TypedDict:

```python
# 타입: "progress" | "log" | "completed" | "error" | "stopped" | "paused"
# completed: {"type", "y_true", "anomaly_scores", "anomaly_maps", "image_paths", "model", "duration_seconds"}
# paused:    {"type", "step", "ckpt_path"}
# stopped:   {"type", "step"}
# error:     {"type", "exception", "traceback"}
```

> v2.0에서는 이 Queue 메시지가 `api/ws/training.py` WebSocket Manager에 의해 소비되어 `WS /ws/training`으로 Push된다. 메시지 필드는 §8.1 참조.

### 10.4 v1.x 탭 Guard 조건

| 탭 | Guard 조건 |
|----|-----------|
| 탭2 | `dataset_path is not None` |
| 탭3 | `dataset_path`, `preprocessing_config`, `model_config` 모두 not None |
| 탭5 | `selected_experiment_id is not None` |

> v2.0에서는 Guard가 React 프론트엔드(Explorer/Vision)에서 구현된다.

### 10.5 v1.x session_state 쓰기 권한

> 상세 내용: 구 06_API_Specification.md §7.2 참조 (git history에서 확인 가능).

---

*다음: [07_Backend_Service_Design.md](./07_Backend_Service_Design.md)*

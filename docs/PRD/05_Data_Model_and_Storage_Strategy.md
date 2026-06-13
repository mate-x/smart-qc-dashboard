# 05. Data Model and Storage Strategy

> **참조 문서**: `00_Global_Context_Document.md` §1 (Core Data Model), §3 (File I/O 계약), §8 (결정 규칙)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **최종수정**: 2026-06-11
> **목적**: 이 시스템의 모든 데이터 스키마(TypeScript 타입 기준), 영속 데이터 및 임시 캐시에 대한 저장 전략, 읽기/쓰기 계약, 원자성 보장, rollback 정책을 구현 가능한 수준으로 명세한다. 04_System_Architecture.md의 파일시스템 레이어 설계를 구체화하고, 07_Backend_Service_Design.md 및 08_AI_ML_Integration.md에서 참조할 스토리지 계약을 선행 정의한다.

---

## 버전 이력

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 초기 작성 — 파일시스템 기반 저장 전략, 원자성 프로토콜, 캐시 정책 |
| v1.1 | 2026-05-26 | §13: 비전검사 대시보드 데이터 저장 전략(Streamlit session_state 기반) 추가 |
| v2.0 | 2026-06-11 | `05_Data_Model.md`(v2.0 TypeScript 타입 역설계) 통합. §3~§5 TypeScript 타입 스키마 섹션 신규 추가. ADR-DS-03: session_state → 서버 메모리로 교체. §11 AnomalyMap 캐시: session_state → `anomaly_map_cache` 서버 인메모리로 교체. §9.2 원자성 프로토콜 호출자: Streamlit → FastAPI 패턴으로 교체. §15 엔티티 관계, §16 PRD↔TS 불일치 대조표 추가. Streamlit 기반 내용 v1.x 참고 섹션으로 이동. |

---

## 목차

1. [설계 원칙 및 ADR](#1-설계-원칙-및-adr)
2. [파일시스템 레이아웃 전체 명세](#2-파일시스템-레이아웃-전체-명세)
3. [스키마 권위 계층 및 타입 파일 위치 맵](#3-스키마-권위-계층-및-타입-파일-위치-맵)
4. [Explorer 타입 스키마](#4-explorer-타입-스키마)
5. [Vision 타입 스키마](#5-vision-타입-스키마)
6. [history.json 상세 명세](#6-historyjson-상세-명세)
7. [configs.yaml 상세 명세](#7-configsyaml-상세-명세)
8. [모델 저장 디렉토리 명세](#8-모델-저장-디렉토리-명세)
9. [모델 저장 3단계 원자성 프로토콜](#9-모델-저장-3단계-원자성-프로토콜)
10. [실험 삭제 프로토콜](#10-실험-삭제-프로토콜)
11. [AnomalyMap 캐시 정책](#11-anomalymap-캐시-정책)
12. [EfficientAD ImageNet Penalty 데이터](#12-efficientad-imagenet-penalty-데이터)
13. [로그 파일 관리](#13-로그-파일-관리)
14. [디스크 용량 모니터링](#14-디스크-용량-모니터링)
15. [엔티티 관계](#15-엔티티-관계)
16. [PRD ↔ TypeScript 불일치 대조표](#16-prd--typescript-불일치-대조표)
17. [구현 체크리스트](#17-구현-체크리스트)

---

## 1. 설계 원칙 및 ADR

### ADR-DS-01: 파일시스템 기반 영속화 + MySQL 인프라 병행

| 항목 | 내용 |
|------|------|
| **결정** | 모델 가중치(`.pth`), 학습 로그(`.log`), 설정(`configs.yaml`), 실험 히스토리(`history.json`)는 파일시스템에 저장한다. MySQL 8.0은 인프라 레이어에 포함하며, 향후 구조화 데이터 저장에 활용한다. |
| **근거** | MVP 단일 사용자 환경(A-01)에서 파일시스템으로 충분하나, MySQL을 인프라에 포함해 확장성을 확보한다. |
| **트레이드오프** | 동시성 제어 불가 → 단일 사용자 가정으로 수용. 전체 히스토리 로드 시 파일 전체 읽기 필요 → 최대 실험 수 제한 없음(실험 수백 개 이하에서 성능 문제 없음). |

### ADR-DS-02: 모든 파일 쓰기는 원자적으로 수행

| 항목 | 내용 |
|------|------|
| **결정** | 단일 파일 쓰기는 `.tmp` → `rename` 패턴. 다단계 저장(모델 + 설정 + 히스토리)은 §9의 3단계 프로토콜을 따른다. |
| **근거** | 쓰기 도중 프로세스 종료 시 부분 쓰기로 인한 데이터 손상 방지. `os.replace()` (POSIX rename)는 원자적 보장. |

### ADR-DS-03: 서버 메모리와 영속 스토리지의 엄격한 경계 (v2.0)

| 항목 | 내용 |
|------|------|
| **결정** | FastAPI 서버 인메모리 상태(`TrainingManager`, `InspectionManager`, `anomaly_map_cache`)는 임시 캐시 전용. 영속이 필요한 데이터는 반드시 파일에 저장한다. 서버 메모리에만 있는 데이터는 프로세스 재시작 시 소실됨을 전제로 설계한다. |
| **근거** | FastAPI 단일 인스턴스(A-01). 서버 재시작 시 history.json에서 실험 레코드를 재로드해 상태를 복원한다. |
| **v1.x 변경** | v1.x: `st.session_state`는 임시 캐시 전용. v2.0: `session_state` → FastAPI 서버 메모리 상태로 교체. |

---

## 2. 파일시스템 레이아웃 전체 명세

```
{WORKDIR}/                          # Docker: /app, 로컬: 프로젝트 루트
│
├── configs.yaml                    # [읽기/쓰기] Explorer Config 화면 공유 설정
│                                   # 학습 시작 시점의 스냅샷 기준
│
├── experiments/
│   └── history.json                # [읽기/쓰기] 실험 레코드 배열
│                                   # 초기 미존재 → load_history()가 [] 반환
│                                   # Explorer: 쓰기 / Vision: 읽기 전용 (R-INSP-04)
│
├── models/
│   └── {experiment_id}/            # 실험별 디렉토리 (학습 완료 시 생성)
│       ├── model_state_dict.pth    # PyTorch state_dict
│       └── configs.yaml            # 학습 시점 설정 스냅샷 (루트 configs.yaml 복사본)
│
├── models/checkpoints/             # 일시정지 체크포인트
│   └── {experiment_id}_step{N}.ckpt
│
├── logs/
│   └── {experiment_id}.log         # 학습 로그 (학습 시작 시 생성, append 전용)
│
├── dataset/
│   └── {사용자_데이터셋}/           # [읽기 전용] Docker 볼륨 마운트 대상
│       ├── train/good/
│       ├── test/{class}/
│       └── ground_truth/{class}/
│
└── dataset/
    └── imagenet_penalty/           # [읽기 전용] EfficientAD penalty 데이터 (§12 참조)
        └── *.jpg / *.png
```

### 경로 접근 규칙

| 규칙 | 내용 |
|------|------|
| **R-PATH-01** (00_Global §8) | 모든 경로는 `pathlib.Path` 사용. 문자열 연산(`os.path.join` 포함) 금지. |
| **절대 경로 vs 상대 경로** | 파일 저장 시 `Path("./experiments/history.json")` 형식의 상대 경로 사용. Docker WORKDIR(`/app`)에서 자동 해석됨. |
| **WORKDIR 환경변수** | `WORKDIR` 환경변수가 설정된 경우 `Path(os.environ.get("WORKDIR", "."))` 를 베이스로 사용. 미설정 시 `.` (현재 디렉토리). |

---

## 3. 스키마 권위 계층 및 타입 파일 위치 맵

### 3.1 스키마 권위 계층 (v2.0)

```
TypeScript src/types/*.ts     ← 실제 구현 기준 (§4, §5의 근거)
     ↕ 일치해야 함
FastAPI 응답 스키마            ← 런타임 계약 기준
     ↕ 파생
history.json 파일 구조         ← 영속 저장 기준 (§6 참조)
```

PRD ↔ TypeScript 불일치 목록은 §16 참조.

### 3.2 타입 파일 위치 맵

| 타입 파일 | 레포 | 용도 |
|-----------|------|------|
| `src/types/experiments.ts` | smart-qc-explorer | 실험 레코드, 메트릭 |
| `src/types/dataset.ts` | smart-qc-explorer | 데이터셋 검증 응답 |
| `src/types/config.ts` | smart-qc-explorer | 전처리·모델 설정, 큐 |
| `src/types/modelParams.ts` | smart-qc-explorer | EfficientAD·PatchCore 파라미터 |
| `src/types/training.ts` | smart-qc-explorer | 학습 상태, WS 메시지 |
| `src/types/anomalyMap.ts` | smart-qc-explorer | Anomaly Map 이미지·job |
| `src/types/inspection.ts` | smart-qc-vision | 검사 결과·이력, WS 메시지 |
| `src/types/model.ts` | smart-qc-vision | 실험 목록(읽기 전용), 적용 모델 |
| `src/types/api.ts` | smart-qc-vision | API 응답 래퍼 |

---

## 4. Explorer 타입 스키마

### 4.1 Experiment — history.json 레코드

**파일:** `smart-qc-explorer/src/types/experiments.ts`

```typescript
export interface Experiment {
  experiment_id: string;                        // PK: "{model_type}_{YYYYMMDD}_{HHMMSS}_{4자리난수}"
  name: string;                                 // 사용자 지정명 또는 자동 생성, max 64자
  status: 'completed' | '중단' | '실패' | '건너뜀'; // v2.0: '실패' | '건너뜀' 추가
  created_at: string;                           // ISO 8601, KST

  model_type: 'efficientad' | 'patchcore';
  metrics: ExperimentMetrics | null;            // null if status != 'completed'
  duration_seconds: number | null;              // null if status != 'completed'
  model_path: string | null;                    // null if status != 'completed'
  configs_path?: string | null;                 // 선택적 (저장 경로 지정 시)

  product_name?: string;                        // 검사 제품명
  set_id?: string | null;                       // 배치 비교용 그룹 ID

  preprocessing_method?: string;                // "none" | "homomorphic" | "he" | "clahe"
  preprocessing_params?: Record<string, unknown> | null;
  background_method?: 'none' | 'sam2';          // 배경 제거 방법
  model_params?: Record<string, unknown>;        // 모델별 파라미터
  threshold_method?: string;                    // "percentile" | "absolute"
  threshold_value?: number;
  dataset_path?: string;
  image_size?: number;
  early_stopped?: boolean;                      // 조기 종료 여부
}
```

**status ENUM 상세:**

| 값 | 의미 | metrics | model_path |
|----|------|---------|------------|
| `'completed'` | 정상 완료 | NOT NULL | NOT NULL |
| `'중단'` | 사용자 [학습 중지] 클릭 | NULL | NULL |
| `'실패'` | 오류로 학습 실패 | NULL | NULL |
| `'건너뜀'` | 배치 학습에서 건너뜀 처리 | NULL | NULL |

> **v1.x 대비 변경**: 00_Global_Context 1.1절에서 `status: "completed" | "중단"` 두 값만 정의했으나, TypeScript 타입에 `'실패' | '건너뜀'`이 추가됨. 공식 ENUM은 4가지 값으로 확정.

---

### 4.2 ExperimentMetrics

**파일:** `smart-qc-explorer/src/types/experiments.ts`

```typescript
export interface ExperimentMetrics {
  accuracy?: number;          // 0.0 ~ 1.0
  precision?: number;         // 0.0 ~ 1.0
  recall?: number;            // 0.0 ~ 1.0
  f1_score?: number;          // 0.0 ~ 1.0
  f2_score?: number;          // 0.0 ~ 1.0
  auc?: number;               // 0.0 ~ 1.0 (AUROC)
  confusion_matrix?: ConfusionMatrix;
  anomaly_scores?: number[];  // 테스트 이미지별 raw score 배열
  image_labels?: number[];    // 테스트 이미지별 정답 레이블 (0=정상, 1=결함)
}
```

> **주의**: 모든 필드가 선택적(`?`)으로 정의됨. 완료 실험에서 `metrics !== null`인 경우 필드가 존재하나 TypeScript 계약상 모두 optional. 실제 사용 시 null 체크 필요.

---

### 4.3 ConfusionMatrix

```typescript
export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}
```

---

### 4.4 DatasetValidateResponse

**파일:** `smart-qc-explorer/src/types/dataset.ts`

POST `/api/dataset/validate` 응답 스키마.

```typescript
export interface DatasetValidateResponse {
  dataset_format: string;             // "mvtec" | "oking"
  channels: 1 | 3;                   // 1=Grayscale, 3=RGB
  train_good_count: number;
  test_counts: Record<string, number>; // { "good": N, "<defect_class>": N, ... }
  gt_counts: Record<string, number>;  // { "<defect_class>": N, ... }
  total_test_count: number;
  defect_classes: string[];
  supported_formats: string[];        // ["jpg", "png"] 등
  has_invalid_files: boolean;
  invalid_file_count: number;
  folder_tree: string;                // 폴더 트리 문자열 (표시용)
  has_background_clean: boolean;

  // OK/NG 형식 전용 (dataset_format === "oking" 시만 존재)
  oking_ok_dir?: string | null;
  oking_ng_dir?: string | null;
  oking_ok_count?: number | null;
  oking_ng_count?: number | null;
  train_ratio?: number | null;        // 학습/테스트 분할 비율 (기본 0.8)
}
```

---

### 4.5 PreprocessingConfig

**파일:** `smart-qc-explorer/src/types/config.ts`

```typescript
export interface PreprocessingConfig {
  method: 'none' | 'homomorphic' | 'he' | 'clahe';
  background_method: 'none' | 'sam2';   // v2.0 추가
  resize_mode: 'padding';               // 고정값
  image_size: number;                   // 32 ~ 1024 (32의 배수)
  normalization: 'imagenet';            // v2.0: 'imagenet' 고정
  mean: [number, number, number];       // [0.485, 0.456, 0.406]
  std: [number, number, number];        // [0.229, 0.224, 0.225]
  params: Record<string, unknown> | null;
}
```

**method별 params:**

| method | params |
|--------|--------|
| `'none'` | `null` |
| `'he'` | `{}` (빈 객체) |
| `'homomorphic'` | `{ sigma, gamma_H, gamma_L, normalize }` |
| `'clahe'` | `{ clip_limit }` |

---

### 4.6 ModelConfig

**파일:** `smart-qc-explorer/src/types/config.ts`

```typescript
export interface ModelConfig {
  model_type: 'efficientad' | 'patchcore';
  batch_size: number;                         // 1 ~ 128
  random_seed: number;                        // 0 ~ 2147483647
  threshold_method: 'percentile' | 'absolute';
  threshold_value: number;
  params: EfficientAdParamsState | PatchCoreParamsState;
}
```

> **v1.x 대비 변경**: `image_size`가 `ModelConfig`에서 제거됨 → `PreprocessingConfig.image_size`로 일원화.

---

### 4.7 EfficientAdParamsState

**파일:** `smart-qc-explorer/src/types/modelParams.ts`

```typescript
export interface EfficientAdParamsState {
  model_size: 'small' | 'medium';
  train_steps: number;                // 1 ~ 200000
  optimizer: 'adam' | 'adamw' | 'sgd';
  learning_rate: number;
  weight_decay: number;
  out_channels: 128 | 256 | 384 | 512;
  padding: boolean;
  ae_loss_weight: number;
  autoencoder_lr: number;
  autoencoder_weight_decay: number;
  lr_decay_epochs: number;
  lr_decay_factor: number;
  scheduler: 'StepLR' | 'CosineAnnealingLR';
  use_imagenet_penalty: boolean;
  penalty_batch_size: number;
  early_stopping: boolean;            // v2.0 추가
  patience: number;                   // v2.0 추가
  min_delta: number;                  // v2.0 추가
}
```

---

### 4.8 PatchCoreParamsState

**파일:** `smart-qc-explorer/src/types/modelParams.ts`

```typescript
export interface PatchCoreParamsState {
  backbone: 'wide_resnet50_2' | 'resnet18' | 'resnet50';
  pretrained_source: 'torchvision' | 'local';
  pretrained_path: string | null;
  coreset_sampling_ratio: number;           // 0.01 ~ 1.0
  neighbourhood_kernel_size: 1 | 3 | 5 | 7 | 9;
  max_train: number;                        // 100 ~ 10000
  knn: number;                              // 1 ~ 50
  top_k_ratio: number;                      // 0.0 ~ 1.0
}
```

---

### 4.9 DeviceInfo

**파일:** `smart-qc-explorer/src/types/config.ts`

```typescript
export interface DeviceInfo {
  device: 'cuda' | 'cpu';
  gpu_name?: string;
  vram_gb?: number;
}
```

---

### 4.10 QueueItem

**파일:** `smart-qc-explorer/src/types/config.ts`

```typescript
export interface QueueItem {
  id: string;
  name: string;
  preprocessing_config: PreprocessingConfig;
  model_config: ModelConfig;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  set_id?: string | null;
}
```

---

### 4.11 Training 런타임 타입

**파일:** `smart-qc-explorer/src/types/training.ts`

```typescript
export type TrainingStatus = 'idle' | 'running' | 'paused';

export interface TrainingProgress {
  step: number;
  total: number;
  loss: number;
  elapsed: number;
}

export interface LossPoint {
  step: number;
  loss: number;
}

export interface CheckpointInfo {
  name: string;
  model_type: string;
  created_at: string;
  step?: number;
  total_steps?: number;
  batch_idx?: number;
  total_batches?: number;
  n_patches?: number;
}

export interface TrainingStatusResponse {
  status: TrainingStatus;
  exp_id: string | null;
  batch_mode: boolean;
  batch_total: number;
  progress: TrainingProgress | null;
  current_stage_idx: number | null;
  current_stage_name: string | null;
  log_lines: string[];
  loss_history: LossPoint[];
  last_ckpt_path: string | null;
}
```

---

### 4.12 Explorer WsMessage

**파일:** `smart-qc-explorer/src/types/training.ts`

WS `/ws/training` 수신 메시지 유니온 타입.

```typescript
export type WsMessage =
  | ({ type: 'snapshot' } & TrainingStatusResponse)
  | { type: 'progress'; step: number; total: number; loss: number; elapsed: number }
  | { type: 'log'; message: string }
  | { type: 'stage'; stage_idx: number; stage_name: string }
  | { type: 'paused'; step: number; ckpt_path: string }
  | { type: 'completed'; exp_id: string; auc: number; duration_seconds: number; message: string; early_stopped: boolean }
  | { type: 'stopped'; step: number }
  | { type: 'error'; message: string; traceback: string }
  | { type: 'batch_item_started'; exp_id: string; queue_idx: number }
  | { type: 'batch_item_skipped' }
  | { type: 'batch_item_error'; traceback: string }
  | { type: 'batch_stopped'; step: number }
  | { type: 'batch_completed'; completed: number; failed: number; skipped: number };
```

| type | 발생 시점 |
|------|-----------|
| `snapshot` | WS 연결 직후 — 현재 전체 상태 전송 |
| `progress` | 학습 진행 중 — step/loss 업데이트 |
| `log` | 학습 로그 한 줄 |
| `stage` | 학습 단계 변경 |
| `paused` | 일시정지 완료, 체크포인트 저장 완료 |
| `completed` | 학습 정상 완료 |
| `stopped` | 중단 처리 완료 |
| `error` | 학습 중 오류 발생 |
| `batch_item_started` | 배치 — 다음 큐 항목 시작 |
| `batch_item_skipped` | 배치 — 현재 항목 건너뜀 |
| `batch_item_error` | 배치 — 현재 항목 실패 |
| `batch_stopped` | 배치 — 전체 중단 |
| `batch_completed` | 배치 — 전체 완료 |

---

### 4.13 AnomalyMap 타입

**파일:** `smart-qc-explorer/src/types/anomalyMap.ts`

```typescript
export interface AnomalyImage {
  image_name: string;
  defect_class: string;       // "good" | "crack" | 등
  anomaly_score: number;
  verdict: string;            // "양품" | "불량"
  gt_match: boolean;
  classification: string;     // "TP" | "FP" | "TN" | "FN"
  image_path: string;
}

export interface AnomalyMapImagesResponse {
  images: AnomalyImage[];
  score_max: number;
  score_avg: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface JobStatusResponse {
  status: JobStatus;
  error?: string;
}

export interface AnomalyMapStatus {
  built: boolean;
  image_count: number;
}
```

---

## 5. Vision 타입 스키마

### 5.1 InspectionResult

**파일:** `smart-qc-vision/src/types/inspection.ts`

수동 검사(POST /api/inspection/run 응답) 및 WS result 메시지에 사용.

```typescript
export interface InspectionResult {
  seq: number;                        // 세션 내 순번 (1부터 자동 증가)
  inspected_at: string;               // ISO 8601, KST
  image_name: string;
  image_path: string;                 // 절대 경로 (이미지 서빙에 사용)
  verdict: '양품' | '불량';
  anomaly_score: number;              // round(value, 4)
  was_reshuffled: boolean;
}
```

---

### 5.2 InspectionRecord

**파일:** `smart-qc-vision/src/types/inspection.ts`

GET `/api/inspection/records` 응답 항목. History 화면 테이블 표시용.

```typescript
export interface InspectionRecord {
  seq: number;
  inspected_at: string;
  image_name: string;
  verdict: '양품' | '불량';
  anomaly_score: number;
}
```

> **InspectionResult vs InspectionRecord**: `InspectionResult`는 검사 직후 서버 응답으로 `image_path` + `was_reshuffled` 포함. `InspectionRecord`는 이력 조회용 — 화면 표시에 필요한 5개 필드만 포함.

---

### 5.3 VerdictFilter

```typescript
export type VerdictFilter = '전체' | '양품' | '불량';
```

---

### 5.4 검사 요청·응답 타입

**파일:** `smart-qc-vision/src/types/inspection.ts`

```typescript
export interface InspectionJobStarted {
  job_id: string;
}

export interface RunInspectionRequest {
  defect_only?: boolean;
}

export type InspectionJobStatus =
  | { status: 'pending' | 'running' }
  | { status: 'completed'; result: InspectionResult }
  | { status: 'failed'; error: string };
```

**수동 검사 polling 흐름:**

```
POST /api/inspection/run
  → { job_id }
  → polling GET /api/inspection/job/{job_id} (1초 간격, max 120초)
    → { status: 'pending' | 'running' }
    → { status: 'completed', result: InspectionResult }
    → { status: 'failed', error: ... }
```

---

### 5.5 Vision WsMessage

**파일:** `smart-qc-vision/src/types/inspection.ts`

```typescript
export type WsMessage =
  | {
      type: 'result';
      seq: number;
      inspected_at: string;
      image_name: string;
      image_path: string;
      verdict: '양품' | '불량';
      anomaly_score: number;
      was_reshuffled: boolean;
    }
  | { type: 'defect_stopped' }
  | { type: 'stopped' }
  | { type: 'error'; message: string };
```

| type | 발생 시점 | 클라이언트 처리 |
|------|-----------|----------------|
| `result` | 추론 1회 완료 | `setLastResult()` + `imageStamp = Date.now()` |
| `defect_stopped` | 불량 감지 → 서버 루프 중지 | `setDefectStopped(true)` → DefectPopup |
| `stopped` | 클라이언트 `stop` 명령 서버 확인 | (클라이언트 이미 처리) |
| `error` | 서버 오류 | `setAutoRunning(false)` |

> **Client→Server**: `'start'` (자동 검사 시작), `'stop'` (중지 요청)

---

### 5.6 ExperimentRecord (Vision 읽기 전용)

**파일:** `smart-qc-vision/src/types/model.ts`

GET `/api/models` 응답 항목. `history.json` Experiment에서 Vision 필요 필드만 추출.

```typescript
export interface ExperimentRecord {
  experiment_id: string;
  name?: string;
  model_type: string;
  product_name: string;
  background_method: string;
  model_path: string;
  dataset_path: string;
  created_at: string;
  threshold_method: string;
  threshold_value: number;
  metrics: {
    f1_score: number;
    auc: number;
    anomaly_scores: number[];
    image_labels: number[];
  };
  preprocessing_method: string;
  preprocessing_params: Record<string, unknown>;
  image_size: number;
  status: string;
}
```

> Vision에서 읽기 전용 (history.json 쓰기 금지 — R-INSP-04).

---

### 5.7 ActiveModel

**파일:** `smart-qc-vision/src/types/model.ts`

`inspectionStore.activeModel`의 타입.

```typescript
export interface ActiveModel {
  experiment_id: string;
  name: string;
  model_path: string;
  model_type: string;
  product_name: string;
  background_method: string;
  threshold: number;                    // 계산된 단일 값 (method+value → raw score 변환 완료)
  dataset_path: string;
  preprocessing_config: {
    method: string;
    params: Record<string, unknown>;
    image_size: number;
  };
  score_min: number;                    // Min-Max 정규화용
  score_max: number;
  device: string;                       // "cuda" | "cpu"
}
```

**Experiment vs ActiveModel 주요 차이:**

| 필드 | Experiment | ActiveModel |
|------|-----------|-------------|
| threshold | `threshold_method` + `threshold_value` (분리) | `threshold` (단일 계산값) |
| score 범위 | — | `score_min`, `score_max` |
| device | — | `device` |
| preprocessing | 분리 필드 | 중첩 객체 |

---

### 5.8 API 응답 타입

**파일:** `smart-qc-vision/src/types/api.ts`

```typescript
export interface ApplyModelResponse {
  success: boolean;
  active_model: ActiveModel;
  gpu_warning: string | null;
}

export interface GetActiveModelResponse {
  active_model: ActiveModel | null;
}
```

---

## 6. history.json 상세 명세

### 6.1 파일 구조 예시

```json
[
  {
    "experiment_id": "efficientad_20260508_140023_7f3a",
    "name": "EfficientAD_screw_clahe_v1",
    "status": "completed",
    "created_at": "2026-05-08T14:00:23+09:00",
    "model_type": "efficientad",
    "metrics": {
      "accuracy": 0.9750,
      "precision": 0.9600,
      "recall": 0.9800,
      "f1_score": 0.9699,
      "f2_score": 0.9757,
      "auc": 0.9900,
      "confusion_matrix": { "tp": 98, "fp": 4, "tn": 96, "fn": 2 },
      "anomaly_scores": [0.1234, 0.8765],
      "image_labels": [0, 1]
    },
    "duration_seconds": 1180,
    "model_path": "./models/efficientad_20260508_140023_7f3a/",
    "configs_path": "./models/efficientad_20260508_140023_7f3a/configs.yaml",
    "product_name": "screw",
    "set_id": null,
    "preprocessing_method": "clahe",
    "preprocessing_params": { "clip_limit": 2.0 },
    "background_method": "none",
    "model_params": { "model_size": "medium", "train_steps": 70000 },
    "threshold_method": "percentile",
    "threshold_value": 90.0,
    "dataset_path": "/app/dataset/screw",
    "image_size": 256,
    "early_stopped": false
  }
]
```

### 6.2 읽기 계약

```python
# utils/storage.py

def load_history() -> list[dict]:
    """
    반환: 실험 레코드 리스트. 파일 미존재 시 빈 리스트.
    예외: JSON 파싱 실패 시 빈 리스트 반환 + WARNING 로그 (예외 전파 금지).
    """
    path = Path("./experiments/history.json")
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        _log_warning("history_load_failed", {"path": str(path)})
        return []
```

**중요**: `load_history()`는 절대 예외를 전파하지 않는다.

### 6.3 쓰기 계약 (원자적 Append)

```python
def append_experiment(record: dict) -> None:
    path = Path("./experiments/history.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    records = load_history()
    records.append(record)

    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tmp.replace(path)   # 원자적 rename (R-ATOMIC-01)
    except IOError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ERR_HISTORY_WRITE_FAILED: {e}") from e
```

### 6.4 삭제 계약

```python
def delete_experiment_from_history(experiment_id: str) -> bool:
    records = load_history()
    filtered = [r for r in records if r["experiment_id"] != experiment_id]
    if len(filtered) == len(records):
        return False  # 해당 ID 없음

    path = Path("./experiments/history.json")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return True
```

### 6.5 history.json 무결성 규칙

| 규칙 | 설명 |
|------|------|
| **중복 ID 금지** | `append_experiment()` 호출 전 동일 `experiment_id` 존재 여부 확인 |
| **status ∈ {'중단','실패','건너뜀'} 레코드** | `model_path`, `configs_path`, `metrics`는 반드시 `null` (§9.1 R-05) |
| **배열 타입 보장** | `load_history()` 반환값이 list가 아닌 경우 빈 리스트 처리 |
| **Vision 쓰기 금지** | R-INSP-04: Vision 레포에서 `append_experiment()` 호출 금지 |

---

## 7. configs.yaml 상세 명세

### 7.1 파일 위치 및 역할

| 파일 | 위치 | 역할 |
|------|------|------|
| **공유 설정 파일** | `./configs.yaml` | Explorer Config 화면 파라미터 편집용. 학습 시작 시 스냅샷 생성. |
| **실험 스냅샷** | `./models/{exp_id}/configs.yaml` | 학습 시점 설정 고정 사본. 이후 변경 불가. |

### 7.2 파일 예시 (v2.0)

```yaml
experiment:
  name: "EfficientAD_screw_clahe_v1"
  created_at: "2026-05-08T14:00:23+09:00"

preprocessing:
  method: "clahe"
  background_method: "none"       # v2.0 추가
  resize_mode: "padding"
  image_size: 256
  normalization: "imagenet"
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  params:
    clip_limit: 2.0

model:
  model_type: "efficientad"
  batch_size: 16
  random_seed: 42
  threshold_method: "percentile"
  threshold_value: 90.0
  params:
    model_size: "medium"
    train_steps: 70000
    optimizer: "adam"
    learning_rate: 0.0001
    weight_decay: 0.0001
    out_channels: 384
    padding: false
    ae_loss_weight: 0.5
    autoencoder_lr: 0.0001
    autoencoder_weight_decay: 0.00001
    lr_decay_epochs: 50000
    lr_decay_factor: 0.1
    scheduler: "StepLR"
    use_imagenet_penalty: false
    penalty_batch_size: 8
    early_stopping: false
    patience: 5
    min_delta: 0.001
```

### 7.3 읽기 계약

```python
# utils/config_manager.py

def load_config(path: str | Path = "./configs.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        _log_warning("config_load_failed", {"path": str(p)})
        return {}
```

### 7.4 섹션 업데이트 계약

```python
def save_config_section(
    section: str,
    data: dict,
    path: str | Path = "./configs.yaml"
) -> None:
    """
    기존 파일의 다른 섹션을 보존하면서 지정 섹션만 업데이트.
    원자적 쓰기 적용 (tmpfile → rename).

    section: "experiment" | "preprocessing" | "model"
    path: 루트 configs.yaml 또는 모델 스냅샷 경로
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(p)
    config[section] = data

    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        tmp.replace(p)
    except IOError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ERR_CONFIG_WRITE_FAILED: {e}") from e
```

### 7.5 화면별 configs.yaml 접근 권한 (v2.0)

| 화면 | section | 접근 종류 | 대상 파일 |
|------|---------|-----------|-----------|
| Explorer Config 화면 | `"preprocessing"` + `"model"` | Write | `./configs.yaml` |
| POST /api/training/start (학습 시작) | `"experiment"` + 전체 | Read + 스냅샷 Write | `./configs.yaml` (Read), `./models/{exp_id}/configs.yaml` (Write) |
| GET /api/anomaly-map/{expId}/images | `"preprocessing"`, `"model"` | Read 전용 | `./models/{exp_id}/configs.yaml` |

---

## 8. 모델 저장 디렉토리 명세

### 8.1 디렉토리 구조

```
./models/
└── {experiment_id}/
    ├── model_state_dict.pth        # PyTorch state_dict
    └── configs.yaml                # 학습 시점 설정 스냅샷

./models/checkpoints/
└── {experiment_id}_step{N}.ckpt   # 일시정지 체크포인트
```

### 8.2 model_state_dict.pth

| 항목 | 내용 |
|------|------|
| **저장 방식** | `torch.save(model.state_dict(), path)` |
| **로드 방식** | `model.load_state_dict(torch.load(path, map_location=device))` |
| **EfficientAD 크기** | ≈ 200~400 MB |
| **PatchCore 크기** | ≈ 600~1000 MB |

> `torch.save(model, path)` 형식 금지. `state_dict()`만 저장.

### 8.3 파일 접근 규칙

| 연산 | Explorer | Vision | Dashboard API |
|------|----------|--------|---------------|
| 저장 (save) | POST /api/experiments/{id}/save 경유 | ✗ | ✓ |
| 로드 (run_inference) | ✗ | POST /api/inspection/model 경유 | ✓ |
| 직접 경로 접근 | ✗ | ✗ | ✓ |

### 8.4 디렉토리 생성 규칙

```python
def prepare_model_dir(experiment_id: str) -> Path:
    model_dir = Path(f"./models/{experiment_id}")
    if model_dir.exists():
        raise RuntimeError(f"모델 디렉토리가 이미 존재합니다: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=False)
    return model_dir
```

---

## 9. 모델 저장 3단계 원자성 프로토콜

> 이 절은 07_Backend_Service_Design.md §6 (완료 후처리) 및 08_AI_ML_Integration.md §8과 연동된다.

### 9.1 3단계 순서

```
[단계 1] model_state_dict.pth 저장
    → ./models/{exp_id}/model_state_dict.pth
    → 실패 시: 디렉토리 삭제 → history 미기록 → ERR_MODEL_SAVE_FAILED

[단계 2] configs.yaml 스냅샷 저장
    → ./models/{exp_id}/configs.yaml
    → 실패 시: model_state_dict.pth + 디렉토리 삭제 → history 미기록 → ERR_MODEL_SAVE_FAILED

[단계 3] history.json append
    → experiment record (status="completed", model_path 설정)
    → 실패 시: 2단계까지 성공한 파일 보존 (고립 디렉토리) + ERR_HISTORY_WRITE_FAILED
             → WS error 메시지에 경고 포함: "모델 파일은 저장되었으나 히스토리 기록에 실패했습니다."
```

**3단계의 부분 실패 처리**: `model_state_dict.pth`는 수백 MB의 가치 있는 데이터이므로 히스토리 기록만 실패한 경우 파일 보존 우선.

### 9.2 전체 구현 코드

```python
# utils/storage.py

def save_completed_experiment(
    experiment_id: str,
    model,                  # EfficientAd | Patchcore 인스턴스
    experiment_record: dict # status="completed" 레코드 (model_path, configs_path 포함)
) -> None:
    model_dir = prepare_model_dir(experiment_id)
    pth_path = model_dir / "model_state_dict.pth"
    cfg_path = model_dir / "configs.yaml"

    # 단계 1: model_state_dict.pth
    try:
        torch.save(model.state_dict(), pth_path)
    except Exception as e:
        _cleanup_dir(model_dir)
        raise RuntimeError(f"ERR_MODEL_SAVE_FAILED (단계1): {e}") from e

    # 단계 2: configs.yaml 스냅샷
    try:
        root_config = load_config("./configs.yaml")
        root_config.setdefault("experiment", {})
        root_config["experiment"]["name"] = experiment_record["name"]
        root_config["experiment"]["created_at"] = experiment_record["created_at"]
        save_config_section("experiment", root_config["experiment"], cfg_path)
    except Exception as e:
        _cleanup_dir(model_dir)
        raise RuntimeError(f"ERR_MODEL_SAVE_FAILED (단계2): {e}") from e

    # 단계 3: history.json
    experiment_record["model_path"] = str(model_dir) + "/"
    experiment_record["configs_path"] = str(cfg_path)
    try:
        append_experiment(experiment_record)
    except RuntimeError as e:
        raise RuntimeError(
            f"ERR_HISTORY_WRITE_FAILED: 모델 저장 성공, 히스토리 기록 실패. "
            f"model_path={model_dir}"
        ) from e


def _cleanup_dir(path: Path) -> None:
    import shutil
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
```

### 9.3 호출자 처리 패턴 (v2.0 — FastAPI)

```python
# api/services/training_manager.py — _handle_completed()

try:
    save_completed_experiment(exp_id, model, record)
    # 성공: WS completed 메시지는 이미 브로드캐스트됨
except RuntimeError as e:
    error_msg = str(e)
    if "ERR_HISTORY_WRITE_FAILED" in error_msg:
        # 모델은 저장됨 — WS error 메시지로 클라이언트에 알림
        await _broadcast_to_clients({
            "type": "error",
            "message": f"모델 파일은 저장되었으나 히스토리 기록에 실패했습니다. {error_msg}",
            "traceback": ""
        })
    else:
        await _broadcast_to_clients({
            "type": "error",
            "message": f"모델 저장에 실패했습니다. 디스크 공간을 확인해 주세요. {error_msg}",
            "traceback": ""
        })
```

---

## 10. 실험 삭제 프로토콜

> Explorer Experiments 화면에서 [실험 삭제] 클릭 시 실행.

### 10.1 삭제 순서

```
[단계 1] history.json에서 레코드 제거 (delete_experiment_from_history)
[단계 2] ./models/{exp_id}/ 디렉토리 삭제 (model_path가 NOT NULL인 경우만)
[단계 3] ./logs/{exp_id}.log 파일 삭제 (존재하는 경우만)
[단계 4] 서버: anomaly_map_cache에서 해당 exp_id 캐시 제거 (v2.0)
         [확인 필요: DELETE /api/experiments/{id} 핸들러에서 cache_manager.invalidate(exp_id) 호출]
```

### 10.2 삭제 구현

```python
# utils/storage.py

def delete_experiment(experiment_id: str, model_path: str | None = None) -> None:
    import shutil

    # 단계 1: history.json
    delete_experiment_from_history(experiment_id)

    # 단계 2: 모델 디렉토리
    if model_path:
        model_dir = Path(model_path)
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    # 단계 3: 로그 파일
    log_path = Path(f"./logs/{experiment_id}.log")
    log_path.unlink(missing_ok=True)
```

### 10.3 삭제 시 참조 무결성 (v2.0)

| 조건 | 처리 |
|------|------|
| 삭제 대상 anomaly_maps 캐시가 서버에 존재 | `cache_manager.invalidate(exp_id)` 호출 |
| `status == "중단"/"실패"` 레코드 삭제 | model_path가 null이므로 파일 삭제 단계 건너뜀 |

---

## 11. AnomalyMap 캐시 정책

> **v2.0**: `st.session_state` 캐시 → FastAPI 서버 인메모리 `anomaly_map_cache`로 교체.

### 11.1 캐시 구조 (v2.0)

```python
# api/services/cache_manager.py

# 서버 인메모리 캐시 (모듈 레벨 dict)
_anomaly_map_cache: dict[str, dict] = {}
# key: experiment_id
# value: {
#   "anomaly_maps": np.ndarray (N, H, W) float32,
#   "image_paths":  list[str],
#   "cached_at":    float   # time.time()
# }
```

### 11.2 크기 추정

| 조건 | 크기 |
|------|------|
| 테스트 이미지 100장, image_size=256 | `100 × 256 × 256 × 4 bytes ≈ 25 MB` |
| 테스트 이미지 500장, image_size=256 | `≈ 125 MB` |
| 테스트 이미지 100장, image_size=512 | `≈ 100 MB` |

### 11.3 캐시 수명 및 무효화 조건 (v2.0)

| 이벤트 | 처리 |
|--------|------|
| **GET /api/anomaly-map/{expId}/images 요청** | 캐시 존재 확인. 있으면 재사용, 없으면 빌드 job 실행 필요 |
| **다른 실험 조회** | 기존 캐시 유지 (새 exp_id 키로 별도 캐시) |
| **DELETE /api/experiments/{id}** | 해당 exp_id 캐시 즉시 제거 |
| **서버 재시작** | 캐시 전체 소멸 (서버 메모리 소실) |
| **캐시 상한** | 최대 3개 유지. 초과 시 `cached_at` 기준 LRU eviction |

### 11.4 캐시 관리 구현 (v2.0)

```python
# api/services/cache_manager.py

MAX_ANOMALY_MAP_CACHE = 3

def set_anomaly_map_cache(experiment_id: str, data: dict) -> None:
    """anomaly_map 캐시 저장. 3개 초과 시 LRU eviction."""
    import time
    if len(_anomaly_map_cache) >= MAX_ANOMALY_MAP_CACHE:
        oldest_key = min(
            _anomaly_map_cache,
            key=lambda k: _anomaly_map_cache[k].get("cached_at", 0)
        )
        del _anomaly_map_cache[oldest_key]

    _anomaly_map_cache[experiment_id] = {
        **data,
        "cached_at": time.time()
    }


def get_anomaly_map_cache(experiment_id: str) -> dict | None:
    return _anomaly_map_cache.get(experiment_id)


def invalidate_anomaly_map_cache(experiment_id: str) -> None:
    _anomaly_map_cache.pop(experiment_id, None)
```

---

## 12. EfficientAD ImageNet Penalty 데이터

### 12.1 배경

EfficientAD는 학습 중 Student-Teacher 과적합 방지를 위해 ImageNet 분포의 랜덤 이미지 배치를 추가 사용한다. 학습 전 미리 준비 필요.

### 12.2 경로 규칙

```
{WORKDIR}/dataset/imagenet_penalty/
├── n01440764_0.jpg
...
(최소 1,000장 이상 권장)
```

```python
IMAGENET_PENALTY_DIR = Path("./dataset/imagenet_penalty")
```

### 12.3 존재 여부 검증

```python
def validate_imagenet_penalty_dir() -> tuple[bool, int]:
    d = IMAGENET_PENALTY_DIR
    if not d.exists():
        return False, 0
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    count = sum(1 for f in d.iterdir() if f.suffix.lower() in supported)
    return count > 0, count
```

### 12.4 학습 시작 전 검증 위치 (v2.0)

```python
# api/routers/training.py — POST /api/training/start 핸들러 내부

if body.model_config.model_type == "efficientad":
    ok, count = validate_imagenet_penalty_dir()
    if not ok:
        raise HTTPException(400, detail={
            "code": "ERR_IMAGENET_PENALTY_MISSING",
            "message": f"`{IMAGENET_PENALTY_DIR}` 경로에 이미지를 추가해 주세요."
        })
    if count < 1000:
        pass  # 경고만 — 응답 body에 gpu_warning 유사 필드로 포함
              # [확인 필요: 경고를 응답에 포함시키는 방식]
```

### 12.5 Docker 마운트 방법

```bash
docker run \
  -v /host/imagenet_penalty:/app/dataset/imagenet_penalty:ro \
  ...
```

### 12.6 준비 방법

| 방법 | 설명 |
|------|------|
| ImageNet 서브셋 | ImageNet-1k validation set 중 1,000장 이상 추출 |
| 대체 데이터 | COCO, Open Images 등 자연 이미지 |
| 최소 요건 | 1장 이상(기술적 동작), 1,000장 이상 권장 |
| 금지 | 학습 데이터셋과 동일 이미지 사용 금지 |

---

## 13. 로그 파일 관리

### 13.1 파일 경로 및 생성 시점

```
./logs/{experiment_id}.log
```

- 학습 스레드 시작 직후 **즉시 생성** (append 모드)
- 학습 중단 또는 완료 후에도 **삭제하지 않는다**
- 실험 삭제 시 함께 삭제 (§10.2 단계 3)

### 13.2 로그 포맷

```
{ISO8601_KST}\t[시작] 실험: {experiment_id}
{ISO8601_KST}\t[Step {step}/{total}] Loss: {loss:.4f} | 경과: {elapsed:.1f}s
{ISO8601_KST}\t[완료] 총 소요: {duration}s | AUC: {auc:.4f}
{ISO8601_KST}\t[중단] {step}번째 스텝에서 사용자 중단
{ISO8601_KST}\t[오류] {traceback_summary}
```

### 13.3 로그 쓰기 구현

```python
# utils/storage.py

def get_log_writer(experiment_id: str):
    """append 전용, line-buffered writer 반환. TrainingWorker에서 호출."""
    log_path = Path(f"./logs/{experiment_id}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
```

**line-buffered (`buffering=1`)**: 프로세스 종료 시에도 마지막 줄 보존 보장.

### 13.4 로그 조회

```python
def read_log_tail(experiment_id: str, n_lines: int = 100) -> str:
    log_path = Path(f"./logs/{experiment_id}.log")
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])
```

WS `log` 메시지로 실시간 스트리밍, GET /api/training/status 응답에서 최신 100줄 반환.

---

## 14. 디스크 용량 모니터링

### 14.1 경고 조건

| 조건 | 처리 |
|------|------|
| 학습 시작 전 여유 공간 < 500 MB | HTTP 400 ERR_DISK_SPACE_INSUFFICIENT (학습 차단) |
| 모델 저장 전 여유 공간 < 100 MB | RuntimeError ERR_DISK_SPACE (저장 차단) |

### 14.2 구현

```python
# utils/storage.py

def check_disk_space(
    required_mb: float = 500.0,
    path: str = "."
) -> tuple[bool, float]:
    import shutil
    usage = shutil.disk_usage(path)
    free_mb = usage.free / (1024 ** 2)
    return free_mb >= required_mb, free_mb


def _check_disk_space(model_type: str) -> None:
    """
    POST /api/training/start 핸들러에서 호출 (v2.0).
    100 MB 미만 시 HTTPException 발생.
    500 MB 미만 시 ERR_DISK_SPACE_INSUFFICIENT 발생.
    """
    sufficient, free_mb = check_disk_space(required_mb=500.0)
    if not sufficient:
        raise HTTPException(status_code=400, detail={
            "code": "ERR_DISK_SPACE_INSUFFICIENT",
            "message": f"디스크 여유 공간 {free_mb:.0f} MB. 500 MB 이상 필요."
        })
```

---

## 15. 엔티티 관계

```
[history.json]
  └── Experiment[]
        ├── experiment_id (PK)
        ├── metrics: ExperimentMetrics | null
        └── model_path → [models/{exp_id}/model_state_dict.pth]

[Explorer — configStore]
  ├── preprocessingConfig: PreprocessingConfig
  ├── modelConfig: ModelConfig
  │   └── params: EfficientAdParamsState | PatchCoreParamsState
  ├── deviceInfo: DeviceInfo
  └── queueItems: QueueItem[]
        ├── preprocessing_config: PreprocessingConfig
        └── model_config: ModelConfig

[Explorer — trainingStore]
  └── lossHistory: LossPoint[]

[Explorer — experimentsStore]
  └── selectedExperimentId → Experiment.experiment_id (FK)

[Vision — inspectionStore]
  ├── activeModel: ActiveModel | null
  │   └── experiment_id → Experiment.experiment_id (FK, 읽기 전용)
  └── lastResult: InspectionResult | null

[FastAPI 서버 메모리]
  ├── training_manager.result_queue → WS /ws/training 팬아웃
  ├── inspection_manager.records: InspectionRecord[]  (모델 교체 시 초기화)
  └── anomaly_map_cache: dict[exp_id, AnomalyMapCache]  (최대 3개)
```

**참조 무결성 규칙:**

| 규칙 | 설명 |
|------|------|
| R-01 | `selectedExperimentId`는 반드시 `history.json` 내 존재하는 `experiment_id`를 참조 |
| R-04 | `Experiment.model_path`가 NOT NULL이면 해당 경로에 `.pth`와 `configs.yaml` 존재 |
| R-05 | `status` ∈ {'중단', '실패', '건너뜀'} 레코드의 `metrics`, `model_path`는 NULL |
| R-06 | `activeModel`은 반드시 `status === 'completed'`인 레코드를 참조 |
| R-07 | `activeModel === null`이면 inspection_records, lastResult도 빈값/null |
| R-08 | 모델 교체(POST /api/inspection/model) 시 서버는 inspection_records, lastResult 즉시 초기화 |

---

## 16. PRD ↔ TypeScript 불일치 대조표

> 아래는 00_Global_Context_Document.md v1.x 정의와 v2.0 TypeScript 타입의 실제 차이점. TypeScript 타입을 공식 기준으로 한다.

### 16.1 Experiment.status

| 항목 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| status ENUM | `"completed" \| "중단"` | `'completed' \| '중단' \| '실패' \| '건너뜀'` | TypeScript 기준 채택 (4가지) |

### 16.2 PreprocessingConfig.background_method

| 항목 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| 필드 존재 | 미정의 | `background_method: 'none' \| 'sam2'` | v2.0 신규 필드로 공식화 |

### 16.3 PreprocessingConfig.normalization

| 항목 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| 허용값 | `'imagenet' \| 'custom'` | `'imagenet'` 고정 | TypeScript 기준 (`'imagenet'` 전용) |

### 16.4 EfficientAdParamsState 추가 필드

| 필드 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| `early_stopping` | 미정의 | `boolean` | v2.0 신규 필드 |
| `patience` | 미정의 | `number` | v2.0 신규 필드 |
| `min_delta` | 미정의 | `number` | v2.0 신규 필드 |

### 16.5 ModelConfig.image_size

| 항목 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| `image_size` 위치 | `model_config.image_size` | `ModelConfig`에 없음, `PreprocessingConfig.image_size`로 일원화 | TypeScript 기준 채택 |

### 16.6 InspectionResult vs InspectionRecord 분리

| 항목 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| 검사 레코드 타입 수 | 단일 `inspection_record` | `InspectionResult` + `InspectionRecord` 분리 | TypeScript 기준 (2개 타입) |
| `image_path` | 단일 레코드에 존재 | `InspectionResult`에만 | TypeScript 기준 |
| `anomaly_map_cache` | 레코드에 존재 | 양쪽 모두에 없음 (API 이미지 서빙으로 대체) | 제거 확정 |

### 16.7 ActiveModel 타입

| 항목 | v1.x PRD | TypeScript 구현 | 결정 |
|------|----------|----------------|------|
| threshold 표현 | `threshold_method` + `threshold_value` | `threshold` (단일 계산값) | TypeScript 기준 |
| `score_min`, `score_max` | 미정의 | `number` 필드 존재 | v2.0 신규 필드 |
| `device` | 미정의 | `string` | v2.0 신규 필드 |
| `preprocessing_config` | 분리 필드 | 중첩 객체 | TypeScript 기준 |

### 16.8 Experiment 선택적 필드

| 항목 | v1.x PRD | TypeScript 구현 | 비고 |
|------|----------|----------------|------|
| `preprocessing_method` | NOT NULL | optional (`?`) | 레거시 레코드에 없을 수 있음 |
| `dataset_path` | NOT NULL | optional (`?`) | 동일 |
| `image_size` | NOT NULL | optional (`?`) | 동일 |
| `threshold_method` | NOT NULL | optional (`?`) | 동일 |
| `threshold_value` | NOT NULL | optional (`?`) | 동일 |

---

## 17. 구현 체크리스트

### storage.py

- [ ] `load_history()` — 파일 미존재, JSON 파싱 실패 모두 `[]` 반환
- [ ] `append_experiment()` — `.tmp` → `rename`, IOError 시 `.tmp` 정리
- [ ] `delete_experiment_from_history()` — 해당 ID 없으면 `False` 반환
- [ ] `save_completed_experiment()` — 3단계 원자성 프로토콜 전체 구현
- [ ] `_cleanup_dir()` — `shutil.rmtree(ignore_errors=True)`
- [ ] `delete_experiment()` — history + 모델 디렉토리 + 로그 파일 삭제
- [ ] `get_log_writer()` — append 모드, line-buffered
- [ ] `read_log_tail()` — 최신 N줄 반환
- [ ] `check_disk_space()` — `shutil.disk_usage()` 기반
- [ ] `_check_disk_space()` — 500 MB 미만 시 HTTPException (v2.0)

### config_manager.py

- [ ] `load_config(path)` — 파일 미존재, YAML 파싱 실패 모두 `{}` 반환
- [ ] `save_config_section(section, data, path)` — 기존 섹션 보존, `.tmp` → `rename`
- [ ] `path` 파라미터로 루트 configs.yaml + 실험 스냅샷 양쪽 대응

### cache_manager.py (v2.0)

- [ ] `set_anomaly_map_cache(exp_id, data)` — 최대 3개, LRU eviction (서버 인메모리)
- [ ] `get_anomaly_map_cache(exp_id)` — 없으면 `None`
- [ ] `invalidate_anomaly_map_cache(exp_id)` — 삭제 시 호출 (v2.0 신규)

### 기타 검증

- [ ] `validate_imagenet_penalty_dir()` — 경로 존재 + 이미지 수 반환
- [ ] EfficientAD 학습 시작 전 POST /api/training/start에서 penalty dir 검증
- [ ] 실험 삭제(DELETE /api/experiments/{id}) 시 `cache_manager.invalidate()` 호출
- [ ] `status ∈ {'중단','실패','건너뜀'}` 레코드 저장 시 `model_path`, `configs_path`, `metrics` 모두 `null`

---

---

# v1.x 참고 — Streamlit 세션 기반 구현 (삭제 금지)

> 이하 내용은 v1.x Streamlit 기반 구현 명세입니다. v2.0에서 FastAPI 서버 메모리로 교체됐습니다.

---

### v1.x A. session_state 기반 스키마

```python
# session_state_init.py — 탐색 대시보드 상태 초기화

SESSION_STATE_SCHEMA = {
    "dataset_path":           None,    # str | None
    "dataset_meta":           None,    # dict | None
    "preprocessing_config":   None,    # dict | None
    "model_config":           None,    # dict | None
    "device_info":            None,    # dict | None
    "experiments":            {},      # dict[str, dict]
    "current_run_status":     "idle",  # "idle" | "running" | "paused"
    "selected_experiment_id": None,    # str | None
    "anomaly_map_threshold":  None,    # float | None
}
```

**v1.x → v2.0 대응:**

| v1.x session_state 키 | v2.0 서버 상태 |
|-----------------------|----------------|
| `current_run_status` | `training_manager.status` |
| `_stop_event` | `training_manager.stop_event` |
| `_result_queue` | `training_manager.result_queue` |
| `_anomaly_maps_{exp_id}` | `anomaly_map_cache[exp_id]` (서버 인메모리) |
| `selected_experiment_id` | `experimentsStore.selectedExperimentId` (클라이언트 Zustand) |

---

### v1.x B. 비전검사 세션 스키마 (Streamlit)

```python
# inspection/utils/insp_session_init.py

INSPECTION_SESSION_SCHEMA = {
    "insp_active_model":  None,   # dict | None
    "insp_records":       [],     # list[dict]
    "insp_seq_counter":   0,
    "insp_auto_active":   False,
    "insp_last_result":   None,
    "insp_defect_popup":  False,
    "insp_test_pool":     [],     # list[tuple[str, str]]
    "insp_pool_index":    0,
}
```

**v1.x → v2.0 대응:**

| v1.x session_state 키 | v2.0 서버 상태 |
|-----------------------|----------------|
| `insp_active_model` | `inspection_manager.active_model.metadata` |
| `insp_records` | `inspection_manager.records` |
| `insp_seq_counter` | `inspection_manager.seq_counter` |
| `insp_test_pool` | `inspection_manager.test_pool` |
| `insp_pool_index` | `inspection_manager.pool_index` |
| `insp_auto_active` | `inspectionStore.isAutoRunning` (클라이언트) |
| `insp_defect_popup` | `inspectionStore.defectStopped` (클라이언트) |

---

### v1.x C. inspection_record 단일 스키마 (Streamlit)

```python
# v1.x에서는 InspectionResult + InspectionRecord가 분리되지 않음
inspection_record = {
    "seq":             int,
    "inspected_at":    str,
    "image_name":      str,
    "image_path":      str,
    "verdict":         str,
    "anomaly_score":   float,
    "anomaly_map_cache": "np.ndarray | None",  # v2.0에서 제거
}
```

---

### v1.x D. AnomalyMap 세션 캐시 (session_state 기반)

```python
# session_state 캐시 키 형식
cache_key = f"_anomaly_maps_{experiment_id}"

st.session_state[cache_key] = {
    "anomaly_maps": np.ndarray,   # shape: (N, H, W), float32
    "image_paths":  list[str],
    "cached_at":    float,
}
```

**캐시 eviction (v1.x — session_state 기반):**

```python
# utils/cache_manager.py (v1.x)

def set_anomaly_map_cache(experiment_id: str, data: dict) -> None:
    cache_keys = [k for k in st.session_state if k.startswith("_anomaly_maps_")]
    if len(cache_keys) >= MAX_ANOMALY_MAP_CACHE:
        oldest_key = min(cache_keys, key=lambda k: st.session_state[k].get("cached_at", 0))
        del st.session_state[oldest_key]
    st.session_state[f"_anomaly_maps_{experiment_id}"] = {**data, "cached_at": time.time()}
```

---

### v1.x E. 비전검사 저장 전략 (Streamlit session_state 전용)

```python
# inspection/utils/test_sampler.py (v1.x)

def build_test_pool(dataset_path: str) -> list[tuple[str, str]]:
    test_root = Path(dataset_path) / "test"
    pool = []
    for cls_dir in test_root.iterdir():
        if not cls_dir.is_dir():
            continue
        label = "양품" if cls_dir.name == "good" else "불량"
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                pool.append((str(img_path), label))
    random.shuffle(pool)
    return pool
```

v2.0에서는 `inspection_manager.test_pool`에 저장. `InspectionManager.sample_from_pool()`로 교체.

---

### v1.x F. 표준 안내 메시지 상수

```python
MSG = {
    "NO_DATASET":       "먼저 Dataset 화면에서 데이터 폴더를 설정해 주세요.",
    "NO_MODEL_CONFIG":  "먼저 Config 화면에서 모델 파라미터를 설정해 주세요.",
    "NO_EXPERIMENTS":   "아직 실행된 실험이 없습니다. Training 화면에서 학습을 먼저 실행해 주세요.",
    "NO_SELECTED_EXP":  "Experiments 화면에서 분석할 실험을 먼저 선택해 주세요.",
    "TRAIN_STOPPED":    "학습이 중단되었습니다. 해당 실험은 '중단' 상태로 기록되었습니다.",
}

INSP_MSG = {
    "NO_MODEL":         "선택된 모델이 없습니다. Model Settings 화면에서 모델을 선택해 주세요.",
    "NO_COMPLETED_EXP": "적용 가능한 완료 실험이 없습니다. Explorer에서 학습을 완료해 주세요.",
    "DEFECT_DETECTED":  "불량이 감지되었습니다. 해당 부품을 라인에서 제거하고 확인해 주세요.",
    "MODEL_REPLACED":   "모델이 교체되었습니다. 검사 이력이 초기화되었습니다.",
    "HISTORY_CLEARED":  "검사 이력이 초기화되었습니다.",
    "AUTO_STOPPED":     "불량 감지로 자동 검사가 중지되었습니다. 확인 후 재시작해 주세요.",
    "POOL_RESHUFFLED":  "테스트 이미지 풀을 모두 소진하여 재구성했습니다.",
}
```

---

*이 문서는 07_Backend_Service_Design.md §3 (학습 시작 서비스), §9 (메모리 관리), §12 (InspectionManager) 및 08_AI_ML_Integration.md §8 (학습 완료 후 처리)와 연동된다.*
*다음: [06_API_Specification.md](./06_API_Specification.md)*

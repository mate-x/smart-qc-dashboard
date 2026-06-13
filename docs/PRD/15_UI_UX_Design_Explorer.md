# Smart QC Explorer — UI/UX 설계 문서

> **작성 기준:** React 19 소스 코드 역설계 (smart-qc-explorer/src/)
> **최종 갱신:** 2026-06-11
> **버전:** 2.0

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|---|---|---|
| 2.0 | 2026-06-11 | React 19 + Zustand v5 기준으로 전면 재작성. Streamlit 내용은 섹션 하단 v1.x 참고로 이동 (삭제 금지) |
| 1.0 | 2026-05-18 | 초기 작성 (Streamlit app.py 역설계 기준) |

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **프로젝트 목적** | MVTec AD 형식 데이터셋 대상 EfficientAD / PatchCore 이상 탐지 모델 학습 및 실험 결과 비교·시각화 |
| **핵심 사용자** | 제조 현장 QC 담당 ML 엔지니어 / 연구자 |
| **주요 시나리오** | Dataset 검증 → Config 저장 → Training 실행·모니터링 → Experiments 비교 → AnomalyMap 시각화·내보내기 |
| **UI 유형** | React 19 SPA (smart-qc-explorer, port 5173) |
| **UI 레이아웃** | 상단 Navbar + TabBar, 하단 `<main>` 콘텐츠 영역 |
| **스타일링** | Tailwind CSS v4 (유틸리티 우선) |
| **백엔드** | FastAPI (smart-qc-dashboard, port 8000) |

---

## 2. 전체 화면 구조

```text
src/main.tsx
└── <BrowserRouter>
    └── <App>                         ← App.tsx
        ├── useTrainingWs()           ← /ws/training WS, root 레벨 — 라우트 이동 시에도 유지
        ├── <Navbar />                ← layout/Navbar.tsx
        ├── <TabBar />                ← layout/TabBar.tsx
        └── <main>
            └── <Routes>
                ├── /                 → Tab1Dataset
                ├── /config           → Tab2Config
                ├── /training         → Tab3Training
                ├── /experiments      → Tab4Experiments
                ├── /anomaly-map      → Tab5AnomalyMap
                └── *                 → <Navigate to="/" replace />
```

**TabBar 탭 레이블 ↔ 라우트 매핑:**

| 탭 레이블 | 경로 | 컴포넌트 |
|---|---|---|
| 데이터 | `/` | Tab1Dataset |
| 전처리 / 모델 | `/config` | Tab2Config |
| 학습 | `/training` | Tab3Training |
| 실험 히스토리 | `/experiments` | Tab4Experiments |
| 이상 시각화 | `/anomaly-map` | Tab5AnomalyMap |

---

## 3. 화면별 기능 요약

| 화면 | 목적 | 주요 입력 | 주요 출력 | Zustand 스토어 |
|---|---|---|---|---|
| **Dataset** | 데이터셋 경로 검증 및 제품명 설정 | 경로 text input, 제품명 input, 검증 버튼 | 폴더 트리, 클래스별 이미지 수, 썸네일 | datasetStore, configStore |
| **Config** | 전처리·모델 하이퍼파라미터 설정 및 저장 | PreprocessingForm, ModelConfigForm, 저장 버튼 | 디바이스 정보, 설정 저장 결과, QueueSection | configStore, datasetStore |
| **Training** | 학습 실행 및 실시간 모니터링 | 실험명 input, 시작/일시정지/중단 버튼 | 스테이지 인디케이터, 진행률 바, Loss 곡선, 로그 | trainingStore, configStore, datasetStore |
| **Experiments** | 실험 히스토리 조회·비교·저장 | 행 클릭(선택), 삭제, 저장 경로 input | 성능 지표 카드, 차트 3종, 다중 비교, 배치 비교 | experimentsStore |
| **AnomalyMap** | 이상 영역 시각화 및 결과 내보내기 | Threshold 슬라이더, 결함 유형 필터, 빌드 버튼 | 이미지 그리드 (4-col), 3-패널 상세 시각화, 내보내기 | experimentsStore, anomalyMapStore |

---

## 4. 전역 상태 관리 (Zustand 스토어)

### 4.1 datasetStore (`src/store/datasetStore.ts`)

| 필드 | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `datasetPath` | `string \| null` | `null` | 검증된 데이터셋 루트 경로 |
| `productName` | `string` | `''` | 제품명 (실험 명명에 사용) |
| `datasetMeta` | `DatasetValidateResponse \| null` | `null` | 검증 결과 메타데이터 |

**액션:** `setDataset(path, meta, productName)`, `clearDataset()`

**`DatasetValidateResponse` 주요 필드:**
```typescript
{
  folder_tree: string;           // 폴더 구조 문자열 (<pre> 표시용)
  train_good_count: number;
  test_counts: Record<string, number>;
  gt_counts: Record<string, number>;
  channels: 1 | 3;
  defect_classes: string[];
  has_invalid_files: boolean;
  invalid_file_count: number;
  has_background_clean: boolean; // SAM2 배경 제거 이미지 존재 여부
  dataset_format: 'mvtec' | 'oking' | string;
  oking_ok_count?: number;
  oking_ng_count?: number;
  train_ratio?: number;
}
```

### 4.2 configStore (`src/store/configStore.ts`)

| 필드 | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `preprocessingConfig` | `PreprocessingConfig \| null` | `null` | 저장된 전처리 설정 |
| `modelConfig` | `ModelConfig \| null` | `null` | 저장된 모델 파라미터 |
| `deviceInfo` | `DeviceInfo \| null` | `null` | CUDA/CPU 디바이스 정보 |

**액션:** `setConfigs(pre, model)`, `setDeviceInfo(info)`, `clearConfigs()`

**`PreprocessingConfig` 주요 필드:**
```typescript
{
  method: 'none' | 'homomorphic' | 'he' | 'clahe';
  background_method: 'none' | 'sam2';  // 🆕 신규 필드 (v1.x에 없음)
  resize_mode: 'padding';
  image_size: number;
  normalization: 'imagenet' | 'custom';
  mean: [number, number, number];
  std: [number, number, number];
  params: Record<string, unknown> | null;
}
```

**`ModelConfig` 주요 필드:**
```typescript
{
  model_type: 'efficientad' | 'patchcore';
  batch_size: number;
  random_seed: number;
  threshold_method: 'percentile' | 'absolute';
  threshold_value: number;
  params: EfficientAdParamsState | PatchCoreParamsState;
}
```

### 4.3 trainingStore (`src/store/trainingStore.ts`)

`TrainingStatusResponse`를 extends하며 WS 메시지 수신 시 업데이트됨.

| 필드 | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `status` | `'idle' \| 'running' \| 'paused'` | `'idle'` | 학습 실행 상태 |
| `exp_id` | `string \| null` | `null` | 현재 실행 중인 실험 ID |
| `batch_mode` | `boolean` | `false` | 일괄 학습 실행 중 여부 |
| `batch_total` | `number` | `0` | 일괄 학습 총 항목 수 |
| `progress` | `{step, total, loss, elapsed} \| null` | `null` | 현재 학습 진행 정보 |
| `current_stage_idx` | `number \| null` | `null` | 현재 스테이지 인덱스 |
| `current_stage_name` | `string \| null` | `null` | 현재 스테이지 이름 |
| `log_lines` | `string[]` | `[]` | 실시간 로그 (최대 100줄 유지) |
| `loss_history` | `{step, loss}[]` | `[]` | Loss 곡선 데이터 |
| `last_ckpt_path` | `string \| null` | `null` | 일시정지 시 체크포인트 경로 |
| `batch_done` | `number` | `0` | 완료된 일괄 학습 항목 수 |
| `last_result` | `{level, msg} \| null` | `null` | 마지막 학습 결과 배너 |
| `ws_error` | `string \| null` | `null` | WebSocket 오류 메시지 |
| `batch_queue_signal` | `number` | `0` | QueuePanel 재로드 트리거 (bump 시 +1) |

### 4.4 experimentsStore (`src/store/experimentsStore.ts`)

| 필드 | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `selectedExperimentId` | `string \| null` | `null` | Tab4→Tab5 연결용 선택된 실험 ID |

**액션:** `setSelectedExperimentId(id)`

### 4.5 anomalyMapStore (`src/store/anomalyMapStore.ts`)

| 필드 | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `threshold` | `number \| null` | `null` | Tab5 threshold 값 (화면 이동 시에도 유지) |

**액션:** `setThreshold(value)`

---

## 5. 프로젝트 파일 구조

```text
smart-qc-explorer/src/
├── main.tsx                          # React 19 진입점 (createRoot + BrowserRouter + StrictMode)
├── App.tsx                           # 루트 레이아웃 + 라우터 + useTrainingWs() 마운트
├── api/
│   ├── datasetApi.ts                 # POST /dataset/validate
│   ├── configApi.ts                  # GET /config, POST /config/save, GET /config/queue
│   ├── trainingApi.ts                # POST /training/start|pause|unpause|stop|resume, batch 관련
│   ├── experimentsApi.ts             # GET /experiments, DELETE, POST save
│   └── anomalyMapApi.ts              # build, build-status, images, export, image URL 헬퍼
├── store/
│   ├── datasetStore.ts
│   ├── configStore.ts
│   ├── trainingStore.ts
│   ├── experimentsStore.ts
│   └── anomalyMapStore.ts
├── hooks/
│   └── useTrainingWs.ts              # /ws/training WebSocket 관리 (App.tsx 루트에서 단일 마운트)
├── pages/
│   ├── Tab1Dataset.tsx
│   ├── Tab2Config.tsx
│   ├── Tab3Training.tsx
│   ├── Tab4Experiments.tsx
│   └── Tab5AnomalyMap.tsx
├── components/
│   ├── layout/
│   │   ├── Navbar.tsx
│   │   └── TabBar.tsx
│   ├── tab1/
│   │   ├── ThumbnailGrid.tsx
│   │   └── ClassCountTable.tsx
│   ├── tab2/
│   │   ├── PreprocessingForm.tsx
│   │   ├── ModelConfigForm.tsx
│   │   └── QueueSection.tsx
│   ├── training/
│   │   ├── StageIndicator.tsx
│   │   ├── ProgressSection.tsx
│   │   ├── IdleSection.tsx
│   │   └── QueuePanel.tsx
│   ├── tab4/
│   │   ├── ConfusionMatrixChart.tsx
│   │   ├── RocCurveChart.tsx
│   │   ├── ScoreDistChart.tsx
│   │   ├── ComparisonSection.tsx
│   │   ├── BatchComparisonSection.tsx
│   │   └── experimentUtils.ts
│   └── tab5/
│       ├── ControlBar.tsx
│       ├── ImageGrid.tsx             # ImageCard + DetailPanel 포함
│       └── ExportSection.tsx
└── types/
    ├── dataset.ts
    ├── config.ts
    ├── training.ts
    ├── experiments.ts
    ├── anomalyMap.ts
    └── modelParams.ts
```

---

## 6. 화면 간 의존성 및 진입 제한

| 화면 | 선행 조건 | 미충족 시 동작 |
|---|---|---|
| **Dataset** | 없음 | — |
| **Config** | `datasetMeta !== null` | 상단에 amber 경고 카드 + Dataset 화면 링크 렌더링, 폼 숨김 |
| **Training** | `datasetMeta !== null` AND `preprocessingConfig !== null` AND `modelConfig !== null` | 상단에 amber 경고 카드 렌더링, 학습 시작 버튼 disabled |
| **Experiments** | 없음 (단, 실험 없으면 빈 상태 화면) | "실험 기록이 없습니다. 학습 탭에서 학습을 실행하면 여기에 표시됩니다." |
| **AnomalyMap** | `selectedExperimentId !== null` | 빈 상태 카드 + "실험 히스토리로" 버튼 |

**Config 화면 — `clearConfigs()` 호출 트리거:**
Dataset 화면에서 `handleValidate()` 시 경로가 이전과 다르면 `configStore.clearConfigs()` 호출 → Config 화면의 configStore 상태 초기화

---

## 7. 실험 실행 흐름

```text
[Dataset 화면]
  경로 입력 + 제품명 입력 → "검증" 버튼 클릭
    → POST /dataset/validate
    → 경로 변경 시: configStore.clearConfigs()
    → 성공: datasetStore.setDataset(path, meta, productName)
    → 배너: background_clean 없음 / grayscale / 유효하지 않은 파일 / oking 형식

[Config 화면]
  마운트 시 GET /config → PreprocessingForm + ModelConfigForm 초기값 설정 + setDeviceInfo
  사용자가 폼 수정 → "저장" 버튼 클릭
    → image_size % 32 검증
    → POST /config/save(preConfig, modelConfig)
    → 성공: configStore.setConfigs(pre, model) + 3초 후 saveOk 배지 사라짐

[Training 화면]
  useTrainingWs() — App.tsx 루트에서 /ws/training 연결 (라우트 무관 유지)
    WS connect 시 스냅샷 메시지 수신 → trainingStore.setFromSnapshot()
    WS push 메시지 → trainingStore 각 액션 (updateProgress, addLog, setStage, setCompleted 등)

  IdleSection에서 실험명 입력 → "▶ 학습 시작" 클릭
    → POST /training/start(expName?)
    → WS 메시지로 status: 'running' 수신 → ProgressSection 전환

  ProgressSection에서:
    ⏸ 일시정지 → POST /training/pause
    ▶ 재개     → POST /training/unpause
    ⏹ 중단     → POST /training/stop (단일) / POST /training/batch/stop (배치)
    ⏭ 건너뜀  → POST /training/batch/skip (배치 전용)

  QueuePanel:
    Config 화면 QueueSection에서 항목 추가 → batch_queue_signal bump 시 getQueue() 재로드
    "▶▶ 일괄 학습 시작" → POST /training/batch/start
    WS: setBatchCompleted 수신 → IdleSection last_result 배너 표시

[Experiments 화면]
  마운트 시 GET /experiments → 실험 목록 로드
  행 클릭 → experimentsStore.setSelectedExperimentId(id)
  selected & completed → 상세 결과 섹션 렌더링
  "이 실험으로 Anomaly Map 보기" → navigate('/anomaly-map')
  모델 저장 → POST /experiments/{id}/save(savePath)
  삭제 → DELETE /experiments/{id} → 목록 재로드

[AnomalyMap 화면]
  selectedExperimentId 확인 → GET /experiments (실험 정보 취득)
  GET /anomaly-map/{id}/build-status
  buildStatus.built == false → "Anomaly Map 생성" 버튼 활성화
  "Anomaly Map 생성" → POST /anomaly-map/{id}/build → job_id 취득
    → polling: GET /jobs/{job_id}/status (1초 간격)
    → completed → GET /anomaly-map/{id}/build-status 갱신

  threshold 슬라이더 변경 → 300ms debounce → anomalyMapStore.setThreshold()
                          → GET /anomaly-map/{id}/images?threshold=&class= 재요청

  이미지 카드 클릭 → DetailPanel 열기 (원본 / GT 마스크 / Heatmap 3-panel)
  내보내기:
    CSV → POST /anomaly-map/{id}/export/csv → Blob 다운로드
    ZIP → POST /anomaly-map/{id}/export/zip/prepare → polling 완료 → GET 다운로드
```

---

## 8. Dataset 화면 (`src/pages/Tab1Dataset.tsx`)

### 8.1 컴포넌트 트리

```text
Tab1Dataset
├── [입력 카드 — bg-white rounded-2xl]
│   └── grid grid-cols-2
│       ├── 왼쪽: 폼 영역
│       │   ├── input: 데이터셋 경로 (inputPath 로컬 state)
│       │   ├── input: 제품명 (productNameInput 로컬 state)
│       │   └── button: "검증" (handleValidate)
│       └── 오른쪽: 폴더 트리
│           └── <pre> datasetMeta.folder_tree
│
├── [배너 영역 — 조건부]
│   ├── !has_background_clean → sky 배너 (SAM2 배경 없음 안내)
│   ├── channels === 1       → sky 배너 (Grayscale → RGB 자동 변환 안내)
│   ├── has_invalid_files    → amber 배너 (유효하지 않은 파일 N개)
│   └── dataset_format === 'oking' → sky 배너 (train_ratio 계산 결과 포함)
│
└── [데이터 표시 영역 — grid grid-cols-2]
    ├── <ThumbnailGrid />   ← components/tab1/ThumbnailGrid.tsx
    └── <ClassCountTable /> ← components/tab1/ClassCountTable.tsx
```

### 8.2 Zustand 스토어 바인딩

| 스토어 | 읽기 | 쓰기 |
|---|---|---|
| `datasetStore` | `datasetMeta`, `datasetPath`, `productName` | `setDataset()`, `clearDataset()` |
| `configStore` | — | `clearConfigs()` (경로 변경 시) |

### 8.3 API 연동

| 시점 | API | 설명 |
|---|---|---|
| "검증" 버튼 클릭 | POST /dataset/validate | `{ path, product_name }` 전송 |
| 응답 성공 | — | 경로 변경 시 `clearConfigs()` 후 `setDataset()` |

### 8.4 사용자 인터랙션 흐름

```text
사용자: 경로 입력 → 제품명 입력 → "검증" 클릭
  ↓ loading=true (버튼 disabled, 스피너)
  POST /dataset/validate
  ↓ 성공
  경로 변경 여부 확인 → 변경됨: clearConfigs()
  setDataset(path, res.data, productNameInput)
  배너 조건부 렌더링 → ThumbnailGrid + ClassCountTable 표시
```

---

## 9. Config 화면 (`src/pages/Tab2Config.tsx`)

### 9.1 컴포넌트 트리

```text
Tab2Config
├── [가드] !datasetMeta → amber 경고 카드 ("Dataset 화면에서 경로를 먼저 검증해 주세요") + return
│
├── [최상단 바 — flex justify-between]
│   ├── 왼쪽: 디바이스 레이블
│   │   └── dot indicator + "gpu_name · vram_gb GB" or "CPU 모드"
│   └── 오른쪽: "저장" 버튼 + saveOk 배지 (3s) / saveError 텍스트
│
├── [설정 폼 — grid grid-cols-2]
│   ├── <PreprocessingForm />  ← components/tab2/PreprocessingForm.tsx
│   │   (method, background_method, image_size, normalization 등)
│   └── <ModelConfigForm />    ← components/tab2/ModelConfigForm.tsx
│       (model_type, batch_size, random_seed, threshold, params 등)
│
└── [대기열 아코디언]
    ├── "설정 대기열" 헤더 버튼 (queueOpen toggle)
    └── {queueOpen && <QueueSection />}  ← components/tab2/QueueSection.tsx
```

### 9.2 Zustand 스토어 바인딩

| 스토어 | 읽기 | 쓰기 |
|---|---|---|
| `datasetStore` | `datasetMeta` (가드용) | — |
| `configStore` | `deviceInfo`, `preprocessingConfig`, `modelConfig` | `setConfigs()`, `setDeviceInfo()` |

### 9.3 API 연동

| 시점 | API | 설명 |
|---|---|---|
| 컴포넌트 마운트 (`useEffect`) | GET /config | 서버 현재 설정 로드 + `setDeviceInfo()` |
| "저장" 버튼 클릭 | POST /config/save | `{ preprocessing_config, model_config }` 전송 |
| QueueSection 표시 | GET /config/queue | 대기열 항목 로드 |

### 9.4 DEFAULT 전처리 설정 (`background_method` 신규 필드)

```typescript
DEFAULT_PRE = {
  method: 'none',
  background_method: 'none',  // 🆕 SAM2 배경 제거 방식 ('none' | 'sam2')
  resize_mode: 'padding',
  image_size: 256,
  normalization: 'imagenet',
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  params: null,
}
```

### 9.5 사용자 인터랙션 흐름

```text
마운트 → GET /config → preConfig, modelConfig, deviceInfo → 폼 초기값 설정
사용자: 폼 값 수정
"저장" 클릭:
  → image_size % 32 !== 0 → saveError 표시, 저장 중단
  → POST /config/save
  → 성공: setConfigs(pre, model) + saveOk 배지 3초 표시
  → 실패: saveError 텍스트 표시

"설정 대기열" 헤더 클릭:
  → queueOpen 토글 → QueueSection 표시/숨김
```

---

## 10. Training 화면 (`src/pages/Tab3Training.tsx`)

### 10.1 컴포넌트 트리

```text
Tab3Training
├── [가드] !datasetMeta || !(preprocessingConfig && modelConfig)
│   → amber 경고 카드 렌더링 (Dataset/Config 화면 링크 포함)
│
├── <QueuePanel />   ← components/training/QueuePanel.tsx
│   (항상 최상단 표시; 대기열 없고 batch_mode==false이면 null 반환)
│
└── [흰색 카드 — bg-white rounded-2xl]
    ├── [isRunning && border-b 영역]
    │   └── <StageIndicator />  ← components/training/StageIndicator.tsx
    └── [콘텐츠 영역]
        ├── isRunning → <ProgressSection />  ← components/training/ProgressSection.tsx
        └── !isRunning → <IdleSection />     ← components/training/IdleSection.tsx

isRunning = status === 'running' || status === 'paused'
```

### 10.2 StageIndicator

모델 타입별 스테이지 시퀀스 pill 표시:

| 모델 | 스테이지 |
|---|---|
| EfficientAD | 데이터 로딩 → 모델 초기화 → 학습 루프 → 테스트 추론 → 완료 |
| PatchCore | 데이터 로딩 → 모델 초기화 → 특징 추출 → Coreset 구성 → Memory Bank → 테스트 추론 → 완료 |

- 완료 스테이지: 회색 + ✓ 아이콘
- 현재 스테이지: sky-600 파란 pill + 펄스 dot (running 시)
- 대기 스테이지: 흰 배경 slate 텍스트
- 일시정지 시: amber 배지 추가

### 10.3 ProgressSection

```text
ProgressSection
├── [배치 모드 시] 일괄 학습 진행 표시 (batch_done / batch_total)
├── [진행률 바] Step X / Y, Loss, 경과 시간, 예상 잔여 시간
├── [제어 버튼] ⏸일시정지/▶재개, ⏭건너뜀(배치), ⏹중단
├── [Loss 곡선] Recharts LineChart (max 500점 다운샘플)
└── [로그] <pre> 최근 100줄, 새 줄 수신 시 자동 스크롤
```

### 10.4 IdleSection

```text
IdleSection
├── [last_result 배너] success(green) / warning(amber) / error(red)
├── [ws_error 배너]
├── [학습 시작 섹션]
│   ├── [설정 요약 아코디언] 모델/전처리/image_size/Threshold/배치크기/디바이스/시드
│   ├── [실험명 input + ▶ 학습 시작 버튼]
│   └── !hasConfig 시 amber 경고
└── [체크포인트에서 재개 섹션 (아코디언)]
    └── checkpoints[] 목록 → 각 항목: 이름/모델/날짜/진행도 + [재개][삭제] 버튼
```

### 10.5 QueuePanel

```text
QueuePanel
├── "학습 대기열 (N개)" 헤더
├── [batch_mode && isRunning] 진행 카운터
├── [!isRunning && pendingCount>0] ▶▶ 일괄 학습 시작 버튼
├── [isRunning && batch_mode] ⏭ 이번 건너뜀 / ⏹ 배치 중단 버튼
└── 대기열 테이블: #, 실험명, Set ID, 상태 (completed 항목 제외)
```

### 10.6 Zustand 스토어 바인딩

| 스토어 | 읽기 | 쓰기 (간접: WS 메시지로) |
|---|---|---|
| `trainingStore` | 모든 필드 | WS 메시지 → 각 액션 |
| `datasetStore` | `datasetMeta`, `datasetPath` | — |
| `configStore` | `preprocessingConfig`, `modelConfig`, `deviceInfo` | — |

### 10.7 WebSocket `/ws/training`

| WS 메시지 타입 | 데이터 | 스토어 액션 |
|---|---|---|
| `snapshot` | 전체 TrainingStatusResponse | `setFromSnapshot()` |
| `progress` | `step, total, loss, elapsed` | `updateProgress()` |
| `log` | `message` | `addLog()` |
| `stage` | `idx, name` | `setStage()` |
| `status` | `status` | `setStatus()` |
| `paused` | `ckpt_path` | `setPaused()` |
| `completed` | `exp_id, auc, duration_secs, message, early_stopped` | `setCompleted()` |
| `stopped` | — | `setStopped()` |
| `batch_progress` | `done, total` | — (store에 직접 반영) |
| `batch_completed` | `completed, failed, skipped` | `setBatchCompleted()` |

### 10.8 REST API 연동

| 엔드포인트 | 시점 |
|---|---|
| POST /training/start | IdleSection "▶ 학습 시작" 클릭 |
| POST /training/pause | ProgressSection "⏸ 일시정지" 클릭 |
| POST /training/unpause | ProgressSection "▶ 재개" 클릭 |
| POST /training/stop | ProgressSection "⏹ 중단" 클릭 (단일) |
| POST /training/batch/stop | ProgressSection "⏹ 중단" 클릭 (배치) |
| POST /training/batch/skip | ProgressSection/QueuePanel "⏭ 건너뜀" |
| GET /training/checkpoints | IdleSection 체크포인트 섹션 열기 |
| POST /training/resume | IdleSection "재개" 클릭 |
| DELETE /training/checkpoints/{name} | IdleSection "삭제" 클릭 |
| POST /training/batch/start | QueuePanel "▶▶ 일괄 학습 시작" 클릭 |
| GET /config/queue | QueuePanel `batch_queue_signal` 변경 시 |

---

## 11. Experiments 화면 (`src/pages/Tab4Experiments.tsx`)

### 11.1 컴포넌트 트리

```text
Tab4Experiments
├── [로딩/오류/빈 상태] → 각각 스피너/에러 카드/빈 상태 카드
│
├── [실험 목록 테이블 — bg-white rounded-2xl]
│   ├── 헤더: "실험 목록" + [삭제 버튼 (선택 시)] / [2단계 확인 UI]
│   └── <table>
│       ├── 열: 실험명, 제품, 모델, 파라미터, Accuracy, Precision, Recall, F1, F2, AUC, 실행시각, 상태
│       └── 행 클릭: selectedExperimentId 토글 (재클릭 시 해제)
│           상태 뱃지: completed(emerald), 중단(slate)
│           조기종료 시: "완료 (조기종료)"
│
├── [상세 결과 — selected && completed 시]
│   ├── 성능 지표 카드 (4-col): Accuracy, Precision, Recall, F1
│   ├── 차트 (3-col):
│   │   ├── <ConfusionMatrixChart />
│   │   ├── <RocCurveChart />
│   │   └── <ScoreDistChart thresholdValue={selected.threshold_value} />
│   └── "이 실험으로 Anomaly Map 보기 →" 버튼 → navigate('/anomaly-map')
│
├── [모델 저장 — selected && completed 시]
│   └── 경로 input + "모델 저장" 버튼
│       초기값: model_path 있으면 model_path, 없으면 ./models/{exp_id}/
│
├── [ComparisonSection — completed.length >= 2 시]
│   └── <ComparisonSection completed={completed} />
│
└── [BatchComparisonSection — set_id 가진 실험 존재 시]
    └── <BatchComparisonSection experiments={experiments} />
```

### 11.2 Zustand 스토어 바인딩

| 스토어 | 읽기 | 쓰기 |
|---|---|---|
| `experimentsStore` | `selectedExperimentId` | `setSelectedExperimentId()` |

※ 실험 목록은 로컬 state (`experiments[]`)로 관리 — 스토어에 캐시 없음

### 11.3 API 연동

| 시점 | API |
|---|---|
| 컴포넌트 마운트 | GET /experiments |
| 행 선택 + 삭제 확인 | DELETE /experiments/{id} → 목록 재로드 |
| "모델 저장" 클릭 | POST /experiments/{id}/save `{ path }` |

---

## 12. AnomalyMap 화면 (`src/pages/Tab5AnomalyMap.tsx`)

### 12.1 컴포넌트 트리

```text
Tab5AnomalyMap
├── [가드] !selectedExperimentId → 빈 상태 카드 + "← 실험 히스토리로" 버튼
│
├── [헤더] "← 실험 히스토리" 링크 + "Anomaly Map — {experiment.name}"
│
├── [빌드 섹션 — bg-white rounded-2xl]
│   ├── built: emerald dot + "완료 ({image_count}개 이미지)"
│   ├── !built: "Anomaly Map이 아직 생성되지 않았습니다."
│   ├── "Anomaly Map 생성" / "재생성" 버튼
│   └── building 중: 펄스 텍스트 "모델 추론 중, 잠시 기다려 주세요..."
│
└── [buildStatus.built 시 3개 카드]
    ├── <ControlBar />  ← components/tab5/ControlBar.tsx
    │   ├── Threshold 슬라이더 (0~1.2, step 0.01)
    │   └── 결함 유형 필터 select (전체 + defectClasses[])
    │
    ├── [이미지 그리드 카드]
    │   └── <ImageGrid />  ← components/tab5/ImageGrid.tsx
    │       ├── 통계 바: 전체N, TP, FP, TN, FN, Max Score, Avg Score
    │       ├── grid grid-cols-4 — PAGE_SIZE=20
    │       │   └── <ImageCard /> × N  (triplet 이미지 + classification 뱃지 + score)
    │       ├── 페이지네이션 버튼 (‹ page/total ›)
    │       └── [카드 클릭 시] <DetailPanel />
    │           ├── 3-panel (grid-cols-3):
    │           │   ├── 원본 이미지 (getOriginalImageUrl)
    │           │   ├── GT 마스크 (getGtMaskImageUrl, 없으면 "GT 마스크 없음")
    │           │   └── Anomaly Heatmap + 윤곽선 오버레이 (getHeatmapImageUrl)
    │           ├── 메트릭 카드: Anomaly Score, Threshold, 판정
    │           └── "↓ PNG 다운로드" 버튼
    │
    └── <ExportSection />  ← components/tab5/ExportSection.tsx
        ├── "📊 CSV 다운로드" 버튼
        └── "📦 ZIP 다운로드" 버튼 (비동기 빌드 → 폴링 완료 → 다운로드)
```

### 12.2 Threshold 초기값 계산

실험 최초 선택 시 (`storedThreshold == null`):
- `threshold_method === 'percentile'`: 정상 이미지 score 배열의 percentile(value) 사용
- `threshold_method === 'absolute'`: `threshold_value` 직접 사용
- 정규화: `(rawThr - scoreMin) / (scoreMax - scoreMin)` → `[0, 1]` 클램프

이후 threshold 변경 시 300ms debounce → `anomalyMapStore.setThreshold()` + 이미지 재조회

### 12.3 Zustand 스토어 바인딩

| 스토어 | 읽기 | 쓰기 |
|---|---|---|
| `experimentsStore` | `selectedExperimentId` | — |
| `anomalyMapStore` | `threshold` | `setThreshold()` |

### 12.4 API 연동

| 시점 | API |
|---|---|
| `selectedExperimentId` 변경 시 | GET /experiments (실험 정보 취득) |
| `selectedExperimentId` 변경 시 | GET /anomaly-map/{id}/build-status |
| `buildStatus.built` && threshold/class 변경 시 | GET /anomaly-map/{id}/images |
| "Anomaly Map 생성/재생성" 클릭 | POST /anomaly-map/{id}/build → polling GET /jobs/{job_id}/status (1s) |
| "📊 CSV 다운로드" | POST /anomaly-map/{id}/export/csv → Blob 다운로드 |
| "📦 ZIP 다운로드" | POST /anomaly-map/{id}/export/zip/prepare → polling → GET zip 다운로드 |
| 이미지 카드 표시 | GET /anomaly-map/{id}/images/{path}/triplet (URL 직접 참조) |
| DetailPanel 3-panel | getOriginalImageUrl / getGtMaskImageUrl / getHeatmapImageUrl |

---

## 13. 공통 레이아웃 컴포넌트

### 13.1 Navbar (`src/components/layout/Navbar.tsx`)

```text
<nav class="bg-slate-900">
├── Logo: Smart QC SVG 아이콘 + "Smart QC" 텍스트 (white)
├── 구분선 (slate-700)
├── [Config 칩들 — flex-1 flex-wrap]
│   ├── 모델 칩: "EFFICIENTAD · small" (슬레이트 배경)
│   ├── 전처리 칩: "전처리 · none"
│   └── 데이터셋 카운터: "학습 {train_good_count} / 테스트 {total}"
└── [우측 상태 칩들]
    ├── status==='running' → "학습 중" amber 펄스 칩
    └── deviceInfo → "GPU" (emerald-900) or "CPU" (slate-700) 칩
```

**읽는 스토어:** `datasetStore.datasetMeta`, `configStore.deviceInfo|modelConfig|preprocessingConfig`, `trainingStore.status`

### 13.2 TabBar (`src/components/layout/TabBar.tsx`)

- React Router `<NavLink>` 사용 (isActive → sky-500 border-b + sky-600 텍스트)
- 비활성: transparent border + slate-500 텍스트 → hover 시 slate-900

---

## 14. 화면 간 의존성 상세

```text
Dataset 화면
  └─[datasetPath, productName, datasetMeta]──────────────→ Config 화면 (가드)
  └─[datasetMeta]─────────────────────────────────────────→ Training 화면 (가드)
  └─[clearConfigs() 트리거]───────────────────────────────→ configStore 초기화

Config 화면
  └─[preprocessingConfig, modelConfig]────────────────────→ Training 화면 (가드)
  └─[batch_queue_signal]──────────────────────────────────→ QueuePanel 재로드

Training 화면 (WS)
  └─[학습 완료 시]────────────────────────────────────────→ Experiments 화면 (새 실험 추가)

Experiments 화면
  └─[selectedExperimentId]────────────────────────────────→ AnomalyMap 화면 (가드 + 초기 임계값)
```

---

## 15. 핵심 인터랙션

| 위젯/요소 | 인터랙션 | 동작 |
|---|---|---|
| Dataset "검증" 버튼 | 클릭 → loading | POST /dataset/validate → setDataset |
| Dataset 경로 input | 값 변경 → 이전 검증 결과 유지 (재검증 필요) | — |
| Config "저장" 버튼 | 클릭 → image_size 검증 | POST /config/save → setConfigs → 3s 배지 |
| Config 대기열 아코디언 | 클릭 → toggle | queueOpen state 변경 |
| Training WS 연결 | App.tsx 마운트 시 1회 | snapshot 수신 → trainingStore 초기화 |
| Training "▶ 학습 시작" | 클릭 → startLoading | POST /training/start → WS 이벤트로 상태 전환 |
| Training "⏸ 일시정지" | 클릭 → pauseLoading | POST /training/pause → WS paused 수신 |
| Training "⏭ 건너뜀" | 클릭 | POST /training/batch/skip |
| StageIndicator | WS stage 메시지 | current_stage_idx 업데이트 → pill 하이라이트 |
| Experiments 행 클릭 | 클릭 → 선택/해제 토글 | setSelectedExperimentId |
| Experiments "삭제" | 클릭 → 2단계 확인 UI | → DELETE /experiments/{id} |
| Experiments "Anomaly Map 보기" | 클릭 | navigate('/anomaly-map') |
| AnomalyMap threshold 슬라이더 | 값 변경 → 300ms debounce | setThreshold → GET images 재조회 |
| AnomalyMap 결함 유형 select | 변경 즉시 | GET images 재조회 (class 필터) |
| ImageGrid 카드 클릭 | 클릭 → 선택/해제 토글 | DetailPanel 열기/닫기 |
| DetailPanel "↓ PNG 다운로드" | 클릭 → downloading | triplet 이미지 Blob → anchor download |
| ExportSection "📦 ZIP 다운로드" | 클릭 → zipLoading | prepareZip → polling(1s) → downloadZip |

---

## 16. 상태 기반 UI 반응

| 상태 | 트리거 | UI 반응 |
|---|---|---|
| Dataset 검증 중 | handleValidate() 호출 | 버튼 disabled + loading 텍스트 |
| datasetMeta 없음 | Config/Training 화면 진입 | amber 경고 카드 렌더링 |
| Config 저장 성공 | saveOk=true | green "저장됨" 배지 3초 표시 |
| Config 저장 실패 | saveError | red 텍스트 인라인 표시 |
| Training running | trainingStore.status | isRunning=true → ProgressSection 전환, StageIndicator 표시 |
| Training paused | status==='paused' | StageIndicator에 "일시정지" amber 배지 |
| 학습 완료 | WS setCompleted | status→idle, last_result success 배너 |
| 학습 중단 | WS setStopped | status→idle, last_result warning 배너 |
| WS 연결 오류 | useTrainingWs 오류 | trainingStore.ws_error → IdleSection 에러 배너 |
| Experiments 없음 | experiments.length===0 | 빈 상태 카드 |
| 실험 선택 | selectedExperimentId 변경 | 상세 결과 섹션 렌더링 |
| AnomalyMap 미선택 | !selectedExperimentId | 빈 상태 카드 + "실험 히스토리로" 버튼 |
| 빌드 진행 중 | building=true | 버튼 disabled + 펄스 텍스트 |
| 이미지 로딩 중 | imagesLoading=true | 펄스 텍스트 |

---

## 17. 예외 처리 UX

| 상황 | 위치 | UI 처리 |
|---|---|---|
| 데이터셋 경로 오류 | Dataset 화면 | 로컬 error state → 빨간 텍스트 인라인 |
| image_size % 32 != 0 | Config 화면 저장 | saveError red 텍스트 + 저장 중단 |
| Config 저장 실패 | Config 화면 | saveError red 텍스트 |
| 학습 시작 실패 | IdleSection | startError red 카드 |
| 학습 제어 실패 (pause/stop) | ProgressSection | ctrlError red 텍스트 |
| WS 연결 오류 | useTrainingWs | ws_error → IdleSection 에러 배너 (✕ 닫기 가능) |
| 실험 목록 로드 실패 | Experiments 화면 | red 카드 렌더링 |
| 실험 삭제 실패 | Experiments 삭제 확인 UI | deleteError red 텍스트 인라인 |
| 모델 저장 실패 | Experiments 모델 저장 | saveResult 빨간 카드 |
| 실험 로드 실패 | AnomalyMap 화면 | expError red 카드 |
| 빌드 실패 | AnomalyMap 빌드 섹션 | buildError red 카드 |
| 이미지 조회 실패 | ImageGrid | imagesError red 카드 |
| GT 마스크 없음 | DetailPanel img onError | "GT 마스크 없음" placeholder 표시 |
| CSV 내보내기 실패 | ExportSection | csvError red 텍스트 |
| ZIP 생성/다운로드 실패 | ExportSection | zipError red 텍스트 |

---

## 18. React 렌더링 전략

| 항목 | 구현 내용 |
|---|---|
| **상태 관리** | Zustand v5 (전역 스토어) + React 로컬 state (폼·로딩·에러 등 화면별) |
| **라우팅** | react-router-dom v7 BrowserRouter + NavLink (isActive 기반 스타일) |
| **WS 연결** | App.tsx 루트에서 `useTrainingWs()` 1회 마운트 → 라우트 변경 시에도 연결 유지 |
| **WS 재연결** | connect 시 서버가 snapshot 메시지 전송 → `setFromSnapshot()` 으로 상태 복원 |
| **Threshold debounce** | 로컬 state (즉시 슬라이더 반응) + debouncedThreshold (300ms, API 호출용) |
| **Loss 히스토리 다운샘플** | > 500점 시 고정 비율로 샘플링 (ProgressSection `downsample()`) |
| **배치 학습 큐 새로고침** | `batch_queue_signal` Zustand 필드 bump → QueuePanel `useEffect` 의존 |
| **이미지 URL** | API 헬퍼 함수로 URL 생성 (`getTripletImageUrl` 등) → `<img src>` 직접 참조 |
| **페이지네이션** | PAGE_SIZE=20, 로컬 page state, imagesData 변경 시 page=1 리셋 |
| **ZIP 비동기 빌드** | prepareZip → 1초 폴링 getJobStatus → completed 시 downloadZip |

---

## 19. 반응형 및 레이아웃 특성

| 항목 | 내용 |
|---|---|
| **레이아웃** | 최상단 고정 Navbar + TabBar, 스크롤 가능 `<main>` 영역 |
| **대상 환경** | 데스크탑 브라우저 (ML 실험 환경 기준) |
| **컬럼 분할** | 2열 (`grid-cols-2`), 3열 (`grid-cols-3`), 4열 (`grid-cols-4`) 혼용 |
| **카드 스타일** | `bg-white rounded-2xl border border-slate-200 shadow-sm` 일관 사용 |
| **모바일** | 비공식 지원 범위 (Tailwind 반응형 미적용) |

---

---

# v1.x 참고 (Streamlit 기반 — 삭제 금지)

> 이하 내용은 v1.x Streamlit 구현 기준입니다. v2.0 React UI가 공식 구현입니다.
> 구현 과정에서의 설계 결정, 위젯 키 체계, 상태 관리 패턴 참고용으로 보존합니다.

---

### v1.x 1. 프로젝트 개요 (Streamlit)

| 항목 | 내용 |
|---|---|
| **UI 유형** | Streamlit 기반 단일 사용자 ML 실험 대시보드 |
| **UI 레이아웃** | `st.set_page_config(layout="wide")` — Wide 레이아웃 |
| **사이드바** | `sidebar_state="expanded"` — 기본 펼침 |

---

### v1.x 2. 전체 화면 구조 (Streamlit)

```text
app.py
├── st.set_page_config(title="Smart QC Dashboard", icon="🔍", layout="wide")
├── init_session_state()          ← utils/session_state_init.py
├── Sidebar (components/sidebar.py)
│   ├── 제목: "Smart QC Dashboard"
│   ├── 데이터셋 정보 (dataset_meta)
│   ├── 디바이스 정보 (device_info)
│   └── 학습 상태 (current_run_status)
└── st.tabs([
    "📁 탭1. 데이터 폴더",
    "⚙️ 탭2. 전처리 및 모델 설정",
    "🚀 탭3. 학습",
    "📊 탭4. 실험 히스토리",
    "🗺️ 탭5. 이상 영역 시각화"
])
```

---

### v1.x 3. 탭별 기능 구조 (Streamlit)

| 탭 | 목적 | 주요 입력 위젯 | 주요 출력 | 저장/연동 파일 |
|---|---|---|---|---|
| **탭1** | MVTec AD 폴더 구조 검증 | `st.text_input` (경로), `st.button` (검증) | 폴더 트리, 클래스별 이미지 수, 썸네일 | `session_state["dataset_meta"]` |
| **탭2** | 전처리 방식 설정·미리보기 및 모델 하이퍼파라미터 설정 | `st.radio` (방식/모델 선택), `st.slider` (파라미터), `st.number_input` (image_size 등) | 원본/필터 미리보기 이미지, 디바이스 정보 | `configs.yaml → preprocessing + model`, `session_state["preprocessing_config"]`, `session_state["model_config"]` |
| **탭3** | 학습 실행 및 실시간 모니터링 | `st.text_input` (실험명), `st.button` (시작/중지) | Progress bar, Loss curve, 실시간 로그 | `experiments/history.json`, `models/`, `logs/`, `configs.yaml` |
| **탭4** | 실험 히스토리 비교 분석 | `st.dataframe` (실험 선택), `st.multiselect` (지표 선택) | ROC curve, Confusion matrix, Anomaly score 분포, 다중 실험 비교 차트 | `experiments/history.json` (읽기/삭제), `models/` (모델 저장) |
| **탭5** | 이상 영역 시각화 및 결과 내보내기 | `st.selectbox` (결함 필터), `st.slider` (threshold), `st.dataframe` (이미지 선택) | Original / GT mask / Anomaly heatmap 3-panel, TP/FP/TN/FN 분류 요약 | `results/` (PNG 다운로드), CSV, ZIP 내보내기 |

---

### v1.x 4. 전역 상태 관리 — session_state 스키마

`utils/session_state_init.py`에 정의된 `SESSION_STATE_SCHEMA` 기준.

| Key | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `dataset_path` | `str \| None` | `None` | 검증된 데이터셋 루트 경로 |
| `dataset_meta` | `dict \| None` | `None` | 탭1 검증 결과 메타데이터 |
| `preprocessing_config` | `dict \| None` | `None` | 탭2 저장된 전처리 설정 |
| `model_config` | `dict \| None` | `None` | 탭2 저장된 모델 파라미터 |
| `device_info` | `dict \| None` | `None` | CUDA/CPU 디바이스 정보 (탭2 첫 방문 시 1회 감지) |
| `experiments` | `dict[str, dict]` | `{}` | 탭4 진입 시 `history.json`으로부터 갱신 |
| `current_run_status` | `"idle" \| "running"` | `"idle"` | 학습 실행 상태 |
| `current_exp_id` | `str \| None` | `None` | 현재 실행 중인 실험 ID |
| `_stop_event` | `threading.Event \| None` | `None` | 학습 중지 신호 |
| `_result_queue` | `queue.Queue \| None` | `None` | Worker → UI 메시지 큐 |
| `_worker` | `TrainingWorker \| None` | `None` | 백그라운드 학습 스레드 |
| `_progress` | `dict \| None` | `None` | `{step, total, loss, elapsed}` |
| `_log_lines` | `list[str]` | `[]` | 실시간 로그 (최대 100줄) |
| `_loss_history` | `list[dict]` | `[]` | `{step, loss}` 리스트 (Loss curve 데이터) |
| `selected_experiment_id` | `str \| None` | `None` | 탭4에서 선택된 실험 ID → 탭5에서 참조 |
| `anomaly_map_threshold` | `float \| None` | `None` | 탭5 threshold 슬라이더 값 (탭 간 유지) |

---

### v1.x 5. 설정 및 산출물 파일 구조

```text
smart-qc-dashboard/
├── configs.yaml
├── experiments/
│   └── history.json
├── models/
│   └── {experiment_id}/
│       ├── model_state_dict.pth
│       └── configs.yaml
├── logs/
│   └── {experiment_id}.log
├── results/
└── dataset/
    └── imagenet_penalty/
```

**configs.yaml 섹션 구조:**
```yaml
experiment:
  name: string
  created_at: ISO8601

preprocessing:
  method: none | homomorphic | he | clahe
  resize_mode: padding
  image_size: int
  normalization: imagenet | custom
  mean: [float, float, float]
  std: [float, float, float]
  params: dict | null

model:
  model_type: efficientad | patchcore
  image_size: int
  batch_size: int
  random_seed: int
  threshold_method: percentile | absolute
  threshold_value: float
  params: dict
```

---

### v1.x 6. 탭 간 의존성 및 진입 제한 (Streamlit)

| 탭 | 선행 조건 | 미충족 시 UI 동작 |
|---|---|---|
| **탭1** | 없음 | — |
| **탭2** | `dataset_path` 설정 완료 | `st.warning("먼저 탭1에서 데이터 폴더를 설정해 주세요.")` — 탭 내용 렌더링 중단 |
| **탭3** | `dataset_path`, `preprocessing_config`, `model_config` 모두 완료 | 각각에 대해 `st.warning` 표시, 학습 시작 버튼 비활성화 |
| **탭4** | `experiments` 딕셔너리가 비어있지 않음 | `st.warning("아직 실행된 실험이 없습니다.")` |
| **탭5** | `selected_experiment_id` 설정 완료 | `st.info("탭4에서 분석할 실험을 먼저 선택해 주세요.")` |

---

### v1.x 7. 실험 실행 흐름 (Streamlit)

```text
[탭1] 데이터셋 경로 입력 → st.button "경로 확인"
         ↓ 검증 통과
      session_state["dataset_path"], ["dataset_meta"] 저장
         ↓
[탭2] 전처리 방식/파라미터 설정
                                  모델/하이퍼파라미터 설정
                                  → st.button "설정 저장" (session)
                                  → st.button "configs.yaml 저장"
         ↓
      session_state["preprocessing_config"], session_state["model_config"] 저장
         ↓
[탭3] 실험명 입력 → st.button "학습 시작"
         ↓
      configs.yaml에 experiment 섹션 기록
      TrainingWorker (threading.Thread) 시작
         │
         ├── Queue: {type: "progress", step, total, loss, elapsed}
         ├── Queue: {type: "log", message}
         └── Queue: {type: "completed", y_true, anomaly_scores, anomaly_maps, ...}
                    또는 {type: "error", exception, traceback}
                    또는 {type: "stopped", step}
         ↓
      [완료] 3단계 원자적 저장:
        1. models/{exp_id}/model_state_dict.pth
        2. models/{exp_id}/configs.yaml (스냅샷)
        3. experiments/history.json (append)
         ↓
[탭4] 실험 선택 (st.dataframe, single-row selection)
      → session_state["selected_experiment_id"] 저장
         ↓
[탭5] Threshold 슬라이더 조정
      → LRU Cache miss 시: load_model_for_inference() → run_inference()
      → Original / GT mask / Anomaly heatmap 3-panel 표시
      → CSV / PNG / ZIP 내보내기
```

---

### v1.x 8. 탭1 와이어프레임 (Streamlit)

```text
┌─────────────────────────────────────────────────────────────┐
│  📁 탭1. 데이터 폴더                                         │
│                                                             │
│  데이터셋 경로                                               │
│  [___________________________________________]              │
│   placeholder: "예: /app/dataset/screw"                     │
│   key: "input_dataset_path"                                 │
│                                                             │
│  [경로 확인]  ← st.button (primary, key:"_tab1_validate_btn")│
│                                                             │
│  ── 검증 결과 ────────────────────────────────────────────  │
│                                                             │
│  [✅ 정상] 또는 [❌ 오류 메시지]                              │
│                                                             │
│  ℹ️  Grayscale 이미지가 감지되었습니다. RGB로 자동 변환됩니다.  │
│  ⚠️  유효하지 않은 파일 N개가 감지되었습니다.               │
│                                                             │
│  폴더 구조                                                   │
│  ┌──────────────────────────────────────────┐              │
│  │  screw/                                  │  ← st.code() │
│  │  ├── train/good/   (N 이미지)            │              │
│  │  ├── test/good/    (N 이미지)            │              │
│  │  ├── test/crack/   (N 이미지)            │              │
│  │  └── ground_truth/ (N 마스크)            │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  클래스별 이미지 수                                          │
│  ┌────────────────────────────────────────────┐            │
│  │  결함 유형  | 테스트 이미지 | GT 마스크     │  ← st.dataframe│
│  │  good       |     20       |     0         │            │
│  │  crack       |     15       |    15         │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  대표 이미지 (최대 4열)                                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
│  │ good   │ │ crack  │ │ ...    │ │ ...    │  ← st.image  │
│  └────────┘ └────────┘ └────────┘ └────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**`dataset_meta` 스키마 (v1.x):**
```python
{
    "dataset_path": str,
    "train_good_count": int,
    "test_counts": dict[str, int],
    "gt_counts": dict[str, int],
    "total_test_count": int,
    "channels": 1 | 3,
    "defect_classes": list[str],
    "supported_formats": list[str],
    "has_invalid_files": bool,
    "_invalid_file_count": int,
}
```

---

### v1.x 9. 탭2 와이어프레임 (Streamlit)

```text
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ 탭2. 전처리 및 모델 설정                                 │
│                                                             │
│  ════════════════════════════════════════════════════════   │
│  【전처리 영역】                                             │
│  ════════════════════════════════════════════════════════   │
│                                                             │
│  전처리 방식:                                                │
│  ◉ 없음  ○ Homomorphic  ○ HE  ○ CLAHE                      │
│  ← st.radio(key="tab2_method_label", horizontal=True)       │
│                                                             │
│  [Homomorphic 선택 시]                                      │
│  sigma    [━━━●──────────] 10.0  key:"tab2_sigma"          │
│  gamma_H  [━━━━━━━●──────] 1.5   key:"tab2_gamma_h"        │
│  gamma_L  [━━━●──────────] 0.5   key:"tab2_gamma_l"        │
│  ☑ normalize  ← st.checkbox(key="tab2_normalize")          │
│                                                             │
│  [CLAHE 선택 시]                                            │
│  clipLimit [━━━●──────────] 2.0  key:"tab2_clip_limit"     │
│                                                             │
│  [HE 선택 시]                                               │
│  ℹ️  HE는 별도 파라미터가 없습니다.                          │
│                                                             │
│  image_size: [  256  ▲▼]  key:"tab2_image_size"            │
│  ❌ 32의 배수가 아니면 st.error() 표시                       │
│                                                             │
│  정규화: ◉ ImageNet  ○ 커스텀                               │
│  [커스텀 시] mean/std text_input                            │
│                                                             │
│  전처리 미리보기  ← st.columns(3)                           │
│  [원본 이미지] [필터 적용 후] [(3번째 열)]                   │
│                                                             │
│  ════════════════════════════════════════════════════════   │
│  【모델 영역】                                               │
│  ════════════════════════════════════════════════════════   │
│                                                             │
│  ℹ️  현재 디바이스: CUDA (RTX xxxx), VRAM: xx.x GB          │
│  모델 선택:  ◉ EfficientAD  ○ PatchCore                    │
│                                                             │
│  공통 설정  ← st.columns(2)                                 │
│  [batch_size] [random_seed]                                │
│                                                             │
│  EfficientAD 파라미터 (선택 시)                             │
│  [model_size radio] [learning_rate] [train_steps]          │
│  ▶ 고급 설정 (st.expander)                                  │
│                                                             │
│  PatchCore 파라미터 (선택 시)                               │
│  [backbone select] [coreset_sampling_ratio slider]         │
│  [neighbourhood_kernel select_slider: 1 3 5 7 9]          │
│  ▶ 고급 설정 (st.expander)                                  │
│                                                             │
│  Threshold 설정                                              │
│  방식: ◉ Percentile  ○ Absolute                             │
│  [슬라이더] → 예상 정상/결함 비율 st.metric                  │
│                                                             │
│  ════════════════════════════════════════════════════════   │
│  【설정 저장】                                               │
│  [설정 저장] [configs.yaml 저장] [configs.yaml 불러오기]    │
│  key: tab2_btn_save / tab2_btn_yaml_save / tab2_btn_yaml_load│
└─────────────────────────────────────────────────────────────┘
```

**`preprocessing_config` 스키마 (v1.x):**
```python
{
    "method": "none" | "homomorphic" | "he" | "clahe",
    "resize_mode": "padding",
    "image_size": int,
    "normalization": "imagenet" | "custom",
    "mean": [float, float, float],
    "std": [float, float, float],
    "params": dict | None,
}
```

---

### v1.x 10. 탭3 와이어프레임 (Streamlit)

```text
┌─────────────────────────────────────────────────────────────┐
│  🚀 탭3. 학습                                               │
│                                                             │
│  ── Idle 상태 ─────────────────────────────────────────── │
│  실험 이름  [__________________________________]             │
│  key:"tab3_experiment_name"                                 │
│  ▶ 현재 학습 설정 요약  ← st.expander                        │
│  ⚠️  디스크 100MB 미만: 학습 시작 차단                       │
│  [학습 시작]  ← st.button (primary)                         │
│                                                             │
│  ── Running 상태 ──────────────────────────────────────── │
│  Progress ████████████░░░░░░░░  65%   ← st.progress()      │
│  Loss Curve  ← st.plotly_chart() 실시간 갱신                │
│  실시간 로그  ← st.text_area(disabled=True)                  │
│  [학습 중지]  ← st.button (secondary)                       │
│                                                             │
│  ── 완료/중단 상태 ─────────────────────────────────────── │
│  [완료] ✅ st.success("학습 완료")                           │
│  [중단] ℹ️  st.info("학습이 중단되었습니다.")                │
│  [오류] ❌ st.error(traceback)                               │
└─────────────────────────────────────────────────────────────┘
```

**TrainingWorker 큐 메시지 타입 (v1.x):**

| `type` 값 | 데이터 | UI 반응 |
|---|---|---|
| `"progress"` | `step, total, loss, elapsed` | progress bar + loss curve 갱신 |
| `"log"` | `message` | 로그 텍스트 추가 |
| `"completed"` | `y_true, anomaly_scores, anomaly_maps, image_paths, model, duration_seconds` | 결과 저장 → st.success |
| `"error"` | `exception, traceback` | st.error(traceback) |
| `"stopped"` | `step` | st.info(중단 메시지) → history에 "중단" 기록 |

---

### v1.x 11. 탭4 와이어프레임 (Streamlit)

```text
┌─────────────────────────────────────────────────────────────┐
│  📊 탭4. 실험 히스토리                                       │
│                                                             │
│  실험 목록  (history.json 매 렌더 시 읽기)                   │
│  ← st.dataframe(selection_mode="single-row", key="t5_table")│
│                                                             │
│  [🗑 실험 삭제]  key:"t5_delete_btn"                        │
│  [삭제 확인 시] st.columns([1,1,6]) → [확인] [취소]         │
│                                                             │
│  성능 지표  ← st.columns(4): Accuracy, Precision, Recall, F1│
│  차트  ← st.columns(3): Confusion Matrix, ROC, Score Dist  │
│                                                             │
│  모델 저장                                                   │
│  경로: [________________________] key:"t5_save_path"        │
│  [💾 모델 저장]  key:"t5_save_btn"                          │
│                                                             │
│  ▶ 다중 실험 비교 차트  ← st.expander                        │
│  비교 실험 선택: st.checkbox × N                            │
│  지표 선택: st.multiselect(key="t5_cmp_metrics")            │
│  차트 유형: ◉ 막대 차트  ○ 레이더 차트                      │
└─────────────────────────────────────────────────────────────┘
```

---

### v1.x 12. 탭5 와이어프레임 (Streamlit)

```text
┌─────────────────────────────────────────────────────────────┐
│  🗺️ 탭5. 이상 영역 시각화                                   │
│                                                             │
│  결함 유형 필터: [전체 ▼]  key:"tab5_class_filter"          │
│  Threshold [━━━━━━━━━━━●───] 0.542                         │
│  ← st.slider(key: session_state["anomaly_map_threshold"])   │
│                                                             │
│  점수 요약: Max Score | Mean Score  ← st.metric()×2        │
│  분류 현황: TP | FP | TN | FN       ← st.metric()×4        │
│                                                             │
│  이미지 목록  ← st.dataframe(selection_mode="single-row",  │
│                              key="tab6_image_table")        │
│                                                             │
│  3-패널 시각화 (선택 시)  ← st.columns(3)                   │
│  [원본 이미지] [GT 마스크] [Anomaly Heatmap + 컨투어]       │
│  지표: Anomaly Score | Threshold | 판정  ← st.metric()×3   │
│  [⬇ PNG 저장]  ← st.download_button                        │
│                                                             │
│  내보내기  ← st.columns(2)                                   │
│  [⬇ CSV 내보내기] [ZIP 준비] → [⬇ ZIP 다운로드]            │
│  key: tab6_csv_export / tab6_zip_prepare / tab6_zip_download│
└─────────────────────────────────────────────────────────────┘
```

**Anomaly Map LRU 캐시 (v1.x `cache_manager.py`):**
- 최대 3개 항목 유지
- Key 형식: `"_anomaly_maps_{exp_id}"`
- 캐시 미스 시: `load_model_for_inference()` → `run_inference()` → 캐시 저장
- 추방(Eviction): `cached_at` 타임스탬프 기준 가장 오래된 항목 제거

---

### v1.x 13. 사이드바 (Streamlit `components/sidebar.py`)

```text
┌─────────────────────────────┐
│  Smart QC Dashboard         │  ← st.title()
│  데이터셋 경로               │  ← st.caption()
│  결함 클래스: crack, scratch │  ← st.caption()
│  [학습 240장] [테스트 85장]  │  ← st.metric()×2
│  ✅ CUDA: RTX 3090          │  ← st.success()
│  ⚠️ 학습 실행 중...         │  ← st.info() (running 시)
│  현재 설정: EfficientAD 등  │
└─────────────────────────────┘
```

---

### v1.x 14. 상태 및 설정 관리 구조 (Streamlit)

| 영역 | 관리 방식 | 역할 |
|---|---|---|
| UI 실행 상태 | `st.session_state` | 탭 간 상태 공유, 위젯 키 기반 |
| 전처리 설정 | `configs.yaml → preprocessing` | 영속 저장 (원자적 write) |
| 모델 설정 | `configs.yaml → model` | 영속 저장 (원자적 write) |
| 실험 기록 | `experiments/history.json` | 실험 메타데이터 누적 저장 |
| 학습 로그 | `logs/{exp_id}.log` | 라인 버퍼 방식 순차 기록 |
| 모델 가중치 | `models/{exp_id}/model_state_dict.pth` | 학습 완료 시 저장 |
| 설정 스냅샷 | `models/{exp_id}/configs.yaml` | 실험 시점 불변 기록 |
| Anomaly map 캐시 | `session_state` (LRU, max 3) | 추론 결과 메모리 캐싱 |

---

### v1.x 15. 탭 간 의존성 상세 (Streamlit)

```text
탭1 ──[dataset_path, dataset_meta]─────────────────→ 탭2
탭2 ──[preprocessing_config, model_config]─────────→ 탭3
탭3 ──[experiments, selected_exp_id]───────────────→ 탭4
탭4 ──[selected_experiment_id]─────────────────────→ 탭5
```

---

### v1.x 16. 핵심 인터랙션 (Streamlit)

| 위젯/요소 | 인터랙션 타입 | 동작 |
|---|---|---|
| 탭1 "경로 확인" 버튼 | `st.button` click → rerun | MVTec AD 폴더 구조 검증, `dataset_meta` 저장 |
| 탭2 전처리 방식 radio | `st.radio` change → rerun | 방식별 파라미터 슬라이더 조건부 표시 |
| 탭2 미리보기 | 파라미터 변경 → rerun → `st.image` 갱신 | 실시간 원본/필터 비교 |
| 탭3 "학습 시작" 버튼 | `st.button` click → TrainingWorker 시작 | 백그라운드 학습 시작 |
| 탭3 Progress 갱신 | `_result_queue` poll → rerun | progress bar + loss curve + 로그 실시간 갱신 |
| 탭4 실험 선택 | `st.dataframe` row select → rerun | 상세 차트 렌더링 + `selected_experiment_id` 저장 |
| 탭5 threshold 슬라이더 | `st.slider` change → rerun | TP/FP/TN/FN 재계산, 판정 갱신 |
| 탭5 이미지 선택 | `st.dataframe` row select → rerun | 3-패널 시각화 렌더링 |

---

### v1.x 17~19. Streamlit 렌더링 전략 및 예외 처리

| 항목 | 구현 내용 |
|---|---|
| **세션 상태 초기화** | `app.py` 진입 시 `init_session_state()` 1회 실행 |
| **백그라운드 학습** | `threading.Thread` (TrainingWorker) 사용 |
| **Worker → UI 통신** | `queue.Queue` 폴링 방식 |
| **YAML 원자적 쓰기** | `tmpfile → rename` 방식 (`config_manager.py`) |
| **history.json 원자적 쓰기** | `tmpfile → rename` 방식 (`storage.py`) |
| **모델 저장 프로토콜** | 3단계 순차 저장: `.pth` → `configs.yaml snapshot` → `history.json` |

**예외 처리 (v1.x):**

| 상황 | UI 처리 |
|---|---|
| dataset 구조 오류 | `st.error("MVTec AD 형식의 폴더 구조가 아닙니다.")` |
| image_size 비배수 | `st.error("32의 배수로 입력하세요")` + 저장 차단 |
| 디스크 공간 부족 (<100MB) | `st.error()` + 학습 시작 차단 |
| 디스크 공간 부족 (100~500MB) | `st.warning()` |
| ImageNet penalty 이미지 없음 | `st.error()` + 학습 시작 차단 |

---

### v1.x 20. 프로젝트 파일 구조 전체 (Streamlit)

```text
smart-qc-dashboard/
├── app.py
├── configs.yaml
├── components/
│   ├── __init__.py
│   └── sidebar.py
├── tabs/
│   ├── tab1_data_folder.py
│   ├── tab2_config.py
│   ├── tab3_training.py
│   ├── tab4_history.py
│   └── tab5_anomaly_map.py
├── utils/
│   ├── env_init.py
│   ├── session_state_init.py
│   ├── config_manager.py
│   ├── storage.py
│   ├── cache_manager.py
│   ├── training_worker.py
│   ├── model_factory.py
│   ├── mvtec_dataset.py
│   └── dataset_scanner.py
├── experiments/
│   └── history.json
├── models/
│   └── {experiment_id}/
│       ├── model_state_dict.pth
│       └── configs.yaml
├── logs/
└── dataset/
    └── imagenet_penalty/
```

---

*이 문서는 실제 소스 코드 역설계를 기반으로 작성되었습니다. v2.0 내용은 smart-qc-explorer 소스 기준이며, [확인 필요: ...] 표기 항목은 추가 코드 검토가 필요합니다.*

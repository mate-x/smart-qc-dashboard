# Smart QC Vision — UI/UX 설계 문서

> **작성 기준:** React 19 소스 코드 역설계 (smart-qc-vision/src/)
> **최종 갱신:** 2026-06-11
> **버전:** 2.0

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|---|---|---|
| 2.0 | 2026-06-11 | React + Zustand 기준으로 전면 재작성. Streamlit 내용은 섹션 하단 v1.x 참고로 이동 (삭제 금지) |
| 1.2 | 2026-05-29 | 탭2 대기열 UI, 탭3 단계 인디케이터, 검사탭2 실시간 차트 추가 |
| 1.1 | — | 비전검사 대시보드 탭 구조 추가, 사이드바 전환 버튼 |
| 1.0 | 2026-05-26 | 초기 작성 (Streamlit 기준) |

---

## 목차

- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 전체 화면 구조](#2-전체-화면-구조)
- [3. 화면별 기능 요약](#3-화면별-기능-요약)
- [4. 전역 상태 관리 — inspectionStore](#4-전역-상태-관리--inspectionstore)
- [5. 프로젝트 파일 구조](#5-프로젝트-파일-구조)
- [6. 공통 레이아웃 컴포넌트](#6-공통-레이아웃-컴포넌트)
- [7. Realtime Inspection 화면](#7-realtime-inspection-화면)
- [8. History 화면](#8-history-화면)
- [9. Model Settings 화면](#9-model-settings-화면)
- [10. 이미지 서빙 방식](#10-이미지-서빙-방식)
- [11. WebSocket 생명주기](#11-websocket-생명주기)
- [12. 불량 감지 알림 UI 흐름](#12-불량-감지-알림-ui-흐름)
- [13. 화면 간 의존성 및 진입 제한](#13-화면-간-의존성-및-진입-제한)
- [14. 핵심 인터랙션](#14-핵심-인터랙션)
- [15. 상태 기반 UI 반응](#15-상태-기반-ui-반응)
- [16. 예외 처리 UX](#16-예외-처리-ux)
- [17. API 연동 전체 목록](#17-api-연동-전체-목록)
- [18. 반응형 및 레이아웃 특성](#18-반응형-및-레이아웃-특성)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **프로젝트 목적** | 학습 완료 모델을 활용한 실시간 비전 검사, 검사 이력 조회·분석, 모델 교체 관리 |
| **핵심 사용자** | 제조 현장 비전 검사 운영자 |
| **주요 시나리오** | 모델 선택 → 수동/자동 검사 → 판정 결과 확인 → 이력 분석 → 모델 교체 |
| **UI 유형** | React SPA (smart-qc-vision, port 5173) |
| **백엔드** | FastAPI (smart-qc-dashboard, port 8000) |
| **스타일링** | Tailwind CSS (유틸리티 우선) |
| **라우팅** | react-router-dom v6 |

---

## 2. 전체 화면 구조

```text
src/main.tsx
└── <BrowserRouter>
    └── <App>                             ← App.tsx
        ├── useActiveModel()              ← 앱 최초 1회: GET /api/inspection/model
        ├── <TabBar />                    ← layout/TabBar.tsx (slate-800 상단 바)
        │   ├── [탭 링크 3개]
        │   └── <ModelStatusChip />       ← 우측: 현재 적용 모델 표시
        ├── <GpuWarningBanner />          ← layout/GpuWarningBanner.tsx (조건부 황색 배너)
        └── <main>
            └── <Routes>
                ├── /           → Tab1Realtime
                ├── /history    → Tab2History
                ├── /settings   → Tab3Settings
                └── *           → <Navigate to="/" replace />
```

**TabBar 탭 레이블 ↔ 라우트 매핑:**

| 탭 레이블 | 경로 | 컴포넌트 |
|---|---|---|
| 실시간 검사 | `/` | Tab1Realtime |
| 검사 이력 | `/history` | Tab2History |
| 설정 | `/settings` | Tab3Settings |

> 📌 **코드 기준 주의**: README에는 `/models` 라우트로 표기되어 있으나 실제 코드(App.tsx)는 `/settings`를 사용함.

---

## 3. 화면별 기능 요약

| 화면 | 목적 | Guard | 주요 훅 |
|---|---|---|---|
| **Realtime Inspection** | 수동/자동 비전 검사, 실시간 이미지 표시 | `NoModelGuard` (모달 오버레이) | useManualInspection, useAutoInspection, useDefectOnlyInspection, useInspectionImages |
| **History** | 검사 이력 테이블, KPI 카드, 통계 차트, CSV 내보내기 | `NoModelGuard` (모달 오버레이) | useInspectionRecords, useStatCharts |
| **Model Settings** | 완료 실험 선택·적용, 이미지 소스 경로 설정 | 없음 (항상 접근 가능) | useModels (30초 폴링), useApplyModel, useUpdateSourcePath |

---

## 4. 전역 상태 관리 — inspectionStore

**파일:** `src/store/inspectionStore.ts` (Zustand)

| 필드 | 타입 | 초기값 | 역할 |
|---|---|---|---|
| `activeModel` | `ActiveModel \| null` | `null` | 현재 적용 모델 정보 |
| `gpuWarning` | `string \| null` | `null` | GPU 경고 메시지 (GpuWarningBanner 표시용) |
| `lastResult` | `InspectionResult \| null` | `null` | 최근 검사 결과 (판정 + score) |
| `imageStamp` | `number` | `0` | 이미지 URL cache-bust 값 (`Date.now()`) |
| `isAutoRunning` | `boolean` | `false` | 자동 검사 WS 실행 중 여부 |
| `defectStopped` | `boolean` | `false` | 불량 감지로 자동 검사 중지됨 → DefectPopup 트리거 |
| `reshuffledToast` | `boolean` | `false` | 이미지 풀 재셔플 토스트 표시 여부 |

**액션:**

| 액션 | 효과 |
|---|---|
| `setActiveModel(model, gpuWarning?)` | activeModel + gpuWarning 설정, **lastResult/imageStamp/isAutoRunning/defectStopped 초기화** |
| `setLastResult(result)` | lastResult 업데이트 + `imageStamp = Date.now()` |
| `setAutoRunning(v)` | isAutoRunning 토글 |
| `setDefectStopped(v)` | defectStopped 토글 |
| `clearHistory()` | lastResult + imageStamp만 초기화 (records 제외) |
| `showReshuffledToast()` / `dismissReshuffledToast()` | reshuffledToast 토글 |
| `setActiveModelDatasetPath(path)` | activeModel.dataset_path 업데이트 |
| `setGpuWarning(warning)` | gpuWarning 설정/해제 |

**`ActiveModel` 타입:**
```typescript
{
  experiment_id: string;
  name: string;
  model_type: 'efficientad' | 'patchcore';
  dataset_path: string;
  threshold: number;
}
```

**`InspectionResult` 타입:**
```typescript
{
  verdict: '양품' | '불량';
  anomaly_score: number;
  image_name: string;
  was_reshuffled?: boolean;
}
```

---

## 5. 프로젝트 파일 구조

```text
smart-qc-vision/src/
├── main.tsx                           # ReactDOM.createRoot + BrowserRouter
├── App.tsx                            # useActiveModel() + TabBar + GpuWarningBanner + Routes
│
├── pages/
│   ├── Tab1Realtime.tsx               # 실시간 검사
│   ├── Tab2History.tsx                # 검사 이력
│   └── Tab3Settings.tsx               # 검사 설정 (모델 교체 + 소스 경로)
│
├── components/
│   ├── layout/
│   │   ├── TabBar.tsx                 # 상단 네비게이션 바 (slate-800)
│   │   ├── ModelStatusChip.tsx        # TabBar 우측 현재 모델 칩
│   │   ├── GpuWarningBanner.tsx       # GPU 경고 황색 배너 (조건부)
│   │   └── NoModelGuard.tsx           # 모델 미선택 시 모달 오버레이
│   ├── tab1/
│   │   ├── AutoRunningBanner.tsx      # "자동 검사 진행 중" 황색 배너
│   │   ├── InspectionControls.tsx     # 4개 검사 버튼 + 에러 표시
│   │   ├── VerdictCard.tsx            # 판정 결과 카드 (양품/불량/초기 안내)
│   │   ├── ImagePanel.tsx             # 범용 이미지 패널 (레이블 헤더 + img)
│   │   ├── ImagePanelPlaceholder.tsx  # 이미지 없을 때 placeholder
│   │   ├── AnomalyMapPanel.tsx        # ImagePanel 래퍼 (label="Anomaly Map")
│   │   ├── OverlayPanel.tsx           # ImagePanel 래퍼 (label="이상 영역 오버레이")
│   │   └── DefectPopup.tsx            # 불량 감지 모달 팝업
│   ├── tab2/
│   │   ├── VerdictFilter.tsx          # 판정 필터 라디오 버튼 (전체/양품만/불량만)
│   │   ├── RecordsTable.tsx           # 이력 테이블 (PAGE_SIZE=10, 페이지네이션)
│   │   ├── RecordRow.tsx              # 이력 테이블 단일 행
│   │   ├── KpiCards.tsx               # KPI 카드 4개 (총검사/양품/불량/불량률)
│   │   ├── StatCharts.tsx             # 통계 차트 컨테이너 (단위 선택 + 3분할)
│   │   ├── TimeRangeTable.tsx         # 시간 범위 그룹 테이블 (좌측)
│   │   ├── ScoreHistogram.tsx         # Anomaly Score 히스토그램 (중앙)
│   │   ├── ScoreScatter.tsx           # Anomaly Score 산점도 (우측)
│   │   └── ClearHistoryDialog.tsx     # 이력 초기화 확인 모달
│   └── tab3/
│       ├── ModelTable.tsx             # 실험 목록 테이블
│       ├── ModelRow.tsx               # 실험 테이블 단일 행
│       └── ApplyModelButton.tsx       # 모델 적용 버튼 + 경고 메시지
│
├── hooks/
│   ├── useActiveModel.ts              # 앱 최초 1회 GET /api/inspection/model
│   ├── useModels.ts                   # GET /api/models (30초 폴링)
│   ├── useApplyModel.ts               # POST /api/inspection/model
│   ├── useUpdateSourcePath.ts         # PATCH /api/inspection/source-path
│   ├── useManualInspection.ts         # POST /api/inspection/run + polling
│   ├── useDefectOnlyInspection.ts     # POST /api/inspection/run {defect_only:true} + polling
│   ├── useAutoInspection.ts           # WebSocket /ws/inspection/auto 관리
│   ├── useInspectionImages.ts         # imageStamp 기반 이미지 URL 생성
│   ├── useInspectionRecords.ts        # GET/DELETE /api/inspection/records
│   └── useStatCharts.ts               # N개 단위 그룹핑 + 선택 상태 관리
│
├── api/
│   ├── client.ts                      # axios 인스턴스 (baseURL: localhost:8000)
│   ├── modelsApi.ts                   # /api/models, /api/inspection/model, /api/inspection/source-path
│   ├── inspectionApi.ts               # /api/inspection/run, /api/inspection/job/{id}, 이미지 URL 헬퍼
│   └── recordsApi.ts                  # /api/inspection/records, CSV
│
├── store/
│   └── inspectionStore.ts             # Zustand 전역 스토어 (단일)
│
└── types/
    ├── model.ts                       # ExperimentRecord, ActiveModel
    ├── inspection.ts                  # InspectionResult, InspectionRecord, WsMessage, VerdictFilter
    └── api.ts                         # API 응답 타입
```

---

## 6. 공통 레이아웃 컴포넌트

### 6.1 TabBar (`components/layout/TabBar.tsx`)

```text
<nav class="bg-slate-800 h-13">
├── [좌측] 탭 링크 3개 (react-router-dom Link)
│   활성: font-semibold text-white bg-slate-700
│   비활성: text-slate-400 (hover 없음)
└── [우측] <ModelStatusChip />
    모델 없음: 회색 칩 "모델 미선택"
    모델 있음: 초록 칩 "모델: {name} / {model_type}"
```

**읽는 스토어:** `inspectionStore.activeModel`

### 6.2 GpuWarningBanner (`components/layout/GpuWarningBanner.tsx`)

- `gpuWarning !== null` 일 때만 렌더링
- 황색 배너 (`bg-yellow-100 border-yellow-300`)
- 텍스트: `"⚠️ {gpuWarning}"` + × 닫기 버튼 (`setGpuWarning(null)`)
- TabBar 바로 아래, `<main>` 위에 위치 (App.tsx 기준)
- **트리거:** 모델 적용 시 서버 응답의 `gpu_warning` 필드

### 6.3 NoModelGuard (`components/layout/NoModelGuard.tsx`)

- `activeModel === null` 시 children 렌더링 + 그 위에 모달 오버레이 표시
- `activeModel !== null` 시 children만 렌더링 (guard 없음)
- 오버레이: 반투명 검정 (`bg-black/50`) + 흰 모달 카드
- 모달 내용: "모델 미선택" + "설정 페이지에서 모델을 먼저 선택해 주세요." + "설정 페이지로 이동" 버튼
- 버튼 클릭: `navigate('/settings')`

---

## 7. Realtime Inspection 화면 (`src/pages/Tab1Realtime.tsx`)

### 7.1 컴포넌트 트리

```text
Tab1Realtime
└── <NoModelGuard>          ← activeModel===null 시 오버레이 모달 표시
    ├── <div class="flex flex-col h-full">
    │   ├── {isAutoRunning && <AutoRunningBanner />}
    │   │   └── "🔄 자동 검사 진행 중..." 황색 배너
    │   │
    │   ├── [헤더 행 — flex justify-between]
    │   │   ├── <InspectionControls>          ← 4개 버튼 그룹
    │   │   │   ├── "수동 검사 (1개 검사)"      bg-blue-600
    │   │   │   ├── "▶ 자동 검사 (3초마다 1개)" bg-green-600
    │   │   │   ├── "⏹ 자동 검사 중지"         bg-red-600
    │   │   │   └── "불량만 검사 (1개)"         bg-slate-600
    │   │   └── {lastResult && 판정 pill 배지}  ← 양품(초록)/불량(빨강) 인라인
    │   │
    │   └── [이미지 패널 — grid grid-cols-3]
    │       ├── <ImagePanel url={imageUrl} label="원본 이미지" onRatioDetected />
    │       ├── <AnomalyMapPanel url={anomalyMapUrl} />
    │       └── <OverlayPanel url={overlayUrl} />
    │           (세 패널 모두 동일한 aspectRatio 공유 — imageRatio state)
    │
    ├── {defectStopped && <DefectPopup>}      ← fixed 오버레이 모달
    ├── {reshuffledToast && 재셔플 toast}     ← fixed bottom-right (3s 자동 해제)
    └── {defectOnlyWarning && 경고 toast}    ← fixed bottom-right amber
```

> 📌 **v1.x 대비 변경**: VerdictCard가 별도 열이 아닌 헤더 행 우측 pill 배지로 변경됨. 이미지 패널은 4열이 아닌 3열 (판정 결과 열 제거).

### 7.2 InspectionControls 버튼 상세

| 버튼 레이블 | 색상 | disabled 조건 |
|---|---|---|
| 수동 검사 (1개 검사) | blue-600 | `isLoading \|\| isDefectOnlyLoading \|\| isAutoRunning` |
| ▶ 자동 검사 (3초마다 1개) | green-600 | `isLoading \|\| isDefectOnlyLoading \|\| isAutoRunning` |
| ⏹ 자동 검사 중지 | red-600 | `isLoading \|\| isDefectOnlyLoading \|\| !isAutoRunning` |
| 불량만 검사 (1개) | slate-600 | `isLoading \|\| isDefectOnlyLoading \|\| isAutoRunning` |

- isLoading / isDefectOnlyLoading 시: "⏳ 검사 중... 잠시 기다려 주세요" 텍스트 표시
- error (manualError 또는 defectOnlyError): 빨간 텍스트 인라인 표시

### 7.3 판정 결과 표시

- `lastResult === null`: 헤더 행 우측 빈 공간 (VerdictCard는 직접 사용 안 함)
- `lastResult !== null`: 헤더 행 우측에 pill 배지
  - 양품: `bg-green-100 text-green-800 border-green-300` + "✅ 양품 {score}"
  - 불량: `bg-red-100 text-red-800 border-red-300` + "❌ 불량 {score}"

> 📌 **VerdictCard 컴포넌트**는 Tab1Realtime에서 직접 사용되지 않음. 독립적으로 존재하며 [확인 필요: 다른 위치에서 사용되는지].

### 7.4 이미지 패널 (3종)

| 패널 | label | 이미지 URL |
|---|---|---|
| ImagePanel | "원본 이미지" | `/api/inspection/image/last?t={stamp}` |
| AnomalyMapPanel | "Anomaly Map" | `/api/inspection/anomaly-map/last?t={stamp}` |
| OverlayPanel | "이상 영역 오버레이" | `/api/inspection/overlay/last?t={stamp}` |

- 세 패널의 높이 동기화: `imageRatio` state (원본 이미지 onLoad 시 감지) → `style.aspectRatio` 적용
- 이미지 없음 (`imageStamp === 0`): URL null → `<ImagePanelPlaceholder>` 표시
- 이미지 로드 오류 (`onError`): imgError state → `<ImagePanelPlaceholder>` 전환

### 7.5 Zustand 스토어 바인딩

| 스토어 필드 | 사용 |
|---|---|
| `isAutoRunning` | AutoRunningBanner 조건, 버튼 disabled |
| `lastResult` | 헤더 판정 pill 배지 |
| `defectStopped` | DefectPopup 표시 트리거 |
| `reshuffledToast` | 재셔플 toast 표시 (3초 자동 해제) |
| `imageStamp` | 이미지 URL 생성 (useInspectionImages) |

### 7.6 수동 검사 흐름 (`useManualInspection`)

```text
"수동 검사" 클릭
  → POST /api/inspection/run {}
  → job_id 취득
  → polling: GET /api/inspection/job/{job_id} (1초 간격, max 120초)
    → status === 'completed': setLastResult(result) → imageStamp = Date.now()
    → status === 'failed': setError(msg)
    → 타임아웃: setError('검사 시간이 초과됐습니다.')
```

### 7.7 불량만 검사 흐름 (`useDefectOnlyInspection`)

```text
"불량만 검사 (1개)" 클릭
  → POST /api/inspection/run { defect_only: true }
  → 동일 polling 패턴
  → status === 'failed' && "불량 이미지가 없습니다" 포함:
    → setWarning(msg) → amber toast (3초 자동 해제)
  → 그 외 failed: setError(msg)
```

---

## 8. History 화면 (`src/pages/Tab2History.tsx`)

### 8.1 컴포넌트 트리

```text
Tab2History
└── <NoModelGuard>
    ├── [헤더 행 — flex justify-between]
    │   ├── h2 "검사 이력"
    │   └── "CSV 내보내기" 버튼 (allRecords.length===0 → disabled)
    │
    ├── <VerdictFilterBar>          ← 판정 필터 라디오 (전체/양품만/불량만)
    │   (클라이언트 필터링 — 서버 재요청 없음)
    │
    ├── <RecordsTable records={filteredRecords} isLoading />
    │   ├── 열: 번호, 시각, 이미지명, 판정결과, Anomaly Score
    │   ├── PAGE_SIZE=10, 페이지네이션 (‹ N/M ›)
    │   ├── max-height 226px, 스크롤
    │   └── 빈 상태: "아직 검사 기록이 없습니다."
    │
    ├── <KpiCards allRecords>
    │   └── grid grid-cols-4: 총검사(blue), 양품(green), 불량(red/gray), 불량률(red/gray)
    │
    ├── <StatCharts allRecords threshold={activeModel.threshold}>
    │   ├── 단위 선택 버튼: [20개] [40개] [100개]
    │   └── grid-cols-[1fr_2fr_2fr]
    │       ├── <TimeRangeTable>  ← 시간 범위 그룹 목록 (행 클릭 → 차트 갱신)
    │       ├── <ScoreHistogram>  ← Anomaly Score 분포 히스토그램
    │       └── <ScoreScatter>    ← Anomaly Score 추이 산점도
    │
    └── "🗑 이력 초기화" 버튼 (우측 정렬)
        └── {showClearDialog && <ClearHistoryDialog>}  ← 확인 모달
```

### 8.2 RecordsTable 상세

| 컬럼 | 표시 형식 |
|---|---|
| 번호 | `record.seq` (내림차순 정렬) |
| 시각 | `record.inspected_at` (YYYY-MM-DD HH:MM:SS) |
| 이미지명 | `record.image_name` |
| 판정결과 | 양품 → 초록 스타일 / 불량 → 빨간 스타일 |
| Anomaly Score | `{score:.4f}` |

- 페이지네이션: `filteredRecords` 변경 시 page=1 리셋
- 빈 상태 (records.length===0): "아직 검사 기록이 없습니다."

### 8.3 KpiCards 계산 (클라이언트 순수 계산, API 호출 없음)

```text
total = allRecords.length
good  = allRecords.filter(r => r.verdict === '양품').length
bad   = total - good
defectRate = bad === 0 ? '0.0%' : `${(bad/total*100).toFixed(1)}%`
```

- `bad > 0` 시 불량/불량률 카드 red 색상, 아니면 gray

### 8.4 StatCharts 상세

**단위 (Unit):** 20 / 40 / 100 (기본 20)

**useStatCharts 그룹핑 로직:**
1. allRecords를 seq 오름차순 정렬
2. N개씩 chunkByN → TimeGroup[]
3. 각 그룹의 label = `"YYYY-MM-DD HH:MM~HH:MM"` (첫 레코드~마지막 레코드)
4. 마지막 그룹: `isPartial = chunk.length < N`

**선택 동작:**
- Unit 변경 → 마지막 그룹으로 이동
- allRecords 변경 → 현재 인덱스 유지 (범위 초과 시 클램프)
- TimeRangeTable 행 클릭 → setSelectedGroupIndex

**ScoreHistogram:**
- x: 0~1 (Anomaly Score), y: 빈도
- 정상(blue) / 불량(red) bar 구분
- threshold 수직 점선 표시

**ScoreScatter:**
- x: 1~selectedUnit (인덱스), y: 0~1 (Score)
- 정상(파란점) / 불량(빨간점)
- threshold 수평 빨간 점선 표시
- 앞뒤 점 연결

### 8.5 ClearHistoryDialog

- `showClearDialog=true` 시 fixed 오버레이 모달
- 내용: "이력을 초기화하면 모든 검사 기록이 삭제됩니다."
- 버튼: [취소] (onCancel) / [초기화] bg-red-600 (onConfirm)
- 확인 흐름: `await clearRecords()` → DELETE /api/inspection/records → `clearHistory()` → setAllRecords([])

### 8.6 Zustand 스토어 바인딩

| 스토어 필드 | 사용 |
|---|---|
| `activeModel.threshold` | StatCharts threshold 값 (하이라이트 기준선) |
| `activeModel` | NoModelGuard 가드 조건 |
| `clearHistory()` | 이력 초기화 시 lastResult/imageStamp 초기화 |

### 8.7 API 연동

| 시점 | API |
|---|---|
| 컴포넌트 마운트 | GET /api/inspection/records |
| VerdictFilter 변경 | 서버 재요청 없음 (클라이언트 필터) |
| "🗑 이력 초기화" 확인 | DELETE /api/inspection/records |
| "CSV 내보내기" 클릭 | `window.open('/api/inspection/records/csv')` — 브라우저 다운로드 |

---

## 9. Model Settings 화면 (`src/pages/Tab3Settings.tsx`)

### 9.1 컴포넌트 트리

```text
Tab3Settings
├── h2 "검사 설정"
│
├── [실험 선택 섹션]
│   ├── <ModelTable>
│   │   └── <table>
│   │       ├── 열: (선택 radio), 실험명, 검사 제품, 모델타입, F1, AUC, 실행시각
│   │       ├── 현재 적용 모델 행 하이라이트
│   │       ├── 행 클릭 → setSelectedId(id)
│   │       ├── 로딩: "로딩 중..."
│   │       └── 빈 상태: "사용 가능한 완료된 실험이 없습니다."
│   │
│   └── <ApplyModelButton>
│       ├── {showWarning && amber 경고 텍스트}
│       │   "⚠️ 모델을 교체하면 현재 세션의 모든 검사 이력이 삭제됩니다."
│       │   (selectedId !== null && selectedId !== activeModelId && !isLoading 시만 표시)
│       ├── "모델 적용" 버튼 (blue-600)
│       │   disabled: !selectedId || selectedId===activeModelId || isLoading
│       └── {error && 빨간 에러 텍스트}
│
└── [이미지 소스 경로 섹션] (border-t 구분선)
    ├── "이미지 소스 경로" 레이블
    ├── 안내: "현재: {activeModel.dataset_path}"
    ├── 경로 input + "경로 적용" 버튼
    │   disabled: !activeModel || pathLoading
    └── {pathError && 빨간 에러 텍스트}
```

### 9.2 ModelTable 컬럼 상세

| 컬럼 | 데이터 소스 | 표시 |
|---|---|---|
| (라디오) | 선택 상태 | 현재 active 행 표시 |
| 실험명 | `experiment.experiment_id` | 문자열 |
| 검사 제품 | `experiment.product_name` [확인 필요: 필드명] | 문자열 |
| 모델타입 | `experiment.model_type` | EfficientAD / PatchCore |
| F1 | `experiment.metrics.f1_score` | `{f1:.4f}` |
| AUC | `experiment.metrics.auc` | `{auc:.4f}` |
| 실행시각 | `experiment.created_at` | YYYY-MM-DD HH:MM |

- `status === 'completed'` 실험만 표시 (서버 필터링 또는 클라이언트 필터링 [확인 필요])
- 30초 폴링 (useModels): 새 학습 완료 실험 자동 반영

### 9.3 모델 적용 흐름

```text
행 클릭 → setSelectedId(id)
"모델 적용" 클릭
  → POST /api/inspection/model { experiment_id: selectedId }
  → 응답: { active_model, gpu_warning }
  → setActiveModel(active_model, gpu_warning)
      ← activeModel 갱신 + lastResult/imageStamp/isAutoRunning/defectStopped 초기화
  → gpu_warning 있으면 GpuWarningBanner 표시
  → setSelectedId(null)
```

### 9.4 이미지 소스 경로 설정

```text
경로 input 수정 → "경로 적용" 클릭
  → PATCH /api/inspection/source-path { source_path: ... }
  → 성공: setActiveModelDatasetPath(res.source_path)
      ← activeModel.dataset_path 업데이트
  → 실패: pathError 빨간 텍스트 표시
```

### 9.5 Zustand 스토어 바인딩

| 스토어 필드 | 사용 |
|---|---|
| `activeModel` | 현재 모델 ID 비교 (이미 적용 중 비활성화), dataset_path 표시 |
| `setActiveModel()` | 모델 적용 후 상태 갱신 + 초기화 |
| `setActiveModelDatasetPath()` | 소스 경로 업데이트 |

### 9.6 API 연동

| 시점 | API |
|---|---|
| 컴포넌트 마운트 + 30초 폴링 | GET /api/models |
| "모델 적용" 클릭 | POST /api/inspection/model |
| "경로 적용" 클릭 | PATCH /api/inspection/source-path |

---

## 10. 이미지 서빙 방식

### 10.1 이미지 URL 구조

```typescript
// useInspectionImages.ts + inspectionApi.ts
imageStamp = 0  →  { imageUrl: null, anomalyMapUrl: null, overlayUrl: null }

imageStamp > 0  →  {
  imageUrl:      `http://localhost:8000/api/inspection/image/last?t=${stamp}`,
  anomalyMapUrl: `http://localhost:8000/api/inspection/anomaly-map/last?t=${stamp}`,
  overlayUrl:    `http://localhost:8000/api/inspection/overlay/last?t=${stamp}`,
}
```

### 10.2 Cache-bust 메커니즘

- **imageStamp** = `Date.now()` — `setLastResult()` 호출 시마다 갱신
- URL의 `?t={stamp}` 파라미터가 매 검사마다 달라지므로 브라우저 캐시 무효화
- 수동 검사, 자동 검사(WS result), 불량만 검사 모두 `setLastResult()` 경유

### 10.3 이미지 엔드포인트 상세

| 엔드포인트 | 내용 |
|---|---|
| GET /api/inspection/image/last | 마지막 검사 원본 이미지 |
| GET /api/inspection/anomaly-map/last | Anomaly Map (jet colormap 적용 RGB) |
| GET /api/inspection/overlay/last | 원본 + anomaly 오버레이 이미지 |

- 서버는 항상 "최신 검사" 결과의 이미지를 반환
- 클라이언트는 `?t=` 파라미터로 캐시 무효화만 담당

---

## 11. WebSocket 생명주기 (`/ws/inspection/auto`)

**훅:** `src/hooks/useAutoInspection.ts`

### 11.1 연결 흐름

```text
[1] start() 호출
    기존 소켓 있으면 close() (단일 소켓 보장)
    ↓
[2] new WebSocket('ws://localhost:8000/ws/inspection/auto')
    ↓
[3] onopen: ws.send('start') → setAutoRunning(true)
    ↓
[4] onmessage: JSON.parse(event.data) → WsMessage
    ├── type: 'result'
    │   → setLastResult(msg) → imageStamp = Date.now()
    │   → msg.was_reshuffled: showReshuffledToast() (3초 auto-dismiss)
    ├── type: 'defect_stopped'
    │   → setDefectStopped(true) → DefectPopup 모달 표시
    │   (자동 검사는 서버에서 이미 중지됨)
    ├── type: 'stopped'
    │   → 서버의 stop 확인 (클라이언트는 stop()에서 이미 처리)
    └── type: 'error'
        → setAutoRunning(false)
        → [확인 필요: Phase 3 에러 토스트 미구현]
```

### 11.2 중지 흐름

```text
[방법 1] stop() 호출 (사용자가 "⏹ 자동 검사 중지" 클릭)
  → wsRef.current?.send('stop')
  → setAutoRunning(false)

[방법 2] DefectPopup에서 "🛑 검사 종료" 클릭
  → setDefectStopped(false) 만 호출 (WS는 이미 서버가 중지)

[방법 3] DefectPopup에서 "✅ 확인 및 재개" 클릭
  → setDefectStopped(false)
  → start() (새 WS 연결)
```

### 11.3 재연결

- 현재 구현상 자동 재연결 없음
- `onerror`: setAutoRunning(false) — 사용자가 "▶ 자동 검사" 버튼 재클릭으로 재연결
- `onclose`: setAutoRunning(false) — 동일

### 11.4 컴포넌트 언마운트 처리

```text
Tab1Realtime 언마운트 시 (useEffect cleanup)
  → isAutoRunning === true: wsRef.current?.send('stop') (서버 루프 정리)
  → wsRef.current?.close()
  → wsRef.current = null
```

### 11.5 WsMessage 타입

```typescript
type WsMessage =
  | { type: 'result'; verdict: '양품'|'불량'; anomaly_score: number;
      image_name: string; was_reshuffled?: boolean }
  | { type: 'defect_stopped' }
  | { type: 'stopped' }
  | { type: 'error'; message: string }
```

---

## 12. 불량 감지 알림 UI 흐름

### 12.1 전체 흐름

```text
자동 검사 중 (isAutoRunning === true)
  ↓
WS 메시지 type: 'defect_stopped' 수신
  ↓
inspectionStore.setDefectStopped(true)
  ↓
Tab1Realtime: {defectStopped && <DefectPopup />}
  ↓
DefectPopup 고정 모달 표시 (z-[1000], 배경 bg-black/55)
  ┌─────────────────────────────────────────────┐
  │ ❌ 불량이 감지되었습니다! 자동 검사가 중지됨   │ (red-100 배경)
  │                                             │
  │  [🛑 검사 종료]      [✅ 확인 및 재개]       │
  └─────────────────────────────────────────────┘
```

### 12.2 팝업 버튼 동작

| 버튼 | 동작 |
|---|---|
| "🛑 검사 종료" | `setDefectStopped(false)` → 팝업 닫기 (자동 검사는 이미 중지 상태) |
| "✅ 확인 및 재개" | `setDefectStopped(false)` + `start()` → 새 WS 연결 시작 |

### 12.3 팝업과 마지막 결과

- 팝업이 열려도 이미지 패널은 렌더링 유지 (결과 확인 가능)
- 팝업은 fixed 오버레이이므로 이미지 위에 표시됨

---

## 13. 화면 간 의존성 및 진입 제한

| 화면 | Guard | Guard 동작 |
|---|---|---|
| **Realtime Inspection** | `NoModelGuard` | activeModel=null → 모달 오버레이 + "설정 페이지로 이동" 버튼 |
| **History** | `NoModelGuard` | 동일 |
| **Model Settings** | 없음 | 항상 접근 가능 |

**NoModelGuard 동작:**
- children을 정상 렌더링 후 그 위에 오버레이 표시 (v1.x의 렌더링 중단 방식과 다름)
- 화면 이동 없이 현재 경로 유지, 모달 내 버튼으로 `/settings`로 이동

**모델 적용 후 상태 초기화:**
- `setActiveModel()` 호출 → lastResult/imageStamp/isAutoRunning/defectStopped 초기화
- Tab2의 allRecords는 초기화되지 않음 (서버 records는 유지)
- [확인 필요: 모델 교체 시 서버 records도 초기화되는지]

---

## 14. 핵심 인터랙션

| 위젯/요소 | 인터랙션 | 동작 |
|---|---|---|
| "수동 검사" 버튼 | 클릭 → isLoading | POST /api/inspection/run → polling → setLastResult |
| "▶ 자동 검사" 버튼 | 클릭 | WS 연결 → send('start') → setAutoRunning(true) |
| "⏹ 자동 검사 중지" 버튼 | 클릭 | send('stop') → setAutoRunning(false) |
| "불량만 검사 (1개)" 버튼 | 클릭 → isDefectOnlyLoading | POST /api/inspection/run {defect_only:true} → polling |
| DefectPopup "✅ 확인 및 재개" | 클릭 | setDefectStopped(false) + start() (WS 재연결) |
| DefectPopup "🛑 검사 종료" | 클릭 | setDefectStopped(false) (팝업 닫기) |
| GpuWarningBanner × | 클릭 | setGpuWarning(null) |
| VerdictFilterBar | radio 변경 | verdictFilter state → filteredRecords 재계산 (서버 재요청 없음) |
| RecordsTable 페이지네이션 | ‹/› 클릭 | page state 변경 |
| "🗑 이력 초기화" 버튼 | 클릭 | showClearDialog=true → ClearHistoryDialog 모달 |
| ClearHistoryDialog "초기화" | 클릭 | DELETE /api/inspection/records → clearHistory() |
| StatCharts 단위 버튼 | 클릭 | selectedUnit 변경 → 마지막 그룹으로 이동 |
| TimeRangeTable 행 클릭 | 클릭 | setSelectedGroupIndex → 차트 갱신 |
| "CSV 내보내기" 버튼 | 클릭 | window.open('/api/inspection/records/csv') |
| ModelTable 행 클릭 | 클릭 | setSelectedId(id) |
| ApplyModelButton "모델 적용" | 클릭 | POST /api/inspection/model → setActiveModel |
| "경로 적용" 버튼 | 클릭 | PATCH /api/inspection/source-path → setActiveModelDatasetPath |

---

## 15. 상태 기반 UI 반응

| 상태 | 트리거 | UI 반응 |
|---|---|---|
| `activeModel === null` | 앱 시작 / 미설정 | NoModelGuard 모달 오버레이 표시 |
| `isAutoRunning === true` | WS onopen 후 | AutoRunningBanner 황색 배너, 자동검사 버튼 disabled |
| `lastResult !== null` | setLastResult() | 헤더 판정 pill 배지 표시, 이미지 패널 URL 활성화 |
| `defectStopped === true` | WS 'defect_stopped' | DefectPopup 모달 표시 |
| `reshuffledToast === true` | WS result.was_reshuffled | 우측 하단 toast (3초 후 자동 해제) |
| `gpuWarning !== null` | setActiveModel(gpuWarning) | GpuWarningBanner 황색 배너 |
| 검사 중 (`isLoading`) | 수동/불량만 검사 시작 | "⏳ 검사 중..." 텍스트, 관련 버튼 disabled |
| `imageStamp === 0` | 초기 또는 clearHistory | 이미지 패널 placeholder 표시 |
| `bad > 0` | KpiCards 계산 | 불량/불량률 카드 red 색상 |
| 모델 미선택 (ModelStatusChip) | activeModel 없음 | 회색 칩 "모델 미선택" |
| 모델 선택 완료 | setActiveModel | 초록 칩 "모델: {name} / {type}" |

---

## 16. 예외 처리 UX

| 상황 | 발생 위치 | UI 처리 |
|---|---|---|
| 수동 검사 실패 | useManualInspection | InspectionControls 하단 빨간 텍스트 |
| 불량만 검사 실패 | useDefectOnlyInspection | InspectionControls 하단 빨간 텍스트 |
| 불량만 검사 — 불량 없음 | useDefectOnlyInspection | 우측 하단 amber toast (3초) |
| 수동 검사 타임아웃 (120초) | useManualInspection | "검사 시간이 초과됐습니다." 에러 |
| WS 연결 오류 | useAutoInspection onerror | setAutoRunning(false) [Phase 3: 에러 토스트 미구현] |
| WS 연결 끊김 | useAutoInspection onclose | setAutoRunning(false) |
| WS 서버 에러 메시지 | type: 'error' | setAutoRunning(false) [Phase 3: 토스트 미구현] |
| 이미지 로드 오류 | ImagePanel onError | imgError=true → Placeholder 전환 |
| 모델 목록 로드 실패 | useModels | [확인 필요: 에러 표시 없음, error state만 존재] |
| 모델 적용 실패 | ApplyModelButton | error 빨간 텍스트 인라인 |
| 소스 경로 적용 실패 | Tab3Settings | pathError 빨간 텍스트 인라인 |
| 이력 조회 실패 | useInspectionRecords | [확인 필요: Phase 3 토스트 미구현] |
| activeModel 로드 실패 | useActiveModel | 조용히 처리 (서버 미실행 등) |

---

## 17. API 연동 전체 목록

| 메서드 | 엔드포인트 | 훅/위치 | 호출 시점 |
|---|---|---|---|
| GET | `/api/inspection/model` | useActiveModel | App 마운트 1회 |
| GET | `/api/models` | useModels | Tab3Settings 마운트 + 30초 폴링 |
| POST | `/api/inspection/model` | useApplyModel | "모델 적용" 클릭 |
| PATCH | `/api/inspection/source-path` | useUpdateSourcePath | "경로 적용" 클릭 |
| POST | `/api/inspection/run` | useManualInspection | "수동 검사" 클릭 |
| POST | `/api/inspection/run` | useDefectOnlyInspection | "불량만 검사" 클릭 (body: `{defect_only:true}`) |
| GET | `/api/inspection/job/{id}` | useManualInspection, useDefectOnlyInspection | 검사 후 1초 폴링 |
| GET | `/api/inspection/image/last?t={stamp}` | useInspectionImages | imageStamp 변경 시 (URL 직접 참조) |
| GET | `/api/inspection/anomaly-map/last?t={stamp}` | useInspectionImages | 동일 |
| GET | `/api/inspection/overlay/last?t={stamp}` | useInspectionImages | 동일 |
| GET | `/api/inspection/records` | useInspectionRecords | Tab2History 마운트 |
| DELETE | `/api/inspection/records` | useInspectionRecords | 이력 초기화 확인 |
| GET | `/api/inspection/records/csv` | recordsApi.downloadCsv | "CSV 내보내기" (window.open) |
| WS | `/ws/inspection/auto` | useAutoInspection | "▶ 자동 검사" 클릭 |

---

## 18. 반응형 및 레이아웃 특성

| 항목 | 내용 |
|---|---|
| **레이아웃** | 상단 고정 TabBar (slate-800) + 조건부 GpuWarningBanner + 스크롤 가능 main |
| **main 패딩** | `p-6` |
| **main 높이** | `flex-1 min-h-0 overflow-hidden` |
| **이미지 패널 레이아웃** | `grid grid-cols-3 gap-4` |
| **KPI 카드 레이아웃** | `grid grid-cols-4 gap-3` |
| **통계 차트 레이아웃** | `grid grid-cols-[1fr_2fr_2fr] gap-4` |
| **대상 환경** | 데스크탑 브라우저 |
| **모바일** | 비공식 지원 범위 |
| **패널 높이 동기화** | imageRatio state → 이미지 3패널에 동일 aspectRatio 적용 |

---

---

# v1.x 참고 (Streamlit 기반 — 삭제 금지)

> 이하 내용은 v1.x Streamlit 구현 기준입니다. v2.0 React UI가 공식 구현입니다.
> 설계 결정 참고, 위젯 키 체계, 상태 패턴 등의 이해를 위해 보존합니다.

---

### v1.x A. 전체 레이아웃 구조 (Streamlit)

```
app.py
├── active_dashboard == "explorer"   →  [모델 탐색 대시보드] (기존 탭1~6)
└── active_dashboard == "inspection" →  [비전검사 대시보드]
        ├── 검사 탭1 — 실시간 검사
        ├── 검사 탭2 — 검사 이력 및 통계
        └── 검사 탭3 — 딥러닝 모델 교체
```

**Streamlit 설정:**
- `layout = "wide"`, `initial_sidebar_state = "expanded"`

---

### v1.x B. 사이드바 (Streamlit v1.1)

사이드바는 **대시보드 전환 버튼만** 표시.

```
┌─────────────────────────────────┐
│        Smart QC Platform        │
│  [🔬 모델 탐색 플랫폼]    (버튼) │
│  [🏭 비전검사 플랫폼]     (버튼) │
└─────────────────────────────────┘
```

- 활성 대시보드 버튼: `type="primary"` (파란색)
- 비활성 버튼: `type="secondary"` (회색)

---

### v1.x B-EXP. 모델 탐색 플랫폼 UI 변경 — v1.2

#### 탭2 하단 — 실험 대기열 2분할 UI

```
┌───────────────────────────────────┬─────────────────────────────────┐
│  [좌측] 대기열 테이블              │  [우측] 선택 모델 상세           │
│  순번 │ 실험명 │ 모델 │ 상태       │  실험명: PatchCore_0529_a1b2    │
│   1  │ exp_A │ EAD │ ■ 완료     │  preprocessing: method: clahe   │
│   2  │ exp_B │ PC  │ ■ 진행     │  model: backbone: ...           │
└───────────────────────────────────┴─────────────────────────────────┘
```

**상태 색상:** 대기중(회색) / 진행중(파랑) / 완료(초록) / 실패(빨강) / 건너뜀(주황)

#### 탭3 상단 — 학습 단계 인디케이터

```
✅ ①데이터 로딩  ✅ ②모델 초기화  🔵 ③학습 루프  ○ ④추론  ○ ⑤완료
████████████░░░░░░ Step 42,000 / 70,000 (60.0%)
[⏸ 일시정지]  [⏭ 건너뛰기]  [⏹ 전체 중단]
```

---

### v1.x C. 비전검사 탭 구조 (Streamlit)

```python
insp_tab1, insp_tab2, insp_tab3 = st.tabs([
    "🔍 탭1. 실시간 검사",
    "📋 탭2. 검사 이력 및 통계",
    "🔄 탭3. 딥러닝 모델 교체",
])
```

| 탭 | 핵심 컴포넌트 | Guard |
|----|--------------|-------|
| 탭1 | 검사 버튼 2개 + 4열 결과 + 팝업 | `insp_active_model is None` |
| 탭2 | 이력 테이블 + KPI 카드 4개 + 실시간 차트 | `insp_active_model is None` |
| 탭3 | 완료 실험 목록 + 적용 버튼 | 없음 |

---

### v1.x D. 검사 탭1 — 실시간 검사 와이어프레임 (Streamlit)

```
┌──────────────────────────────────────────────────────────────────────┐
│  [🔍 수동 검사 (1개 검사)]  [▶ 자동 검사 (3초마다 1개)]  [⏹ 자동 검사 중지]│
├──────────────┬───────────────────┬───────────────────┬───────────────┤
│  판정 결과    │  원본 이미지       │  Anomaly Map       │  이상 영역 오버레이│
│  ✅ 양품     │  [이미지 표시]     │  [히트맵 표시]     │  [오버레이 이미지]│
│  Score: 0.23 │  filename.png     │  jet colormap      │               │
└──────────────┴───────────────────┴───────────────────┴───────────────┘
비율: [1, 2, 2, 2]
```

**버튼 조건:**
- 자동 검사 중 `disabled=True`: 수동 검사 버튼
- `insp_auto_active == True`: "⏹ 자동 검사 중지"만 활성

**4열 패널:**
- col1: 판정결과 (`st.success`/`st.error`/`st.info`) + `st.metric("Anomaly Score")`
- col2: `st.image(image_path)` — 원본
- col3: `st.image(heatmap_rgb)` — jet colormap RGB
- col4: `st.image(overlay_image)` — 빨간 반투명 오버레이 (alpha=0.45)

**불량 팝업 (v1.x):**
```
┌────────────────────────────────────────────┐
│ ❌ 불량이 감지되었습니다! 자동 검사가 중지됨.  │ ← st.error()
│  [✅ 확인 및 재개]      [🛑 검사 종료]      │
└────────────────────────────────────────────┘
```
- `insp_defect_popup == True` → 팝업만 렌더 + `return`으로 나머지 UI 숨김 (v2.0은 모달 오버레이)

**자동 검사 배너 (v1.x):**
```python
st.warning("🔄 자동 검사 진행 중...")
```

---

### v1.x E. 검사 탭2 — 검사 이력 및 통계 (Streamlit)

```
┌──────────────────────────────────────────────────────────────┐
│  검사 이력                              [📥 CSV 내보내기]     │
├────┬──────────────────┬──────────────┬────────────┬──────────┤
│ 번호│       시각        │   이미지명    │  판정결과  │  Score   │
│  5 │ 2026-05-26 14:32 │ crack_01.png │ 🔴 불량    │  0.8721  │
├──────────────────────────────────────────────────────────────┤
│  총 검사  │  양품  │  불량  │  불량률  │  ← st.metric()×4    │
└──────────────────────────────────────────────────────────────┘
```

**실시간 통계 차트 3분할 (v1.2):**
```
[단위 선택] [20개]  [40개]  [100개]
┌─────────────────────┬─────────────────────┬──────────────────────┐
│  시간 범위 테이블     │  Anomaly Score 히스토그램│  Anomaly Score 산점도  │
│  2026-06-24         │                     │                      │
│  14:00~14:01 ← 선택 │  y ▌ ▌              │  1.0 · · ·           │
│  14:01~14:02        │  ▌ ▌ ▌ ▌            │  - - - threshold - - │
│  14:02~14:03 (진행) │  0───── x           │  0.0     1  5  10 x  │
└─────────────────────┴─────────────────────┴──────────────────────┘
```

**차트 규칙:**

| 항목 | 히스토그램 | 산점도 |
|------|-----------|-------|
| x축 | 0~1 (Score) | 1~N (인덱스) |
| y축 | 동적 | 0~1 고정 |
| threshold | 수직 점선 | 수평 빨간 점선 |

- KPI 카드 검사 없음 시도 0 표시 (Guard 미적용)
- CSV: `len(insp_records)==0` 시 `disabled=True`

---

### v1.x F. 검사 탭3 — 딥러닝 모델 교체 와이어프레임 (Streamlit)

```
┌──────────────────────────────────────────────────────────────┐
│  딥러닝 모델 교체                                              │
│  ⚠️ 모델을 교체하면 현재 세션의 모든 검사 이력이 삭제됩니다.   │
├───────────────┬────────────┬──────┬──────┬──────────────────┤
│    실험명      │  모델 타입  │  F1  │  AUC │     실행 시각    │
│ exp_A ✅현재   │ PatchCore  │ 0.95 │ 0.98 │ 2026-05-25 10:00 │
└───────────────┴────────────┴──────┴──────┴──────────────────┘
│              [✅ 이 모델로 검사 시작]                         │
└──────────────────────────────────────────────────────────────┘
```

- 선택: `selection_mode="single-row"`, `on_select="rerun"`
- 버튼: `type="primary"`, 미선택 시 `disabled=True`
- 경고 메시지: 항상 표시 (이력 없어도)

---

### v1.x G. 공통 UI 규칙 (Streamlit)

| Guard 유형 | Streamlit 컴포넌트 | 예시 |
|------------|-------------------|------|
| 모델 미선택 | `st.info(...)` | "검사에 사용할 모델이 선택되지 않았습니다." |
| 이력 없음 | `st.info(...)` | "아직 검사 기록이 없습니다." |
| 완료 실험 없음 | `st.info(...)` | "사용 가능한 완료된 실험이 없습니다." |
| 불량 감지 팝업 | `st.error(...)` | "불량이 감지되었습니다!" |

**색상 규칙:**

| 의미 | Streamlit 표현 | 색상 |
|------|---------------|------|
| 양품 | `st.success()` | 초록 |
| 불량 | `st.error()` | 빨간 |
| 정보/안내 | `st.info()` | 파란 |
| 경고 | `st.warning()` | 노란 |

**초기 상태 처리:**

| 상황 | 표시 |
|------|------|
| 앱 첫 진입, 모델 미선택 | 탭1·탭2: Guard 메시지. 탭3: 실험 목록 표시 |
| 모델 선택 직후, 검사 전 | 탭1: 버튼 활성화 + "검사 버튼을 눌러 시작하세요." 안내 |
| 첫 검사 완료 | 탭1: 4열 결과 표시. 탭2: 이력 1개 + KPI 갱신 |

---

*이 문서는 실제 소스 코드 역설계를 기반으로 작성되었습니다. [확인 필요: ...] 표기 항목은 추가 코드 검토가 필요합니다.*

# 04. System Architecture

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [03_Functional_Requirements.md](./03_Functional_Requirements.md)
> **버전**: v2.0
> **작성일**: 2026-05-08
> **최종수정**: 2026-06-11
> **후속 문서**: [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)
> **중요**: 이 문서는 05~14번 PRD 파일 전체가 참조하는 아키텍처 기준이다. 컴포넌트명·파일 경로·인터페이스 시그니처는 이 문서에서 확정되며, 이후 파일에서 변경 불가.
> **v2.0 주의**: ADR-01 폐기 및 FastAPI 공식화. 3개 레포(Explorer/Vision/Dashboard) 구조로 전환. 공식 UI = React. Streamlit = 개발자 보조 도구.

---

<!-- v2.0: 버전 히스토리 추가 -->
## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-08 | 최초 작성 — Streamlit 단독 앱, 모델 탐색 대시보드 전용 |
| v1.1 | 2026-05-26 | 이중 대시보드 구조. B.10 비전검사 대시보드 아키텍처 추가 |
| v2.0 | 2026-06-11 | ADR-01 폐기 및 FastAPI 공식화; B.1 3개 레포 다이어그램으로 전면 교체; B.2 Presentation 레이어 React/Streamlit 분리; B.5 WebSocket(/ws/training) 흐름 추가; B.8.1 React→FastAPI→ML→WS 시퀀스로 교체; B.10 Vision React + WebSocket(/ws/inspection/auto) 기준으로 교체; Streamlit 다이어그램 v1.x 참고 섹션으로 이동 |

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Detailed Specification](#b-detailed-specification)
  - [B.1 전체 시스템 아키텍처 (v2.0)](#b1-전체-시스템-아키텍처-v20)
  - [B.2 레이어 구조](#b2-레이어-구조)
  - [B.3 모듈 책임 명세](#b3-모듈-책임-명세)
  - [B.4 모듈 간 의존성 그래프](#b4-모듈-간-의존성-그래프)
  - [B.5 학습 비동기 처리 아키텍처 (v2.0 — WebSocket)](#b5-학습-비동기-처리-아키텍처-v20--websocket)
  - [B.6 Streamlit 생명주기와 상태 관리 (v1.x 참고)](#b6-streamlit-생명주기와-상태-관리-v1x-참고)
  - [B.7 훈련 상태 머신](#b7-훈련-상태-머신)
  - [B.8 주요 시퀀스 다이어그램](#b8-주요-시퀀스-다이어그램)
  - [B.9 에러 전파 경로](#b9-에러-전파-경로)
  - [B.10 비전검사 아키텍처 (v2.0 — Vision React + WebSocket)](#b10-비전검사-아키텍처-v20--vision-react--websocket)
- [C. System & Data Design](#c-system--data-design)
- [D. API Contracts](#d-api-contracts)
- [E. AI/ML Details](#e-aiml-details)
- [F. Non-Functional Requirements](#f-non-functional-requirements)
- [G. Observability](#g-observability)
- [H. QA & Validation](#h-qa--validation)
- [I. Implementation Plan](#i-implementation-plan)

---

## A. Objective & Scope

### A.1 이 문서의 목적

이 시스템 전체의 컴포넌트 구성, 레이어 책임, 모듈 간 인터페이스, 데이터 흐름, 동시성 모델을 확정한다. 이후 05~14번 문서는 이 문서의 결정을 참조하고, 이 문서와 충돌하는 설계를 포함할 수 없다.

<!-- v2.0: ADR-01 폐기 및 신규 ADR-01(FastAPI 도입)로 교체. ADR-04 적용 범위 Streamlit 내부로 축소. -->
### A.2 핵심 아키텍처 결정 사항 (ADR)

| ADR ID | 결정 | 근거 | 상태 |
|--------|------|------|------|
| ~~ADR-01~~ | ~~REST API 서버 없음. Streamlit 단일 프로세스~~ | ~~단일 사용자(가정 A-01), 네트워크 레이턴시 제거~~ | **폐기 (v2.0)** |
| **ADR-01 (v2.0)** | **FastAPI 서버 도입. Explorer/Vision React ↔ HTTP/WebSocket ↔ FastAPI :8000** | React 기반 프론트엔드 분리를 위해 REST API 및 WebSocket 인터페이스 필요 | **확정** |
| ADR-02 | 학습 루프는 백그라운드 스레드 + `queue.Queue` | Streamlit/FastAPI 메인 스레드 블로킹 방지, UI 갱신 유지 | **유효** |
| ADR-03 | 영속 저장은 파일시스템 전용 (JSON + YAML + .pth) | RDBMS 불필요, 단일 사용자, Docker 볼륨 마운트로 영속성 | **유효** |
| ADR-04 | session_state는 메인 스레드에서만 Write | **v2.0: Streamlit 보조 도구 내부에만 적용.** FastAPI는 thread-safe하게 설계 | **범위 축소** |
| ADR-05 | 전처리 파이프라인은 `utils/image_utils.py` 단일 구현 | Config 미리보기와 학습 루프가 동일 코드 사용, 불일치 방지 | **유효** |
| ADR-06 | Anomalib v1.0+ Engine API 사용 | `utils/model_factory.py` 래퍼로 추상화, 버전 업 시 래퍼만 수정 | **유효** |
| ADR-07 (v2.0) | Explorer 공식 UI = React (smart-qc-explorer). Streamlit = 개발자 디버그 전용 | 프론트엔드 분리 후 React가 더 나은 UX 제공 | **확정** |
| ADR-08 (v2.0) | Vision 공식 UI = React (smart-qc-vision). WebSocket으로 자동 검사 Push | 실시간 검사 결과 Push가 필요. Streamlit time.sleep 방식 대체 | **확정** |

---

## B. Detailed Specification

---

<!-- v2.0: [B.1] Streamlit 단독 다이어그램 → 3개 레포(Explorer/Vision/Dashboard) 구조로 전면 교체. 기존 다이어그램은 B.1.x로 이동 -->
### B.1 전체 시스템 아키텍처 (v2.0)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              사용자 브라우저                                   │
│                                                                               │
│  ┌───────────────────────────────┐   ┌────────────────────────────────────┐  │
│  │    smart-qc-explorer          │   │    smart-qc-vision                 │  │
│  │    React 19 + Vite + TS       │   │    React + Vite + TS               │  │
│  │    http://localhost:5173       │   │    http://localhost:5173           │  │
│  │                               │   │                                    │  │
│  │  ┌─────────────────────────┐  │   │  ┌───────────────────────────┐    │  │
│  │  │ Zustand Stores (v5)     │  │   │  │ Zustand Store             │    │  │
│  │  │ datasetStore            │  │   │  │ inspectionStore           │    │  │
│  │  │ configStore             │  │   │  └───────────────────────────┘    │  │
│  │  │ trainingStore           │  │   │                                    │  │
│  │  │ experimentsStore        │  │   │  Pages:                            │  │
│  │  │ anomalyMapStore         │  │   │  Tab1Realtime  (/             )   │  │
│  │  └─────────────────────────┘  │   │  Tab2History   (/history)         │  │
│  │                               │   │  Tab3Setting   (/models)          │  │
│  │  Pages:                       │   │                                    │  │
│  │  Tab1Dataset  (/)             │   │  hooks/                            │  │
│  │  Tab2Config   (/config)       │   │  useAutoInspection (WS)           │  │
│  │  Tab3Training (/training)     │   │  useManualInspection (POST)       │  │
│  │  Tab4Experiments(/experiments)│   │  useModels (30s polling)          │  │
│  │  Tab5AnomalyMap(/anomaly-map) │   │                                    │  │
│  │                               │   │  api/                              │  │
│  │  hooks/                       │   │  modelsApi  inspectionApi         │  │
│  │  useTrainingWs (WS)           │   │  recordsApi                       │  │
│  │                               │   │                                    │  │
│  │  api/                         │   │                                    │  │
│  │  datasetApi  configApi        │   │                                    │  │
│  │  trainingApi experimentsApi   │   │                                    │  │
│  │  anomalyMapApi                │   │                                    │  │
│  └──────────────┬────────────────┘   └──────────────┬─────────────────────┘  │
└─────────────────┼─────────────────────────────────── ┼─────────────────────────┘
                  │ HTTP REST / WebSocket               │ HTTP REST / WebSocket
                  │ baseURL: http://localhost:8000      │ baseURL: http://localhost:8000
┌─────────────────▼─────────────────────────────────────▼────────────────────┐
│                     smart-qc-dashboard (FastAPI :8000)                       │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  api/ — FastAPI 라우터 레이어                                         │    │
│  │                                                                       │    │
│  │  /api/dataset/validate      /api/config           /api/queue        │    │
│  │  /api/training/start        /api/training/pause   /api/training/stop│    │
│  │  /api/training/resume       /api/training/checkpoints               │    │
│  │  /api/training/batch/*      /api/experiments      /api/anomaly-map/ │    │
│  │  /api/models                /api/inspection/model                   │    │
│  │  /api/inspection/run        /api/inspection/records                 │    │
│  │  /api/inspection/image/*    /api/inspection/anomaly-map/*           │    │
│  │                                                                       │    │
│  │  WebSocket 엔드포인트:                                                │    │
│  │  /ws/training          — 학습 진행상황 실시간 Push → Explorer        │    │
│  │  /ws/inspection/auto   — 자동 검사 결과 실시간 Push → Vision         │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                  │                                            │
│  ┌───────────────────────────────▼────────────────────────────────────┐     │
│  │  학습 비동기 레이어 (ADR-02 유지)                                     │     │
│  │  TrainingWorker Thread ←→ queue.Queue ←→ WebSocket Manager         │     │
│  │  threading.Event (stop_event, pause_event)                          │     │
│  │  utils/checkpoint_manager.py  (일시정지 시 .ckpt 저장)               │     │
│  └───────────────────────────────┬────────────────────────────────────┘     │
│                                  │                                            │
│  ┌───────────────────────────────▼────────────────────────────────────┐     │
│  │  ML 레이어 (Anomalib / PyTorch)                                      │     │
│  │  EfficientAD Engine  │  PatchCore Engine  │  Evaluator              │     │
│  │  run_inference() ← Explorer AnomalyMap / Vision 수동·자동 검사      │     │
│  └───────────────────────────────┬────────────────────────────────────┘     │
│                                  │                                            │
│  ┌───────────────────────────────▼────────────────────────────────────┐     │
│  │  [보조] Streamlit UI (개발자 디버그용 — 공식 UI 아님)                  │     │
│  │  streamlit run app.py  →  http://localhost:8501                      │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │ 파일시스템 읽기/쓰기
┌───────────────────────────────────────▼───────────────────────────────────────┐
│                            파일시스템 레이어                                    │
│                                                                                │
│  /app/dataset/           읽기 전용 (볼륨 마운트)                                │
│  /app/experiments/       history.json (Explorer 쓰기 / Vision 읽기 전용)      │
│  /app/models/{exp_id}/   model_state_dict.pth, configs.yaml                  │
│  /app/logs/{exp_id}.log  학습 로그                                              │
│  /app/configs.yaml       공유 설정 (Explorer Config 화면 R/W)                 │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  데이터베이스 레이어 (인프라 포함, 현재 미사용)                                    │
│  MySQL 8.0  (컨테이너: smart-qc-db)  호스트 포트: 3307 → 컨테이너: 3306        │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

### B.1.x v1.x 참고 — Streamlit 단독 아키텍처 다이어그램

> **v1.x 참고 전용**: 아래는 v1.x Streamlit 단독 구조다. v2.0 공식 구조는 B.1절을 참조한다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        사용자 브라우저                                  │
│                   http://localhost:8501                               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ WebSocket (Streamlit 내장)
┌────────────────────────────▼─────────────────────────────────────────┐
│                       Streamlit 프로세스                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  app.py — init_session_state() / render_sidebar() / st.tabs │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────────────┐  │
│  │  tab1    │  tab2    │  tab3    │  tab4    │  tab5             │  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────────────┘  │
│       └──────────┴──────────┴──────────┴──────────┘                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   utils/ 레이어                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  학습 비동기 레이어 (ADR-02)                                   │    │
│  │  TrainingWorker Thread ←→ queue.Queue ←→ 메인 스레드 UI       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

<!-- v2.0: [B.2] Presentation 레이어를 React(공식 UI)와 Streamlit(보조 도구)으로 분리. FastAPI Service Layer 추가 -->
### B.2 레이어 구조

| 레이어 | 위치 | 역할 | 외부 의존성 | v2.0 상태 |
|--------|------|------|------------|-----------|
| **Presentation Layer — React (공식)** | `smart-qc-explorer/src/`, `smart-qc-vision/src/` | UI 렌더링, 사용자 입력 처리, Zustand store Read/Write, API 호출 | React 19, Vite, Zustand v5, Axios, react-router-dom | **공식 UI** |
| **Presentation Layer — Streamlit (보조)** | `app.py`, `tabs/`, `components/`, `inspection/` | 개발자 디버그용 UI. session_state Read/Write | Streamlit ≥ 1.30 | 보조 도구 (비공식 UI) |
| **API Layer** | `api/` (FastAPI) | REST API 엔드포인트, WebSocket 관리, 요청/응답 직렬화 | FastAPI, uvicorn, WebSocket | **v2.0 신규** |
| **Service Layer** | `utils/` | 비즈니스 로직, 파일 I/O, 이미지 처리, 메트릭 계산 | PyYAML, PIL, OpenCV, scikit-learn, torch | **유효** |
| **ML Layer** | `utils/model_factory.py` → Anomalib | 모델 초기화, 학습 루프, 추론 | Anomalib ≥ 1.0, PyTorch ≥ 2.1 | **유효** |
| **Storage Layer** | 파일시스템 (`./experiments/`, `./models/`, `./logs/`) + MySQL 8.0 | 영속 데이터 저장 | MySQL 8.0 (컨테이너, 현재 미사용) | **유효** |

**레이어 간 의존성 규칙 (v2.0)**:
- React Presentation → API Layer: HTTP REST / WebSocket (허용)
- React Presentation → Service/ML/Storage: **금지** (반드시 API Layer 경유)
- API Layer → Service: 허용
- API Layer → ML: 허용
- Service → ML: 허용
- Service → Storage: 허용
- ML → Storage: 학습 완료 시 Service가 저장 (직접 쓰기 금지)
- Streamlit Presentation → Service: 허용 (보조 도구 내부 한정)
- Streamlit Presentation → ML: **금지** (반드시 Service 경유)

---

### B.3 모듈 책임 명세

<!-- v2.0: FastAPI API 레이어 모듈 추가. 기존 Streamlit 탭 모듈은 v1.x 보조 도구로 역할 변경 표기 -->

#### B.3.1 FastAPI 진입점 (v2.0)

##### `api/main.py`

```python
책임:
  - FastAPI 앱 인스턴스 생성
  - CORS 미들웨어 설정 (localhost:5173 허용)
  - 라우터 등록 (dataset, config, training, experiments, anomaly_map, inspection)
  - WebSocket 엔드포인트 등록 (/ws/training, /ws/inspection/auto)
  - uvicorn 실행 (포트 8000)

실행 명령: uvicorn api.main:app --reload --port 8000

금지:
  - 비즈니스 로직 직접 구현
  - ML 코드 직접 호출
  - 파일 I/O 직접 접근
```

##### `api/` 라우터 모듈 구조

| 파일 | 엔드포인트 | 주요 책임 |
|------|-----------|-----------|
| `api/routers/dataset.py` | `POST /api/dataset/validate`, `GET /api/dataset/thumbnail/{class}` | 데이터셋 경로 검증, 메타 정보 반환, 썸네일 서빙 |
| `api/routers/config.py` | `GET/POST /api/config`, `POST /api/config/preview` | 설정 조회·저장, Threshold 미리보기 |
| `api/routers/queue.py` | `GET/POST /api/queue`, `DELETE /api/queue/{id}` | 배치 학습 큐 관리 |
| `api/routers/training.py` | `POST /api/training/{start,pause,unpause,stop,resume}`, `GET /api/training/checkpoints`, `POST /api/training/batch/*` | 학습 제어, 체크포인트 관리 |
| `api/routers/experiments.py` | `GET /api/experiments`, `POST /api/experiments/{id}/save`, `DELETE /api/experiments/{id}` | 실험 히스토리 조회·저장·삭제 |
| `api/routers/anomaly_map.py` | `GET/POST /api/anomaly-map/{expId}/*`, `GET /api/anomaly-map/job/{jobId}` | Anomaly Map 빌드 job, 이미지 서빙, CSV/ZIP 내보내기 |
| `api/routers/inspection.py` | `GET/POST /api/inspection/model`, `POST /api/inspection/run`, `GET/DELETE /api/inspection/records`, `GET /api/inspection/*/last` | 비전검사 모델 적용, 수동 검사, 이력 관리, 이미지 서빙 |
| `api/ws/training_ws.py` | `WS /ws/training` | 학습 진행상황 실시간 Push (TrainingWorker → queue → WebSocket → Explorer) |
| `api/ws/inspection_ws.py` | `WS /ws/inspection/auto` | 자동 검사 결과 실시간 Push → Vision |

---

#### B.3.2 Streamlit 진입점 (v1.x 보조 도구)

##### `app.py` (Streamlit — 개발자 보조 도구)

```python
책임:
  - st.set_page_config() 호출 (1회)
  - init_session_state() 호출 (매 rerun, 멱등)
  - render_sidebar() 호출
  - active_dashboard에 따라 모델 탐색 또는 비전검사 대시보드 분기 렌더링

금지:
  - 비즈니스 로직 직접 구현
  - ML 코드 직접 호출
```

---

#### B.3.3 Streamlit 탭 모듈 — `tabs/` (v1.x 보조 도구)

각 탭 파일의 공통 구조:

```python
# tabs/tab{n}_*.py 공통 패턴 (Streamlit 보조 도구 내부)

def render():
    """탭 렌더링 진입 함수."""
    _guard()
    _render_ui()
    _handle_events()

def _guard() -> bool:
    """선행 session_state 키 확인."""

def _render_ui():
    """Streamlit 컴포넌트 렌더링."""

def _handle_events():
    """버튼 클릭 등 부작용 처리."""
```

| 파일 | 공개 함수 | 주요 책임 |
|------|----------|-----------|
| `tab1_data_folder.py` | `render()` | 경로 입력, MVTec AD 검증, dataset_meta 구성 |
| `tab2_config.py` | `render()` | 전처리·모델 파라미터 UI, 미리보기 |
| `tab3_training.py` | `render()` | 실험명 입력, 학습 시작/일시정지/재시작/중지, Progress Bar, Loss 곡선 |
| `tab4_history.py` | `render()` | 히스토리 테이블, 상세 결과 차트, 비교, 모델 저장 |
| `tab5_anomaly_map.py` | `render()` | 이미지 목록, Threshold 슬라이더, 3분할 시각화 |
| `inspection/inspection_app.py` | `render()` | 비전검사 대시보드 진입점 (Streamlit 보조 도구) |
| `inspection/tabs/insp_tab1_realtime.py` | `render()` | 실시간 검사 탭 (Streamlit 보조 도구) |
| `inspection/tabs/insp_tab2_history.py` | `render()` | 검사 이력 테이블 + KPI 카드 (Streamlit 보조 도구) |
| `inspection/tabs/insp_tab3_model.py` | `render()` | 딥러닝 모델 교체 (Streamlit 보조 도구) |

---

#### B.3.4 공유 유틸리티 모듈 (`utils/`)

##### `utils/image_utils.py`

```python
책임: 이미지 전처리 파이프라인 전체 구현 (ADR-05)
      Explorer Config 미리보기 / 학습 루프 / Vision 추론 모두 동일 코드 사용

공개 함수:
  def load_image(path: str) -> PIL.Image
  def apply_filter(image, method, params) -> PIL.Image
  def resize_with_padding(image, target_size) -> PIL.Image
  def normalize_to_tensor(image, mean, std) -> torch.Tensor
  def apply_preprocessing(image_path, config) -> tuple[PIL.Image, torch.Tensor]
  def tensor_to_display_image(tensor) -> PIL.Image
  def anomaly_map_to_heatmap(anomaly_map, colormap) -> PIL.Image
  def create_triplet_image(original, gt_mask, heatmap) -> PIL.Image
```

---

##### `utils/metrics.py`

```python
책임: 이상 탐지 평가 메트릭 계산

공개 함수:
  def compute_metrics(y_true, anomaly_scores, threshold) -> dict
      # accuracy, precision, recall, f1_score, f2_score, auc, confusion_matrix
  def compute_roc_curve(y_true, anomaly_scores) -> tuple[np.ndarray, np.ndarray, float]
  def compute_threshold_from_percentile(normal_scores, percentile) -> float
```

---

##### `utils/model_factory.py`

```python
책임: Anomalib 모델 초기화 래퍼 (ADR-06)

공개 함수:
  def create_trainer(model_config, preprocessing_config, dataset_path,
                     device, exp_id, stop_event, result_queue) -> TrainingWorker

  def load_model_for_inference(exp_id, model_path, model_config, device) -> object

  def run_inference(model, image_tensor) -> tuple[float, np.ndarray]
      # 반환: (anomaly_score, anomaly_map (H, W))
```

---

##### `utils/training_worker.py`

```python
책임: 백그라운드 학습 스레드 실행 로직 (ADR-02)

공개 클래스:
  class TrainingWorker(threading.Thread):
      """
      stop_event: threading.Event
      result_queue: queue.Queue

      Queue 메시지 형식 (v2.0: WebSocket Manager가 구독):
        {"type": "progress", "step": int, "total": int, "loss": float, "elapsed": float}
        {"type": "log", "message": str}
        {"type": "completed", "y_true": list, "anomaly_scores": list, ...}
        {"type": "paused", "ckpt_path": str}
        {"type": "error", "exception": Exception, "traceback": str}
        {"type": "stopped"}
      """
```

---

##### `utils/config_manager.py`

```python
책임: configs.yaml 파일 읽기/쓰기

공개 함수:
  def load_config(path: str = "./configs.yaml") -> dict
  def save_config_section(section, data, path) -> None  # R-ATOMIC-01 준수
  def get_preprocessing_config(path) -> dict | None
  def get_model_config(path) -> dict | None
```

---

##### `utils/storage.py`

```python
책임: history.json 읽기/쓰기

공개 함수:
  def load_history() -> list[dict]
      # 파일 없으면 [] 반환 (예외 금지)
  def save_history(records: list[dict]) -> None
      # R-ATOMIC-01: tmpfile → rename
  def append_experiment(record: dict) -> None
      # load → append → save
  def delete_experiment(exp_id: str) -> None
```

---

#### B.3.5 컴포넌트 모듈

##### `components/sidebar.py` (Streamlit 보조 도구)

```python
책임: 대시보드 전환 버튼 렌더링 (v1.1 이후)

공개 함수:
  def render_sidebar() -> None:
      """[모델 탐색 대시보드] / [비전검사 대시보드] 전환 버튼 2개 렌더링."""
```

---

### B.4 모듈 간 의존성 그래프

<!-- v2.0: API 레이어와 React 프론트엔드 의존성 추가 -->

#### B.4.1 v2.0 공식 의존성 (React ↔ FastAPI)

```
smart-qc-explorer (React)
  ├── api/client.ts (Axios, baseURL: localhost:8000)
  ├── api/datasetApi.ts        → POST /api/dataset/validate
  ├── api/configApi.ts         → GET/POST /api/config, /api/queue
  ├── api/trainingApi.ts       → POST /api/training/*, WS /ws/training
  │     └── hooks/useTrainingWs.ts (WebSocket 연결 + 메시지 디스패치)
  ├── api/experimentsApi.ts    → GET /api/experiments, POST/DELETE
  └── api/anomalyMapApi.ts     → GET/POST /api/anomaly-map/*

smart-qc-vision (React)
  ├── api/client.ts (Axios, baseURL: localhost:8000)
  ├── api/modelsApi.ts         → GET /api/models, POST /api/inspection/model
  ├── api/inspectionApi.ts     → POST /api/inspection/run, 이미지 URL
  │     └── hooks/useAutoInspection.ts (WS /ws/inspection/auto)
  └── api/recordsApi.ts        → GET/DELETE /api/inspection/records

smart-qc-dashboard (FastAPI api/)
  ├── api/routers/dataset.py   → utils/image_utils.py, utils/storage.py
  ├── api/routers/config.py    → utils/config_manager.py
  ├── api/routers/training.py  → utils/model_factory.py
  │                              utils/training_worker.py
  │                              api/ws/training_ws.py (queue → WebSocket)
  ├── api/routers/experiments.py → utils/storage.py, utils/metrics.py
  ├── api/routers/anomaly_map.py → utils/model_factory.py, utils/image_utils.py
  └── api/routers/inspection.py  → utils/model_factory.py, utils/image_utils.py
                                   api/ws/inspection_ws.py

금지된 의존성 (v2.0):
  React → utils/* 직접 (반드시 FastAPI API 경유)
  api/routers/* → Anomalib 직접 import (반드시 model_factory 경유)
  api/routers/* → 파일시스템 직접 쓰기 (반드시 storage.py / config_manager 경유)
```

#### B.4.2 v1.x 참고 — Streamlit 의존성 그래프 (보조 도구 내부)

```
app.py (Streamlit)
  ├── utils/session_state_init.py
  ├── components/sidebar.py
  ├── tabs/tab1_data_folder.py → utils/image_utils.py, utils/messages.py
  ├── tabs/tab2_config.py      → utils/image_utils.py, utils/config_manager.py
  ├── tabs/tab3_training.py    → utils/model_factory.py → utils/training_worker.py
  │                              utils/metrics.py, utils/config_manager.py
  ├── tabs/tab4_history.py     → utils/config_manager.py, utils/metrics.py
  └── tabs/tab5_anomaly_map.py → utils/model_factory.py, utils/image_utils.py
```

---

<!-- v2.0: [B.5] WebSocket(/ws/training) 기반 흐름 추가. 기존 Streamlit 흐름은 B.5.x로 이동 -->
### B.5 학습 비동기 처리 아키텍처 (v2.0 — WebSocket)

이 시스템의 가장 복잡한 동시성 문제: FastAPI 메인 이벤트 루프를 블로킹하지 않고 수십 분의 학습 루프를 실행하고, 진행상황을 Explorer React에 실시간 전달하는 방법.

#### B.5.1 스레드 모델 (v2.0 — WebSocket 방식)

```
Explorer (React)                FastAPI 메인 스레드              백그라운드 스레드 (TrainingWorker)
────────────────────────────────────────────────────────────────────────────────────────────────
[학습 시작 버튼 클릭]
POST /api/training/start
─────────────────────────►
                            stop_event = threading.Event()
                            result_queue = queue.Queue()
                            worker = TrainingWorker(
                                config=...,
                                stop_event=stop_event,
                                result_queue=result_queue
                            )
                            worker.daemon = True
                            worker.start()
                            return {"status": "started", "exp_id": ...}
◄─────────────────────────
WS /ws/training 연결
─────────────────────────►
                            WebSocket Manager 등록
                                                                  [worker.run() 시작]
                                                                    random seed 고정
                                                                    DataLoader 구성
                                                                    모델 초기화
                                                                    for step in loop:
                            [queue 소비 루프 — 비동기]               result_queue.put({
                            msg = await queue.get_async()              "type": "progress",
                            await ws.send_json(msg)                    "step": step, ...
                            ◄──────────────────────                })
[trainingStore 갱신]                                               result_queue.put({
progress, lossHistory 업데이트                                         "type": "log", ...
◄─────────────────────────                                        })

[⏸ 일시정지]
POST /api/training/pause                                           if pause_event.is_set():
─────────────────────────►                                           save_checkpoint(...)
                            pause_event.set()                        result_queue.put({
                                                                         "type": "paused",
                                                                         "ckpt_path": ...
                                                                     })
                                                                     while pause_event.is_set():
                                                                         time.sleep(0.5)

[▶ 재개]
POST /api/training/unpause
─────────────────────────►
                            pause_event.clear()

[⏹ 중단]
POST /api/training/stop
─────────────────────────►
                            stop_event.set()
                            pause_event.clear()
                                                                   result_queue.put({
                                                                       "type": "stopped"
                                                                   })

                                                                  [학습 완료]
                            [queue에서 completed 수신]              result_queue.put({
                            metrics 계산                                "type": "completed",
                            history.json append                        "y_true": ...,
                            model .pth 저장                            "anomaly_scores": ...
                            await ws.send_json({                   })
                                "type": "completed", ...
                            })
◄─────────────────────────
[trainingStore 최종 갱신]
status = 'idle'
WS 연결 해제
```

#### B.5.2 WebSocket 메시지 프로토콜 (/ws/training)

| 방향 | type | 필드 | 설명 |
|------|------|------|------|
| Server → Client | `progress` | `step`, `total`, `loss`, `elapsed`, `stage` | 학습 진행상황 |
| Server → Client | `log` | `message` | 학습 로그 한 줄 |
| Server → Client | `paused` | `ckpt_path` | 일시정지 완료, 체크포인트 경로 |
| Server → Client | `completed` | `exp_id`, `metrics`, `duration_seconds` | 학습 완료 |
| Server → Client | `stopped` | — | 사용자 중단 완료 |
| Server → Client | `error` | `message`, `traceback` | 예외 발생 |

#### B.5.3 스레드 안전성 규칙 (v2.0)

| 규칙 | 설명 |
|------|------|
| **R-THREAD-01** | `TrainingWorker`는 FastAPI 요청/WebSocket 객체에 직접 접근하지 않는다. `result_queue`로만 통신한다. |
| **R-THREAD-02** | FastAPI WebSocket Manager는 `result_queue`를 비동기로 소비하여 클라이언트에 Push한다. |
| **R-THREAD-03** | `worker.daemon = True`로 설정하여 FastAPI 프로세스 종료 시 워커 스레드도 함께 종료된다. |
| **R-THREAD-04** | 동시에 실행 중인 학습은 1개뿐이다. 학습 중 `POST /api/training/start` 재요청 시 409 Conflict 반환. |
| **R-THREAD-05** | 배치 학습(queue) 실행 중에는 단일 학습 엔드포인트 비활성화. |

---

### B.6 Streamlit 생명주기와 상태 관리 (v1.x 참고 — Streamlit 보조 도구 내부)

> **v1.x 참고 전용**: 아래는 v1.x Streamlit 기반 생명주기 설명이다. v2.0 공식 아키텍처에서는 해당 없음.

```
[Streamlit Rerun 사이클]
사용자 인터랙션 발생
  → 전체 app.py 재실행
  → init_session_state() (이미 있는 키는 무시)
  → render_sidebar()
  → st.tabs() 렌더링
  → 활성 탭의 render() 실행
  → UI 상태: st.session_state에서 읽기
  → 완료 후 화면 업데이트
```

#### B.6.1 멱등성 보장 규칙 (Streamlit 보조 도구 내부)

| 규칙 | 설명 |
|------|------|
| **R-IDEM-01** | 버튼 클릭 없이 rerun 시 부작용 없음 |
| **R-IDEM-02** | 디바이스 감지는 `device_info is None`인 경우에만 실행 |
| **R-IDEM-03** | 학습 워커 시작은 `st.button("학습 시작")` 블록 내부에서만 |
| **R-IDEM-04** | configs.yaml 쓰기는 저장 버튼 클릭 시에만 |

#### B.6.2 v1.x 학습 비동기 처리 — Streamlit 방식 (보조 도구 참고)

```
메인 스레드 (Streamlit)               백그라운드 스레드 (TrainingWorker)
─────────────────────────────────────────────────────────────────────
[학습 시작 버튼]
  worker.start()
  session_state.current_run_status = "running"
  st.rerun()
[매 rerun]
  queue 드레인 → Progress Bar / Loss 차트 갱신
  time.sleep(0.3) → st.rerun()
[⏸ 일시정지]
  pause_event.set()
[▶ 재시작]
  pause_event.clear()
  st.rerun()
[⏹ 중지]
  stop_event.set()
                                       time.sleep(3) 자동 검사 타이머
                                       [확인 필요: v2.0 WS /ws/inspection/auto로 대체됐는지]
```

---

### B.7 훈련 상태 머신

```
                    [POST /api/training/start]
         ┌──────────────────────────────────────────┐
         │                                          │
    ┌────▼────┐                               ┌─────▼──────┐
    │  idle   │◄──────────────────────────────│  running   │
    └─────────┘   [완료/중단/에러 처리 완료]   └──┬──────┬──┘
                                                  │      │
                              ┌───────────────────┤      │
                              │                   │ [POST /api/training/pause]
                    [POST /api/training/stop]      │
                              │                   ▼
                         ┌────▼──────┐   ┌────────────────┐
                         │  stopped  │   │    paused      │
                         └────┬──────┘   └────────┬───────┘
                              │            [POST /api/training/unpause]
                              │                   │ → running
                [handle_stopped() 실행]            │
                [history.json append]   ┌─────────▼──────┐
                [status="중단"]         │   completed    │
                              │         └────────┬───────┘
                              │        [metrics 계산, 파일 저장]
                              │        [history.json append]
                              └──────┬─────────────┘
                                     │ WS 전송 후 idle 전환
                                ┌────▼────┐
                                │  idle   │
                                └─────────┘

[에러 발생]
  running → idle (WS error 메시지 전송, history 미기록)
```

---

<!-- v2.0: [B.8.1] 학습 시작 시퀀스를 Streamlit 기반 → React→FastAPI→ML→WebSocket 기준으로 교체. 기존 시퀀스는 B.8.1.x로 이동 -->
### B.8 주요 시퀀스 다이어그램

#### B.8.1 학습 시작 ~ 완료 시퀀스 (v2.0 — React → FastAPI → ML → WebSocket)

```
Explorer(React)    FastAPI(:8000)    TrainingWorker       파일시스템
    │                   │                  │                 │
    │ POST /api/        │                  │                 │
    │ training/start    │                  │                 │
    ├──────────────────►│                  │                 │
    │                   │ worker = create()│                 │
    │                   ├─────────────────►│                 │
    │                   │ worker.start()   │                 │
    │                   ├─────────────────►│                 │
    │                   │                  │ 학습 루프 시작   │
    │ 200 {exp_id}       │                  │                 │
    │◄──────────────────┤                  │                 │
    │                   │                  │                 │
    │ WS /ws/training   │                  │                 │
    ├──────────────────►│                  │                 │
    │ (연결 유지)        │                  │                 │
    │                   │   queue.put(     │                 │
    │                   │   progress)      │                 │
    │                   │◄─────────────────┤                 │
    │ WS:{type:progress,│                  │                 │
    │  step,loss,...}   │                  │                 │
    │◄──────────────────┤                  │                 │
    │ trainingStore 갱신 │                  │                 │
    │ (반복)             │                  │                 │
    │                   │   queue.put(     │                 │
    │                   │   completed)     │                 │
    │                   │◄─────────────────┤                 │
    │                   │ metrics 계산      │                 │
    │                   │                  │  history.json   │
    │                   ├────────────────────────────────────►
    │                   │                  │  .pth 저장      │
    │                   ├────────────────────────────────────►
    │ WS:{type:         │                  │                 │
    │  completed,       │                  │                 │
    │  metrics,...}     │                  │                 │
    │◄──────────────────┤                  │                 │
    │ WS 연결 해제       │                  │                 │
    │ trainingStore     │                  │                 │
    │  status='idle'    │                  │                 │
```

---

#### B.8.1.x v1.x 참고 — 학습 시퀀스 (Streamlit 보조 도구)

> **v1.x 참고 전용**

```
사용자          app.py          tab3_training.py   TrainingWorker     파일시스템
  │                │                 │                    │               │
  │ [학습 시작]    │                 │                    │               │
  ├──────────────►│                 │                    │               │
  │               │  탭3.render()   │                    │               │
  │               ├────────────────►│                    │               │
  │               │                │  worker.start()     │               │
  │               │                ├───────────────────►│               │
  │               │                │                    │ 학습 루프      │
  │               │                │  status="running"  │               │
  │               │                │  st.rerun()        │               │
  │  화면 갱신     │◄───────────────┤                    │               │
  │◄──────────────┤                │                    │               │
  │               │                │  queue.put(progress│               │
  │               │                │◄───────────────────┤               │
  │               │  (0.3s 마다)   │ queue 드레인 → 갱신 │               │
  │               │                │  st.rerun()        │               │
  │               │◄───────────────┤                    │               │
  │◄──────────────┤  (반복)         │                    │               │
  │               │                │  queue.put(completed)              │
  │               │                │◄───────────────────┤               │
  │               │                │ metrics 계산        │  history.json │
  │               │                ├─────────────────────────────────────►
  │  완료 알림     │◄───────────────┤                    │               │
  │◄──────────────┤                │                    │               │
```

---

#### B.8.2 Explorer AnomalyMap 시퀀스 (v2.0)

```
Explorer(React)    FastAPI(:8000)    ML Layer          파일시스템
    │                   │                │                 │
    │ GET /api/anomaly- │                │                 │
    │  map/{expId}/status               │                 │
    ├──────────────────►│                │                 │
    │ {status:"none"}   │                │                 │
    │◄──────────────────┤                │                 │
    │ POST /api/anomaly-│                │                 │
    │  map/{expId}/build│                │                 │
    ├──────────────────►│                │                 │
    │ {job_id: "..."}   │                │                 │
    │◄──────────────────┤                │                 │
    │ [1초마다 폴링]     │                │                 │
    │ GET /api/anomaly- │                │                 │
    │  map/job/{jobId}  │                │                 │
    ├──────────────────►│                │                 │
    │                   │ load_model()   │                 │
    │                   ├───────────────►│                 │
    │                   │               │  .pth 로드       │
    │                   │               ├─────────────────►│
    │                   │               │◄─────────────────┤
    │                   │ run_inference()│                 │
    │                   ├───────────────►│                 │
    │                   │◄──────────────┤                 │
    │                   │  캐시 저장     │                 │
    │ {status:"complete"}               │                 │
    │◄──────────────────┤                │                 │
    │ GET /api/anomaly- │                │                 │
    │  map/{expId}/images               │                 │
    ├──────────────────►│                │                 │
    │ 이미지 목록 + 통계 │                │                 │
    │◄──────────────────┤                │                 │
```

---

### B.9 에러 전파 경로

<!-- v2.0: 에러 전파를 FastAPI HTTP 응답 기준으로 업데이트 -->

| 발생 위치 | 예외 타입 | 전파 경로 | 최종 처리 |
|-----------|-----------|-----------|-----------|
| `POST /api/dataset/validate` — 경로 없음 | `FileNotFoundError` | FastAPI 400 | Explorer Dataset 화면 오류 표시 |
| `POST /api/config` — 잘못된 파라미터 | `ValidationError` | FastAPI 422 | Explorer Config 화면 오류 표시 |
| `POST /api/training/start` — 학습 중 | 학습 충돌 | FastAPI 409 | Explorer Training 화면 "이미 실행 중" |
| `TrainingWorker.run()` — GPU OOM | `torch.cuda.OutOfMemoryError` | queue → WS `{type:"error"}` | Explorer WS 메시지 수신 → 오류 표시 |
| `TrainingWorker.run()` — 일반 예외 | `Exception` | queue → WS `{type:"error"}` | Explorer WS 메시지 수신 → 오류 표시 |
| `GET /api/anomaly-map/job/{id}` — 추론 실패 | 추론 예외 | FastAPI 500 | Explorer AnomalyMap 화면 오류 표시 |
| `POST /api/inspection/run` — 모델 없음 | 모델 미선택 | FastAPI 400 | Vision Realtime 화면 안내 |
| `POST /api/inspection/run` — 추론 실패 | `Exception` | FastAPI 500 | Vision 오류 토스트 |
| `WS /ws/inspection/auto` — 불량 감지 | 비즈니스 로직 | WS `{type:"defect_stopped"}` | Vision 자동 중지 + 팝업 |
| `storage.save_history()` — 디스크 풀 | `OSError` | FastAPI 500 | Explorer Training 화면 오류 |

---

<!-- v2.0: [B.10] Vision React + WebSocket(/ws/inspection/auto) 기준으로 전면 교체. 기존 Streamlit 내용은 B.10.x로 이동 -->
### B.10 비전검사 아키텍처 (v2.0 — Vision React + WebSocket)

#### B.10.1 비전검사 아키텍처 결정 (ADR — v2.0)

| ADR ID | 결정 | 근거 | 상태 |
|--------|------|------|------|
| ADR-INSP-01 | 검사 이력은 FastAPI 서버 메모리에 저장 (세션 한정, 파일 영속 없음) | 현장 운영 단순화. 서버 재시작 시 초기화 | 유효 (저장 위치 변경) |
| ADR-INSP-02 | history.json 읽기 전용 (Vision/검사 라우터) | 비전검사는 학습 실험 생성 안 함 | 유효 |
| ADR-INSP-03 | 자동 검사는 WS `/ws/inspection/auto` Push 방식 | v1.x `time.sleep(3)+st.rerun()` 대체. 서버 주도 Push로 타이밍 일관성 확보 | v2.0 변경 |
| ADR-INSP-04 | 모델 로드는 서버 인메모리 캐싱 | activeModel 변경 시에만 재로드. 동일 모델 연속 추론 시 재로드 방지 | 유효 |
| ADR-INSP-05 | test_pool은 모델 교체 시 재구성, 서버 메모리 관리 | dataset_path는 실험 레코드에 저장됨 | 유효 |

---

#### B.10.2 비전검사 전체 흐름 (v2.0)

```
Vision (React)         FastAPI(:8000)             ML Layer
    │                       │                          │
    │ GET /api/inspection/   │                          │
    │  model                │                          │
    ├──────────────────────►│                          │
    │ {activeModel or null}  │                          │
    │◄──────────────────────┤                          │
    │                       │                          │
    │ [수동 검사]             │                          │
    │ POST /api/inspection/  │                          │
    │  run                  │                          │
    ├──────────────────────►│                          │
    │                       │ apply_preprocessing()    │
    │                       │ run_inference(model)     │
    │                       ├─────────────────────────►│
    │                       │◄─────────────────────────┤
    │                       │ {score, verdict}         │
    │                       │ records.append(record)   │
    │ {seq, verdict,         │                          │
    │  anomaly_score}       │                          │
    │◄──────────────────────┤                          │
    │ GET /api/inspection/   │                          │
    │  image/last           │                          │
    ├──────────────────────►│                          │
    │ 원본 이미지 바이너리    │                          │
    │◄──────────────────────┤                          │
    │ GET /api/inspection/   │                          │
    │  anomaly-map/last     │                          │
    ├──────────────────────►│                          │
    │ Anomaly Map 이미지     │                          │
    │◄──────────────────────┤                          │
    │                       │                          │
    │ [자동 검사 시작]        │                          │
    │ WS /ws/inspection/auto│                          │
    ├──────────────────────►│                          │
    │ (연결 유지)             │                          │
    │                       │ loop: apply_preprocessing│
    │                       │       run_inference()    │
    │                       ├─────────────────────────►│
    │                       │◄─────────────────────────┤
    │                       │ if verdict=="불량":       │
    │                       │   type="defect_stopped"  │
    │ WS:{type:"result",    │ else:                    │
    │  seq, verdict, ...}   │   type="result"          │
    │◄──────────────────────┤                          │
    │ lastResult 갱신        │                          │
    │ 이미지 URL 갱신         │                          │
    │                       │                          │
    │ WS:{type:             │                          │
    │  "defect_stopped"}    │                          │
    │◄──────────────────────┤                          │
    │ 불량 팝업 표시          │                          │
    │ WS 연결 해제            │                          │
```

---

#### B.10.3 WebSocket 메시지 프로토콜 (/ws/inspection/auto)

| 방향 | type | 필드 | 설명 |
|------|------|------|------|
| Server → Client | `result` | `seq`, `inspected_at`, `image_name`, `image_path`, `verdict`, `anomaly_score`, `was_reshuffled` | 정상 추론 결과 |
| Server → Client | `defect_stopped` | — | 불량 감지, 자동 검사 중지 |
| Server → Client | `stopped` | — | 사용자 요청 또는 서버 측 중지 |
| Server → Client | `error` | `message` | 추론 오류 |

---

#### B.10.4 Vision 화면별 상태 흐름

```
[Realtime Inspection 화면 — Tab1Realtime.tsx]
  진입 시:
    useActiveModel → GET /api/inspection/model → inspectionStore.activeModel 갱신
    activeModel == null → NoModelGuard 렌더링 (검사 버튼 비활성)

  [수동 검사] → useManualInspection → POST /api/inspection/run
    → lastResult 갱신
    → useInspectionImages → 이미지 URL 갱신 (cache-bust 파라미터)
    → verdict == "불량" → 불량 팝업 표시

  [자동 검사] → useAutoInspection → WS /ws/inspection/auto
    type=="result" → lastResult 갱신, 이미지 갱신
    type=="defect_stopped" → autoActive=false, 팝업 표시
    type=="stopped" → autoActive=false

[History 화면 — Tab2History.tsx]
  useInspectionRecords → GET /api/inspection/records
  KPI 카드: 총/양품/불량 수, 불량률 (분모=0이면 "-")
  CSV 내보내기 → GET /api/inspection/records/csv
  이력 초기화 → DELETE /api/inspection/records

[Model Settings 화면 — Tab3Setting.tsx]
  useModels → GET /api/models (30초 폴링) → 완료 실험 목록
  현재 적용 모델 배지 표시
  [적용] → useApplyModel → POST /api/inspection/model
    → inspectionStore 갱신 + 이력 초기화 (서버에서 DELETE records 자동 실행)
```

---

#### B.10.5 Vision Zustand Store 구조

```typescript
// src/store/inspectionStore.ts
interface InspectionState {
  // 적용 모델
  activeModel: ActiveModel | null;

  // 검사 이력
  records: InspectionRecord[];

  // 마지막 추론 결과
  lastResult: InspectionResult | null;

  // 자동 검사 상태
  autoActive: boolean;

  // 이미지 URL 캐시 무효화 토큰
  imageCacheBust: number;
}
```

---

#### B.10.x v1.x 참고 — 비전검사 대시보드 아키텍처 (Streamlit 보조 도구)

> **v1.x 참고 전용**: 아래는 v1.x Streamlit 기반 비전검사 아키텍처다.

##### B.10.x.1 대시보드 라우팅 (Streamlit)

```python
# app.py (Streamlit 보조 도구)
if st.session_state.active_dashboard == "inspection":
    from inspection.inspection_app import render as render_inspection
    render_inspection()
else:
    tab1, tab2, ... = st.tabs([...])
    ...
```

##### B.10.x.2 v1.x 비전검사 데이터 흐름

```
test_pool 구성 (모델 교체 시)
  ↓
[수동 검사] 또는 [자동 검사] 버튼
  ↓
test_sampler.sample(pool, index) → (image_path, verdict_label)
  ↓
image_utils.apply_preprocessing(image_path, preprocessing_params)
  ↓
model_factory.run_inference(model, tensor) → {anomaly_score, anomaly_map}
  ↓
score > threshold → verdict = "불량" | "양품"
  ↓
insp_records.append(inspection_record) + insp_last_result 갱신
  ↓
불량이면: insp_defect_popup = True + insp_auto_active = False → st.rerun()
양품이면: [자동] time.sleep(3) → st.rerun() | [수동] 대기
          [확인 필요: time.sleep(3) → WS /ws/inspection/auto 대체 여부]
```

##### B.10.x.3 v1.x ADR-INSP-03: `time.sleep(3) + st.rerun()`

```python
# inspection/tabs/insp_tab1_realtime.py (Streamlit 보조 도구)
if insp_auto_active:
    # 수동 검사 흐름 실행
    # ...
    if verdict == "양품":
        time.sleep(3)
        st.rerun()
    else:
        insp_defect_popup = True
        insp_auto_active = False
        st.rerun()
```

##### B.10.x.4 v1.x sidebar.py (v1.1)

```python
# components/sidebar.py (Streamlit 보조 도구)
def render_sidebar():
    with st.sidebar:
        if st.button("🔬 모델 탐색 대시보드", use_container_width=True,
                     type="primary" if st.session_state.active_dashboard == "explorer" else "secondary"):
            st.session_state.active_dashboard = "explorer"
            st.rerun()
        if st.button("🏭 비전검사 대시보드", use_container_width=True,
                     type="primary" if st.session_state.active_dashboard == "inspection" else "secondary"):
            st.session_state.active_dashboard = "inspection"
            st.rerun()
```

---

## C. System & Data Design

### C.1 디렉토리 구조 (확정)

<!-- v2.0: 3개 레포 구조로 디렉토리 명세 업데이트 -->

#### smart-qc-dashboard (FastAPI 백엔드)

```
/app/                               # Docker WORKDIR (로컬: ./smart-qc-dashboard/)
├── api/                            # FastAPI 라우터 (v2.0 추가)
│   ├── main.py                     # FastAPI 진입점
│   ├── routers/
│   │   ├── dataset.py
│   │   ├── config.py
│   │   ├── queue.py
│   │   ├── training.py
│   │   ├── experiments.py
│   │   ├── anomaly_map.py
│   │   └── inspection.py
│   └── ws/
│       ├── training_ws.py          # /ws/training WebSocket
│       └── inspection_ws.py        # /ws/inspection/auto WebSocket
│
├── app.py                          # Streamlit 진입점 (개발자 보조 도구)
├── requirements.txt
├── Dockerfile
├── docker-compose.base.yml
├── docker-compose.yml
├── docker-compose.cpu.yml
├── .env
├── configs.yaml
│
├── tabs/                           # Streamlit 탭 (보조 도구)
│   ├── tab1_data_folder.py
│   ├── tab2_config.py
│   ├── tab3_training.py
│   ├── tab4_history.py
│   └── tab5_anomaly_map.py
│
├── inspection/                     # Streamlit 비전검사 (보조 도구)
│   ├── inspection_app.py
│   ├── tabs/
│   │   ├── insp_tab1_realtime.py
│   │   ├── insp_tab2_history.py
│   │   └── insp_tab3_model.py
│   └── utils/
│       ├── insp_session_init.py
│       └── test_sampler.py
│
├── utils/                          # 공유 유틸리티 (API + Streamlit 공용)
│   ├── session_state_init.py
│   ├── config_manager.py
│   ├── messages.py
│   ├── image_utils.py
│   ├── metrics.py
│   ├── model_factory.py
│   ├── training_worker.py
│   ├── storage.py
│   ├── cache_manager.py
│   ├── checkpoint_manager.py
│   ├── dataset_converter.py
│   └── image_utils.py
│
├── components/
│   └── sidebar.py
│
├── experiments/
│   └── history.json
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

#### smart-qc-explorer (Explorer 프론트엔드)

```
smart-qc-explorer/
├── src/
│   ├── pages/           # Tab1Dataset / Tab2Config / Tab3Training / Tab4Experiments / Tab5AnomalyMap
│   ├── components/      # layout / tab1 / config / training / tab4 / tab5
│   ├── hooks/           # useTrainingWs
│   ├── api/             # client / datasetApi / configApi / trainingApi / experimentsApi / anomalyMapApi
│   ├── store/           # datasetStore / configStore / trainingStore / experimentsStore / anomalyMapStore
│   └── types/           # dataset / config / training / experiments / anomalyMap
└── package.json
```

#### smart-qc-vision (Vision 프론트엔드)

```
smart-qc-vision/
├── src/
│   ├── pages/           # Tab1Realtime / Tab2History / Tab3Setting
│   ├── components/      # layout / tab1 / tab2 / tab3
│   ├── hooks/           # useActiveModel / useModels / useApplyModel / useManualInspection / useAutoInspection / useInspectionImages / useInspectionRecords / useStatCharts
│   ├── api/             # client / modelsApi / inspectionApi / recordsApi
│   ├── store/           # inspectionStore
│   └── types/           # model / inspection / api
└── package.json
```

---

### C.2 Zustand Store 전체 키 목록 (v2.0 — Explorer)

| Store | 키 | 타입 | 초기값 | Write 화면 | Read 화면 |
|-------|----|----|--------|-----------|---------|
| `datasetStore` | `datasetPath` | `string\|null` | null | Dataset | Config, Training |
| `datasetStore` | `productName` | `string` | `''` | Dataset | Training |
| `datasetStore` | `datasetMeta` | `object\|null` | null | Dataset | Config |
| `configStore` | `preprocessingConfig` | `object\|null` | null | Config | Training |
| `configStore` | `modelConfig` | `object\|null` | null | Config | Training |
| `configStore` | `deviceInfo` | `object\|null` | null | Config (API) | Training |
| `configStore` | `queueItems` | `array` | `[]` | Config | Training |
| `trainingStore` | `status` | `string` | `'idle'` | Training (WS) | Training |
| `trainingStore` | `progress` | `object\|null` | null | Training (WS) | Training |
| `trainingStore` | `lossHistory` | `array` | `[]` | Training (WS) | Training |
| `trainingStore` | `logs` | `array` | `[]` | Training (WS) | Training |
| `experimentsStore` | `selectedExperimentId` | `string\|null` | null | Experiments | AnomalyMap |
| `anomalyMapStore` | `threshold` | `number\|null` | null | AnomalyMap | AnomalyMap |

### C.3 session_state 전체 키 목록 (v1.x — Streamlit 보조 도구)

> **v1.x 참고 전용**: 00_Global_Context_Document.md 3.1절 SESSION_STATE_SCHEMA 참조.

---

## D. API Contracts

<!-- v2.0: "N/A — REST API 없음"에서 FastAPI 엔드포인트 기준 명세로 전환. 상세 명세는 06_API_Specification.md 참조 -->

> **v2.0**: REST API 상세 명세는 [06_API_Specification.md](./06_API_Specification.md)에서 다룬다. 이 절에서는 아키텍처 수준의 API 계약만 확정한다.

### D.1 공통 설정

| 항목 | 값 |
|------|----|
| **baseURL** | `http://localhost:8000` |
| **Content-Type** | `application/json` |
| **timeout** | 30초 (Axios 기본) |
| **CORS 허용 origin** | `http://localhost:5173` (Explorer / Vision 공통) |

### D.2 엔드포인트 그룹 요약

| 그룹 | 경로 접두사 | 사용 클라이언트 |
|------|------------|----------------|
| 데이터셋 | `/api/dataset/` | Explorer |
| 설정·큐 | `/api/config`, `/api/queue` | Explorer |
| 학습 제어 | `/api/training/` | Explorer |
| 실험 히스토리 | `/api/experiments` | Explorer |
| Anomaly Map | `/api/anomaly-map/` | Explorer |
| 모델 목록 | `/api/models` | Vision |
| 비전검사 | `/api/inspection/` | Vision |
| WebSocket — 학습 | `/ws/training` | Explorer |
| WebSocket — 자동검사 | `/ws/inspection/auto` | Vision |

### D.3 공통 에러 코드

| HTTP 코드 | 의미 | 예시 조건 |
|-----------|------|-----------|
| 400 | Bad Request — 요청 파라미터 오류 | 잘못된 경로, 필수 필드 누락 |
| 404 | Not Found — 리소스 없음 | 존재하지 않는 exp_id |
| 409 | Conflict — 상태 충돌 | 학습 중 재시작 요청 |
| 422 | Unprocessable Entity — 유효성 검사 실패 | Pydantic 스키마 불일치 |
| 500 | Internal Server Error — 서버 예외 | ML 추론 실패, 파일시스템 오류 |

---

## E. AI/ML Details

```
N/A — 이 문서는 아키텍처 범위이다.
      Anomalib Engine 초기화·학습 루프·추론 상세 구현은
      08_AI_ML_Integration.md에서 다룬다.
      이 문서에서는:
      - ML 레이어가 utils/model_factory.py와 utils/training_worker.py를 통해
        Presentation/API 레이어로부터 격리된다는 아키텍처 결정만 확정한다.
      - v2.0: ML 레이어는 FastAPI API Layer에서만 호출된다.
        React 프론트엔드는 ML에 직접 접근하지 않는다.
```

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체 상속.

<!-- v2.0: React UI / WebSocket 관련 NFR 추가 -->

아키텍처 수준에서 추가로 명시:

| 항목 | 요구사항 | 아키텍처 결정 |
|------|----------|--------------|
| **UI 비블로킹** | 학습 중 화면 전환 가능 | ADR-02: 백그라운드 스레드 + WS Push |
| **FastAPI 응답성** | API 응답 < 200ms (ML 처리 제외) | 비즈니스 로직을 utils로 분리 |
| **WebSocket 학습 지연** | step 이벤트 → Explorer UI 갱신 < 1초 | queue 소비 루프를 비동기로 구현 |
| **파일 쓰기 안전성** | 부분 쓰기로 인한 데이터 손상 없음 | R-ATOMIC-01: tmpfile → rename |
| **메모리 누수** | 학습 완료 후 모델 객체 GC 보장 | `del model; torch.cuda.empty_cache()` |
| **CORS 보안** | Explorer/Vision 외 origin 차단 | FastAPI CORSMiddleware allowedOrigins 설정 |
| **포트** | FastAPI 8000/tcp, React 5173/tcp 고정 | Dockerfile EXPOSE / Vite 설정 |
| **자동 검사 타이밍** | 3초 간격 오차 0.5초 이내 | ADR-INSP-03: WS 서버 주도 타이밍 |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards) 전체 상속.

<!-- v2.0: FastAPI / WebSocket 관련 관측 포인트 추가 -->

아키텍처 수준에서 추가:

| 관측 포인트 | 위치 | 방법 |
|------------|------|------|
| FastAPI 시작 확인 | `api/main.py` startup | `logger.info("FastAPI started on :8000")` |
| 파일시스템 마운트 확인 | `api/main.py` startup | `Path("./experiments").mkdir(parents=True, exist_ok=True)` |
| 학습 스레드 생존 여부 | `GET /api/training/status` | `worker.is_alive()` 확인 |
| WebSocket 연결 수 | `api/ws/training_ws.py` | 연결 수 로깅 (training_ws manager) |
| 자동 검사 WS 연결 | `api/ws/inspection_ws.py` | 연결/해제 이벤트 로깅 |
| 메모리 사용 | FastAPI 미들웨어 | 모델 캐시 사이즈 주기적 확인 |

---

## H. QA & Validation

### H.1 아키텍처 검증 기준 (v2.0)

<!-- v2.0: FastAPI/React 기준으로 검증 기준 업데이트 -->

| # | 기준 | 검증 방법 |
|---|------|-----------|
| ARC-01 | `api/routers/*`에서 Anomalib 직접 import 없음 | `grep -r "from anomalib" api/routers/` → 결과 없음 |
| ARC-02 | `api/routers/*`에서 파일시스템 직접 쓰기 없음 | `grep -r "open(" api/routers/` → storage.py 경유만 허용 |
| ARC-03 | `utils/*`에서 FastAPI Request/Response 직접 사용 없음 | `grep -r "from fastapi" utils/` → 결과 없음 |
| ARC-04 | TrainingWorker에서 FastAPI 객체 접근 없음 | `grep -r "fastapi" utils/training_worker.py` → 결과 없음 |
| ARC-05 | result_queue 메시지 타입이 B.5.2절 명세 외 없음 | training_worker.py 코드 리뷰 |
| ARC-06 | 학습 중 POST /api/training/start 재요청 시 409 반환 | curl 테스트 |
| ARC-07 | WS /ws/training 메시지가 B.5.2절 프로토콜 준수 | Explorer useTrainingWs 단위 테스트 |
| ARC-08 | WS /ws/inspection/auto 메시지가 B.10.3절 프로토콜 준수 | Vision useAutoInspection 단위 테스트 |

### H.2 v1.x 아키텍처 검증 기준 (Streamlit 보조 도구)

| # | 기준 | 검증 방법 |
|---|------|-----------|
| ARC-v1-01 | `tabs/*`에서 Anomalib 직접 import 없음 | `grep -r "from anomalib" tabs/` |
| ARC-v1-02 | `utils/*`에서 `st.session_state` Write 없음 | `grep -r "session_state\." utils/` → 읽기만 |
| ARC-v1-03 | TrainingWorker에서 `st.session_state` 접근 없음 | `grep -r "session_state" utils/training_worker.py` |

### H.3 Given-When-Then 시나리오 (v2.0)

#### TC-ARC-01: 학습 중 화면 전환 비블로킹

```
Given:  Explorer Training 화면에서 학습이 진행 중이다
        WS /ws/training 연결이 활성화되어 있다
When:   사용자가 Experiments 탭을 클릭한다
Then:   Experiments 화면이 1초 이내에 렌더링된다
        TrainingWorker 스레드는 중단 없이 계속 실행된다
        Training 화면으로 돌아오면 최신 progress가 표시된다
```

#### TC-ARC-02: 자동 검사 불량 감지

```
Given:  Vision Realtime 화면에서 자동 검사가 실행 중이다
        WS /ws/inspection/auto 연결이 활성화되어 있다
When:   추론 결과 verdict == "불량"이 발생한다
Then:   서버가 WS {type: "defect_stopped"} 메시지를 전송한다
        Vision 클라이언트가 autoActive = false로 갱신한다
        불량 팝업이 0.5초 이내에 표시된다
        WS 연결이 해제된다
```

#### TC-ARC-03: 동시 학습 방지

```
Given:  학습이 진행 중이다 (TrainingWorker running)
When:   POST /api/training/start 를 다시 요청한다
Then:   HTTP 409 Conflict 응답 반환
        기존 학습은 중단 없이 계속 실행된다
```

#### TC-ARC-04: history.json 원자적 쓰기

```
Given:  학습이 완료되어 history.json에 실험 레코드를 append한다
When:   쓰기 도중 프로세스가 강제 종료된다
Then:   tmpfile(.tmp)이 남아있을 수 있으나 history.json은 이전 유효 상태를 유지
        다음 실행 시 tmpfile을 삭제하고 정상 동작
```

---

## I. Implementation Plan

```
N/A — 전체 구현 계획은 14_Deployment_and_Release_Plan.md에서 다룬다.
      이 문서의 아키텍처 결정은 구현 시작 전 팀 전체가 합의해야 하는 사전 조건이다.

      v2.0 기준 구현 전제 조건:
      - ADR-01 (v2.0) FastAPI 도입이 확정됐으므로, api/ 디렉토리를 먼저 구성해야 한다.
      - React 프론트엔드는 API Layer(FastAPI)를 통해서만 데이터에 접근한다.
      - WS /ws/training 구현 없이 Explorer Training 화면을 구현하면
        실시간 학습 모니터링이 불가능하다.
      - WS /ws/inspection/auto 구현 없이 Vision 자동 검사를 구현하면
        불량 감지 즉시 중지 기능이 동작하지 않는다.
      - utils/ 레이어는 FastAPI와 Streamlit(보조 도구) 양쪽에서 공유되므로
        FastAPI 의존성(Request/Response)을 utils/에 직접 추가하면 안 된다.
```

---

*다음 문서*: [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)

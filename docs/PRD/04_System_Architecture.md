# 04. System Architecture

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [03_Functional_Requirements.md](./03_Functional_Requirements.md)
> **버전**: v1.1
> **작성일**: 2026-05-08
> **최종수정**: 2026-05-26
> **후속 문서**: [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)
> **중요**: 이 문서는 05~14번 PRD 파일 전체가 참조하는 아키텍처 기준이다. 컴포넌트명·파일 경로·인터페이스 시그니처는 이 문서에서 확정되며, 이후 파일에서 변경 불가.
> **v1.1 주의**: v1.1: 이중 대시보드 구조로 확장. B.1~B.9는 모델 탐색 대시보드 전용. B.10 이후는 비전검사 대시보드 아키텍처.

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Detailed Specification](#b-detailed-specification)
  - [B.1 전체 시스템 아키텍처](#b1-전체-시스템-아키텍처)
  - [B.2 레이어 구조](#b2-레이어-구조)
  - [B.3 모듈 책임 명세](#b3-모듈-책임-명세)
  - [B.4 모듈 간 의존성 그래프](#b4-모듈-간-의존성-그래프)
  - [B.5 학습 비동기 처리 아키텍처](#b5-학습-비동기-처리-아키텍처)
  - [B.6 Streamlit 생명주기와 상태 관리](#b6-streamlit-생명주기와-상태-관리)
  - [B.7 훈련 상태 머신](#b7-훈련-상태-머신)
  - [B.8 주요 시퀀스 다이어그램](#b8-주요-시퀀스-다이어그램)
  - [B.9 에러 전파 경로](#b9-에러-전파-경로)
  - [B.10 비전검사 대시보드 아키텍처](#b10-비전검사-대시보드-아키텍처)
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

### A.2 핵심 아키텍처 결정 사항 (ADR)

| ADR ID | 결정 | 근거 |
|--------|------|------|
| ADR-01 | REST API 서버 없음. Streamlit 단일 프로세스 | 단일 사용자(가정 A-01), 네트워크 레이턴시 제거 |
| ADR-02 | 학습 루프는 백그라운드 스레드 + `queue.Queue` | Streamlit 메인 스레드 블로킹 방지, UI 갱신 유지 |
| ADR-03 | 영속 저장은 파일시스템 전용 (JSON + YAML + .pth) | RDBMS 불필요, 단일 사용자, Docker 볼륨 마운트로 영속성 |
| ADR-04 | session_state는 메인 스레드에서만 Write | Streamlit 스레드 안전성 보장 (FR-CMN-01 멱등성) |
| ADR-05 | 전처리 파이프라인은 `utils/image_utils.py` 단일 구현 | 탭2 미리보기와 탭4 학습 루프가 동일 코드 사용, 불일치 방지 |
| ADR-06 | Anomalib v1.0+ Engine API 사용 | `model_factory.py` 래퍼로 추상화, 버전 업 시 래퍼만 수정 |

---

## B. Detailed Specification

---

### B.1 전체 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                        사용자 브라우저                                  │
│                   http://localhost:8501                               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ WebSocket (Streamlit 내장)
┌────────────────────────────▼─────────────────────────────────────────┐
│                       Streamlit 프로세스                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                        app.py                               │    │
│  │   init_session_state()  │  st.set_page_config()            │    │
│  │   render_sidebar()      │  st.tabs([탭1~탭6])               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             │                                        │
│  ┌──────────┬──────────┬───▼──────┬──────────┬──────────┬───────┐  │
│  │  tab1    │  tab2    │  tab3    │  tab4    │  tab5    │ tab6  │  │
│  │_data_    │_preproc_ │_model_   │_training │_history_ │_anomaly│  │
│  │folder.py │params.py │params.py │.py       │.py       │_map.py│  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴───┬───┘  │
│       │          │          │          │          │         │       │
│  ┌────▼──────────▼──────────▼──────────▼──────────▼─────────▼────┐  │
│  │                       utils/ 레이어                             │  │
│  │  session_state_init  config_manager  messages  image_utils    │  │
│  │  metrics             model_factory                            │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │            components/ 레이어                                  │   │
│  │            sidebar.py                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              학습 비동기 레이어 (ADR-02)                         │   │
│  │   TrainingWorker Thread  ←→  queue.Queue  ←→  메인 스레드 UI  │   │
│  │   threading.Event (stop_event)                               │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
└────────────────────────────┬┘────────────────────────────────────────┘
                             │ Python function call
┌────────────────────────────▼─────────────────────────────────────────┐
│                        ML 레이어                                       │
│   Anomalib Engine (EfficientAD)  │  Anomalib Engine (PatchCore)      │
│   PyTorch ≥ 2.1                  │  torchvision (backbone)           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ 파일시스템 읽기/쓰기
┌────────────────────────────▼─────────────────────────────────────────┐
│                       파일시스템 레이어                                  │
│                                                                      │
│  /app/dataset/           읽기 전용 (볼륨 마운트)                         │
│  /app/experiments/       history.json (읽기/쓰기)                     │
│  /app/models/{exp_id}/   model_state_dict.pth, configs.yaml          │
│  /app/logs/{exp_id}.log  학습 로그 (쓰기 전용)                           │
│  /app/configs.yaml       공유 설정 (읽기/쓰기)                           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                       데이터베이스 레이어                                │
│                                                                      │
│  MySQL 8.0  (컨테이너: smart-qc-db)                                   │
│  호스트 포트: 3307  →  컨테이너 포트: 3306                               │
│  DB: smart-qc                                                        │
│  볼륨: db_data:/var/lib/mysql                                         │
│  접속 (앱 내부): db:3306   접속 (호스트): localhost:3307               │
└──────────────────────────────────────────────────────────────────────┘
```

---

### B.2 레이어 구조

| 레이어 | 위치 | 역할 | 외부 의존성 |
|--------|------|------|------------|
| **Presentation Layer** | `app.py`, `tabs/`, `components/` | UI 렌더링, 사용자 입력 처리, session_state Read/Write | Streamlit |
| **Service Layer** | `utils/` | 비즈니스 로직, 파일 I/O, 이미지 처리, 메트릭 계산 | PyYAML, PIL, OpenCV, scikit-learn, torch |
| **ML Layer** | `utils/model_factory.py` → Anomalib | 모델 초기화, 학습 루프, 추론 | Anomalib ≥ 1.0, PyTorch ≥ 2.1 |
| **Storage Layer** | 파일시스템 (`./experiments/`, `./models/`, `./logs/`) + MySQL 8.0 (`smart-qc-db`) | 영속 데이터 저장 | MySQL 8.0 (컨테이너) |

**레이어 간 의존성 규칙**:
- Presentation → Service: 허용
- Presentation → ML: 금지 (반드시 Service 경유)
- Presentation → Storage: 금지 (반드시 Service 경유)
- Service → ML: 허용
- Service → Storage: 허용
- ML → Storage: 학습 중간 체크포인트 없음 (완료 시 Service가 저장)

---

### B.3 모듈 책임 명세

#### B.3.1 진입점

##### `app.py`

```python
책임:
  - st.set_page_config() 호출 (1회)
  - init_session_state() 호출 (매 rerun, 멱등)
  - render_sidebar() 호출
  - st.tabs() 로 6탭 생성
  - 각 탭 내부에서 tab{n}_*.render() 호출

금지:
  - 비즈니스 로직 직접 구현
  - 파일 I/O 직접 호출
  - ML 코드 직접 호출

공개 함수: 없음 (진입점 파일)
```

---

#### B.3.2 탭 모듈 (`tabs/`)

각 탭 파일의 공통 구조:

```python
# tabs/tab{n}_*.py 공통 패턴

def render():
    """
    탭 렌더링 진입 함수.
    app.py의 st.tabs() 내부에서 호출된다.
    Guard 조건 미충족 시 st.warning() + return으로 조기 종료.
    """
    _guard()       # Guard 조건 확인
    _render_ui()   # UI 컴포넌트 렌더링
    _handle_events() # 버튼 클릭 등 이벤트 처리

def _guard() -> bool:
    """선행 session_state 키 확인. 미충족 시 return."""

def _render_ui():
    """Streamlit 컴포넌트 렌더링."""

def _handle_events():
    """버튼 클릭 등 부작용(파일 쓰기, ML 호출) 처리."""
```

| 파일 | 공개 함수 | 주요 책임 |
|------|----------|-----------|
| `tab1_data_folder.py` | `render()` | 경로 입력, MVTec AD 검증, dataset_meta 구성, 트리/테이블/썸네일 렌더링 |
| `tab2_preprocessing.py` | `render()` | 전처리 라디오, 파라미터 UI, 미리보기, preprocessing_config 저장 |
| `tab3_model_params.py` | `render()` | 모델 라디오, 파라미터 UI, 디바이스 감지, model_config 저장 |
| `tab4_training.py` | `render()` | 실험명 입력, 학습 시작/중지, Progress Bar, Loss 곡선, 로그 |
| `tab5_history.py` | `render()` | 히스토리 테이블, 상세 결과 차트, 비교 차트, 모델 저장, 삭제 |
| `tab6_anomaly_map.py` | `render()` | 이미지 목록 테이블, Threshold 슬라이더, 3분할 시각화, PNG 저장 |
| `inspection/inspection_app.py` | `render()` | 비전검사 대시보드 진입점. render() 함수 제공. active_dashboard == "inspection" 시 app.py에서 호출 |
| `inspection/tabs/insp_tab1_realtime.py` | `render()` | 실시간 검사 탭. 수동/자동 검사 버튼, 추론 실행, 결과 표시, 불량 팝업 |
| `inspection/tabs/insp_tab2_history.py` | `render()` | 검사 이력 테이블(5컬럼) + KPI 카드 |
| `inspection/tabs/insp_tab3_model.py` | `render()` | 딥러닝 모델 교체. history.json 읽기 전용, 완료 실험 목록 |
| `inspection/utils/insp_session_init.py` | — | INSPECTION_SESSION_SCHEMA 초기화 |
| `inspection/utils/test_sampler.py` | — | test_pool 구성(dataset_path/test/ 스캔), 랜덤 샘플링 |

---

#### B.3.3 유틸리티 모듈 (`utils/`)

##### `utils/session_state_init.py`

```python
책임: session_state 초기화 스키마 정의 및 초기화 실행

공개 상수:
  SESSION_STATE_SCHEMA: dict  # 00_Global_Context 3.1절 스키마

공개 함수:
  def init_session_state() -> None:
      """
      st.session_state에 SESSION_STATE_SCHEMA의 키가 없는 경우에만 기본값 설정.
      이미 존재하는 키는 덮어쓰지 않는다 (멱등).
      """
```

---

##### `utils/config_manager.py`

```python
책임: configs.yaml 파일 읽기/쓰기

공개 함수:
  def load_config(path: str = "./configs.yaml") -> dict:
      """
      YAML 파일 로드. 파일 없으면 {} 반환 (예외 발생 금지).
      파싱 실패 시 ERR_CONFIG_LOAD_FAILED 포함 ConfigLoadError 발생.
      반환: dict (빈 dict 포함 항상 dict 타입)
      """

  def save_config_section(
      section: str,
      data: dict,
      path: str = "./configs.yaml"
  ) -> None:
      """
      기존 파일 로드 후 section 키만 data로 교체하여 저장.
      R-ATOMIC-01: tmpfile(.tmp) 생성 → rename 방식.
      존재하는 다른 섹션은 보존.
      """

  def get_preprocessing_config(path: str = "./configs.yaml") -> dict | None:
      """load_config() 후 'preprocessing' 섹션 반환. 없으면 None."""

  def get_model_config(path: str = "./configs.yaml") -> dict | None:
      """load_config() 후 'model' 섹션 반환. 없으면 None."""
```

---

##### `utils/messages.py`

```python
책임: 모든 안내·오류 메시지 문자열 상수 관리

공개 상수:
  MSG: dict[str, str]  # 00_Global_Context 3.4절 전체
  ERR: dict[str, str]  # 00_Global_Context 3.5절 전체 오류 코드

# 사용법: st.warning(MSG["NO_DATASET"])
#         raise AppError(ERR["ERR_DATASET_NOT_FOUND"])
```

---

##### `utils/image_utils.py`

```python
책임: 이미지 전처리 파이프라인 전체 구현 (ADR-05)

공개 함수:
  def load_image(path: str) -> PIL.Image:
      """
      이미지 파일 로드.
      mode == "L" → convert("RGB") 자동 수행 (Grayscale 처리).
      mode == "RGBA" → convert("RGB") 수행.
      지원 포맷: {".jpg", ".jpeg", ".png", ".bmp"}
      """

  def apply_filter(
      image: PIL.Image,
      method: str,           # "none" | "homomorphic" | "he" | "clahe"
      params: dict | None    # 1.3절 preprocessing_params 구조
  ) -> PIL.Image:
      """
      선택된 전처리 필터 적용.
      method == "none": 원본 반환
      method == "homomorphic": Homomorphic Filter 적용
      method == "he": Histogram Equalization 적용
      method == "clahe": CLAHE 적용
      입력: PIL.Image (RGB), 출력: PIL.Image (RGB)
      OpenCV 사용 시: PIL → numpy → cv2 → PIL 변환
      """

  def resize_with_padding(
      image: PIL.Image,
      target_size: int       # image_size (정사각형)
  ) -> PIL.Image:
      """
      비율 유지 Resize 후 검정(0) 패딩으로 target_size×target_size 생성.
      resize_mode는 항상 "padding" (R에 따른 고정값).
      """

  def normalize_to_tensor(
      image: PIL.Image,
      mean: list[float],     # [R, G, B]
      std: list[float]       # [R, G, B]
  ) -> torch.Tensor:
      """
      PIL.Image → torch.Tensor (C, H, W), float32, 정규화 적용.
      """

  def apply_preprocessing(
      image_path: str,
      config: dict           # 1.6절 preprocessing_config 구조
  ) -> tuple[PIL.Image, torch.Tensor]:
      """
      load_image → apply_filter → resize_with_padding → normalize_to_tensor
      파이프라인 전체 실행.
      반환: (미리보기용 PIL.Image, 학습용 torch.Tensor)
      PIL.Image는 정규화 미적용 (시각화 목적).
      """

  def tensor_to_display_image(tensor: torch.Tensor) -> PIL.Image:
      """역정규화 후 PIL.Image 변환 (미리보기 표시용)."""

  def anomaly_map_to_heatmap(
      anomaly_map: np.ndarray,    # (H, W) float32
      colormap: int = cv2.COLORMAP_JET
  ) -> PIL.Image:
      """Anomaly Score 2D 배열을 jet colormap 히트맵 PIL.Image로 변환."""

  def create_triplet_image(
      original: PIL.Image,
      gt_mask: PIL.Image | None,  # None이면 검정 마스크
      heatmap: PIL.Image
  ) -> PIL.Image:
      """
      원본 / GT 마스크 / Heatmap 3개를 가로로 이어 붙인 단일 PIL.Image 생성.
      세 패널 동일 크기.
      """
```

---

##### `utils/metrics.py`

```python
책임: 이상 탐지 평가 메트릭 계산

공개 함수:
  def compute_metrics(
      y_true: list[int],           # 0=정상, 1=결함
      anomaly_scores: list[float],
      threshold: float
  ) -> dict:
      """
      반환 dict (1.2절 metrics 스키마):
        accuracy, precision, recall, f1_score, f2_score, auc,
        confusion_matrix(tp,fp,tn,fn), anomaly_scores, image_labels
      sklearn.metrics 사용: roc_auc_score, precision_recall_fscore_support
      f2_score: beta=2 fbeta_score
      """

  def compute_roc_curve(
      y_true: list[int],
      anomaly_scores: list[float]
  ) -> tuple[np.ndarray, np.ndarray, float]:
      """
      반환: (fpr, tpr, auc)
      sklearn.metrics.roc_curve, roc_auc_score 사용
      """

  def compute_threshold_from_percentile(
      normal_scores: list[float],
      percentile: float              # 0.0 ~ 100.0
  ) -> float:
      """
      정상 이미지 Anomaly Score의 percentile 백분위수를 threshold로 반환.
      np.percentile(normal_scores, percentile)
      """
```

---

##### `utils/model_factory.py`

```python
책임: Anomalib 모델 초기화 래퍼 (ADR-06)

공개 함수:
  def create_trainer(
      model_config: dict,           # 1.7절 model_config 구조
      preprocessing_config: dict,   # 1.6절 preprocessing_config 구조
      dataset_path: str,
      device: str,                  # "cuda" | "cpu"
      exp_id: str,
      stop_event: threading.Event,
      result_queue: queue.Queue
  ) -> TrainingWorker:
      """
      model_config.model_type에 따라 EfficientAD 또는 PatchCore TrainingWorker 생성.
      반환된 TrainingWorker는 threading.Thread로 실행된다.
      """

  def load_model_for_inference(
      exp_id: str,
      model_path: str,              # experiment.model_path
      model_config: dict,
      device: str
  ) -> object:                      # Anomalib model object
      """
      저장된 model_state_dict.pth + configs.yaml 로드 후 추론용 모델 반환.
      탭6 Anomaly Map 생성 시 사용.
      """

  def run_inference(
      model: object,
      image_tensor: torch.Tensor    # (1, C, H, W)
  ) -> np.ndarray:                  # (H, W) anomaly map
      """단일 이미지 추론. Anomaly Map (H, W) 반환."""
```

---

##### `utils/training_worker.py`

```python
책임: 백그라운드 학습 스레드 실행 로직 (ADR-02)

공개 클래스:
  class TrainingWorker(threading.Thread):
      """
      백그라운드에서 Anomalib 학습 루프를 실행한다.
      stop_event: threading.Event — 외부에서 set() 시 루프 종료
      result_queue: queue.Queue — 진행 상황 메인 스레드에 전송

      Queue에 넣는 메시지 형식:
        {"type": "progress", "step": int, "total": int, "loss": float, "elapsed": float}
        {"type": "log", "message": str}
        {"type": "completed", "y_true": list, "scores": list, "model": object}
        {"type": "error", "exception": Exception, "traceback": str}
        {"type": "stopped"}
      """

      def run(self) -> None:
          """
          1. random seed 고정 (R-SEED-01)
          2. DataLoader 구성 (image_utils.apply_preprocessing 사용)
          3. 모델 초기화
          4. 학습 루프:
             - stop_event.is_set() 체크 (매 step)
             - 진행 상황 result_queue.put()
          5. 완료: result_queue.put({"type": "completed", ...})
          6. 중단: result_queue.put({"type": "stopped"})
          7. 예외: result_queue.put({"type": "error", ...})
          """
```

---

#### B.3.4 컴포넌트 모듈 (`components/`)

##### `components/sidebar.py`

```python
책임: 사이드바 공통 컴포넌트 렌더링

공개 함수:
  def render_sidebar() -> None:
      """
      FR-CMN-02 명세대로 데이터셋/디바이스/현재설정 섹션 렌더링.
      각 섹션은 해당 session_state 키가 None이 아닌 경우에만 렌더링.
      """
```

---

### B.4 모듈 간 의존성 그래프

```
app.py
  ├── utils/session_state_init.py
  ├── components/sidebar.py
  │     └── (session_state 읽기만)
  ├── tabs/tab1_data_folder.py
  │     └── utils/image_utils.py (썸네일)
  │     └── utils/messages.py
  ├── tabs/tab2_preprocessing.py
  │     └── utils/image_utils.py (미리보기 전처리)
  │     └── utils/config_manager.py
  │     └── utils/messages.py
  ├── tabs/tab3_model_params.py
  │     └── utils/config_manager.py
  │     └── utils/messages.py
  ├── tabs/tab4_training.py
  │     └── utils/model_factory.py
  │     │     └── utils/training_worker.py
  │     │           └── utils/image_utils.py
  │     │           └── (Anomalib, PyTorch)
  │     └── utils/metrics.py
  │     └── utils/config_manager.py
  │     └── utils/messages.py
  ├── tabs/tab5_history.py
  │     └── utils/config_manager.py (history.json 로드)
  │     └── utils/metrics.py (ROC 계산)
  │     └── utils/messages.py
  └── tabs/tab6_anomaly_map.py
        └── utils/model_factory.py (추론)
        └── utils/image_utils.py (heatmap, triplet)
        └── utils/messages.py

금지된 의존성:
  tabs/* → Anomalib 직접 import (반드시 model_factory 경유)
  tabs/* → 파일시스템 직접 접근 (반드시 config_manager 경유)
  utils/* → tabs/* (역방향 의존성 금지)
  utils/* → components/* (역방향 의존성 금지)
```

---

### B.5 학습 비동기 처리 아키텍처

이 시스템의 가장 복잡한 동시성 문제: Streamlit 메인 스레드를 블로킹하지 않고 수십 분의 학습 루프를 실행하는 방법.

#### B.5.1 스레드 모델

```
메인 스레드 (Streamlit)               백그라운드 스레드 (TrainingWorker)
─────────────────────────────────────────────────────────────────────
[학습 시작 버튼 클릭]
  stop_event = threading.Event()
  result_queue = queue.Queue()
  worker = TrainingWorker(
      config=...,
      stop_event=stop_event,
      result_queue=result_queue
  )
  worker.daemon = True
  worker.start()
  session_state.current_run_status = "running"
  st.rerun()
                                       [worker.run() 시작]
                                         random seed 고정
                                         DataLoader 구성
                                         모델 초기화
                                         for step in training_loop:
[매 rerun 사이클]                           if stop_event.is_set():
  while not result_queue.empty():            result_queue.put({"type":"stopped"})
    msg = result_queue.get_nowait()          return
    if msg["type"] == "progress":          loss = train_step()
      update_progress_bar(msg)             if step % 500 == 0:
    elif msg["type"] == "log":               result_queue.put({
      append_log(msg)                          "type":"progress",
    elif msg["type"] == "completed":           "step": step, ...
      handle_completion(msg)               })
      break                             result_queue.put({"type":"log", ...})
    elif msg["type"] == "error":
      handle_error(msg)                [학습 완료]
      break                            result_queue.put({
    elif msg["type"] == "stopped":       "type":"completed",
      handle_stopped()                   "y_true": [...],
      break                             "scores": [...],
  time.sleep(0.3)                        "model": model_object
  st.rerun()                           })

[학습 중지 버튼 클릭]
  stop_event.set()
  # 다음 rerun에서 "stopped" 메시지 처리
```

#### B.5.2 스레드 안전성 규칙

| 규칙 | 설명 |
|------|------|
| **R-THREAD-01** | `st.session_state` Write는 메인 스레드에서만 수행한다. TrainingWorker는 절대 session_state에 직접 쓰지 않는다. |
| **R-THREAD-02** | 메인 스레드와 백그라운드 스레드의 유일한 통신 수단은 `result_queue`(Queue)와 `stop_event`(Event)이다. |
| **R-THREAD-03** | `worker.daemon = True`로 설정하여 Streamlit 프로세스 종료 시 워커 스레드도 함께 종료된다. |
| **R-THREAD-04** | `result_queue.get_nowait()`를 사용한다. `get(block=True)`는 메인 스레드를 블로킹하므로 금지. |
| **R-THREAD-05** | 메인 스레드는 `time.sleep(0.3)` 후 `st.rerun()`으로 UI를 갱신한다. sleep 시간은 0.3초 고정. |

#### B.5.3 session_state 학습 관련 키

```python
# 학습 시작 전 초기화
session_state.current_run_status = "running"
session_state.current_exp_id = exp_id
session_state._stop_event = stop_event        # threading.Event
session_state._result_queue = result_queue    # queue.Queue
session_state._progress = {"step": 0, "total": total_steps, "loss": None, "elapsed": 0.0}
session_state._log_lines = []                 # list[str], 최대 100줄
session_state._loss_history = []              # list[dict] {"step":int, "loss":float}
session_state._training_start_time = time.time()  # elapsed 실시간 계산용

# 학습 완료/중단 후 초기화
session_state.current_run_status = "idle"
session_state.current_exp_id = None
session_state._stop_event = None
session_state._result_queue = None
# _progress, _log_lines, _loss_history는 마지막 상태 유지 (UI에 표시)
```

접두사 `_`가 붙은 키는 내부 상태 키이며, 탭 간 공유 불필요. 탭4 내부에서만 사용한다.

---

### B.6 Streamlit 생명주기와 상태 관리

Streamlit은 사용자 인터랙션(버튼 클릭, 슬라이더 변경 등)마다 전체 스크립트를 재실행(rerun)한다. 이 특성을 고려한 설계 규칙:

```
[Streamlit Rerun 사이클]

사용자 인터랙션 발생
  → 전체 app.py 재실행
  → init_session_state() (이미 있는 키는 무시)
  → render_sidebar()
  → st.tabs() 렌더링
  → 활성 탭의 render() 실행
  → UI 상태: st.session_state에서 읽기
  → 버튼 클릭 등 이벤트: if st.button(...): 블록 내부 실행
  → 완료 후 화면 업데이트
```

#### B.6.1 멱등성 보장 규칙

| 규칙 | 설명 | 위반 예 |
|------|------|---------|
| **R-IDEM-01** | 버튼 클릭 없이 rerun 시 부작용 없음 | 탭 함수 최상위에서 파일 쓰기 금지 |
| **R-IDEM-02** | 디바이스 감지는 `device_info is None`인 경우에만 실행 | 매 rerun마다 `torch.cuda.is_available()` 호출 금지 |
| **R-IDEM-03** | 학습 워커 시작은 `st.button("학습 시작")` 블록 내부에서만 | 조건부 없이 `worker.start()` 최상위 호출 금지 |
| **R-IDEM-04** | configs.yaml 쓰기는 저장 버튼 클릭 시에만 | 슬라이더 변경만으로 파일 쓰기 금지 |

#### B.6.2 st.session_state 키 생명주기

```
생성:   init_session_state() 호출 시 (앱 시작)
갱신:   각 탭의 _handle_events() 내부
삭제:   삭제하지 않음 (None으로 초기화)
리셋:   탭1에서 새 경로 입력 시, 이전 dataset_path와 다르면
        dataset_meta, preprocessing_config, model_config 모두 None 리셋
```

---

### B.7 훈련 상태 머신

```
                    [학습 시작 버튼 클릭]
         ┌──────────────────────────────────────────┐
         │                                          │
    ┌────▼────┐                               ┌─────▼──────┐
    │  idle   │◄──────────────────────────────│  running   │
    └─────────┘   [완료/중단/에러 처리 완료]   └─────┬──────┘
                                                    │
                              ┌─────────────────────┤
                              │                     │
                    [stop_event.set()]    [학습 루프 정상 종료]
                              │                     │
                         ┌────▼──────┐    ┌─────────▼──────┐
                         │  stopped  │    │   completed    │
                         └────┬──────┘    └────────┬───────┘
                              │                    │
                [handle_stopped() 실행]  [handle_completion() 실행]
                [history.json append]   [metrics 계산, 파일 저장]
                [status="중단"]         [history.json append]
                              │                    │
                              └──────┬─────────────┘
                                     │ current_run_status = "idle"
                                ┌────▼────┐
                                │  idle   │
                                └─────────┘

[에러 발생]
  running → idle (handle_error 실행, st.error() 표시, history 미기록)
```

상태값은 `session_state.current_run_status`에 저장. 허용 값: `"idle"` | `"running"` | `"stopped"` | `"completed"`.
`"stopped"`와 `"completed"`는 처리 완료 후 즉시 `"idle"`로 전환.

---

### B.8 주요 시퀀스 다이어그램

#### B.8.1 학습 시작 ~ 완료 시퀀스

```
사용자          app.py          tab4_training.py       TrainingWorker      파일시스템
  │                │                    │                      │               │
  │ [학습 시작]    │                    │                      │               │
  ├──────────────►│                    │                      │               │
  │               │  tab4.render()     │                      │               │
  │               ├───────────────────►│                      │               │
  │               │                   │  worker = create()    │               │
  │               │                   ├─────────────────────►│               │
  │               │                   │  worker.start()       │               │
  │               │                   ├─────────────────────►│               │
  │               │                   │                      │ 학습 루프 시작  │
  │               │                   │  status="running"     │               │
  │               │                   │  st.rerun()           │               │
  │               │◄──────────────────┤                      │               │
  │  화면 갱신     │                    │                      │               │
  │◄──────────────┤                    │                      │               │
  │               │                   │   queue.put(progress) │               │
  │               │                   │◄─────────────────────┤               │
  │   [1초마다]    │                    │                      │               │
  │               │  tab4.render()     │                      │               │
  │               ├───────────────────►│                      │               │
  │               │                   │ queue 드레인           │               │
  │               │                   │ Progress Bar 갱신     │               │
  │               │                   │ Loss 차트 갱신        │               │
  │               │                   │ st.rerun()            │               │
  │  화면 갱신     │◄──────────────────┤                      │               │
  │◄──────────────┤  (반복)            │                      │               │
  │               │                   │   queue.put(completed)│               │
  │               │                   │◄─────────────────────┤               │
  │               │                   │ metrics 계산           │               │
  │               │                   │ experiments 구성       │               │
  │               │                   │                       │ history.json  │
  │               │                   ├───────────────────────────────────────►
  │               │                   │                       │ .pth 저장     │
  │               │                   ├───────────────────────────────────────►
  │               │                   │ status="idle"         │               │
  │               │                   │ st.success()          │               │
  │  완료 알림     │◄──────────────────┤                      │               │
  │◄──────────────┤                    │                      │               │
```

#### B.8.2 탭6 Anomaly Map 시퀀스

```
사용자      tab6_anomaly_map.py    model_factory.py    image_utils.py    파일시스템
  │                 │                    │                   │               │
  │ [이미지 선택]   │                    │                   │               │
  ├────────────────►│                    │                   │               │
  │                 │ load_model_for_inference()             │               │
  │                 ├───────────────────►│                   │               │
  │                 │                   │ model_state_dict   │               │
  │                 │                   ├───────────────────────────────────►│
  │                 │                   │◄──────────────────────────────────┤
  │                 │ load_image(path)   │                   │               │
  │                 ├───────────────────────────────────────►│               │
  │                 │ apply_preprocessing()                  │               │
  │                 ├───────────────────────────────────────►│               │
  │                 │◄──────────────────────────────────────┤               │
  │                 │ run_inference(model, tensor)           │               │
  │                 ├───────────────────►│                   │               │
  │                 │◄──────────────────┤ anomaly_map (H,W) │               │
  │                 │ anomaly_map_to_heatmap()               │               │
  │                 ├───────────────────────────────────────►│               │
  │                 │◄──────────────────────────────────────┤               │
  │                 │ create_triplet_image()                 │               │
  │                 ├───────────────────────────────────────►│               │
  │                 │◄──────────────────────────────────────┤               │
  │  3분할 시각화   │                    │                   │               │
  │◄────────────────┤                    │                   │               │
```

---

### B.9 에러 전파 경로

| 발생 위치 | 예외 타입 | 전파 경로 | 최종 처리 |
|-----------|-----------|-----------|-----------|
| `image_utils.load_image()` | `FileNotFoundError` | tab1 → `st.error()` | dataset_path = None |
| `image_utils.apply_filter()` | `cv2.error` | tab2 → `st.error()` | 미리보기 미갱신 |
| `config_manager.load_config()` | `yaml.YAMLError` | tab2/3 → `st.error(ERR_CONFIG_LOAD_FAILED)` | UI 상태 유지 |
| `TrainingWorker.run()` — GPU OOM | `torch.cuda.OutOfMemoryError` | queue → tab4 → `st.error()` | status="idle" |
| `TrainingWorker.run()` — 일반 예외 | `Exception` | queue(type="error") → tab4 → `st.error()` | status="idle" |
| `model_factory.load_model_for_inference()` | `FileNotFoundError` | tab6 → `st.error()` | 시각화 미렌더링 |
| `save_history()` — 디스크 풀 | `OSError` | tab4 → `st.error(ERR_MODEL_SAVE_FAILED)` | session_state 유지 |

모든 예외는 각 탭의 `_handle_events()` 또는 `render()` 상단 `try-except`에서 포착된다. 처리되지 않은 예외가 Streamlit 메인 루프까지 전파되면 페이지 에러 화면이 표시된다.

---

## C. System & Data Design

### C.1 디렉토리 구조 (확정)

```
/app/                               # Docker WORKDIR (로컬: ./smart-qc-dashboard/)
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.base.yml
├── docker-compose.yml
├── docker-compose.cpu.yml
├── .env
├── configs.yaml                    # 공유 설정 파일 (탭2/3 Write)
│
├── tabs/
│   ├── __init__.py
│   ├── tab1_data_folder.py
│   ├── tab2_preprocessing.py
│   ├── tab3_model_params.py
│   ├── tab4_training.py
│   ├── tab5_history.py
│   └── tab6_anomaly_map.py
│
├── utils/
│   ├── __init__.py
│   ├── session_state_init.py
│   ├── config_manager.py
│   ├── messages.py
│   ├── image_utils.py
│   ├── metrics.py
│   ├── model_factory.py
│   └── training_worker.py
│
├── components/
│   ├── __init__.py
│   └── sidebar.py
│
├── experiments/
│   └── history.json                # 자동 생성
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

### C.2 session_state 전체 키 목록 (확정)

| 키 | 타입 | 초기값 | Write 탭 | Read 탭 |
|----|------|--------|----------|---------|
| `dataset_path` | `str\|None` | None | 탭1 | 탭2, 탭4 |
| `dataset_meta` | `dict\|None` | None | 탭1 | 탭2, 탭6 |
| `preprocessing_config` | `dict\|None` | None | 탭2 | 탭3, 탭4 |
| `model_config` | `dict\|None` | None | 탭3 | 탭4 |
| `device_info` | `dict\|None` | None | 탭3 | 탭4, sidebar |
| `experiments` | `dict` | `{}` | 탭4, 탭5 | 탭5, 탭6 |
| `current_run_status` | `str` | `"idle"` | 탭4 | 탭4 |
| `current_exp_id` | `str\|None` | None | 탭4 | 탭4 |
| `selected_experiment_id` | `str\|None` | None | 탭5 | 탭6 |
| `anomaly_map_threshold` | `float\|None` | None | 탭6 | 탭6 |
| `_stop_event` | `Event\|None` | None | 탭4 내부 | 탭4 내부 |
| `_result_queue` | `Queue\|None` | None | 탭4 내부 | 탭4 내부 |
| `_progress` | `dict\|None` | None | 탭4 내부 | 탭4 내부 |
| `_log_lines` | `list` | `[]` | 탭4 내부 | 탭4 내부 |
| `_loss_history` | `list` | `[]` | 탭4 내부 | 탭4 내부 |

---

## D. API Contracts

```
N/A — REST API 없음. 모듈 간 인터페이스는 B.3절에서 Python 함수 시그니처로 확정.
```

---

## E. AI/ML Details

```
N/A — 이 문서는 아키텍처 범위이다.
      Anomalib Engine 초기화·학습 루프·추론 상세 구현은
      08_AI_ML_Integration.md에서 다룬다.
      이 문서에서는 ML 레이어가 model_factory.py와 training_worker.py를 통해
      Presentation/Service 레이어로부터 격리된다는 아키텍처 결정만 확정한다.
```

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체 상속.

아키텍처 수준에서 추가로 명시:

| 항목 | 요구사항 | 아키텍처 결정 |
|------|----------|--------------|
| **UI 비블로킹** | 학습 중 탭 전환 가능 | ADR-02: 백그라운드 스레드 |
| **rerun 성능** | rerun 1회 < 200ms (학습 중 제외) | 탭 함수에서 무거운 연산 금지 (버튼 블록 내부만) |
| **파일 쓰기 안전성** | 부분 쓰기로 인한 데이터 손상 없음 | R-ATOMIC-01: tmpfile → rename |
| **메모리 누수** | 학습 완료 후 모델 객체 GC 보장 | `del model; torch.cuda.empty_cache()` |
| **포트** | 8501/tcp 고정 | Dockerfile EXPOSE 8501 |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards) 전체 상속.

아키텍처 수준에서 추가:

| 관측 포인트 | 위치 | 방법 |
|------------|------|------|
| 모듈 초기화 실패 | `app.py` `try-except` | `st.error()` + 로그 |
| 파일시스템 마운트 확인 | `app.py` 시작 시 | `Path("./experiments").mkdir(parents=True, exist_ok=True)` |
| 스레드 생존 여부 | 탭4 rerun 사이클 | `worker.is_alive()` 확인 후 미아 스레드 감지 |

---

## H. QA & Validation

### H.1 아키텍처 검증 기준

| # | 기준 | 검증 방법 |
|---|------|-----------|
| ARC-01 | `tabs/*`에서 Anomalib 직접 import 없음 | `grep -r "from anomalib" tabs/` → 결과 없음 |
| ARC-02 | `tabs/*`에서 파일시스템 직접 쓰기 없음 | `grep -r "open(" tabs/` → config_manager 경유만 허용 |
| ARC-03 | `utils/*`에서 `st.session_state` Write 없음 | `grep -r "session_state\." utils/` → 읽기만 허용 |
| ARC-04 | TrainingWorker에서 `st.session_state` 접근 없음 | `grep -r "session_state" utils/training_worker.py` → 결과 없음 |
| ARC-05 | result_queue 메시지 타입이 B.3.3절 명세 외 없음 | training_worker.py 코드 리뷰 |
| ARC-06 | 학습 중 [학습 시작] 버튼 disabled | `current_run_status == "running"` 조건 확인 |

### H.2 Given-When-Then 시나리오

#### TC-ARC-02: 탭 전환 중 학습 비블로킹

```
Given:  탭4에서 학습이 진행 중이다 (current_run_status == "running")
        TrainingWorker 스레드가 실행 중이다
When:   사용자가 탭1 탭을 클릭한다
Then:   탭1이 1초 이내에 렌더링된다
        TrainingWorker 스레드는 중단 없이 계속 실행된다
        result_queue에 progress 메시지가 계속 쌓인다
        다시 탭4로 돌아오면 최신 진행 상황이 표시된다
```

#### TC-ARC-04: 브라우저 새로고침 후 스레드 안전성

```
Given:  학습이 진행 중이다 (TrainingWorker running)
When:   사용자가 브라우저를 새로고침한다
Then:   session_state가 초기화된다 (current_run_status = "idle")
        TrainingWorker 스레드는 daemon=True이므로 Streamlit 프로세스와 함께 종료
        새로운 세션에서 탭4는 "idle" 상태로 렌더링
        history.json에는 중단된 실험 레코드가 없음 (새로고침 시 handle_stopped 미실행)
        → 탭4에 "학습 중 새로고침 시 학습 상태를 확인할 수 없습니다." 안내 텍스트 상시 표시
```

#### TC-ARC-01: 멱등성 검증

```
Given:  탭2에서 preprocessing_config가 이미 저장된 상태이다
When:   사용자가 탭2를 클릭하여 rerun이 발생한다 (버튼 클릭 없음)
Then:   configs.yaml 파일의 수정 시각이 변경되지 않는다
        session_state.preprocessing_config 값이 변경되지 않는다
        st.success() 메시지가 표시되지 않는다
```

---

## I. Implementation Plan

```
N/A — 전체 구현 계획은 14_Deployment_and_Release_Plan.md에서 다룬다.
      이 문서의 아키텍처 결정은 구현 시작 전 팀 전체가 합의해야 하는 사전 조건이다.

      이 문서가 확정되기 전에 구현을 시작하면 안 되는 이유:
      - 탭4 학습 비동기 구조(B.5절)를 모르고 tab4_training.py를 구현하면
        UI 블로킹 버그가 발생한다.
      - 모듈 의존성 규칙(B.4절)을 모르고 tabs/*에서 Anomalib를 직접 import하면
        리팩터링이 필요하다.
      - 레이어 규칙(B.2절)을 모르고 utils/*에서 session_state를 Write하면
        스레드 안전성이 깨진다.
```

---

### B.10 비전검사 대시보드 아키텍처

#### B.10.1 대시보드 라우팅

app.py는 session_state.active_dashboard 값에 따라 렌더링을 분기한다:

```python
if st.session_state.active_dashboard == "inspection":
    from inspection.inspection_app import render as render_inspection
    render_inspection()
else:
    # 기존 모델 탐색 대시보드 탭1~6 렌더링
    tab1, tab2, ... = st.tabs([...])
    ...
```

#### B.10.2 비전검사 아키텍처 결정 (ADR)

| ADR ID | 결정 | 근거 |
|--------|------|------|
| ADR-INSP-01 | 검사 이력은 session_state에만 저장 (영속 없음) | 현장 운영 단순화. 앱 재시작 시 초기화 |
| ADR-INSP-02 | history.json 읽기 전용 | 비전검사는 학습 실험을 생성하지 않음 |
| ADR-INSP-03 | 자동 검사는 time.sleep(3) + st.rerun() | 별도 스레드 없음. Streamlit 단일 스레드 모델 준수 |
| ADR-INSP-04 | 모델 로드는 st.cache_resource로 캐싱 | 동일 모델 연속 추론 시 재로드 방지. 모델 교체 시에만 캐시 무효화 |
| ADR-INSP-05 | test_pool은 모델 교체 시 재구성 | dataset_path는 실험 레코드에 저장됨. 교체된 모델의 데이터셋 사용 |

#### B.10.3 비전검사 데이터 흐름

```
test_pool 구성 (UC-13, 모델 교체 시)
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
```

#### B.10.4 sidebar.py 변경 (v1.1)

기존: 데이터셋·디바이스·현재 설정 정보 섹션 렌더링
변경: 대시보드 전환 버튼 2개만 렌더링

```python
# components/sidebar.py (v1.1)
def render_sidebar():
    with st.sidebar:
        if st.button("🔬 모델 탐색 대시보드",
                     use_container_width=True,
                     type="primary" if st.session_state.active_dashboard == "explorer" else "secondary"):
            st.session_state.active_dashboard = "explorer"
            st.rerun()
        if st.button("🏭 비전검사 대시보드",
                     use_container_width=True,
                     type="primary" if st.session_state.active_dashboard == "inspection" else "secondary"):
            st.session_state.active_dashboard = "inspection"
            st.rerun()
```

---

*다음 문서*: [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)

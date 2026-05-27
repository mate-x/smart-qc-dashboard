# Smart QC Dashboard — UI/UX 설계 문서

> **작성 기준:** 실제 Streamlit 소스 코드 역설계 (app.py, tabs/, components/, utils/, configs.yaml)  
> **최종 갱신:** 2026-05-18  
> **버전:** 1.0

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **프로젝트 목적** | MVTec AD 형식 데이터셋을 대상으로 EfficientAD / PatchCore 이상 탐지 모델을 학습하고, 실험 결과를 비교·시각화하는 단일 사용자 ML 실험 대시보드 |
| **핵심 사용자** | 제조 현장 품질 검사(QC) 담당 ML 엔지니어 / 연구자 |
| **주요 사용 시나리오** | 데이터 폴더 검증 → 전처리 파라미터 설정 → 모델 하이퍼파라미터 설정 → 학습 실행 및 모니터링 → 실험 히스토리 비교 → 이상 영역 시각화 및 결과 내보내기 |
| **UI 유형** | Streamlit 기반 단일 사용자 ML 실험 대시보드 |
| **UI 레이아웃** | `st.set_page_config(layout="wide")` — Wide 레이아웃 |
| **사이드바** | `sidebar_state="expanded"` — 기본 펼침 |

---

## 2. 전체 화면 구조

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

## 3. 탭별 기능 구조

| 탭 | 목적 | 주요 입력 위젯 | 주요 출력 | 저장/연동 파일 |
|---|---|---|---|---|
| **탭1** | MVTec AD 폴더 구조 검증 | `st.text_input` (경로), `st.button` (검증) | 폴더 트리, 클래스별 이미지 수, 썸네일 | `session_state["dataset_meta"]` |
| **탭2** | 전처리 방식 설정·미리보기 및 모델 하이퍼파라미터 설정 | `st.radio` (방식/모델 선택), `st.slider` (파라미터), `st.number_input` (image_size 등) | 원본/필터 미리보기 이미지, 디바이스 정보 | `configs.yaml → preprocessing + model`, `session_state["preprocessing_config"]`, `session_state["model_config"]` |
| **탭3** | 학습 실행 및 실시간 모니터링 | `st.text_input` (실험명), `st.button` (시작/중지) | Progress bar, Loss curve, 실시간 로그 | `experiments/history.json`, `models/`, `logs/`, `configs.yaml` |
| **탭4** | 실험 히스토리 비교 분석 | `st.dataframe` (실험 선택), `st.multiselect` (지표 선택) | ROC curve, Confusion matrix, Anomaly score 분포, 다중 실험 비교 차트 | `experiments/history.json` (읽기/삭제), `models/` (모델 저장) |
| **탭5** | 이상 영역 시각화 및 결과 내보내기 | `st.selectbox` (결함 필터), `st.slider` (threshold), `st.dataframe` (이미지 선택) | Original / GT mask / Anomaly heatmap 3-panel, TP/FP/TN/FN 분류 요약 | `results/` (PNG 다운로드), CSV, ZIP 내보내기 |

---

## 4. 전역 상태 관리 (session_state 스키마)

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

## 5. 설정 및 산출물 파일 구조

```text
smart-qc-dashboard/
├── configs.yaml                    ← 현재 실험 설정 (preprocessing + model + experiment)
├── experiments/
│   └── history.json                ← 실험 기록 누적 저장 (append only)
├── models/
│   └── {experiment_id}/
│       ├── model_state_dict.pth    ← 학습 완료 모델 가중치
│       └── configs.yaml            ← 실험 시점 설정 스냅샷 (불변)
├── logs/
│   └── {experiment_id}.log         ← 학습 로그 (line-buffered)
├── results/                        ← (예약; 현재 코드 상 직접 쓰기 없음)
└── dataset/
    └── imagenet_penalty/           ← EfficientAD ImageNet penalty 이미지
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

## 6. 탭 간 의존성 및 진입 제한

| 탭 | 선행 조건 | 미충족 시 UI 동작 |
|---|---|---|
| **탭1** | 없음 | — |
| **탭2** | `dataset_path` 설정 완료 (`dataset_meta` not None) | `st.warning("먼저 탭1에서 데이터 폴더를 설정해 주세요.")` — 탭 내용 렌더링 중단 |
| **탭3** | `dataset_path`, `preprocessing_config`, `model_config` 모두 완료 | 각각에 대해 `st.warning` 표시, 학습 시작 버튼 비활성화 |
| **탭4** | `experiments` 딕셔너리가 비어있지 않음 | `st.warning("아직 실행된 실험이 없습니다. 탭3에서 학습을 먼저 실행해 주세요.")` |
| **탭5** | `selected_experiment_id` 설정 완료 (`experiments`에 해당 ID 존재) | `st.info("탭4에서 분석할 실험을 먼저 선택해 주세요.")` |

---

## 7. 실험 실행 흐름

```text
[탭1] 데이터셋 경로 입력 → st.button "경로 확인"
         ↓ 검증 통과
      session_state["dataset_path"], ["dataset_meta"] 저장
         ↓
[탭2] 전처리 방식/파라미터 설정
                                  모델/하이퍼파라미터 설정
                                  → st.button "설정 저장" (session)
                                    — preprocessing_config + model_config 동시 저장
                                  → st.button "configs.yaml 저장"
                                    — preprocessing + model 섹션 동시 저장
         ↓
      session_state["preprocessing_config"], session_state["model_config"] 저장
         ↓
[탭3] 실험명 입력 → st.button "학습 시작"
         ↓
      configs.yaml에 experiment 섹션 기록
      TrainingWorker (threading.Thread) 시작
         │
         ├── Queue: {type: "progress", step, total, loss, elapsed}
         │      → st.progress() 갱신, st.plotly_chart() Loss curve 갱신
         │
         ├── Queue: {type: "log", message}
         │      → st.text_area() 로그 갱신 (최대 100줄)
         │
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
      → Confusion matrix, ROC curve, Anomaly score 분포 표시
      → 다중 실험 비교 (st.expander)
      → session_state["selected_experiment_id"] 저장
         ↓
[탭5] Threshold 슬라이더 조정
      → LRU Cache miss 시: load_model_for_inference() → run_inference()
      → Original / GT mask / Anomaly heatmap 3-panel 표시
      → CSV / PNG / ZIP 내보내기
```

---

## 8. 탭1 — 데이터 폴더 와이어프레임

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
│  │(150px) │ │(150px) │ │        │ │        │              │
│  └────────┘ └────────┘ └────────┘ └────────┘              │
│   col1       col2       col3       col4                     │
└─────────────────────────────────────────────────────────────┘
```

**주요 session_state 흐름:**
- `st.text_input` → `session_state["input_dataset_path"]` (위젯 내부)
- 검증 통과 → `session_state["dataset_path"]`, `session_state["dataset_meta"]` 저장

**`dataset_meta` 스키마:**
```python
{
    "dataset_path": str,
    "train_good_count": int,
    "test_counts": dict[str, int],      # {class_name: count}
    "gt_counts": dict[str, int],
    "total_test_count": int,
    "channels": 1 | 3,
    "defect_classes": list[str],        # "good" 포함
    "supported_formats": list[str],
    "has_invalid_files": bool,
    "_invalid_file_count": int,
}
```

---

## 9. 탭2 — 전처리 및 모델 설정 와이어프레임

구 탭2(전처리 설정)와 구 탭3(모델 파라미터)를 통합한 신 탭2 와이어프레임.  
설계 문서(docs/UI/탭2_02.txt, 탭3_02.txt) 기준과 실제 코드 구현 비교:

| 요소 | 설계 문서 | 실제 구현 |
|---|---|---|
| CLAHE clipLimit 슬라이더 | 미언급 | ✅ 구현됨 (0.1–40.0, default 2.0) |
| HE 파라미터 | 미언급 | `st.info("HE는 별도 파라미터 없음")` |
| 전처리 미리보기 열 수 | col1(1/2), col2(1/2) | `st.columns(3)` — 3열 (코드 기준 우선) |
| 디바이스 표시 | `ℹ️ 현재 디바이스: CUDA (RTX xxxx)` | 동일 (`device_info` 첫 방문 1회 감지) |
| 공통 설정 2열 | batch_size, random_seed | `st.columns(2)` — image_size는 전처리 영역 소유 |
| EfficientAD 고급 설정 | `st.expander` 접힘 | ✅ 구현됨 |
| PatchCore 고급 설정 | `st.expander` 접힘 | ✅ 구현됨 |
| neighbourhood_kernel `select_slider` | `◀ 1 3 ●5 7 9 ▶` | ✅ `st.select_slider(options=[1,3,5,7,9])` |
| "configs.yaml 불러오기" 동작 | 버튼 클릭 → 즉시 로드 | 클릭 → 경로 text_input + 확인/취소 버튼 출현 |
| 저장 버튼 | 전처리·모델 각각 별도 버튼 | 통합 `설정 저장` 버튼 1개 (primary) |

> **widget key 설계 주의**: 구 탭2와 구 탭3의 widget key가 단일 탭 내에서 공존하므로, 기존 key(`t2_*`, `tab3_*`) 재사용 금지. 통합 탭2 전용 key 체계로 재설계 필요 (코드 상에서 확인 필요).

```text
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ 탭2. 전처리 및 모델 설정                    │
│                                                             │
│  ⚠️  먼저 탭1에서 데이터 폴더를 설정해 주세요.               │
│  (dataset_meta == None 일 때 여기서 중단)                   │
│                                                             │
│  ════════════════════════════════════════════════════════   │
│  【전처리 영역】                                             │
│  ════════════════════════════════════════════════════════   │
│                                                             │
│  전처리 방식:                                                │
│  ◉ 없음  ○ Homomorphic  ○ HE  ○ CLAHE                      │
│  ← st.radio(key="tab2_method_label", horizontal=True)       │
│  ──────────────────────────────────────────────────────     │
│                                                             │
│  [Homomorphic 선택 시]                                      │
│  sigma    [━━━●──────────────] 10.0  key:"tab2_sigma"      │
│           (0.1 ~ 50.0, step 0.1)                           │
│  gamma_H  [━━━━━━━●──────────] 1.5   key:"tab2_gamma_h"    │
│           (1.0 ~ 3.0, step 0.1)                            │
│  gamma_L  [━━━●──────────────] 0.5   key:"tab2_gamma_l"    │
│           (0.1 ~ 1.0, step 0.05)                           │
│  ☑ normalize  ← st.checkbox(key="tab2_normalize")          │
│                                                             │
│  [CLAHE 선택 시]                                            │
│  clipLimit [━━━●──────────────] 2.0  key:"tab2_clip_limit" │
│            (0.1 ~ 40.0, step 0.1)                          │
│                                                             │
│  [HE 선택 시]                                               │
│  ℹ️  HE는 별도 파라미터가 없습니다.                          │
│                                                             │
│  ──────────────────────────────────────────────────────     │
│  이미지 크기 및 정규화  (전처리 영역 단독 소유)               │
│                                                             │
│  image_size: [  256  ▲▼]  key:"tab2_image_size"            │
│              (32 ~ 1024, step 32)                           │
│  ❌ 32의 배수가 아니면 st.error() 표시                       │
│                                                             │
│  정규화: ◉ ImageNet  ○ 커스텀                               │
│  ← st.radio(key="tab2_norm_label", horizontal=True)         │
│                                                             │
│  [커스텀 선택 시]                                           │
│  mean: [____________________________] key:"tab2_mean"       │
│  std:  [____________________________] key:"tab2_std"        │
│                                                             │
│  ──────────────────────────────────────────────────────     │
│  전처리 미리보기  ← st.columns(3)                           │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │              │ │              │ │              │        │
│  │  원본 이미지  │ │  필터 적용 후 │ │  (3번째 열)  │        │
│  │              │ │              │ │              │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│   col1             col2             col3                    │
│   ← st.image()     ← st.image()     ← (코드 확인 필요)     │
│                                                             │
│  ════════════════════════════════════════════════════════   │
│  【모델 영역】                                               │
│  ════════════════════════════════════════════════════════   │
│                                                             │
│  ℹ️  현재 디바이스: CUDA (RTX xxxx), VRAM: xx.x GB          │
│  (또는: ℹ️  현재 디바이스: CPU)                              │
│  ← device_info 첫 방문 1회 감지 후 표시 (탭2 Write)         │
│                                                             │
│  모델 선택:  ◉ EfficientAD  ○ PatchCore                    │
│  ← st.radio(key="tab2_model_label", horizontal=True)        │
│  ──────────────────────────────────────────────────────     │
│  공통 설정  ← st.columns(2)                                 │
│                                                             │
│  ┌────────────────────────┐ ┌────────────────────────┐     │
│  │  batch_size            │ │  random_seed           │     │
│  │ [   16  ▲▼]           │ │ [   42  ▲▼]           │     │
│  │ key:"tab2_batch_size"  │ │ key:"tab2_random_seed" │     │
│  └────────────────────────┘ └────────────────────────┘     │
│   col1                       col2                          │
│                                                             │
│  ── EfficientAD 선택 시 ────────────────────────────────── │
│  EfficientAD 파라미터  ← st.columns(2)                      │
│                                                             │
│  ┌──────────────────────────┐ ┌──────────────────────────┐  │
│  │ model_size:              │ │ learning_rate: [1e-4 ▲▼] │  │
│  │ ◉ small  ○ medium        │ │  key:"tab2_ead_lr"       │  │
│  │ key:"tab2_ead_model_size" │ │ weight_decay: [1e-4 ▲▼]  │  │
│  │ train_steps: [70000 ▲▼]  │ │  key:"tab2_ead_wd"       │  │
│  │ key:"tab2_ead_train_steps"│ │ ☐ padding               │  │
│  │ optimizer: [adam      ▼] │ │  key:"tab2_ead_padding"  │  │
│  │ key:"tab2_ead_optimizer"  │ │                          │  │
│  │ out_channels: [384    ▼] │ │                          │  │
│  │ key:"tab2_ead_out_channels"│ │                         │  │
│  └──────────────────────────┘ └──────────────────────────┘  │
│   col1                         col2                         │
│                                                             │
│  AE Loss 비중 (ae_loss_weight)                              │
│  [━━━━━━━━━━━●───────────────────] 0.5                     │
│  key:"tab2_ead_ae_weight"  (0 ~ 1, step 0.01)              │
│                                                             │
│  ▶ 고급 설정 (Advanced Settings)  ← st.expander            │
│  ┌──────────────────────────┐ ┌──────────────────────────┐  │
│  │ autoencoder_lr  [▲▼]    │ │ lr_decay_factor [●───]   │  │
│  │ key:"tab2_ead_ae_lr"     │ │ key:"tab2_ead_decay_f"   │  │
│  │ autoencoder_wd  [▲▼]    │ │ scheduler: [StepLR    ▼] │  │
│  │ key:"tab2_ead_ae_wd"     │ │ key:"tab2_ead_sched"     │  │
│  │ lr_decay_epochs [▲▼]    │ │ ☐ use_imagenet_penalty   │  │
│  │ key:"tab2_ead_decay_ep"  │ │ key:"tab2_ead_use_penalty"│ │
│  │                          │ │ penalty_batch_size [▲▼]  │  │
│  │                          │ │ key:"tab2_ead_pen_bs"    │  │
│  └──────────────────────────┘ └──────────────────────────┘  │
│   adv1                         adv2                         │
│                                                             │
│  ── PatchCore 선택 시 ──────────────────────────────────── │
│  PatchCore 파라미터  ← st.columns(2)                        │
│                                                             │
│  ┌──────────────────────────┐ ┌──────────────────────────┐  │
│  │ backbone:                │ │ coreset_sampling_ratio:  │  │
│  │ [wide_resnet50_2      ▼] │ │ [━━━●──────────────] 0.1 │  │
│  │ key:"tab2_pc_backbone"   │ │ key:"tab2_pc_coreset"    │  │
│  │ pretrained:              │ │ neighbourhood_kernel:    │  │
│  │ ◉ torchvision ○ 로컬경로 │ │ ◀ 1  3  ●5  7  9 ▶      │  │
│  │ key:"tab2_pc_pretrained" │ │ key:"tab2_pc_kernel"     │  │
│  │ [로컬경로 선택 시]        │ │ (st.select_slider)       │  │
│  │ [경로 text_input]        │ │                          │  │
│  └──────────────────────────┘ └──────────────────────────┘  │
│   col1                         col2                         │
│                                                             │
│  ▶ 고급 설정 (Advanced Settings)  ← st.expander            │
│  ┌──────────────────────────┐ ┌──────────────────────────┐  │
│  │ max_train: [1000    ▲▼]  │ │ knn:      [9    ▲▼]      │  │
│  │ key:"tab2_pc_max_train"  │ │ key:"tab2_pc_knn"        │  │
│  │ (100 ~ 10000)             │ │ top_k_ratio: [●───] 0.1  │  │
│  │                          │ │ key:"tab2_pc_top_k"      │  │
│  └──────────────────────────┘ └──────────────────────────┘  │
│   adv1                         adv2                         │
│                                                             │
│  ──────────────────────────────────────────────────────     │
│  Threshold 설정                                              │
│  방식: ◉ Percentile (백분위)  ○ Absolute (절대값)            │
│  ← st.radio(key="tab2_threshold_label", horizontal=True)    │
│                                                             │
│  [Percentile 선택 시]                                       │
│  [━━━━━━━━━━━━━━━━━━━●──────────────] 95.0                 │
│  key:"tab2_threshold_pct"  (0 ~ 100, step 0.5)             │
│                                                             │
│  [Absolute 선택 시]                                         │
│  [━━━━━━━━━━━●──────────────] 0.5                          │
│  key:"tab2_threshold_abs"  (0 ~ 1, step 0.01)              │
│                                                             │
│  ← st.columns(2)                                            │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │  예상 정상 판정 비율 │  │  예상 결함 판정 비율 │            │
│  │      95.0%         │  │       5.0%         │            │
│  └────────────────────┘  └────────────────────┘            │
│   col1 (st.metric)        col2 (st.metric)                 │
│                                                             │
│  ════════════════════════════════════════════════════════   │
│  【설정 저장】  ← st.columns(3)                              │
│  ════════════════════════════════════════════════════════   │
│                                                             │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐  │
│  │   설정 저장      │ │configs.yaml │ │   configs.yaml   │  │
│  │  🔵 (primary)   │ │     저장     │ │     불러오기      │  │
│  └─────────────────┘ └─────────────┘ └──────────────────┘  │
│  key:"tab2_btn_save" key:"tab2_btn_yaml_save"               │
│                      key:"tab2_btn_yaml_load"               │
│   col1 (1/3)         col2 (1/3)       col3 (1/3)           │
│                                                             │
│  [불러오기 클릭 시 조건부 렌더링]                             │
│  YAML 경로: [./configs.yaml___________] key:"tab2_load_path"│
│  [확인] key:"tab2_btn_load_confirm"                         │
│  [취소] key:"tab2_btn_load_cancel"                          │
└─────────────────────────────────────────────────────────────┘
```

**"설정 저장" 버튼 동작**: `preprocessing_config` + `model_config` 동시 저장.  
**"configs.yaml 저장" 버튼 동작**: `preprocessing` 섹션 + `model` 섹션 동시 저장.  
**"configs.yaml 불러오기" 버튼 동작**: `preprocessing` 섹션 + `model` 섹션 동시 로드.

**`preprocessing_config` 스키마:**
```python
{
    "method": "none" | "homomorphic" | "he" | "clahe",
    "resize_mode": "padding",   # 항상 "padding"
    "image_size": int,          # 전처리 영역 단독 소유
    "normalization": "imagenet" | "custom",
    "mean": [float, float, float],
    "std": [float, float, float],
    "params": dict | None,
}
```

**`model_config` 스키마:**
```python
{
    "model_type": "efficientad" | "patchcore",
    "batch_size": int,
    "random_seed": int,
    "threshold_method": "percentile" | "absolute",
    "threshold_value": float,
    "params": dict,   # 모델별 파라미터
}
```

---

## 10. 탭3 — 학습 실행 와이어프레임

```text
┌─────────────────────────────────────────────────────────────┐
│  🚀 탭3. 학습                                               │
│                                                             │
│  [선행 조건 미충족 시]                                       │
│  ⚠️  먼저 탭1/탭2 설정을 완료해 주세요.                  │
│  ← 각 미완료 항목에 대해 별도 st.warning()                   │
│                                                             │
│  ── Idle 상태 ─────────────────────────────────────────── │
│                                                             │
│  실험 이름                                                   │
│  [__________________________________]  (max 64자)           │
│  key:"tab4_experiment_name"                                 │
│  placeholder: "예: EfficientAD CLAHE clip2.0 실험"          │
│                                                             │
│  ▶ 현재 학습 설정 요약  ← st.expander                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  ← st.columns(2)                                   │    │
│  │  [모델/전처리 설정 요약]    [디바이스/데이터셋 정보]   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ⚠️  디스크 공간 100MB 미만: 학습 시작 차단 st.error()       │
│  ⚠️  디스크 공간 100~500MB: st.warning()                    │
│  ⚠️  use_imagenet_penalty=True이나 이미지 없음: st.error()   │
│                                                             │
│  [학습 시작]  ← st.button (primary)                         │
│                                                             │
│  ── Running 상태 (current_run_status == "running") ──────  │
│                                                             │
│  ℹ️  학습 중... {step}/{total} steps | Loss: {loss:.4f} | {elapsed} │
│                                                             │
│  Progress                                                   │
│  ████████████░░░░░░░░  65%   ← st.progress()               │
│  {step} / {total} steps                                    │
│                                                             │
│  Loss Curve                                                 │
│  ┌──────────────────────────────────────────────────┐      │
│  │                                                  │      │
│  │         ← st.plotly_chart() 실시간 갱신          │      │
│  │           (x: step, y: loss, 선형 차트)           │      │
│  │                                                  │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  실시간 로그                                                 │
│  ┌──────────────────────────────────────────────────┐      │
│  │  [Epoch 1/10] Loss: 0.3421 ...                  │      │
│  │  Validation running...                          │      │
│  │  ...                                            │      │
│  └──────────────────────────────────────────────────┘      │
│  ← st.text_area(disabled=True, key="tab4_log_area")        │
│    (최근 50줄 표시, 최대 100줄 누적)                         │
│                                                             │
│  [학습 중지]  ← st.button (secondary)                       │
│                                                             │
│  ── 완료/중단 상태 ──────────────────────────────────────  │
│  [완료] ✅ st.success("학습 완료")                           │
│  [중단] ℹ️  st.info("학습이 중단되었습니다. 히스토리에 기록됨") │
│  [오류] ❌ st.error(traceback)                               │
└─────────────────────────────────────────────────────────────┘
```

**백그라운드 학습 스레드 (TrainingWorker) 큐 메시지 타입:**

| `type` 값 | 데이터 | UI 반응 |
|---|---|---|
| `"progress"` | `step, total, loss, elapsed` | progress bar + loss curve 갱신 |
| `"log"` | `message` | 로그 텍스트 추가 |
| `"completed"` | `y_true, anomaly_scores, anomaly_maps, image_paths, model, duration_seconds` | 결과 저장 → st.success |
| `"error"` | `exception, traceback` | st.error(traceback) |
| `"stopped"` | `step` | st.info(중단 메시지) → history에 "중단" 기록 |

---

## 11. 탭4 — 실험 히스토리 와이어프레임

```text
┌─────────────────────────────────────────────────────────────┐
│  📊 탭4. 실험 히스토리                                       │
│                                                             │
│  [실험 없을 시]                                              │
│  ⚠️  아직 실행된 실험이 없습니다. 탭3에서 학습을 먼저 실행해 주세요. │
│                                                             │
│  실험 목록  (history.json 매 렌더 시 읽기)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  실험명 | 모델 | 파라미터 | Acc | F1 | AUC | 실행시각 │  │
│  │  ──── row1 (completed) ────────────────────────────  │  │
│  │  ──── row2 (중단 → 회색 텍스트 스타일) ─────────────  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ← st.dataframe(selection_mode="single-row",               │
│                 on_select="rerun", key="t5_table")          │
│                                                             │
│  [🗑 실험 삭제]  ← st.button(secondary, key="t5_delete_btn") │
│  (선택 없으면 disabled=True)                                 │
│                                                             │
│  [삭제 확인 시 조건부 렌더링]                                 │
│  ⚠️  "실험 '{name}'을 삭제하시겠습니까? 모델 파일도 삭제됩니다." │
│  ← st.columns([1, 1, 6])                                    │
│  [확인] key:"t5_delete_confirm"   [취소] key:"t5_delete_cancel" │
│                                                             │
│  ── 선택된 실험 상세 (status == "completed" 일 때만) ─────  │
│                                                             │
│  성능 지표  ← st.columns(4)                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Accuracy │ │Precision │ │  Recall  │ │    F1    │      │
│  │  0.954   │ │  0.923   │ │  0.981   │ │  0.951   │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│  ← st.metric()                                             │
│                                                             │
│  차트  ← st.columns(3)                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │ Confusion    │ │ ROC Curve    │ │ Anomaly Score    │   │
│  │ Matrix       │ │ (AUC: 0.98)  │ │ Distribution     │   │
│  │ (Heatmap)    │ │              │ │ (Histogram)      │   │
│  └──────────────┘ └──────────────┘ └──────────────────┘   │
│  ← st.plotly_chart()×3                                     │
│                                                             │
│  모델 저장                                                   │
│  경로: [./models/exp_id_________] key:"t5_save_path"       │
│  [💾 모델 저장]  ← st.button(primary, key="t5_save_btn")   │
│                                                             │
│  ── 다중 실험 비교 (completed 실험 2개 이상일 때) ────────   │
│                                                             │
│  ▶ 다중 실험 비교 차트  ← st.expander                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  비교할 실험 선택: ☑ exp1  ☑ exp2  ☐ exp3  ...       │  │
│  │  ← st.columns(min(len(completed), 4)) → st.checkbox  │  │
│  │                                                      │  │
│  │  지표 선택: [Accuracy ×] [F1 ×] +추가               │  │
│  │  ← st.multiselect(key="t5_cmp_metrics",              │  │
│  │                   default=["Accuracy","F1"])          │  │
│  │                                                      │  │
│  │  차트 유형: ◉ 막대 차트  ○ 레이더 차트                 │  │
│  │  ← st.radio(key="t5_cmp_type", horizontal=True)      │  │
│  │  ℹ️  레이더 차트: 2개 이상 지표 필요                   │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │         ← st.plotly_chart() 비교 차트          │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. 탭5 — 이상 영역 시각화 와이어프레임

```text
┌─────────────────────────────────────────────────────────────┐
│  🗺️ 탭5. 이상 영역 시각화                                   │
│                                                             │
│  [selected_experiment_id == None 일 때]                     │
│  ℹ️  탭4에서 분석할 실험을 먼저 선택해 주세요.               │
│                                                             │
│  [중단 상태 실험 선택 시]                                    │
│  ⚠️  이 실험은 중단 상태입니다.                              │
│                                                             │
│  ── 필터 및 제어 ──────────────────────────────────────── │
│                                                             │
│  결함 유형 필터:  [전체 ▼]                                   │
│  ← st.selectbox(key="tab6_class_filter",                    │
│                 options=["전체"] + defect_classes)          │
│                                                             │
│  Threshold                                                  │
│  [━━━━━━━━━━━━━━━━━━━●──────────────] 0.542                │
│  ← st.slider(0 ~ slider_max, step=0.001, format="%.3f")    │
│    (session_state["anomaly_map_threshold"]로 유지)           │
│                                                             │
│  점수 요약  ← st.columns(2)                                  │
│  ┌──────────────────┐ ┌──────────────────┐                 │
│  │  Max Score       │ │  Mean Score      │                 │
│  │  0.873           │ │  0.412           │                 │
│  └──────────────────┘ └──────────────────┘                 │
│  ← st.metric()×2                                           │
│                                                             │
│  분류 현황  ← st.columns(4)                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
│  │   TP   │ │   FP   │ │   TN   │ │   FN   │              │
│  │   12   │ │    3   │ │   18   │ │    2   │              │
│  └────────┘ └────────┘ └────────┘ └────────┘              │
│  ← st.metric()×4                                           │
│                                                             │
│  ── 이미지 목록 ──────────────────────────────────────── │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 이미지명 | 결함 유형 | Anomaly Score | 판정 | GT일치 | 오분류 │  │
│  │ img001   │  crack    │    0.723     │ 결함 │   ✓   │      │  │
│  │ img002   │  good     │    0.201     │ 정상 │   ✓   │      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ← st.dataframe(selection_mode="single-row",               │
│                 on_select="rerun", key="tab6_image_table")  │
│                                                             │
│  ── 이미지 선택 시 3-패널 시각화 ─────────────────────────  │
│                                                             │
│  ← st.columns(3)                                            │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │
│  │                  │ │                  │ │                  │ │
│  │   원본 이미지     │ │   GT 마스크       │ │   Anomaly        │ │
│  │   ← st.image()   │ │   (없으면 빈 화면)│ │   Heatmap        │ │
│  │                  │ │   ← st.image()   │ │   + 컨투어 오버레이│ │
│  │                  │ │                  │ │   ← st.image()   │ │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘ │
│   col1                col2                col3                 │
│                                                               │
│  지표  ← st.columns(3)                                        │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐    │
│  │ Anomaly Score  │ │   Threshold    │ │      판정       │    │
│  │    0.723       │ │    0.542       │ │      결함       │    │
│  └────────────────┘ └────────────────┘ └────────────────┘    │
│  ← st.metric()×3                                             │
│                                                               │
│  [⬇ PNG 저장]  ← st.download_button(key=f"tab6_dl_{stem}")  │
│                                                               │
│  ── 내보내기 ────────────────────────────────────────────── │
│                                                               │
│  ← st.columns(2)                                              │
│  ┌────────────────────────┐ ┌────────────────────────────┐   │
│  │  [⬇ CSV 내보내기]      │ │  [ZIP 준비]                │   │
│  │  ← st.download_button  │ │  ← st.button               │   │
│  │  key:"tab6_csv_export" │ │  key:"tab6_zip_prepare"    │   │
│  │                        │ │                            │   │
│  │                        │ │  [ZIP 준비 완료 시]         │   │
│  │                        │ │  [⬇ ZIP 다운로드]          │   │
│  │                        │ │  key:"tab6_zip_download"   │   │
│  └────────────────────────┘ └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Anomaly Map LRU 캐시 (`cache_manager.py`):**
- 최대 3개 항목 유지
- Key 형식: `"_anomaly_maps_{exp_id}"`
- 캐시 미스 시: `load_model_for_inference()` → `run_inference()` → 캐시 저장
- 추방(Eviction): `cached_at` 타임스탬프 기준 가장 오래된 항목 제거

---

## 13. 사이드바 (components/sidebar.py)

```text
┌─────────────────────────────┐
│  Smart QC Dashboard         │  ← st.title()
│  ─────────────────────────  │
│  데이터셋                    │  ← st.subheader()
│  경로: /app/dataset/screw   │  ← st.caption()
│  결함 클래스: crack, scratch │  ← st.caption()
│                             │
│  ┌──────────┐ ┌──────────┐  │  ← st.columns(2)
│  │  학습    │ │  테스트  │  │
│  │  240장   │ │  85장    │  │  ← st.metric()×2
│  └──────────┘ └──────────┘  │
│                             │
│  디바이스                    │  ← st.subheader()
│  ✅ CUDA: RTX 3090          │  ← st.success() (CUDA 시)
│  (또는: ℹ️ CPU 모드)         │  ← st.info() (CPU 시)
│                             │
│  ⚠️ 학습 실행 중...          │  ← st.info() (running 시)
│                             │
│  현재 설정                   │  ← st.subheader()
│  모델: EfficientAD          │
│  전처리: Homomorphic         │
│  image_size: 256            │
└─────────────────────────────┘
```

**참조하는 session_state:**
- `dataset_meta` → 데이터셋 정보 표시
- `device_info` → 디바이스 정보 표시
- `model_config` → 현재 설정 표시
- `current_run_status` → 학습 상태 경고 표시

---

## 14. 상태 및 설정 관리 구조

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

## 15. 탭 간 의존성 상세

```text
탭1 ──[dataset_path, dataset_meta]─────────────────→ 탭2
탭2 ──[preprocessing_config, model_config]─────────→ 탭3
탭3 ──[experiments, selected_exp_id]───────────────→ 탭4
탭4 ──[selected_experiment_id]─────────────────────→ 탭5
```

- **단방향 의존성**: 이전 탭 완료 없이 이후 탭 진입 가능하나 기능 차단
- **탭4→탭5 연결**: `st.dataframe` 행 선택 시 `session_state["selected_experiment_id"]` 갱신 → 탭5에서 읽어 추론 수행

---

## 16. 핵심 인터랙션

| 위젯/요소 | 인터랙션 타입 | 동작 |
|---|---|---|
| 탭1 "경로 확인" 버튼 | `st.button` click → rerun | MVTec AD 폴더 구조 4단계 검증, `dataset_meta` 저장 |
| 탭2 전처리 방식 radio | `st.radio` change → rerun | 방식별 파라미터 슬라이더 조건부 표시 |
| 탭2 미리보기 | 파라미터 변경 → rerun → `st.image` 갱신 | 실시간 원본/필터 비교 |
| 탭2 모델 선택 radio | `st.radio` change → rerun | EfficientAD/PatchCore 파라미터 섹션 전환 |
| 탭2 threshold 슬라이더 | `st.slider` change → rerun → `st.metric` 갱신 | 예상 정상/결함 비율 실시간 표시 |
| 탭3 "학습 시작" 버튼 | `st.button` click → TrainingWorker 시작 → rerun loop | 백그라운드 학습 시작 |
| 탭3 Progress 갱신 | `_result_queue` poll → rerun | progress bar + loss curve + 로그 실시간 갱신 |
| 탭3 "학습 중지" 버튼 | `st.button` click → `_stop_event.set()` | Worker에 중지 신호 전달 |
| 탭4 실험 선택 | `st.dataframe` row select → rerun | 상세 차트 렌더링 + `selected_experiment_id` 저장 |
| 탭4 다중 비교 | `st.checkbox` + `st.multiselect` + `st.radio` → rerun | Plotly 비교 차트 동적 갱신 |
| 탭5 threshold 슬라이더 | `st.slider` change → rerun | TP/FP/TN/FN 재계산, 테이블 판정 갱신 |
| 탭5 이미지 선택 | `st.dataframe` row select → rerun | 3-패널 시각화 렌더링 |

---

## 17. 상태 기반 UI 반응

| 상태 | 트리거 | UI 반응 |
|---|---|---|
| 로딩 중 | 추론 실행, 캐시 미스 | `st.spinner()` (구현 여부 불명확 — 코드 상 직접 확인 필요) |
| 학습 중 | `current_run_status == "running"` | progress bar + loss curve + log 활성화, "학습 중지" 버튼 표시 |
| 오류 | Exception 발생 | `st.error(traceback)` |
| 데이터 없음 | `experiments == {}` | `st.warning("실험 없음")` |
| 성공 | 설정 저장 완료 | `st.success()` (탭2 설정 저장, 탭4 모델 저장 등) |
| 유효하지 않은 설정 | image_size % 32 != 0 | `st.error()` + 저장 버튼 차단 |

---

## 18. 예외 처리 UX

| 상황 | 발생 위치 | UI 처리 |
|---|---|---|
| CUDA 미감지 | 탭2 `device_info` 감지 | 사이드바 `st.info("CPU 모드")`, 탭2 ℹ️ "현재 디바이스: CPU" |
| dataset 구조 오류 | 탭1 경로 검증 | `st.error("MVTec AD 형식의 폴더 구조가 아닙니다.")` |
| image_size 비배수 | 탭2 | `st.error("32의 배수로 입력하세요")` + 저장 차단 |
| 커스텀 정규화 파싱 실패 | 탭2 mean/std 입력 | `st.error("유효하지 않은 형식입니다.")` |
| configs.yaml 로드 실패 | 탭2 불러오기 | 코드 상에서 확인되지 않음 (예외 처리 구현 여부 불명) |
| 디스크 공간 부족 (<100MB) | 탭3 학습 시작 전 | `st.error()` + 학습 시작 차단 |
| 디스크 공간 부족 (100~500MB) | 탭3 학습 시작 전 | `st.warning()` |
| ImageNet penalty 이미지 없음 | 탭3 (use_imagenet_penalty=True 시) | `st.error()` + 학습 시작 차단 |
| ImageNet penalty 이미지 <1000 | 탭2 고급 설정, 탭3 | `st.warning()` |
| 실험 삭제 실패 | 탭4 삭제 확인 후 | `st.error()` |
| 모델 저장 실패 | 탭4 모델 저장 | `st.error()` |
| 추론 오류 | 탭5 캐시 미스 → 추론 | `st.error()` |
| 모델 경로 없음 | 탭5 (중단 실험) | `st.warning("model_path 없음")` |

---

## 19. Streamlit 렌더링 및 상태 전략

| 항목 | 구현 내용 |
|---|---|
| **세션 상태 초기화** | `app.py` 진입 시 `init_session_state()` 1회 실행 |
| **백그라운드 학습** | `threading.Thread` (TrainingWorker) 사용 |
| **Worker → UI 통신** | `queue.Queue` 폴링 방식 — 탭3 렌더링 루프마다 `_result_queue.get_nowait()` |
| **rerun 발생 지점** | 모든 `st.button`, `st.dataframe` 행 선택, `st.radio`/`st.slider` 변경 시 |
| **캐싱 전략** | `st.cache_data` / `st.cache_resource` 사용 여부: 코드 상에서 확인되지 않음. Anomaly map은 session_state 기반 커스텀 LRU 캐시 사용 |
| **Plotly 실시간 갱신** | rerun 시 `_loss_history` session_state 데이터로 매번 재생성 (`st.plotly_chart`) |
| **탭 간 상태 공유** | `session_state`를 통한 단방향 전달 (탭1→2→3→4→5) |
| **YAML 원자적 쓰기** | `tmpfile → rename` 방식 (`config_manager.py`) |
| **history.json 원자적 쓰기** | `tmpfile → rename` 방식 (`storage.py`) |
| **모델 저장 프로토콜** | 3단계 순차 저장: `.pth` → `configs.yaml snapshot` → `history.json` |

---

## 20. 프로젝트 파일 구조 전체

```text
smart-qc-dashboard/
├── app.py                          # Streamlit 진입점
├── configs.yaml                    # 현재 실험 설정 (수정 가능)
├── components/
│   ├── __init__.py
│   └── sidebar.py                  # 사이드바 렌더링
├── tabs/
│   ├── __init__.py
│   ├── tab1_data_folder.py         # 데이터셋 검증
│   ├── tab2_config.py              # 탭2 (전처리 및 모델 설정)
│   ├── tab3_training.py            # 탭3 (학습 실행/모니터링)
│   ├── tab4_history.py             # 탭4 (실험 히스토리)
│   └── tab5_anomaly_map.py         # 탭5 (이상 영역 시각화)
├── utils/
│   ├── __init__.py
│   ├── env_init.py                 # 필수 디렉토리 초기화
│   ├── session_state_init.py       # 세션 상태 스키마 정의
│   ├── config_manager.py           # YAML 설정 I/O
│   ├── storage.py                  # 실험 기록/모델 파일 관리
│   ├── cache_manager.py            # Anomaly map LRU 캐시
│   ├── messages.py                 # UI 메시지 상수
│   ├── image_utils.py              # 이미지 처리 유틸리티
│   ├── metrics.py                  # 성능 지표 계산
│   ├── model_factory.py            # 모델 생성/추론
│   ├── mvtec_dataset.py            # MVTec AD 데이터셋 로더
│   ├── training_worker.py          # 백그라운드 학습 스레드
│   ├── path_validator.py           # 경로 검증
│   └── dataset_scanner.py          # 데이터셋 스캔
├── experiments/
│   └── history.json                # 실험 기록 (누적 append)
├── models/
│   └── {experiment_id}/
│       ├── model_state_dict.pth
│       └── configs.yaml
├── logs/
│   └── {experiment_id}.log
├── results/                        # (예약 디렉토리; 현재 직접 쓰기 없음)
└── dataset/
    └── imagenet_penalty/           # EfficientAD ImageNet penalty 이미지
```

---

## 21. 반응형 및 레이아웃 특성

| 항목 | 내용 |
|---|---|
| **레이아웃** | `layout="wide"` — 전체 브라우저 너비 사용 |
| **대상 환경** | 데스크탑 브라우저 (ML 실험 환경 기준) |
| **모바일** | Streamlit wide layout 특성상 가독성 저하 예상 — 공식 지원 범위 아님 |
| **탭 내 주요 컬럼 분할** | 2열, 3열, 4열 혼용 (`st.columns()`) |
| **Plotly 차트 크기** | `use_container_width=True` 여부: 코드 상에서 확인되지 않음 |

---

*이 문서는 실제 소스 코드 역설계를 기반으로 작성되었습니다. "코드 상에서 확인되지 않음" 표기 항목은 추가 코드 검토가 필요합니다.*

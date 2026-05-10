# 07. Backend Service Design

> **참조 문서**: `04_System_Architecture.md` §B.5 (비동기 처리 아키텍처), `06_API_Specification.md` §5 (tab4 소비 알고리즘), `08_AI_ML_Integration.md` §B.4~B.6 (학습 구현)
> **버전**: v1.0
> **작성일**: 2026-05-09
> **목적**: utils/ 서비스 레이어의 비즈니스 로직 흐름을 구현 가능한 수준으로 명세한다. 아래 4가지 영역이 이 문서에서 확정된다:
> 1. experiment_id 생성 규칙 (코드 포함)
> 2. 학습 시작 버튼 핸들러 전체 흐름
> 3. TrainingWorker 생성자 인스턴스 변수 명세
> 4. 탭6 이상 감지 서비스 (모델 재로드 → 추론 → 캐시 → 시각화) 파이프라인
>
> **역할 분리**: 04.B.5의 스레드 모델, 04.B.7의 상태 머신, 06.§5의 tab4 polling loop, 08.B.4~B.6의 학습 루프 구현은 이 문서에서 반복하지 않고 참조만 한다.

---

## 목차

1. [서비스 책임 범위](#1-서비스-책임-범위)
2. [experiment_id 생성 서비스](#2-experiment_id-생성-서비스)
3. [학습 시작 서비스](#3-학습-시작-서비스)
4. [TrainingWorker 생성자 명세](#4-trainingworker-생성자-명세)
5. [학습 중 Progress 보고 주기](#5-학습-중-progress-보고-주기)
6. [학습 완료/중단 후처리 흐름](#6-학습-완료중단-후처리-흐름)
7. [탭6 이상 감지 서비스](#7-탭6-이상-감지-서비스)
8. [단일 워커 보장](#8-단일-워커-보장)
9. [메모리 관리 규칙](#9-메모리-관리-규칙)
10. [서비스 데이터 흐름 요약](#10-서비스-데이터-흐름-요약)
11. [구현 체크리스트](#11-구현-체크리스트)

---

## 1. 서비스 책임 범위

### 이 문서가 확정하는 것

| 서비스 | 위치 | 핵심 결정 사항 |
|--------|------|----------------|
| ID 생성 | `utils/training_worker.py` 또는 `tabs/tab4_training.py` | experiment_id 생성 코드 확정 |
| 학습 시작 | `tabs/tab4_training.py._handle_start_training()` | 버튼 클릭부터 worker.start()까지 전체 순서 |
| TrainingWorker 초기화 | `utils/training_worker.py.TrainingWorker.__init__()` | 생성자 파라미터와 인스턴스 변수 전체 |
| 이상 감지 서비스 | `tabs/tab6_anomaly_map.py` + `utils/model_factory.py` | 모델 재로드·추론·캐시·시각화 파이프라인 |

### 이 문서가 다루지 않는 것 (참조 문서)

| 내용 | 참조 |
|------|------|
| 비동기 스레드 모델 설계 | `04_System_Architecture.md §B.5` |
| 학습 상태 머신 | `04_System_Architecture.md §B.7` |
| tab4 Queue 소비 polling loop | `06_API_Specification.md §5` |
| stop_event 경쟁 조건 처리 | `06_API_Specification.md §6` |
| EfficientAD / PatchCore 학습 루프 코드 | `08_AI_ML_Integration.md §B.4, B.5, B.6` |
| 3단계 원자성 저장 프로토콜 | `05_Data_Model_and_Storage_Strategy.md §6` |

---

## 2. experiment_id 생성 서비스

### 2.1 생성 규칙

`00_Global_Context_Document.md §8` R-NAMING-03, R-ID-01 기준:

```
형식: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_소문자_16진수}
예:   efficientad_20260509_143022_3b9f
      patchcore_20260509_143022_a1c4
```

### 2.2 생성 코드 (확정)

```python
# tabs/tab4_training.py 내부 또는 utils/training_worker.py

import uuid
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

def generate_experiment_id(model_type: str) -> str:
    """
    R-NAMING-03: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_소문자_16진수}
    R-ID-01: uuid.uuid4().hex[:4]로 4자리 난수 생성.
    R-TIME-01: KST (UTC+9) 기준 타임스탬프.
    """
    now = datetime.now(tz=KST)
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")
    rand_part = uuid.uuid4().hex[:4]           # 예: "3b9f"
    return f"{model_type}_{date_part}_{time_part}_{rand_part}"


def generate_created_at() -> str:
    """
    R-TIME-01: ISO 8601 KST 형식.
    예: "2026-05-09T14:30:22+09:00"
    """
    return datetime.now(tz=KST).isoformat()
```

### 2.3 experiment_id 충돌 가능성

같은 초에 두 실험을 시작해도 `uuid4().hex[:4]` (65,536가지)로 충돌 확률은 무시 가능하다.
단, `storage.prepare_model_dir(experiment_id)` 호출 시 디렉토리 이미 존재하면 RuntimeError가 발생하므로 실질적 충돌 방지가 보장된다.

---

## 3. 학습 시작 서비스

### 3.1 사전 조건 체크 순서

[학습 시작] 버튼 클릭 시 아래 순서대로 검증한다. 첫 번째 실패에서 즉시 `st.stop()`.

```
① current_run_status == "idle"          (단일 워커 보장 — §8 참조)
② dataset_path is not None
③ preprocessing_config is not None
④ model_config is not None
⑤ 디스크 여유 공간 체크 (check_disk_before_save)
⑥ [EfficientAD만] validate_imagenet_penalty_dir()
```

### 3.2 학습 시작 핸들러 전체 구현

```python
# tabs/tab4_training.py

def _handle_start_training(experiment_name: str) -> None:
    """
    [학습 시작] 버튼 클릭 시 호출.
    experiment_name: 사용자 입력 또는 자동 생성 실험명.
    """
    # ── 사전 조건 검증 ────────────────────────────────────────────────────────
    if st.session_state["current_run_status"] != "idle":
        st.warning("이미 학습이 진행 중입니다.")
        st.stop()

    model_config: dict = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str = st.session_state["dataset_path"]
    device_info: dict = st.session_state.get("device_info", {"device": "cpu"})

    try:
        check_disk_before_save(model_config["model_type"])
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if model_config["model_type"] == "efficientad":
        ok, count = validate_imagenet_penalty_dir()
        if not ok:
            st.error(
                "EfficientAD 학습에 필요한 ImageNet penalty 데이터가 없습니다. "
                f"`{IMAGENET_PENALTY_DIR}` 경로에 이미지를 추가해 주세요."
            )
            st.stop()
        elif count < 1000:
            st.warning(f"ImageNet penalty 이미지가 {count}장입니다. 1,000장 이상 권장합니다.")

    # ── experiment_id 및 메타데이터 생성 ──────────────────────────────────────
    exp_id = generate_experiment_id(model_config["model_type"])
    created_at = generate_created_at()

    # 실험명 미입력 시 자동 생성
    if not experiment_name.strip():
        experiment_name = f"{model_config['model_type'].upper()} {exp_id[-4:]}"

    # ── configs.yaml experiment 섹션 저장 ──────────────────────────────────────
    save_config_section(
        section="experiment",
        data={"name": experiment_name, "created_at": created_at},
        path="./configs.yaml"
    )

    # ── 동시성 객체 생성 ──────────────────────────────────────────────────────
    stop_event = threading.Event()
    result_queue = queue.Queue()

    # ── TrainingWorker 생성 및 시작 ────────────────────────────────────────────
    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device_info["device"],
        stop_event=stop_event,
        result_queue=result_queue
    )
    worker.daemon = True   # R-THREAD-03: Streamlit 프로세스 종료 시 함께 종료
    worker.start()

    # ── session_state 갱신 (메인 스레드 전용 — ADR-04) ──────────────────────────
    st.session_state["current_run_status"] = "running"
    st.session_state["current_exp_id"] = exp_id
    st.session_state["_stop_event"] = stop_event
    st.session_state["_result_queue"] = result_queue
    st.session_state["_worker"] = worker
    st.session_state["_progress"] = {
        "step": 0,
        "total": _get_total_steps(model_config),
        "loss": None,
        "elapsed": 0.0
    }
    st.session_state["_log_lines"] = []
    st.session_state["_loss_history"] = []

    st.rerun()   # running UI로 전환


def _get_total_steps(model_config: dict) -> int:
    """
    Progress Bar의 분모값 결정.
    EfficientAD: train_steps (정수)
    PatchCore: 1 (에포크 개념 없음, feature 추출 완료 = 1스텝)
    """
    if model_config["model_type"] == "efficientad":
        return model_config["params"]["train_steps"]
    return 1  # PatchCore
```

### 3.3 자동 생성 실험명 형식

| 조건 | 형식 | 예 |
|------|------|---|
| 사용자 입력 있음 | 입력값 그대로 (최대 64자) | "스크류 CLAHE 기본 실험" |
| 사용자 입력 없음 | `{MODEL_TYPE} {rand_4자리}` | "EFFICIENTAD 3b9f" |

---

## 4. TrainingWorker 생성자 명세

### 4.1 생성자 파라미터 및 인스턴스 변수

```python
# utils/training_worker.py

import threading
import queue
import time
import traceback
import random
import numpy as np
import torch

class TrainingWorker(threading.Thread):
    """
    백그라운드 학습 스레드.
    외부 → 내부 통신: stop_event.set()
    내부 → 외부 통신: result_queue.put(QueueMessage)
    """

    def __init__(
        self,
        experiment_id: str,
        model_config: dict,           # 00_Global §1.7 model_config 스키마
        preprocessing_config: dict,   # 00_Global §1.6 preprocessing_config 스키마
        dataset_path: str,            # MVTec AD 데이터셋 루트 경로
        device: str,                  # "cuda" | "cpu"
        stop_event: threading.Event,
        result_queue: queue.Queue
    ) -> None:
        super().__init__(name=f"TrainingWorker-{experiment_id}")

        # 외부 주입 파라미터
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path = dataset_path
        self.device = device
        self.stop_event = stop_event
        self.result_queue = result_queue

        # 내부 상태 (run() 실행 중 설정)
        self._model = None          # EfficientAd | Patchcore — 학습 완료 후 설정
        self._start_time: float = 0.0
        self._log_writer = None     # get_log_writer() 반환 파일 객체
```

### 4.2 인스턴스 변수 생명주기

| 변수 | 설정 시점 | 해제 시점 |
|------|-----------|-----------|
| `_model` | 모델 초기화 완료 후 | `"completed"` 메시지 전송 후 메인 스레드에서 `del msg["model"]` |
| `_start_time` | `run()` 진입 직후 | 스레드 종료 시 자동 소멸 |
| `_log_writer` | `run()` 진입 직후 open | `run()` 종료 전 `finally` 블록에서 `close()` |

### 4.3 run() 골격 구조

```python
def run(self) -> None:
    self._start_time = time.time()
    self._log_writer = get_log_writer(self.experiment_id)

    try:
        self._run_impl()
    except Exception as e:
        self.result_queue.put({
            "type": "error",
            "exception": e,
            "traceback": traceback.format_exc()
        })
        self._write_log(f"[오류] {traceback.format_exc()[:500]}")
    finally:
        if self._log_writer:
            self._log_writer.close()

def _write_log(self, message: str) -> None:
    """타임스탬프 + 메시지를 로그 파일과 Queue(log 메시지)에 동시 기록."""
    from datetime import datetime, timezone, timedelta
    KST = timezone(timedelta(hours=9))
    ts = datetime.now(tz=KST).strftime("%Y-%m-%dT%H:%M:%S+09:00")
    line = f"{ts}\t{message}"
    if self._log_writer:
        self._log_writer.write(line + "\n")
    self.result_queue.put({"type": "log", "message": message})
```

---

## 5. 학습 중 Progress 보고 주기

### 5.1 모델별 보고 전략

| 모델 | 보고 시점 | 이유 |
|------|-----------|------|
| **EfficientAD** | 매 100 step (첫 번째 step에서 즉시 1회) | UI 반응성 향상 — 초기 진행 확인 가능 |
| **PatchCore** | 루프 진입 직전 1회(total 확정) + 배치마다 1회 | 학습이 단일 에포크이므로 스텝 개념 없음 |

### 5.2 PatchCore progress 메시지 예시

```python
# PatchCore 학습 루프 내부 (08_AI_ML_Integration.md §B.5 참조)

# feature 추출 완료
self.result_queue.put({
    "type": "progress",
    "step": 0,
    "total": 1,
    "loss": 0.0,          # PatchCore는 loss 없음, 0.0 고정
    "elapsed": round(time.time() - self._start_time, 1)
})
self._write_log("[PatchCore] 특징 추출 완료. Coreset 구성 중...")

# coreset 구성 완료 = 학습 완료
self.result_queue.put({
    "type": "progress",
    "step": 1,
    "total": 1,
    "loss": 0.0,
    "elapsed": round(time.time() - self._start_time, 1)
})
```

### 5.3 학습 시작 로그 포맷

```python
# run() 진입 직후 첫 번째 로그
self._write_log(f"[시작] 실험: {self.experiment_id}")
self._write_log(
    f"[설정] 모델: {self.model_config['model_type']} | "
    f"이미지 크기: {self.model_config['image_size']} | "
    f"디바이스: {self.device}"
)
```

---

## 6. 학습 완료/중단 후처리 흐름

> 상세 구현은 `06_API_Specification.md §5.2`에 있다. 이 절에서는 서비스 흐름 관점의 순서도만 확정한다.

### 6.1 완료 후처리 순서 (메인 스레드)

```
[completed 메시지 수신]
  │
  ├─ 1. compute_threshold() → float threshold
  │       percentile: np.percentile(정상_scores, threshold_value)
  │       absolute:   float(threshold_value)
  │
  ├─ 2. compute_metrics(y_true, anomaly_scores, threshold)
  │       → metrics dict (00_Global §1.2 스키마)
  │
  ├─ 3. _build_experiment_record() → record dict
  │       status="completed", metrics, duration_seconds 포함
  │
  ├─ 4. check_disk_before_save(model_type)  ← 저장 전 재검증
  │
  ├─ 5. save_completed_experiment(exp_id, model, record)
  │       (05_Data_Model §6 — 3단계 원자성 프로토콜)
  │
  ├─ 6. session_state["experiments"][exp_id] = record
  │
  ├─ 7. session_state["_last_result"] = {"level": "success", "text": "학습이 완료되었습니다. AUC: {auc:.4f} | 소요 시간: {분}분 {초}초"}
  │       ※ st.success()를 직접 호출하지 않는다.
  │         _reset_run_state() 직후 st.rerun()이 실행되어 현재 렌더 트리가 폐기되므로,
  │         메시지를 session_state에 저장했다가 다음 rerun의 _show_last_result()에서 표시한다.
  │
  └─ 8. _reset_run_state()
         current_run_status = "idle"
         del msg["model"]
         torch.cuda.empty_cache()
```

### 6.2 중단 후처리 순서 (메인 스레드)

```
[stopped 메시지 수신]
  │
  ├─ 1. _build_experiment_record()
  │       status="중단", metrics=None, model_path=None, configs_path=None
  │
  ├─ 2. append_experiment(record)
  │       (history.json append — 05_Data_Model §3.3)
  │
  ├─ 3. session_state["experiments"][exp_id] = record
  │
  ├─ 4. session_state["_last_result"] = {"level": "warning", "text": MSG["TRAIN_STOPPED"] + step_suffix}
  │       ※ st.warning()을 직접 호출하지 않는다. (§6.1 step7 주석 참고)
  │
  └─ 5. _reset_run_state()
```

### 6.3 오류 후처리 순서 (메인 스레드)

```
[error 메시지 수신]
  │
  ├─ 1. session_state["_last_result"] = {"level": "error", "text": traceback 포함 메시지}
  │
  └─ 2. _reset_run_state()
```

### 6.4 _show_last_result() — 알림 지연 표시

```
[다음 rerun — status="idle" 렌더 경로]
  │
  └─ _show_last_result()
       result = session_state.pop("_last_result", None)
       level == "success" → st.success(text)
       level == "warning" → st.warning(text)
       level == "error"   → st.error(text)
```

> **설계 의도**: 핸들러(`_handle_completed` 등)는 `st.rerun()` 직전에 실행된다.
> Streamlit에서 `st.rerun()` 이후 이전 렌더 트리는 폐기되므로, 핸들러 내부에서
> `st.success/warning/error`를 직접 호출해도 사용자에게 표시되지 않는다.
> `_last_result`에 저장하면 다음 렌더 사이클까지 메시지가 보존된다.

### 6.5 _build_experiment_record() 구현

```python
# tabs/tab4_training.py

def _build_experiment_record(
    exp_id: str,
    status: str,                    # "completed" | "중단"
    metrics: dict | None,
    duration_seconds: int | None
) -> dict:
    """
    00_Global §1.1 experiment 스키마에 맞는 레코드 생성.
    status="중단" 시 metrics, model_path, configs_path 강제 None 설정.
    """
    model_config: dict = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str = st.session_state["dataset_path"]
    exp_cfg = load_config("./configs.yaml").get("experiment", {})

    record = {
        "experiment_id": exp_id,
        "name": exp_cfg.get("name", exp_id),
        "status": status,
        "created_at": exp_cfg.get("created_at", generate_created_at()),
        "model_type": model_config["model_type"],
        "preprocessing_method": preprocessing_config["method"],
        "preprocessing_params": preprocessing_config.get("params"),
        "model_params": model_config["params"],
        "threshold_method": model_config["threshold_method"],
        "threshold_value": model_config["threshold_value"],
        "dataset_path": dataset_path,
        "image_size": model_config["image_size"],
        "duration_seconds": duration_seconds,
        # 완료 시에만 채워짐
        "metrics": metrics,
        "model_path": None,       # save_completed_experiment()가 설정
        "configs_path": None,     # save_completed_experiment()가 설정
    }

    # status="중단" 불변 조건 (00_Global §2 R-05)
    if status == "중단":
        record["metrics"] = None
        record["model_path"] = None
        record["configs_path"] = None

    return record
```

---

## 7. 탭6 이상 감지 서비스

### 7.1 전체 파이프라인

```
[탭6 진입 — selected_experiment_id 확정]
  │
  ├─ [캐시 확인]
  │    get_anomaly_map_cache(exp_id)
  │    │
  │    ├─ 캐시 HIT → anomaly_maps, image_paths 재사용 (모델 로드 생략)
  │    │
  │    └─ 캐시 MISS
  │         │
  │         ├─ 1. 실험 레코드 로드 (session_state.experiments[exp_id])
  │         │
  │         ├─ 2. load_model_for_inference(exp_id, model_path, model_config, device)
  │         │       └─ model_state_dict.pth 로드 → 모델 eval 모드 설정
  │         │
  │         ├─ 3. 테스트 이미지 목록 수집 (MVTec AD test/ 디렉토리 스캔)
  │         │
  │         ├─ 4. 전체 테스트 이미지 일괄 추론 (_run_batch_inference)
  │         │       └─ apply_preprocessing() → run_inference() → anomaly_map
  │         │
  │         └─ 5. set_anomaly_map_cache(exp_id, {...})
  │
  ├─ [이미지 목록 테이블 렌더링]
  │    결함 유형 필터 (selectbox), 정상/결함 구분
  │
  ├─ [이미지 선택]
  │    선택된 image_path, anomaly_map 추출
  │
  ├─ [Threshold 슬라이더]
  │    anomaly_map_threshold 갱신 → 이진 마스크 재계산
  │
  └─ [3분할 시각화]
       create_triplet_image(original, gt_mask, heatmap)
       → st.image() 표시
       → [PNG 저장] 버튼
```

### 7.2 load_model_for_inference() 구현

```python
# utils/model_factory.py

def load_model_for_inference(
    exp_id: str,
    model_path: str,         # experiment_record["model_path"]
    model_config: dict,      # 실험 스냅샷 configs.yaml의 model 섹션
    device: str
) -> object:
    """
    저장된 model_state_dict.pth 로드 후 추론 가능한 모델 반환.
    반환된 모델은 eval() 모드.

    Raises:
        RuntimeError: pth 파일 없거나 state_dict 불일치 시
    """
    pth_path = Path(model_path) / "model_state_dict.pth"
    if not pth_path.exists():
        raise RuntimeError(
            f"ERR_MODEL_FILE_NOT_FOUND: {pth_path} — "
            "모델 파일이 존재하지 않습니다."
        )

    model_type = model_config["model_type"]

    # 모델 객체 재생성 (저장 시와 동일한 파라미터로)
    if model_type == "efficientad":
        model = _build_efficientad(model_config)
    elif model_type == "patchcore":
        model = _build_patchcore(model_config)
    else:
        raise RuntimeError(f"알 수 없는 model_type: {model_type}")

    try:
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"ERR_MODEL_LOAD_FAILED: state_dict 불일치. "
            f"학습 파라미터와 로드 파라미터가 다를 수 있습니다. {e}"
        ) from e

    model.to(device)
    model.eval()
    return model
```

### 7.3 일괄 추론 서비스 (_run_batch_inference)

```python
# tabs/tab6_anomaly_map.py 또는 utils/model_factory.py

def run_batch_inference(
    model: object,
    image_paths: list[str],
    preprocessing_config: dict,
    device: str
) -> np.ndarray:
    """
    테스트 이미지 전체에 대해 anomaly_map 생성.
    반환: np.ndarray shape (N, H, W), dtype=float32

    진행 중 st.progress() 갱신 (Streamlit context에서 호출).
    """
    anomaly_maps = []
    total = len(image_paths)
    progress_bar = st.progress(0, text="추론 중...")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            _, tensor = apply_preprocessing(img_path, preprocessing_config)
            tensor = tensor.unsqueeze(0).to(device)   # (1, C, H, W)
            anomaly_map = run_inference(model, tensor) # (H, W)
            anomaly_maps.append(anomaly_map)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                progress_bar.progress(
                    (i + 1) / total,
                    text=f"추론 중... {i+1}/{total}"
                )

    progress_bar.empty()
    return np.stack(anomaly_maps, axis=0).astype(np.float32)
```

### 7.4 Threshold 슬라이더 연동

```python
# tabs/tab6_anomaly_map.py

def _render_threshold_section(anomaly_map: np.ndarray) -> float:
    """
    Threshold 슬라이더 렌더링.
    반환: 현재 threshold float 값 (session_state 갱신 포함).
    """
    exp_id = st.session_state["selected_experiment_id"]
    exp_record = st.session_state["experiments"][exp_id]
    default_threshold = float(exp_record.get("metrics", {}).get(
        "threshold_value",
        st.session_state["model_config"]["threshold_value"]
    ))

    # 슬라이더 범위: anomaly_map 전체의 min~max
    cache = get_anomaly_map_cache(exp_id)
    all_maps: np.ndarray = cache["anomaly_maps"]  # (N, H, W)
    score_min = float(all_maps.min())
    score_max = float(all_maps.max())

    current = st.session_state.get("anomaly_map_threshold") or default_threshold
    current = max(score_min, min(score_max, current))  # 범위 클램핑

    threshold = st.slider(
        "이상 감지 Threshold",
        min_value=score_min,
        max_value=score_max,
        value=current,
        step=(score_max - score_min) / 200,
        format="%.4f"
    )
    st.session_state["anomaly_map_threshold"] = threshold
    return threshold
```

### 7.5 3분할 시각화 구성

```python
# tabs/tab6_anomaly_map.py

def _render_visualization(
    selected_idx: int,
    cache: dict,
    threshold: float,
    preprocessing_config: dict
) -> None:
    """
    선택된 이미지에 대해 원본 / GT 마스크 / Heatmap 3분할 시각화.
    """
    image_path: str = cache["image_paths"][selected_idx]
    anomaly_map: np.ndarray = cache["anomaly_maps"][selected_idx]  # (H, W)

    # 원본 이미지 (전처리 미적용 원본)
    original_pil = load_image(image_path)

    # GT 마스크 로드 (없으면 검정 마스크)
    gt_mask = _load_gt_mask(image_path)   # PIL.Image | None

    # Heatmap (이진 마스크 오버레이 포함)
    heatmap_pil = anomaly_map_to_heatmap(anomaly_map)
    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    heatmap_with_mask = _overlay_binary_mask(heatmap_pil, binary_mask)

    # 3분할 이미지 생성
    triplet = create_triplet_image(
        original=original_pil,
        gt_mask=gt_mask,
        heatmap=heatmap_with_mask
    )

    st.image(triplet, caption="원본 이미지 | GT 마스크 | Anomaly Heatmap", use_column_width=True)

    # Anomaly Score 표시
    score = float(anomaly_map.max())
    col1, col2 = st.columns(2)
    col1.metric("Anomaly Score", f"{score:.4f}")
    col2.metric("Threshold", f"{threshold:.4f}")
    is_defect = score > threshold
    st.markdown(f"**판정**: {'🔴 결함' if is_defect else '🟢 정상'}")

    # PNG 저장 버튼
    if st.button("PNG 저장"):
        _save_triplet_png(triplet, image_path)


def _load_gt_mask(image_path: str) -> PIL.Image.Image | None:
    """
    image_path 기준으로 ground_truth/ 경로의 마스크 로드.
    마스크 없으면 None 반환 (A-10: 검정 마스크로 처리).

    경로 변환 예:
      test/crack/000.png → ground_truth/crack/000_mask.png
    """
    p = Path(image_path)
    # test/{class}/{filename} → ground_truth/{class}/{stem}_mask{suffix}
    parts = list(p.parts)
    try:
        test_idx = parts.index("test")
    except ValueError:
        return None

    gt_parts = parts.copy()
    gt_parts[test_idx] = "ground_truth"
    gt_parts[-1] = p.stem + "_mask" + p.suffix
    gt_path = Path(*gt_parts)

    if not gt_path.exists():
        return None

    mask = PIL.Image.open(gt_path).convert("L")  # Grayscale
    return mask.resize((p.stat().st_size,) * 2)  # anomaly_map과 동일 크기로 리사이즈


def _save_triplet_png(triplet: PIL.Image.Image, source_path: str) -> None:
    """
    ./results/{exp_id}/{source_filename}_triplet.png 저장.
    저장 완료 후 st.success() 표시.
    """
    exp_id = st.session_state["selected_experiment_id"]
    result_dir = Path(f"./results/{exp_id}")
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(source_path).stem + "_triplet.png"
    save_path = result_dir / filename
    triplet.save(save_path, format="PNG")

    size_kb = save_path.stat().st_size / 1024
    st.success(f"저장 완료: {save_path} ({size_kb:.1f} KB)")
```

### 7.6 탭6에서 모델 로드 시점 (캐시 MISS 처리)

```python
# tabs/tab6_anomaly_map.py — render() 상단

def render() -> None:
    _guard()  # selected_experiment_id is None → st.stop()

    exp_id: str = st.session_state["selected_experiment_id"]
    exp_record: dict = st.session_state["experiments"].get(exp_id)
    if exp_record is None:
        st.error(MSG["NO_SELECTED_EXP"])
        return

    if exp_record["status"] == "중단":
        st.warning("중단된 실험은 이상 영역 시각화를 지원하지 않습니다.")
        return

    # 캐시 확인
    cache = get_anomaly_map_cache(exp_id)
    if cache is None:
        # 캐시 MISS: 모델 로드 + 전체 추론
        with st.spinner("모델 로드 및 전체 이미지 추론 중..."):
            try:
                exp_configs = load_config(exp_record["configs_path"])
                stored_model_config = exp_configs.get("model", {})
                device = st.session_state.get("device_info", {}).get("device", "cpu")

                model = load_model_for_inference(
                    exp_id=exp_id,
                    model_path=exp_record["model_path"],
                    model_config=stored_model_config,
                    device=device
                )

                image_paths = _collect_test_image_paths(
                    exp_record["dataset_path"]
                )
                anomaly_maps = run_batch_inference(
                    model=model,
                    image_paths=image_paths,
                    preprocessing_config=exp_configs.get("preprocessing", {}),
                    device=device
                )

                set_anomaly_map_cache(exp_id, {
                    "anomaly_maps": anomaly_maps,
                    "image_paths": image_paths
                })
                cache = get_anomaly_map_cache(exp_id)

                # 추론 완료 후 모델 메모리 해제
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                st.error(f"모델 로드 실패: {e}")
                return

    # 이하 캐시 데이터 기반 UI 렌더링 ...
```

---

## 8. 단일 워커 보장

### 8.1 중복 학습 시작 방지

```python
# tabs/tab4_training.py — _render_idle_ui()

def _render_idle_ui() -> None:
    """
    current_run_status == "idle" 상태 UI.
    학습 중에는 [학습 시작] 버튼이 아예 렌더링되지 않음 (R-UI-02).
    """
    # 실험명 입력
    experiment_name = st.text_input(
        "실험명 (비워두면 자동 생성)",
        max_chars=64,
        placeholder="예: EfficientAD CLAHE clip2.0 실험"
    )

    if st.button("학습 시작", type="primary"):
        _handle_start_training(experiment_name)
```

```python
# tabs/tab4_training.py — _render_running_ui()

def _render_running_ui() -> None:
    """
    current_run_status == "running" 상태 UI.
    [학습 시작] 버튼 없음 — DOM에서 완전히 제외 (R-UI-02).
    [학습 중지] 버튼만 렌더링.
    """
    st.info("🔄 학습이 진행 중입니다. 탭을 전환해도 학습은 계속됩니다.")
    # Progress Bar, Loss 곡선, 로그 (06.§5.3 참조)
    # [학습 중지] 버튼
```

### 8.2 동시 학습 불가 설계 이유

- `current_run_status == "running"` 중에는 [학습 시작] 버튼이 렌더링되지 않음 → 클릭 자체 불가.
- `_handle_start_training()` 최상위에서 `current_run_status != "idle"` 체크 → 이중 방어.
- `session_state["_worker"]`에 단일 스레드 참조 유지.

---

## 9. 메모리 관리 규칙

### 9.1 GPU 메모리 해제 시점

| 이벤트 | 처리 |
|--------|------|
| 학습 완료 (`_handle_completed()`) | `del msg["model"]` + `torch.cuda.empty_cache()` |
| 탭6 캐시 MISS 추론 완료 | `del model` + `torch.cuda.empty_cache()` (device == "cuda"인 경우만) |
| 학습 중단 (`_handle_stopped()`) | 모델 객체 없음 (TrainingWorker가 중단 시 model 미전달) |
| 학습 오류 (`_handle_error()`) | 모델 객체 없음 |

### 9.2 해제 코드 패턴

```python
# 학습 완료 후 메인 스레드
del msg["model"]
if st.session_state.get("device_info", {}).get("device") == "cuda":
    torch.cuda.empty_cache()

# 탭6 추론 완료 후
del model
if device == "cuda":
    torch.cuda.empty_cache()
```

### 9.3 anomaly_map numpy 배열 메모리

- `np.ndarray (N, 256, 256)` float32 ≈ 25 MB (100장 기준) — session_state에 캐시 유지.
- 캐시 eviction은 `cache_manager.py`가 자동 처리 (최대 3개 — 06.§2.2 참조).
- 명시적 해제 불필요: Python GC가 eviction 시 참조 해제.

---

## 10. 서비스 데이터 흐름 요약

```
[탭4 학습 시작 서비스]

사용자 클릭
  → generate_experiment_id()          # R-NAMING-03, R-ID-01
  → save_config_section("experiment") # configs.yaml 기록
  → TrainingWorker(...)               # 생성자 파라미터 §4.1
  → worker.daemon = True
  → worker.start()                    # 백그라운드 학습 시작
  → session_state 갱신 (8개 키)
  → st.rerun()


[TrainingWorker 내부 흐름]

worker.run()
  → _write_log("[시작]")
  → _run_impl()
       → random seed 고정 (R-SEED-01)
       → MVTecDataset 구성 (08.B.2)
       → 모델 초기화 (08.B.4 또는 B.5)
       → 학습 루프
           → stop_event.is_set() 체크
           → train_step()
           → result_queue.put(progress/log)
       → _run_full_test_inference()   (08.B.7)
       → result_queue.put(completed)


[탭6 이상 감지 서비스]

탭6 진입
  → get_anomaly_map_cache(exp_id)
       HIT: 캐시 재사용
       MISS:
         → load_model_for_inference()  # pth 로드
         → run_batch_inference()       # 전체 이미지 추론
         → set_anomaly_map_cache()     # 캐시 저장
         → del model + empty_cache()  # 메모리 해제
  → 이미지 선택 UI
  → Threshold 슬라이더
  → create_triplet_image()
  → st.image()
```

---

## 11. 구현 체크리스트

### experiment_id 생성

- [ ] `generate_experiment_id(model_type)` — R-NAMING-03 형식 준수
- [ ] `generate_created_at()` — ISO 8601 KST 형식
- [ ] `uuid.uuid4().hex[:4]` — 4자리 소문자 16진수

### 학습 시작 핸들러

- [ ] 사전 조건 6단계 순서대로 검증
- [ ] `worker.daemon = True` 설정
- [ ] `session_state["_worker"]` 저장 (미아 스레드 감지용 — 06.§6.3)
- [ ] `_get_total_steps()` — EfficientAD: train_steps, PatchCore: 1
- [ ] 실험명 미입력 시 자동 생성 (`"EFFICIENTAD {rand_4자리}"` 형식)

### TrainingWorker

- [ ] `__init__()` — §4.1 파라미터 전체 포함
- [ ] `run()` — `finally` 블록에서 `_log_writer.close()`
- [ ] `_write_log()` — 파일 + Queue(log 메시지) 동시 기록
- [ ] EfficientAD progress: 매 500 step
- [ ] PatchCore progress: feature 추출 완료 1회 + coreset 완료 1회

### 탭6 이상 감지

- [ ] 캐시 HIT 시 모델 로드 생략
- [ ] 캐시 MISS 시 전체 추론 후 `set_anomaly_map_cache()` 저장
- [ ] 추론 완료 후 `del model` + `torch.cuda.empty_cache()`
- [ ] `_load_gt_mask()` — 마스크 없으면 `None` 반환 (A-10)
- [ ] `_save_triplet_png()` — `./results/{exp_id}/` 디렉토리 자동 생성
- [ ] 중단 실험 진입 시 시각화 비활성화 안내

### 메모리

- [ ] `_handle_completed()` — `del msg["model"]` + `torch.cuda.empty_cache()`
- [ ] 탭6 추론 완료 후 모델 해제 (device == "cuda"인 경우만)

---

*이 문서는 04_System_Architecture.md §B.5 (비동기 모델), 06_API_Specification.md §5 (polling loop), 08_AI_ML_Integration.md §B.4~B.6 (학습 구현)과 연동된다.*
*다음: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md) — Docker 설정, AWS 배포, 볼륨 마운트 전략*

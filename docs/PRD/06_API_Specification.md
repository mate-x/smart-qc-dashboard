# 06. API Specification (Internal Interface Contracts)

> **참조 문서**: `04_System_Architecture.md` §B.3 (모듈 책임 명세), `05_Data_Model_and_Storage_Strategy.md`
> **버전**: v1.0
> **작성일**: 2026-05-09
> **목적**: 이 시스템은 REST API가 없다. 이 문서는 `utils/` 레이어의 모든 공개 함수 인터페이스, 에러 계약, Queue 메시지 프로토콜, tab4 소비 알고리즘을 단일 참조점으로 확정한다.
>
> **04와의 역할 분리**: 04.B.3에서 확정된 함수 시그니처는 이 문서에서 반복하지 않는다. 이 문서는 04에서 미정의된 항목(에러 계약, Queue 프로토콜, tab4 소비 루프, 신규 모듈)만 추가로 명세한다.

---

## 목차

1. [확정 모듈 파일 목록](#1-확정-모듈-파일-목록)
2. [신규 모듈 인터페이스](#2-신규-모듈-인터페이스)
3. [공개 함수 에러 계약 통합표](#3-공개-함수-에러-계약-통합표)
4. [Queue 메시지 프로토콜 공식 명세](#4-queue-메시지-프로토콜-공식-명세)
5. [tab4 Queue 소비 알고리즘](#5-tab4-queue-소비-알고리즘)
6. [stop_event 경쟁 조건 처리 규칙](#6-stop_event-경쟁-조건-처리-규칙)
7. [탭 Guard 조건 및 session_state 쓰기 권한](#7-탭-guard-조건-및-session_state-쓰기-권한)
8. [모듈 간 호출 권한 매트릭스](#8-모듈-간-호출-권한-매트릭스)
9. [구현 검증 체크리스트](#9-구현-검증-체크리스트)

---

## 1. 확정 모듈 파일 목록

04_System_Architecture.md의 디렉토리 구조에 05에서 추가된 파일을 포함하여 최종 확정한다.

```
utils/
├── __init__.py
├── session_state_init.py   # 04.B.3.3 확정
├── config_manager.py       # 04.B.3.3 + 05.§4 확정
├── messages.py             # 04.B.3.3 확정
├── image_utils.py          # 04.B.3.3 + 08 확정
├── metrics.py              # 04.B.3.3 + 08 확정
├── model_factory.py        # 04.B.3.3 + 08 확정
├── training_worker.py      # 04.B.3.3 + 08 확정
├── storage.py              # 05 신규 추가 — §2.1 참조
└── cache_manager.py        # 05 신규 추가 — §2.2 참조
```

> `storage.py`와 `cache_manager.py`는 04 작성 시점에 `config_manager.py` 내 단순 함수로 설계됐으나, 05에서 책임 분리 원칙에 따라 독립 파일로 분리 확정됐다.

---

## 2. 신규 모듈 인터페이스

04.B.3에 포함되지 않은 두 모듈의 전체 공개 인터페이스를 명세한다.

### 2.1 `utils/storage.py`

```python
"""
책임: history.json 읽기/쓰기, 모델 파일 저장/삭제, 로그 파일 접근, 디스크 모니터링.
금지: st.session_state 직접 접근, Anomalib/torch 직접 import (model 인자만 수신).
"""

# 상수
IMAGENET_PENALTY_DIR: Path = Path("./dataset/imagenet_penalty")

# ── history.json ──────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    """실험 레코드 전체 로드. 파일 미존재·파싱 실패 모두 [] 반환. 예외 전파 없음."""

def append_experiment(record: dict) -> None:
    """
    레코드 append 후 원자적 쓰기.
    Raises: RuntimeError("ERR_HISTORY_WRITE_FAILED: ...") — IOError 발생 시.
    """

def delete_experiment_from_history(experiment_id: str) -> bool:
    """
    해당 ID 제거 후 원자적 쓰기.
    Returns: True(제거 성공) | False(ID 없음).
    """

# ── 모델 저장/삭제 ─────────────────────────────────────────────────────────────

def prepare_model_dir(experiment_id: str) -> Path:
    """
    ./models/{experiment_id}/ 생성 후 Path 반환.
    Raises: RuntimeError — 동일 경로 이미 존재 시 (중복 ID 방지).
    """

def save_completed_experiment(
    experiment_id: str,
    model: object,            # EfficientAd | Patchcore 인스턴스
    experiment_record: dict   # status="completed" 레코드
) -> None:
    """
    3단계 원자성 프로토콜 실행 (05.§6 참조).
    Raises: RuntimeError("ERR_MODEL_SAVE_FAILED ...") — 단계1/2 실패 시 (파일 정리 후 raise).
    Raises: RuntimeError("ERR_HISTORY_WRITE_FAILED ...") — 단계3 실패 시 (파일 보존 후 raise).
    """

def delete_experiment(experiment_id: str, model_path: str | None = None) -> None:
    """
    history.json 제거 + 모델 디렉토리 삭제 + 로그 파일 삭제.
    model_path=None이면 파일 삭제 생략 (status="중단" 레코드용).
    """

def validate_imagenet_penalty_dir() -> tuple[bool, int]:
    """
    Returns: (이미지 존재 여부, 이미지 수).
    EfficientAD 학습 시작 전 tab4에서 호출.
    """

# ── 로그 파일 ──────────────────────────────────────────────────────────────────

def get_log_writer(experiment_id: str):
    """
    ./logs/{experiment_id}.log append 모드 파일 객체 반환 (line-buffered).
    백그라운드 스레드(TrainingWorker)에서 직접 호출.
    반환된 파일 객체는 호출자가 close() 책임.
    """

def read_log_tail(experiment_id: str, n_lines: int = 100) -> str:
    """최신 n_lines줄 반환. 파일 미존재 시 빈 문자열."""

# ── 디스크 모니터링 ────────────────────────────────────────────────────────────

def check_disk_space(required_mb: float = 500.0, path: str = ".") -> tuple[bool, float]:
    """Returns: (충분 여부, 여유 공간 MB)."""

def check_disk_before_save(model_type: str) -> None:
    """
    100 MB 미만: RuntimeError("ERR_DISK_SPACE: ...") raise.
    500 MB 미만: st.warning() 표시 (저장은 허용).
    Streamlit context에서만 호출 가능 (st.warning 사용).
    """
```

---

### 2.2 `utils/cache_manager.py`

```python
"""
책임: session_state 기반 anomaly_map 캐시 CRUD.
주의: session_state Write 함수이므로 메인 스레드(Streamlit context)에서만 호출.
"""

MAX_ANOMALY_MAP_CACHE: int = 3  # 동시 보유 최대 실험 수

def set_anomaly_map_cache(experiment_id: str, data: dict) -> None:
    """
    캐시 저장. 기존 캐시가 MAX_ANOMALY_MAP_CACHE개 이상이면 cached_at 기준
    가장 오래된 캐시 자동 제거 후 저장.

    data 구조:
        {
            "anomaly_maps": np.ndarray,  # shape (N, H, W), float32
            "image_paths":  list[str]    # 테스트 이미지 경로 목록
        }
    cached_at(float)은 이 함수가 time.time()으로 자동 추가.
    """

def get_anomaly_map_cache(experiment_id: str) -> dict | None:
    """
    캐시 반환. 없으면 None.
    반환 dict: {"anomaly_maps": np.ndarray, "image_paths": list[str], "cached_at": float}
    """

def invalidate_anomaly_map_cache(experiment_id: str) -> None:
    """
    특정 실험의 캐시 제거. 키 없으면 no-op.
    실험 삭제(storage.delete_experiment) 직후 호출.
    """
```

---

## 3. 공개 함수 에러 계약 통합표

> 각 함수가 예외를 raise하는지, 안전하게 기본값을 반환하는지 통합 정리.
> "예외 없음"은 내부 오류 발생 시 기본값 반환 + WARNING 로그를 의미.

| 모듈 | 함수 | 실패 시 동작 | 비고 |
|------|------|-------------|------|
| `config_manager` | `load_config()` | `{}` 반환, 예외 없음 | YAML 파싱 실패 포함 |
| `config_manager` | `save_config_section()` | `RuntimeError` raise | ERR_CONFIG_WRITE_FAILED |
| `storage` | `load_history()` | `[]` 반환, 예외 없음 | JSON 파싱 실패 포함 |
| `storage` | `append_experiment()` | `RuntimeError` raise | ERR_HISTORY_WRITE_FAILED |
| `storage` | `delete_experiment_from_history()` | `False` 반환 | ID 없으면 no-op |
| `storage` | `prepare_model_dir()` | `RuntimeError` raise | 중복 경로 존재 시 |
| `storage` | `save_completed_experiment()` | `RuntimeError` raise | 단계별 에러코드 다름 (05.§6) |
| `storage` | `delete_experiment()` | 예외 없음 | `ignore_errors=True` |
| `storage` | `validate_imagenet_penalty_dir()` | `(False, 0)` 반환 | 디렉토리 없으면 |
| `storage` | `read_log_tail()` | `""` 반환 | 파일 없으면 |
| `storage` | `check_disk_space()` | 예외 없음 | `shutil.disk_usage` 실패 시 `(False, 0.0)` |
| `storage` | `check_disk_before_save()` | `RuntimeError` raise (100MB↓) | 500MB↓는 warning만 |
| `image_utils` | `load_image()` | `FileNotFoundError` raise | 호출자(탭)가 처리 |
| `image_utils` | `apply_filter()` | `RuntimeError` raise | cv2 오류 래핑 |
| `image_utils` | `apply_preprocessing()` | 내부 예외 전파 | load_image + apply_filter 연쇄 |
| `metrics` | `compute_metrics()` | 예외 없음 | 단일 클래스 데이터 시 auc=0.0 |
| `metrics` | `compute_roc_curve()` | `([], [], 0.0)` 반환 | 단일 클래스 시 |
| `model_factory` | `load_model_for_inference()` | `RuntimeError` raise | 파일 없거나 state_dict 불일치 |
| `model_factory` | `run_inference()` | `RuntimeError` raise | CUDA OOM 등 torch 예외 래핑 |
| `cache_manager` | `set_anomaly_map_cache()` | 예외 없음 | session_state 쓰기 실패 없음 |
| `cache_manager` | `get_anomaly_map_cache()` | `None` 반환 | |
| `cache_manager` | `invalidate_anomaly_map_cache()` | 예외 없음 | no-op if not exists |

**원칙**: UI 렌더링 경로(탭 render 함수 최상위 호출)에 있는 함수는 예외를 raise하지 않는다. 학습/저장 경로의 함수는 명시적으로 raise하고 탭의 try-except에서 처리한다.

---

## 4. Queue 메시지 프로토콜 공식 명세

> `TrainingWorker`(백그라운드 스레드)가 `result_queue`(queue.Queue)에 put하는 메시지의 공식 스키마.
> 04.B.3.3과 08에서 비공식으로 기술된 내용을 TypedDict 형식으로 확정한다.

### 4.1 메시지 타입 정의

```python
from typing import TypedDict, Literal

class ProgressMessage(TypedDict):
    type: Literal["progress"]
    step: int           # 현재 스텝 (1-based)
    total: int          # 전체 스텝 수 (EfficientAD: train_steps, PatchCore: 1)
    loss: float         # 현재 스텝 loss 값 (PatchCore는 0.0 고정)
    elapsed: float      # 학습 시작부터 경과 시간 (초)

class LogMessage(TypedDict):
    type: Literal["log"]
    message: str        # 로그 파일에도 기록할 텍스트 (ISO8601 타임스탬프 미포함)

class CompletedMessage(TypedDict):
    type: Literal["completed"]
    y_true: list[int]                      # 테스트 이미지 정답 레이블 (0=정상, 1=결함)
    anomaly_scores: list[float]            # 테스트 이미지별 anomaly score
    anomaly_maps: dict[str, np.ndarray]    # image_path → anomaly map (H, W), float32
    image_paths: list[str]                 # 테스트 이미지 경로 목록 (anomaly_maps 키 순서)
    model: object                          # 학습 완료된 EfficientAd | Patchcore 인스턴스
    duration_seconds: int                  # 학습 소요 시간 (초, int 변환)

class ErrorMessage(TypedDict):
    type: Literal["error"]
    exception: Exception    # 발생한 예외 객체
    traceback: str          # traceback.format_exc() 결과

class StoppedMessage(TypedDict):
    type: Literal["stopped"]
    step: int               # 중단된 시점의 스텝 번호

QueueMessage = ProgressMessage | LogMessage | CompletedMessage | ErrorMessage | StoppedMessage
```

### 4.2 메시지 발생 시점 규칙

| 메시지 타입 | 발생 조건 | 발생 횟수 |
|------------|-----------|-----------|
| `"progress"` | EfficientAD: 매 500 step. PatchCore: feature 추출 완료, coreset 구성 완료 (총 2회) | N회 |
| `"log"` | 학습 시작, 완료, 중단, 오류, 주요 단계 전환 | N회 |
| `"completed"` | 학습 루프 정상 종료 직후 | **정확히 1회** |
| `"error"` | 예외 발생 시 (except 블록 내) | **정확히 1회** |
| `"stopped"` | `stop_event.is_set()` 확인 후 루프 탈출 시 | **정확히 1회** |

**종료 메시지 불변 조건**: `"completed"`, `"error"`, `"stopped"` 중 정확히 하나만 전송된다. 이 세 메시지 중 하나가 Queue에 들어오면 학습 스레드는 곧 종료된다.

### 4.3 메시지 생성 코드 패턴 (TrainingWorker 내부)

```python
# utils/training_worker.py 내부 패턴

def run(self) -> None:
    try:
        self._run_impl()
    except Exception as e:
        self.result_queue.put({
            "type": "error",
            "exception": e,
            "traceback": traceback.format_exc()
        })

def _run_impl(self) -> None:
    # ... 학습 루프 ...
    for step in range(1, total_steps + 1):
        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped", "step": step})
            return  # ← 즉시 반환, completed 미전송

        loss = self._train_step(step)

        if step % 500 == 0 or step == total_steps:
            self.result_queue.put({
                "type": "progress",
                "step": step, "total": total_steps,
                "loss": round(loss, 6),
                "elapsed": round(time.time() - self._start_time, 1)
            })

    # 테스트 추론
    y_true, anomaly_scores, _ = self._run_full_test_inference()

    self.result_queue.put({
        "type": "completed",
        "y_true": y_true,
        "anomaly_scores": anomaly_scores,
        "model": self._model,
        "duration_seconds": int(time.time() - self._start_time)
    })
```

---

## 5. tab4 Queue 소비 알고리즘

> 04.B.5에서 메인 스레드의 rerun 사이클을 개략적으로 기술했으나, tab4_training.py 구현자가 직접 참조할 수 있는 수준의 알고리즘을 이 절에서 확정한다.

### 5.1 전체 polling loop 구조

```python
# tabs/tab4_training.py

def render() -> None:
    _guard()

    status = st.session_state["current_run_status"]

    if status == "idle":
        _render_idle_ui()
    elif status == "running":
        _render_running_ui()
        _drain_queue()           # Queue 드레인 후 UI 갱신
        _schedule_rerun()        # 1초 후 재실행 예약
    # "stopped" / "completed" 상태는 drain 중 _handle_terminal() 호출로 처리됨
    # → 처리 완료 후 status = "idle" 로 전환되므로 여기서 별도 분기 불필요


def _drain_queue() -> None:
    """
    Queue에 쌓인 메시지를 모두 소비한다.
    종료 메시지(completed/error/stopped)를 만나면 즉시 처리 후 드레인 중단.
    """
    q: queue.Queue = st.session_state.get("_result_queue")
    if q is None:
        return

    while True:
        try:
            msg: QueueMessage = q.get_nowait()   # 비블로킹
        except queue.Empty:
            break  # 더 이상 메시지 없음 → 다음 rerun 대기

        msg_type = msg["type"]

        if msg_type == "progress":
            _handle_progress(msg)

        elif msg_type == "log":
            _handle_log(msg)

        elif msg_type == "completed":
            _handle_completed(msg)  # 메인 스레드에서 저장 처리
            break  # 종료 메시지 → 드레인 중단

        elif msg_type == "error":
            _handle_error(msg)
            break

        elif msg_type == "stopped":
            _handle_stopped(msg)
            break


def _schedule_rerun() -> None:
    """1초 대기 후 st.rerun() 호출."""
    time.sleep(1.0)   # R-THREAD-05: 1.0초 고정
    st.rerun()
```

### 5.2 각 메시지 처리 함수

```python
def _handle_progress(msg: ProgressMessage) -> None:
    """
    session_state._progress 갱신.
    session_state._loss_history 에 {"step": step, "loss": loss} append.
    """
    st.session_state["_progress"] = {
        "step": msg["step"],
        "total": msg["total"],
        "loss": msg["loss"],
        "elapsed": msg["elapsed"]
    }
    st.session_state["_loss_history"].append({
        "step": msg["step"],
        "loss": msg["loss"]
    })


def _handle_log(msg: LogMessage) -> None:
    """
    session_state._log_lines 에 타임스탬프 포함 줄 append.
    최대 100줄 유지 (초과분 앞에서 제거).
    """
    timestamp = datetime.now(tz=KST).strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg['message']}"
    lines: list = st.session_state["_log_lines"]
    lines.append(line)
    if len(lines) > 100:
        st.session_state["_log_lines"] = lines[-100:]


def _handle_completed(msg: CompletedMessage) -> None:
    """
    1. compute_metrics() 호출로 metrics dict 생성
    2. experiment_record 구성
    3. save_completed_experiment() 호출 (3단계 저장)
    4. session_state.experiments 갱신
    5. session_state["_last_result"] 저장 (알림 지연 표시 — 07_Backend §6.4)
    6. current_run_status = "idle"
    """
    exp_id: str = st.session_state["current_exp_id"]
    model_config: dict = st.session_state["model_config"]

    threshold = _compute_threshold(
        msg["anomaly_scores"],
        model_config["threshold_method"],
        model_config["threshold_value"]
    )
    metrics = compute_metrics(msg["y_true"], msg["anomaly_scores"], threshold)

    record = _build_experiment_record(
        exp_id=exp_id,
        status="completed",
        metrics=metrics,
        duration_seconds=msg["duration_seconds"]
    )

    try:
        check_disk_before_save(model_config["model_type"])
        save_completed_experiment(exp_id, msg["model"], record)
        st.session_state["experiments"][exp_id] = record
        mins, secs = divmod(msg["duration_seconds"], 60)
        auc = metrics.get("auc", 0.0)
        st.session_state["_last_result"] = {
            "level": "success",
            "text": f"학습이 완료되었습니다. AUC: {auc:.4f} | 소요 시간: {mins}분 {secs}초",
        }
    except RuntimeError as e:
        error_msg = str(e)
        if "ERR_HISTORY_WRITE_FAILED" in error_msg:
            st.session_state["_last_result"] = {
                "level": "warning",
                "text": f"모델 파일은 저장되었으나 히스토리 기록에 실패했습니다. {error_msg}",
            }
        else:
            st.session_state["_last_result"] = {
                "level": "error",
                "text": f"모델 저장에 실패했습니다. 디스크 공간을 확인해 주세요. {error_msg}",
            }
    finally:
        _reset_run_state()   # current_run_status = "idle"
        del msg["model"]     # GC 즉시 유도
        torch.cuda.empty_cache()


def _handle_error(msg: ErrorMessage) -> None:
    """
    session_state["_last_result"] 저장 (level="error").
    current_run_status = "idle".
    history.json에 기록하지 않음.
    """
    st.session_state["_last_result"] = {
        "level": "error",
        "text": f"학습 중 오류가 발생했습니다.\n{msg['traceback']}",
    }
    _reset_run_state()


def _handle_stopped(msg: StoppedMessage) -> None:
    """
    status="중단" 레코드 생성 후 history.json append.
    metrics, model_path, configs_path 모두 null.
    session_state["_last_result"] 저장 (level="warning").
    current_run_status = "idle".
    """
    exp_id: str = st.session_state["current_exp_id"]
    record = _build_experiment_record(
        exp_id=exp_id,
        status="중단",
        metrics=None,
        duration_seconds=None
    )
    try:
        append_experiment(record)
        st.session_state["experiments"][exp_id] = record
    except RuntimeError:
        pass  # 중단 레코드 저장 실패는 치명적이지 않음
    step = msg.get("step", 0)
    st.session_state["_last_result"] = {
        "level": "warning",
        "text": MSG["TRAIN_STOPPED"] + (f" ({step:,} step 완료 후 중단)" if step else ""),
    }
    _reset_run_state()


def _reset_run_state() -> None:
    """학습 종료 후 내부 상태 초기화."""
    st.session_state["current_run_status"] = "idle"
    st.session_state["current_exp_id"] = None
    st.session_state["_stop_event"] = None
    st.session_state["_result_queue"] = None
```

### 5.3 _render_running_ui 에서의 UI 갱신

```python
def _render_running_ui() -> None:
    """
    학습 중 Progress Bar, Loss 곡선, 로그 텍스트 렌더링.
    _drain_queue() 호출 전에 실행되어 현재 session_state 값을 표시.
    """
    progress = st.session_state.get("_progress", {})
    step = progress.get("step", 0)
    total = progress.get("total", 1)
    loss = progress.get("loss")
    elapsed = progress.get("elapsed", 0)

    # Progress Bar
    pct = step / total if total > 0 else 0
    st.progress(pct, text=f"Step {step:,} / {total:,} ({pct*100:.1f}%) | 경과: {elapsed:.0f}s")

    # Loss 곡선 (Plotly)
    loss_history = st.session_state.get("_loss_history", [])
    if loss_history:
        df = pd.DataFrame(loss_history)
        fig = px.line(df, x="step", y="loss", title="학습 Loss 곡선")
        st.plotly_chart(fig, use_container_width=True)

    # 로그 텍스트
    log_text = "\n".join(st.session_state.get("_log_lines", []))
    st.text_area("학습 로그", value=log_text, height=200, disabled=True)

    # 학습 중지 버튼
    if st.button("학습 중지", type="secondary"):
        stop_event: threading.Event = st.session_state.get("_stop_event")
        if stop_event:
            stop_event.set()
        st.info("중지 신호를 전송했습니다. 현재 스텝 완료 후 중단됩니다.")
```

---

## 6. stop_event 경쟁 조건 처리 규칙

> 04.B.5에서 기술된 스레드 모델에서 발생할 수 있는 경쟁 조건을 확정한다.

### 6.1 경쟁 조건 시나리오

```
시나리오: 사용자가 [학습 중지]를 클릭한 직후, 마지막 학습 스텝이 완료되어
         TrainingWorker가 completed 메시지를 Queue에 put하려는 순간이 겹침.

가능한 Queue 상태:
  A. stopped만 있음     → _handle_stopped() 호출
  B. completed만 있음   → _handle_completed() 호출
  C. completed + stopped 순서 → completed 먼저 드레인
  D. stopped + completed 순서 → stopped 먼저 드레인 → break → completed 잔류
```

### 6.2 우선순위 규칙 (확정)

| 규칙 | 내용 |
|------|------|
| **R-RACE-01** | `_drain_queue()`는 Queue를 순서대로 처리한다. 첫 번째 종료 메시지(completed/error/stopped)를 처리한 즉시 break한다. |
| **R-RACE-02** | 시나리오 D의 경우 — stopped가 먼저 처리되어 "중단" 레코드가 기록된다. Queue에 남은 completed 메시지는 다음 rerun에서 드레인되지 않는다. 이유: `_reset_run_state()` 후 `_result_queue = None`으로 초기화되므로 Q 참조 소멸. |
| **R-RACE-03** | 시나리오 C의 경우 — completed가 먼저 처리되어 "완료" 레코드가 기록된다. 이후 stopped 메시지는 소멸. |
| **R-RACE-04** | 사용자의 [학습 중지] 클릭 의도가 항상 우선한다는 보장은 없다. stop_event.set() 이후 TrainingWorker가 이미 completed를 put했다면 "완료"로 처리되어 모델이 저장된다. 이것은 정상 동작이다. |

**요약**: 경쟁 조건은 Queue의 메시지 도착 순서로 자연 해결된다. 별도의 락이나 플래그 없이 "먼저 도착한 종료 메시지가 우선"이다.

### 6.3 미아 스레드(orphan thread) 처리

```python
# tabs/tab4_training.py — render() 최상위 guard

def _guard() -> None:
    """
    미아 스레드 감지: worker가 살아있는데 result_queue가 None인 경우.
    브라우저 새로고침 후 재진입 시 발생 가능.
    """
    worker = st.session_state.get("_worker")
    if worker is not None and worker.is_alive():
        q = st.session_state.get("_result_queue")
        if q is None:
            # 새로고침으로 Queue 참조 소멸 — 스레드는 daemon이므로 강제 종료 불가
            # 상태만 초기화하고 UI에 안내
            _reset_run_state()
            st.info(
                "새로고침으로 인해 학습 상태를 확인할 수 없습니다. "
                "새로 학습을 시작하거나 탭5에서 히스토리를 확인하세요."
            )
```

> **참고**: `worker.daemon = True` (04.B.5.2 R-THREAD-03)이므로 Streamlit 프로세스가 살아있는 한 학습은 계속된다. 새로고침 후 탭5에서 history.json을 확인하면 완료 레코드가 있을 수 있다.

---

## 7. 탭 Guard 조건 및 session_state 쓰기 권한

### 7.1 탭별 Guard 조건 (진입 차단 기준)

| 탭 | Guard 조건 (미충족 시 진입 차단) | 차단 메시지 |
|----|--------------------------------|-------------|
| **탭1** | 없음 | — |
| **탭2** | `dataset_path is not None` | `MSG["NO_DATASET"]` |
| **탭3** | `preprocessing_config is not None` | `MSG["NO_PREPROCESSING"]` |
| **탭4** | `dataset_path is not None` AND `preprocessing_config is not None` AND `model_config is not None` | 미충족 항목별 메시지 |
| **탭5** | Guard 없음. `experiments == {}` 이면 `MSG["NO_EXPERIMENTS"]` 표시 후 렌더링 계속 | — |
| **탭6** | `selected_experiment_id is not None` | `MSG["NO_SELECTED_EXP"]` |

**차단 구현 패턴**:
```python
def _guard() -> None:
    if st.session_state["dataset_path"] is None:
        st.warning(MSG["NO_DATASET"])
        st.stop()
```

`st.stop()`은 이후 코드 실행을 중단하므로 `return`보다 안전하다.

### 7.2 탭별 session_state 쓰기 권한 (확정)

| session_state 키 | Write 탭 | 조건 |
|-----------------|----------|------|
| `dataset_path` | 탭1 | 경로 검증 성공 시 |
| `dataset_meta` | 탭1 | 경로 검증 성공 시 |
| `preprocessing_config` | 탭2 | [설정 저장] 버튼 클릭 시 |
| `model_config` | 탭3 | [설정 저장] 버튼 클릭 시 |
| `device_info` | 탭3 | 탭3 최초 진입 시 1회 (idempotent) |
| `experiments` | 탭4 (추가), 탭5 (삭제) | 학습 완료/중단, 실험 삭제 |
| `current_run_status` | 탭4 | 학습 시작/완료/중단 |
| `current_exp_id` | 탭4 | 학습 시작 시 설정, 완료/중단 시 None |
| `selected_experiment_id` | 탭5 | 실험 행 클릭 시 |
| `anomaly_map_threshold` | 탭6 | Threshold 슬라이더 변경 시 |
| `_stop_event` | 탭4 내부 | 학습 시작 시 |
| `_result_queue` | 탭4 내부 | 학습 시작 시 |
| `_progress` | 탭4 내부 | `_handle_progress()` 호출 시 |
| `_log_lines` | 탭4 내부 | `_handle_log()` 호출 시 |
| `_loss_history` | 탭4 내부 | `_handle_progress()` 호출 시 |
| `_anomaly_maps_{exp_id}` | 탭6 (`cache_manager`) | 캐시 미스 시 추론 후 |

### 7.3 탭1 데이터셋 경로 변경 시 연쇄 초기화

```python
# tabs/tab1_data_folder.py

def _handle_path_change(new_path: str) -> None:
    """
    새 경로가 기존 dataset_path와 다를 때 하위 session_state 초기화.
    이전 설정이 새 데이터셋에 맞지 않을 수 있으므로 리셋.
    """
    if new_path != st.session_state.get("dataset_path"):
        st.session_state["preprocessing_config"] = None
        st.session_state["model_config"] = None
        st.session_state["device_info"] = None
        # experiments, selected_experiment_id는 유지 (이전 실험 보존)
```

---

## 8. 모듈 간 호출 권한 매트릭스

> 04.B.4의 "금지된 의존성"을 표 형식으로 확장한다.

| 호출자 \ 피호출자 | session_state | Anomalib | 파일시스템 직접 | storage.py | config_manager.py | cache_manager.py |
|-----------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `tabs/*` | Read ✅ / Write ✅ | ❌ 금지 | ❌ 금지 | ✅ | ✅ | ✅ |
| `utils/image_utils.py` | ❌ 금지 | ❌ 금지 | Read ✅ | ❌ | ❌ | ❌ |
| `utils/metrics.py` | ❌ 금지 | ❌ 금지 | ❌ 금지 | ❌ | ❌ | ❌ |
| `utils/model_factory.py` | ❌ 금지 | ✅ | ❌ 금지 | ✅ (penalty dir) | ✅ (config read) | ❌ |
| `utils/training_worker.py` | ❌ 금지 | ✅ | Write ✅ (log only) | ✅ (log writer) | ❌ | ❌ |
| `utils/storage.py` | ❌ 금지 | ❌ 금지 | Read/Write ✅ | — | ✅ (config read) | ❌ |
| `utils/config_manager.py` | ❌ 금지 | ❌ 금지 | Read/Write ✅ | ❌ | — | ❌ |
| `utils/cache_manager.py` | Write ✅ | ❌ 금지 | ❌ 금지 | ❌ | ❌ | — |
| `components/sidebar.py` | Read ✅ | ❌ 금지 | ❌ 금지 | ❌ | ❌ | ❌ |

**특이사항**:
- `training_worker.py`의 파일시스템 쓰기는 **로그 파일(`./logs/`)만** 허용. 모델/히스토리 저장은 반드시 완료 후 메인 스레드(탭4)가 `storage.save_completed_experiment()` 경유.
- `cache_manager.py`는 session_state를 직접 쓰는 유일한 utils 모듈 — 메인 스레드에서만 호출해야 함.

---

## 9. 구현 검증 체크리스트

> 이 문서 기준으로 구현 완료 여부를 확인하는 체크리스트.

### Queue 프로토콜

- [ ] TrainingWorker가 `"completed"` / `"error"` / `"stopped"` 중 정확히 하나만 전송
- [ ] `"completed"` 전송 후 스레드 정상 종료 (추가 메시지 없음)
- [ ] `"stopped"` 메시지에 `step` 필드 포함
- [ ] `"error"` 메시지에 `traceback` 필드 포함 (str 타입)

### tab4 polling loop

- [ ] `q.get_nowait()` 사용 (블로킹 get 금지)
- [ ] 종료 메시지 처리 후 즉시 `break`
- [ ] `_reset_run_state()` 이후 `_result_queue = None` 처리
- [ ] `time.sleep(1.0)` 후 `st.rerun()` 호출 (1.0초 고정)
- [ ] `_handle_completed()` 내 `del msg["model"]` + `torch.cuda.empty_cache()`

### 에러 계약

- [ ] `load_history()` — 어떤 예외도 전파하지 않음
- [ ] `load_config()` — 어떤 예외도 전파하지 않음
- [ ] `compute_metrics()` — 단일 클래스 데이터 시 `auc=0.0` 반환
- [ ] `check_disk_before_save()` — Streamlit context에서만 호출

### Guard 조건

- [ ] 탭2: `dataset_path is None` 시 `st.stop()`
- [ ] 탭3: `preprocessing_config is None` 시 `st.stop()`
- [ ] 탭4: 3개 조건 모두 확인 후 학습 시작 버튼 활성화
- [ ] 탭6: `selected_experiment_id is None` 시 `st.stop()`

### 경쟁 조건

- [ ] `_drain_queue()` 종료 메시지 처리 후 즉시 break
- [ ] `_reset_run_state()` 호출 직후 `_result_queue = None` 설정
- [ ] 미아 스레드 감지 로직 (`_guard()` 내 `worker.is_alive()` 확인)

### 신규 모듈 파일 생성

- [ ] `utils/storage.py` 생성 (§2.1 전체 구현)
- [ ] `utils/cache_manager.py` 생성 (§2.2 전체 구현)
- [ ] 두 파일 모두 `utils/__init__.py`에 import 추가

---

*이 문서는 04_System_Architecture.md §B.3 (모듈 시그니처)와 05_Data_Model_and_Storage_Strategy.md §3~§10 (스토리지 구현)의 보완 명세이다.*
*다음: [07_Backend_Service_Design.md](./07_Backend_Service_Design.md)*

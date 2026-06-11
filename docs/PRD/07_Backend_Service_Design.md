# 07. Backend Service Design

> **참조 문서**: `04_System_Architecture.md` §B.5 (비동기 처리 아키텍처), `06_API_Specification.md` (FastAPI 엔드포인트 명세), `08_AI_ML_Integration.md` §B.4~B.6 (학습 구현)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **최종수정**: 2026-06-11
> **목적**: FastAPI 서비스 레이어의 비즈니스 로직 흐름을 구현 가능한 수준으로 명세한다. 아래 4가지 영역이 이 문서에서 확정된다:
> 1. experiment_id 생성 규칙 (코드 포함)
> 2. 학습 시작 API 핸들러 전체 흐름 (POST /api/training/start)
> 3. TrainingWorker 생성자 인스턴스 변수 명세
> 4. AnomalyMap 비동기 빌드 서비스 파이프라인
>
> **역할 분리**: 04.B.5의 스레드 모델, 05_Data_Model.md의 TypeScript 타입 명세, 08.B.4~B.6의 학습 루프 구현은 이 문서에서 반복하지 않고 참조만 한다.

---

## 버전 이력

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 초기 작성 — Streamlit utils/ 서비스 레이어 명세 (experiment_id, TrainingWorker, 탭5 추론 서비스) |
| v1.1 | 2026-05-26 | §12~§14: 비전검사 추론 서비스(Streamlit session_state 기반) 추가 |
| v2.0 | 2026-06-11 | FastAPI 라우터/서비스 구조로 전면 재작성. TrainingManager·InspectionManager 서버 상태 관리 도입. session_state → 서버 메모리로 교체. WS 브로드캐스트로 클라이언트 갱신 교체. AnomalyMap 비동기 job 패턴 추가 (§7). Streamlit 서비스 레이어 → 문서 하단 v1.x 참고로 이동. |

---

## 목차

1. [서비스 책임 범위](#1-서비스-책임-범위)
2. [experiment_id 생성 서비스](#2-experiment_id-생성-서비스)
3. [학습 시작 서비스 (POST /api/training/start)](#3-학습-시작-서비스)
4. [TrainingWorker 생성자 명세](#4-trainingworker-생성자-명세)
5. [학습 중 Progress 보고 주기](#5-학습-중-progress-보고-주기)
6. [학습 완료/중단 후처리 흐름](#6-학습-완료중단-후처리-흐름)
7. [AnomalyMap 비동기 빌드 서비스](#7-anomalymap-비동기-빌드-서비스)
8. [TrainingManager — 단일 워커 보장](#8-trainingmanager--단일-워커-보장)
9. [메모리 관리 규칙](#9-메모리-관리-규칙)
10. [서비스 데이터 흐름 요약](#10-서비스-데이터-흐름-요약)
11. [구현 체크리스트](#11-구현-체크리스트)
12. [InspectionManager — 추론 서비스](#12-inspectionmanager--추론-서비스)
13. [자동 검사 WebSocket 설계](#13-자동-검사-websocket-설계)
14. [검사 서비스 체크리스트](#14-검사-서비스-체크리스트)

---

## 1. 서비스 책임 범위

### 이 문서가 확정하는 것 (v2.0)

| 서비스 | 위치 (v2.0) | 핵심 결정 사항 |
|--------|-------------|----------------|
| ID 생성 | `api/services/training_service.py` | experiment_id 생성 코드 확정 |
| 학습 시작 | `api/routers/training.py` — `POST /api/training/start` 핸들러 | 요청 수신부터 `TrainingWorker.start()`까지 전체 순서 |
| TrainingManager | `api/services/training_manager.py` | 서버 싱글턴 — status, worker, ws_clients 관리 |
| TrainingWorker | `utils/training_worker.py` | 백그라운드 ML 학습 스레드 (v1.x에서 동일) |
| WS 브로드캐스트 | `api/ws/training_ws.py` — `WS /ws/training` | queue → WebSocket 팬아웃 |
| AnomalyMap 빌드 | `api/routers/anomaly_map.py` — `POST /api/anomaly-map/{expId}/build` | 비동기 job 패턴 |
| InspectionManager | `api/services/inspection_manager.py` | 서버 싱글턴 — active_model, records, test_pool 관리 |
| 수동 검사 | `api/routers/inspection.py` — `POST /api/inspection/run` | 동기 추론 1회 실행 |
| 자동 검사 WS | `api/routers/inspection.py` — `WS /ws/inspection/auto` | asyncio 서버 주도 루프 |

### 이 문서가 다루지 않는 것 (참조 문서)

| 내용 | 참조 |
|------|------|
| 비동기 스레드 모델 설계 | `04_System_Architecture.md §B.5` |
| 전체 API 엔드포인트 명세 (Request/Response 스키마) | `06_API_Specification.md` |
| EfficientAD / PatchCore 학습 루프 코드 | `08_AI_ML_Integration.md §B.4, B.5, B.6` |
| 3단계 원자성 저장 프로토콜 | `05_Data_Model_and_Storage_Strategy.md §6` |
| TypeScript 타입과 PRD 스키마 비교 | `05_Data_Model.md` |

---

## 2. experiment_id 생성 서비스

### 2.1 생성 규칙

`00_Global_Context_Document.md §8` R-NAMING-03, R-ID-01 기준:

```
형식: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_소문자_16진수}
예:   efficientad_20260509_143022_3b9f
      patchcore_20260509_143022_a1c4
```

### 2.2 생성 코드 (확정 — v2.0 위치 업데이트)

```python
# api/services/training_service.py
# (v1.x: tabs/tab3_training.py 에서 이동)

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
`storage.prepare_model_dir(experiment_id)` 호출 시 디렉토리 이미 존재하면 RuntimeError 발생 → 실질적 충돌 방지.

---

## 3. 학습 시작 서비스

> **v2.0**: Streamlit `_handle_start_training()` → FastAPI `POST /api/training/start` 핸들러로 전면 교체.

### 3.1 사전 조건 체크 순서

`POST /api/training/start` 요청 수신 시 아래 순서대로 검증한다. 첫 번째 실패에서 즉시 HTTP 에러 반환.

```
① training_manager.status == "idle"        → 아니면 409 Conflict
② body.dataset_path 유효성 (경로 존재 여부) → 아니면 400 ERR_DATASET_NOT_FOUND
③ body.preprocessing_config not None       → 아니면 400 ERR_PREPROCESSING_CONFIG_MISSING
④ body.model_config not None               → 아니면 400 ERR_MODEL_CONFIG_MISSING
⑤ 디스크 여유 공간 체크                   → < 500MB면 400 ERR_DISK_SPACE_INSUFFICIENT
⑥ [EfficientAD만] ImageNet penalty 디렉토리 검증
```

### 3.2 학습 시작 핸들러 전체 구현 (v2.0)

```python
# api/routers/training.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.services.training_manager import training_manager
from api.services.training_service import generate_experiment_id, generate_created_at
from utils.training_worker import TrainingWorker
from utils.config_manager import save_config_section
import threading, queue

router = APIRouter(prefix="/api/training")

@router.post("/start")
async def start_training(
    body: TrainingStartRequest,
    background_tasks: BackgroundTasks
):
    """
    POST /api/training/start
    학습 시작 요청을 받아 TrainingWorker를 백그라운드에서 실행한다.
    """
    # ── 사전 조건 검증 ────────────────────────────────────────────────────────
    if training_manager.status != "idle":
        raise HTTPException(status_code=409, detail={"code": "ERR_TRAINING_ALREADY_RUNNING"})

    _check_disk_space(body.model_config.model_type)

    if body.model_config.model_type == "efficientad":
        _validate_imagenet_penalty_dir()

    # ── experiment_id 및 메타데이터 생성 ──────────────────────────────────────
    exp_id = generate_experiment_id(body.model_config.model_type)
    created_at = generate_created_at()

    experiment_name = body.experiment_name or \
        f"{body.model_config.model_type.upper()} {exp_id[-4:]}"

    # ── configs.yaml experiment 섹션 저장 ────────────────────────────────────
    save_config_section(
        section="experiment",
        data={"name": experiment_name, "created_at": created_at}
    )

    # ── 동시성 객체 생성 ──────────────────────────────────────────────────────
    stop_event   = threading.Event()
    pause_event  = threading.Event()
    result_queue = queue.Queue()

    # ── TrainingWorker 생성 및 시작 ────────────────────────────────────────────
    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=body.model_config.dict(),
        preprocessing_config=body.preprocessing_config.dict(),
        dataset_path=body.dataset_path,
        device=_get_device(),           # cuda 또는 cpu 자동 감지
        stop_event=stop_event,
        pause_event=pause_event,
        result_queue=result_queue,
    )
    worker.daemon = True
    worker.start()

    # ── TrainingManager 상태 갱신 ────────────────────────────────────────────
    training_manager.set_running(
        exp_id=exp_id,
        worker=worker,
        stop_event=stop_event,
        pause_event=pause_event,
        result_queue=result_queue,
    )

    # ── 백그라운드: result_queue → WebSocket 팬아웃 태스크 등록 ────────────────
    background_tasks.add_task(_consume_and_broadcast, training_manager)

    return {"exp_id": exp_id, "created_at": created_at}
```

### 3.3 자동 생성 실험명 형식

| 조건 | 형식 | 예 |
|------|------|---|
| `body.experiment_name` 입력 있음 | 입력값 그대로 (최대 64자) | "스크류 CLAHE 기본 실험" |
| `body.experiment_name` 없음 | `{MODEL_TYPE} {rand_4자리}` | "EFFICIENTAD 3b9f" |

---

## 4. TrainingWorker 생성자 명세

> `TrainingWorker` 클래스 자체는 v1.x와 동일하다. Streamlit session_state 대신 `result_queue`를 통해 FastAPI 서비스 레이어와 통신한다.

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
    외부 → 내부 통신: stop_event.set(), pause_event.set()/clear()
    내부 → 외부 통신: result_queue.put(QueueMessage)
      ※ v2.0: result_queue 메시지는 TrainingManager가 읽어 WebSocket 클라이언트에 브로드캐스트한다.
              session_state 직접 쓰기 없음.
    """

    def __init__(
        self,
        experiment_id: str,
        model_config: dict,              # 00_Global §1.7 model_config 스키마
        preprocessing_config: dict,      # 00_Global §1.6 preprocessing_config 스키마
        dataset_path: str,               # MVTec AD 데이터셋 루트 경로
        device: str,                     # "cuda" | "cpu"
        stop_event: threading.Event,
        result_queue: queue.Queue,
        pause_event: threading.Event | None = None,
        # EfficientAD resume 파라미터
        start_step: int = 0,
        student_state_dict: dict | None = None,
        autoencoder_state_dict: dict | None = None,
        optimizer_st_state_dict: dict | None = None,
        optimizer_ae_state_dict: dict | None = None,
        scheduler_st_state_dict: dict | None = None,
        scheduler_ae_state_dict: dict | None = None,
        loss_history: list | None = None,
        # PatchCore resume 파라미터
        start_batch_idx: int = 0,
        accumulated_features: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(name=f"TrainingWorker-{experiment_id}")

        self.experiment_id        = experiment_id
        self.model_config         = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path         = dataset_path
        self.device               = device
        self.stop_event           = stop_event
        self.result_queue         = result_queue
        self.pause_event          = pause_event if pause_event is not None else threading.Event()

        # EfficientAD resume
        self.start_step               = start_step
        self.student_state_dict       = student_state_dict
        self.autoencoder_state_dict   = autoencoder_state_dict
        self.optimizer_st_state_dict  = optimizer_st_state_dict
        self.optimizer_ae_state_dict  = optimizer_ae_state_dict
        self.scheduler_st_state_dict  = scheduler_st_state_dict
        self.scheduler_ae_state_dict  = scheduler_ae_state_dict
        self.loss_history: list       = list(loss_history) if loss_history else []

        # PatchCore resume
        self.start_batch_idx      = start_batch_idx
        self.accumulated_features = accumulated_features

        # 내부 상태 (run() 실행 중 설정)
        self._model = None
        self._start_time: float = 0.0
        self._log_writer = None
```

### 4.2 인스턴스 변수 생명주기

| 변수 | 설정 시점 | 해제 시점 |
|------|-----------|-----------|
| `_model` | 모델 초기화 완료 후 | `"completed"` 메시지 수신 후 TrainingManager에서 `del msg["model"]` |
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
    """타임스탬프 + 메시지를 로그 파일과 result_queue(log 메시지)에 동시 기록."""
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

> result_queue에 넣어진 메시지는 TrainingManager의 `_consume_and_broadcast()`가 읽어 WS /ws/training 클라이언트에 팬아웃한다. (§8.2 참조)

### 5.1 모델별 보고 전략

| 모델 | 보고 시점 | 이유 |
|------|-----------|------|
| **EfficientAD** | 매 100 step (첫 번째 step에서 즉시 1회) | UI 반응성 향상 (A-08) |
| **PatchCore** | feature 추출 완료 1회 + coreset 완료 1회 | 학습이 단일 에포크이므로 step 개념 없음 |

### 5.2 result_queue 메시지 타입 전체 목록

WS /ws/training 으로 팬아웃되는 13가지 메시지 타입 (05_Data_Model.md 3.3절 TypeScript 타입 기준):

| type | 방향 | 설명 |
|------|------|------|
| `snapshot` | Server→Client | WS 연결 직후 현재 상태 전체 스냅샷 |
| `progress` | Server→Client | 학습 step 진행률 + loss + 경과 시간 |
| `log` | Server→Client | 학습 로그 1줄 |
| `stage` | Server→Client | 현재 학습 단계명 변경 |
| `paused` | Server→Client | 일시정지 완료 + 체크포인트 경로 |
| `completed` | Server→Client | 학습 완료 + auc + duration_seconds |
| `stopped` | Server→Client | 사용자 중단 완료 |
| `error` | Server→Client | 예외 발생 + traceback |
| `batch_item_started` | Server→Client | 배치 항목 시작 |
| `batch_item_skipped` | Server→Client | 배치 항목 건너뜀 |
| `batch_item_error` | Server→Client | 배치 항목 실패 |
| `batch_stopped` | Server→Client | 배치 학습 사용자 중단 |
| `batch_completed` | Server→Client | 배치 학습 완료 통계 |

### 5.3 PatchCore progress 메시지 예시

```python
# PatchCore 학습 루프 내부 (08_AI_ML_Integration.md §B.5 참조)
self.result_queue.put({
    "type": "progress",
    "step": 0, "total": 1,
    "loss": 0.0,  # PatchCore는 loss 없음, 0.0 고정
    "elapsed": round(time.time() - self._start_time, 1)
})
```

---

## 6. 학습 완료/중단 후처리 흐름

> **v2.0**: Streamlit `st.rerun()` + session_state 갱신 → WebSocket 브로드캐스트 + 서버 상태 갱신으로 전면 교체.

### 6.1 result_queue → WebSocket 브로드캐스트 루프

```python
# api/services/training_manager.py

async def _consume_and_broadcast(self) -> None:
    """
    BackgroundTask로 실행.
    result_queue에서 메시지를 읽어 연결된 모든 WS 클라이언트에 브로드캐스트한다.
    completed / stopped / error 메시지 수신 시 후처리 후 루프 종료.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    while True:
        try:
            msg = await loop.run_in_executor(
                None, lambda: self.result_queue.get(timeout=1.0)
            )
        except queue.Empty:
            if self.status == "idle":
                break
            continue

        # WS 팬아웃
        dead_clients = set()
        for ws in list(self.ws_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead_clients.add(ws)
        self.ws_clients -= dead_clients

        # 후처리: 종료 메시지에서 상태 전이
        if msg["type"] == "completed":
            await self._handle_completed(msg)
            break
        elif msg["type"] == "stopped":
            await self._handle_stopped(msg)
            break
        elif msg["type"] == "error":
            await self._handle_error(msg)
            break
        elif msg["type"] == "paused":
            self.status = "paused"
            self.last_ckpt_path = msg.get("ckpt_path")
```

### 6.2 완료 후처리 순서 (v2.0 — TrainingManager)

```
[completed 메시지 수신 — _handle_completed(msg)]
  │
  ├─ 1. compute_threshold() → float threshold
  │       percentile: np.percentile(정상_scores, threshold_value)
  │       absolute:   float(threshold_value)
  │
  ├─ 2. compute_metrics(y_true, anomaly_scores, threshold)
  │       → metrics dict (00_Global §1.2 스키마)
  │
  ├─ 3. _build_experiment_record()
  │       status="completed", metrics, duration_seconds 포함
  │
  ├─ 4. save_completed_experiment(exp_id, model, record)
  │       (05_Data_Model §6 — 3단계 원자성 프로토콜)
  │       model_path, configs_path 갱신 후 history.json append
  │
  ├─ 5. del msg["model"]  +  torch.cuda.empty_cache()  (§9.1)
  │
  └─ 6. _reset_training_state()
         training_manager.status = "idle"
         training_manager.current_exp_id = None
         (클라이언트는 completed WS 메시지에서 auc / duration_seconds 수신)
```

### 6.3 중단 후처리 순서 (v2.0)

```
[stopped 메시지 수신 — _handle_stopped(msg)]
  │
  ├─ 1. _build_experiment_record()
  │       status="중단", metrics=None, model_path=None
  │
  ├─ 2. append_experiment(record)  (history.json)
  │
  └─ 3. _reset_training_state()
         (클라이언트는 stopped WS 메시지 수신)
```

### 6.4 오류 후처리 순서 (v2.0)

```
[error 메시지 수신 — _handle_error(msg)]
  │
  ├─ 1. _build_experiment_record() status="실패"
  │       [확인 필요: status="실패" 레코드가 history.json에 기록되는지 확인]
  │
  └─ 2. _reset_training_state()
         (클라이언트는 error WS 메시지에서 traceback 수신)
```

### 6.5 일시정지 후처리 순서 (v2.0)

```
[paused 메시지 수신]
  │
  ├─ 1. training_manager.status = "paused"
  ├─ 2. training_manager.last_ckpt_path = msg["ckpt_path"]
  └─ (루프 계속 — paused는 종료 메시지가 아님)
     클라이언트: paused WS 메시지 수신 → Training 화면 일시정지 UI 표시
```

### 6.6 _build_experiment_record() 구현 (v2.0)

```python
# api/services/training_service.py

def build_experiment_record(
    exp_id: str,
    status: str,                    # "completed" | "중단" | "실패"
    metrics: dict | None,
    duration_seconds: int | None,
    model_config: dict,
    preprocessing_config: dict,
    dataset_path: str,
    experiment_name: str,
    created_at: str,
) -> dict:
    """
    00_Global §1.1 experiment 스키마에 맞는 레코드 생성.
    v2.0: session_state 대신 파라미터로 직접 받음.
    status != "completed" 시 metrics, model_path, configs_path 강제 None.
    """
    record = {
        "experiment_id": exp_id,
        "name": experiment_name,
        "status": status,
        "created_at": created_at,
        "model_type": model_config["model_type"],
        "preprocessing_method": preprocessing_config["method"],
        "background_method": preprocessing_config.get("background_method", "none"),
        "preprocessing_params": preprocessing_config.get("params"),
        "model_params": model_config["params"],
        "threshold_method": model_config["threshold_method"],
        "threshold_value": model_config["threshold_value"],
        "dataset_path": dataset_path,
        "image_size": preprocessing_config["image_size"],
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "model_path": None,
        "configs_path": None,
        "early_stopped": None,
    }

    # R-05: status != "completed" 불변 조건
    if status != "completed":
        record["metrics"] = None
        record["model_path"] = None
        record["configs_path"] = None

    return record
```

---

## 6b. checkpoint_manager 모듈 명세

> `utils/checkpoint_manager.py` — 일시정지 기능의 영속 레이어 (v1.x에서 동일, 유지)

### 6b.1 공개 함수

| 함수 | 시그니처 | 설명 |
|------|----------|------|
| `save_checkpoint` | `(data: dict, exp_id: str, label: int) -> Path` | 체크포인트를 `.ckpt`로 저장하고 경로 반환 |
| `load_checkpoint` | `(path: str \| Path) -> dict` | 체크포인트 파일 로드 후 dict 반환 |
| `list_checkpoints` | `() -> list[Path]` | 저장된 `.ckpt` 파일을 수정시간 역순으로 반환 |
| `delete_checkpoint` | `(path: str \| Path) -> bool` | 체크포인트 파일 삭제. 성공 시 True |

### 6b.2 저장 경로 규칙

```
CHECKPOINT_DIR = Path("./models/checkpoints")
파일명: {exp_id}_step{label}.ckpt
  - EfficientAD: label = 현재 step 번호
  - PatchCore:   label = 현재 batch_idx
```

### 6b.3 체크포인트 데이터 스키마

**공통 필드**:
```
experiment_id        str
model_type           "efficientad" | "patchcore"
model_config         dict  (00_Global §1.7)
preprocessing_config dict  (00_Global §1.6)
dataset_path         str
created_at           str   (ISO 8601, KST)
```

**EfficientAD 추가 필드**:
```
step, total_steps, loss_history,
student_state_dict, autoencoder_state_dict,
optimizer_st_state_dict, optimizer_ae_state_dict,
scheduler_st_state_dict, scheduler_ae_state_dict
```

**PatchCore 추가 필드**:
```
batch_idx, total_batches, accumulated_features (torch.Tensor)
```

### 6b.4 저장 메커니즘

`torch.save(data, path)` 사용. 로드: `torch.load(path, map_location="cpu", weights_only=False)`.

---

## 7. AnomalyMap 비동기 빌드 서비스

> **v2.0**: Streamlit `run_batch_inference()` 동기 블로킹 → FastAPI 비동기 job 패턴으로 전면 교체.

### 7.1 전체 파이프라인 (v2.0 — 비동기 job)

```
[Explorer: POST /api/anomaly-map/{expId}/build]
  │
  ├─ 1. GET /api/anomaly-map/{expId}/status 먼저 확인
  │       status == "ready" → 빌드 불필요, 바로 /images 조회
  │       status == "not_built" → 빌드 필요
  │
  ├─ 2. POST /api/anomaly-map/{expId}/build
  │       → job_id 즉시 반환 ({"job_id": "anomalymap_3b9f_..."})
  │       → 백그라운드 스레드에서 빌드 실행
  │
  ├─ 3. GET /api/anomaly-map/job/{jobId} polling (1초 간격)
  │       → {"status": "running", "progress": 45, "total": 100}
  │       → {"status": "completed"}
  │       → {"status": "failed", "error": "..."}
  │
  ├─ 4. GET /api/anomaly-map/{expId}/images
  │       → AnomalyMapImagesResponse (이미지 목록 + TP/FP/TN/FN 통계)
  │
  ├─ 5. GET /api/anomaly-map/{expId}/image/{path}/triplet
  │       → Triplet PNG (원본/GT마스크/Heatmap)
  │
  └─ 6. 내보내기:
         GET  /api/anomaly-map/{expId}/export/csv
         POST /api/anomaly-map/{expId}/export/zip → job polling
         GET  /api/anomaly-map/zip/{jobId}        → 파일 다운로드
```

### 7.2 빌드 서비스 핸들러 (v2.0)

```python
# api/routers/anomaly_map.py

@router.post("/api/anomaly-map/{exp_id}/build")
async def build_anomaly_map(exp_id: str, background_tasks: BackgroundTasks):
    """
    비동기 Anomaly Map 빌드 job 시작.
    즉시 job_id 반환. 실제 빌드는 백그라운드 스레드에서 실행.
    """
    # 실험 레코드 확인
    experiment = load_experiment(exp_id)
    if experiment is None:
        raise HTTPException(404, detail={"code": "ERR_EXPERIMENT_NOT_FOUND"})
    if experiment["status"] != "completed":
        raise HTTPException(400, detail={"code": "ERR_EXPERIMENT_NOT_COMPLETED"})

    # 이미 빌드 완료된 경우 캐시 반환
    if job_registry.is_built(exp_id):
        return {"job_id": None, "status": "already_built"}

    job_id = f"anomalymap_{exp_id[-4:]}_{uuid.uuid4().hex[:4]}"
    job_registry.create(job_id, exp_id)

    background_tasks.add_task(_run_build_job, job_id, exp_id, experiment)
    return {"job_id": job_id}


async def _run_build_job(job_id: str, exp_id: str, experiment: dict) -> None:
    """
    백그라운드에서 모델 로드 → 전체 추론 → 캐시 저장.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _build_sync, job_id, exp_id, experiment)
        job_registry.set_completed(job_id)
    except Exception as e:
        job_registry.set_failed(job_id, str(e))


def _build_sync(job_id: str, exp_id: str, experiment: dict) -> None:
    """
    동기 빌드 실행 (별도 스레드에서 호출).
    """
    device = _get_device()
    model = load_model_for_inference(
        exp_id=exp_id,
        model_path=experiment["model_path"],
        model_config=load_config(experiment["configs_path"]).get("model", {}),
        device=device
    )
    image_paths = _collect_test_image_paths(experiment["dataset_path"])
    total = len(image_paths)

    anomaly_maps = []
    for i, img_path in enumerate(image_paths):
        _, tensor = apply_preprocessing(img_path, experiment.get("preprocessing_params"))
        tensor = tensor.unsqueeze(0).to(device)
        anomaly_map = run_inference(model, tensor)
        anomaly_maps.append(anomaly_map)

        if (i + 1) % 5 == 0 or (i + 1) == total:
            job_registry.update_progress(job_id, progress=i + 1, total=total)

    anomaly_map_cache.set(exp_id, {
        "anomaly_maps": np.stack(anomaly_maps, axis=0).astype(np.float32),
        "image_paths":  image_paths,
    })

    # 추론 완료 후 메모리 해제
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
```

### 7.3 Threshold 슬라이더 연동 (v2.0)

Explorer AnomalyMap 화면은 Threshold 값을 클라이언트 state (`anomalyMapStore.threshold`)로 관리한다.
서버 API는 stateless — 슬라이더 값은 요청 파라미터로 전달:

```
GET /api/anomaly-map/{expId}/images?threshold=0.42
→ 서버가 해당 threshold 기준으로 TP/FP/TN/FN 분류 후 반환
```

> 슬라이더 300ms debounce는 Explorer React 클라이언트 측에서 처리한다.

### 7.4 load_model_for_inference() — 모델 로드 함수 (유지)

```python
# utils/model_factory.py

def load_model_for_inference(
    exp_id: str,
    model_path: str,
    model_config: dict,
    device: str
) -> object:
    """
    저장된 model_state_dict.pth 로드 후 추론 가능한 모델 반환 (eval() 모드).
    Raises RuntimeError: pth 파일 없거나 state_dict 불일치 시.
    """
    pth_path = Path(model_path) / "model_state_dict.pth"
    if not pth_path.exists():
        raise RuntimeError(f"ERR_MODEL_FILE_NOT_FOUND: {pth_path}")

    model_type = model_config["model_type"]
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
        raise RuntimeError(f"ERR_MODEL_LOAD_FAILED: {e}") from e

    model.to(device)
    model.eval()
    return model
```

---

## 8. TrainingManager — 단일 워커 보장

> **v2.0**: `session_state["current_run_status"] != "idle"` 체크 → `TrainingManager` 싱글턴 상태 체크로 교체.

### 8.1 TrainingManager 클래스 명세

```python
# api/services/training_manager.py

from dataclasses import dataclass, field
import threading, queue
from fastapi import WebSocket

@dataclass
class TrainingManager:
    """
    FastAPI 서버 싱글턴. 모듈 레벨에서 1회 초기화.
    단일 사용자 환경(A-01) 기준 — 동시 학습 1개만 허용.
    """
    status: str = "idle"              # "idle" | "running" | "paused"
    current_exp_id: str | None = None
    worker: threading.Thread | None = None
    stop_event: threading.Event | None = None
    pause_event: threading.Event | None = None
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    last_ckpt_path: str | None = None
    ws_clients: set = field(default_factory=set)  # set[WebSocket]

    def set_running(self, exp_id, worker, stop_event, pause_event, result_queue):
        self.status = "running"
        self.current_exp_id = exp_id
        self.worker = worker
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.result_queue = result_queue

    def reset(self):
        self.status = "idle"
        self.current_exp_id = None
        self.worker = None
        self.stop_event = None
        self.pause_event = None
        self.result_queue = queue.Queue()
        self.last_ckpt_path = None

    def get_snapshot(self) -> dict:
        """신규 WS 연결 시 현재 상태 전체를 담은 snapshot 메시지 반환."""
        return {
            "type": "snapshot",
            "status": self.status,
            "exp_id": self.current_exp_id,
            "batch_mode": False,  # [확인 필요: 배치 모드 상태 추적 방식]
            "last_ckpt_path": self.last_ckpt_path,
        }


# 모듈 레벨 싱글턴
training_manager = TrainingManager()
```

### 8.2 WebSocket 핸들러 — 연결 및 스냅샷 전송

```python
# api/routers/training.py

@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    await websocket.accept()
    training_manager.ws_clients.add(websocket)
    try:
        # 연결 즉시 현재 상태 스냅샷 전송
        await websocket.send_json(training_manager.get_snapshot())
        # 클라이언트 disconnect 대기
        while True:
            await websocket.receive_text()  # ping/pong 또는 연결 유지
    except WebSocketDisconnect:
        pass
    finally:
        training_manager.ws_clients.discard(websocket)
```

### 8.3 중복 학습 방지

```
POST /api/training/start 요청 시:
  training_manager.status != "idle"
    → HTTPException(409, {"code": "ERR_TRAINING_ALREADY_RUNNING"})
    → Explorer Training 화면에 에러 메시지 인라인 표시
```

---

## 9. 메모리 관리 규칙

### 9.1 GPU 메모리 해제 시점

| 이벤트 | 처리 |
|--------|------|
| 학습 완료 (`_handle_completed()`) | `del msg["model"]` + `torch.cuda.empty_cache()` |
| AnomalyMap 빌드 완료 (`_build_sync()`) | `del model` + `torch.cuda.empty_cache()` (device == "cuda"인 경우만) |
| 학습 중단/오류 | TrainingWorker가 model 객체 미전달 → 해제 불필요 |
| 비전검사 모델 교체 | 이전 모델 참조 해제 (`del inspection_manager.active_model.model`) + `torch.cuda.empty_cache()` |

### 9.2 해제 코드 패턴

```python
# 학습 완료 후 (TrainingManager._handle_completed)
del msg["model"]
if _get_device() == "cuda":
    torch.cuda.empty_cache()

# AnomalyMap 빌드 완료 후 (_build_sync)
del model
if device == "cuda":
    torch.cuda.empty_cache()
```

### 9.3 AnomalyMap 캐시 메모리

- `np.ndarray (N, H, W)` float32 ≈ 25 MB (100장, 256×256 기준)
- 서버 메모리 내 `anomaly_map_cache` 딕셔너리 유지 (exp_id → 캐시)
- 캐시 eviction: 최대 3개 실험 유지. `cache_manager.py`가 자동 처리.
- 명시적 해제 불필요: Python GC가 eviction 시 참조 해제.

---

## 10. 서비스 데이터 흐름 요약

```
[학습 시작 서비스 — v2.0]

Explorer → POST /api/training/start
  → generate_experiment_id()
  → save_config_section("experiment")
  → TrainingWorker(...)
  → worker.daemon = True
  → worker.start()
  → training_manager.set_running(...)
  → BackgroundTask: _consume_and_broadcast(training_manager)
  ← { exp_id, created_at }

Explorer → WS /ws/training (연결)
  ← snapshot 메시지 (즉시)
  ← progress / log / stage 메시지 (실시간 Push)
  ← completed / stopped / error 메시지 (종료)


[AnomalyMap 빌드 서비스 — v2.0]

Explorer → POST /api/anomaly-map/{expId}/build
  → job_id 즉시 반환
  → BackgroundTask: _run_build_job
       → _build_sync (별도 스레드)
           → load_model_for_inference()
           → run_batch_inference()
           → anomaly_map_cache.set()
           → del model + empty_cache()

Explorer → GET /api/anomaly-map/job/{jobId} polling
  ← { status: "running", progress, total }
  ← { status: "completed" }

Explorer → GET /api/anomaly-map/{expId}/images
  ← AnomalyMapImagesResponse (이미지 목록 + 통계)


[검사 서비스 — v2.0]

Vision → POST /api/inspection/model { experiment_id }
  → load_model_for_inference()
  → inspection_manager.set_active_model(model, metadata)
  → inspection_manager.reset_records()
  ← { active_model, gpu_warning }

Vision → POST /api/inspection/run
  → inspection_manager.run_single()
  → run_inference() → score, anomaly_map
  → verdict 판정 → InspectionResult 생성
  ← { seq, inspected_at, image_name, verdict, anomaly_score }

Vision → WS /ws/inspection/auto (연결)
  → Client 전송: "start"
  → [asyncio 루프]
      run_inference() → send result
      if 불량: send defect_stopped + break
      await asyncio.sleep(3)
      check for "stop" message
  ← result 메시지 (3초마다)
  ← defect_stopped 메시지 (불량 시)
  ← stopped 메시지 (사용자 중지 시)
```

---

## 11. 구현 체크리스트

### experiment_id 생성

- [ ] `generate_experiment_id(model_type)` — R-NAMING-03 형식 준수
- [ ] `generate_created_at()` — ISO 8601 KST 형식
- [ ] `uuid.uuid4().hex[:4]` — 4자리 소문자 16진수

### 학습 시작 핸들러 (POST /api/training/start)

- [ ] `training_manager.status != "idle"` → 409 Conflict
- [ ] 사전 조건 5단계 순서대로 검증
- [ ] `worker.daemon = True` 설정
- [ ] `training_manager.set_running()` 호출
- [ ] `BackgroundTasks.add_task(_consume_and_broadcast)` 등록
- [ ] 응답: `{ exp_id, created_at }`

### TrainingWorker

- [ ] `__init__()` — §4.1 파라미터 전체 포함
- [ ] `run()` — `finally` 블록에서 `_log_writer.close()`
- [ ] `_write_log()` — 파일 + result_queue 동시 기록
- [ ] EfficientAD progress: 매 100 step
- [ ] PatchCore progress: feature 추출 완료 1회 + coreset 완료 1회
- [ ] result_queue 메시지: session_state 직접 접근 없음

### AnomalyMap 빌드 서비스

- [ ] POST /api/anomaly-map/{expId}/build → 즉시 job_id 반환
- [ ] 백그라운드에서 `_build_sync()` 실행
- [ ] job_registry 진행률 업데이트 (5장마다)
- [ ] 빌드 완료 후 `del model` + `torch.cuda.empty_cache()`
- [ ] GET /api/anomaly-map/job/{jobId} → polling 응답

### TrainingManager

- [ ] 모듈 레벨 싱글턴 1개만 생성
- [ ] `get_snapshot()` — 신규 WS 연결 즉시 전송
- [ ] `ws_clients` 관리 — disconnect 시 자동 제거

### 메모리

- [ ] 학습 완료: `del msg["model"]` + `torch.cuda.empty_cache()`
- [ ] AnomalyMap 빌드 완료: 모델 해제
- [ ] anomaly_map_cache 최대 3개 유지

---

## 12. InspectionManager — 추론 서비스

> **v2.0**: `st.cache_resource` + Streamlit session_state → FastAPI 서버 싱글턴 `InspectionManager`로 전면 교체.

### 12.1 InspectionManager 클래스 명세

```python
# api/services/inspection_manager.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class LoadedModel:
    """PyTorch 모델 객체 + 메타데이터 컨테이너."""
    model: Any                          # torch.nn.Module (eval() 모드)
    metadata: dict                      # ActiveModel 스키마 (05_Data_Model.md 3.2절)
    device: str


@dataclass
class InspectionManager:
    """
    FastAPI 서버 싱글턴. 비전검사 서버 상태 전체를 관리.
    단일 사용자 환경(A-01): activeModel 1개, records 서버 메모리 보관.
    """
    active_model: LoadedModel | None = None
    records: list = field(default_factory=list)  # list[dict] — InspectionRecord 스키마
    seq_counter: int = 0
    test_pool: list = field(default_factory=list)  # list[tuple[str, str]] — (image_path, label)
    pool_index: int = 0

    def set_active_model(self, loaded_model: LoadedModel) -> None:
        """
        모델 교체 + 이력 초기화 (R-INSP-04).
        이전 모델 GPU 메모리 해제 선행.
        """
        if self.active_model is not None:
            del self.active_model.model
            if self.active_model.device == "cuda":
                import torch; torch.cuda.empty_cache()
        self.active_model = loaded_model
        self.reset_records()

    def reset_records(self) -> None:
        """records, seq_counter, test_pool, pool_index 초기화."""
        self.records = []
        self.seq_counter = 0
        self.test_pool = []
        self.pool_index = 0

    def sample_from_pool(self) -> tuple[str, str]:
        """test_pool에서 순환 샘플링. 소진 시 재셔플 후 인덱스 리셋 (A-16)."""
        import random
        if not self.test_pool:
            raise RuntimeError("ERR_INSP_TEST_POOL_EMPTY")
        if self.pool_index >= len(self.test_pool):
            random.shuffle(self.test_pool)
            self.pool_index = 0
        item = self.test_pool[self.pool_index]
        self.pool_index += 1
        was_reshuffled = (self.pool_index == 1 and len(self.test_pool) > 0)
        return item[0], item[1], was_reshuffled


# 모듈 레벨 싱글턴
inspection_manager = InspectionManager()
```

### 12.2 모델 적용 핸들러 (POST /api/inspection/model)

```python
# api/routers/inspection.py

@router.post("/api/inspection/model")
async def apply_model(body: ApplyModelRequest):
    """
    POST /api/inspection/model { experiment_id }
    모델 로드 + 이력 초기화 + test_pool 구성.
    """
    experiment = load_experiment(body.experiment_id)
    if experiment is None or experiment["status"] != "completed":
        raise HTTPException(400, {"code": "ERR_INSP_MODEL_NOT_COMPLETED"})

    device = _get_device()
    configs = load_config(experiment["configs_path"])

    model = load_model_for_inference(
        exp_id=body.experiment_id,
        model_path=experiment["model_path"],
        model_config=configs.get("model", {}),
        device=device,
    )

    # ActiveModel 메타데이터 구성 (05_Data_Model.md §3.2 ActiveModel 스키마)
    threshold_raw = _compute_threshold(
        method=experiment["threshold_method"],
        value=experiment["threshold_value"],
        anomaly_scores=experiment["metrics"]["anomaly_scores"],
        image_labels=experiment["metrics"]["image_labels"],
    )
    score_min = float(min(experiment["metrics"]["anomaly_scores"]))
    score_max = float(max(experiment["metrics"]["anomaly_scores"]))

    metadata = {
        "experiment_id": body.experiment_id,
        "name": experiment.get("name", body.experiment_id),
        "model_path": experiment["model_path"],
        "model_type": experiment["model_type"],
        "product_name": experiment.get("product_name", ""),
        "background_method": experiment.get("background_method", "none"),
        "threshold": threshold_raw,       # 단일 계산값 (05_Data_Model.md 3.2절 ActiveModel.threshold)
        "dataset_path": experiment["dataset_path"],
        "preprocessing_config": {
            "method": experiment["preprocessing_method"],
            "params": experiment.get("preprocessing_params"),
            "image_size": experiment["image_size"],
        },
        "score_min": score_min,
        "score_max": score_max,
        "device": device,
    }

    loaded = LoadedModel(model=model, metadata=metadata, device=device)
    inspection_manager.set_active_model(loaded)

    # test_pool 구성 (A-16)
    pool = _build_test_pool(experiment["dataset_path"])
    inspection_manager.test_pool = pool
    inspection_manager.pool_index = 0

    gpu_warning = _check_gpu_warning(device)
    return {"active_model": metadata, "gpu_warning": gpu_warning}
```

### 12.3 수동 검사 핸들러 (POST /api/inspection/run)

```python
@router.post("/api/inspection/run")
async def run_inspection():
    """
    POST /api/inspection/run
    test_pool에서 이미지 1장 샘플링 → 추론 → InspectionResult 반환.
    [확인 필요: 동기 응답 vs 비동기 job 패턴 확인 필요]
    """
    import asyncio, numpy as np

    if inspection_manager.active_model is None:
        raise HTTPException(400, {"code": "ERR_INSP_NO_MODEL"})

    active = inspection_manager.active_model
    image_path, gt_label, was_reshuffled = inspection_manager.sample_from_pool()

    # 추론 (별도 스레드 — ML 코드는 동기적)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _run_inference_sync,
        active, image_path, was_reshuffled
    )

    # 이력 저장
    seq = inspection_manager.seq_counter + 1
    inspection_manager.seq_counter = seq
    record = {
        "seq": seq,
        "inspected_at": generate_created_at(),
        "image_name": Path(image_path).name,
        "verdict": result["verdict"],
        "anomaly_score": round(result["anomaly_score"], 4),
    }
    inspection_manager.records.append(record)

    # 이미지 서버 사이드 캐시 (GET /api/inspection/image/last 용)
    _cache_last_result(
        image_path=image_path,
        anomaly_map=result["anomaly_map"],
    )

    return {
        "seq": seq,
        "inspected_at": record["inspected_at"],
        "image_name": record["image_name"],
        "image_path": image_path,
        "verdict": record["verdict"],
        "anomaly_score": record["anomaly_score"],
        "was_reshuffled": was_reshuffled,
    }


def _run_inference_sync(active: LoadedModel, image_path: str, was_reshuffled: bool) -> dict:
    """동기 추론 1회 실행 — executor에서 호출."""
    import numpy as np
    _, tensor = apply_preprocessing(image_path, active.metadata["preprocessing_config"])
    tensor = tensor.unsqueeze(0).to(active.device)
    anomaly_map = run_inference(active.model, tensor)  # (H, W) float32
    anomaly_score = float(np.max(anomaly_map))
    verdict = "불량" if anomaly_score >= active.metadata["threshold"] else "양품"
    return {"anomaly_score": anomaly_score, "verdict": verdict, "anomaly_map": anomaly_map}
```

### 12.4 이미지 서빙 엔드포인트

검사 후 마지막 이미지를 Vision UI에 제공하는 3개 엔드포인트:

| 엔드포인트 | 반환 | 설명 |
|------------|------|------|
| `GET /api/inspection/image/last` | JPEG | 원본 이미지 (cache-bust: `?t={imageStamp}`) |
| `GET /api/inspection/anomaly-map/last` | JPEG | Anomaly Map 히트맵 |
| `GET /api/inspection/overlay/last` | JPEG | 오버레이 이미지 |

```python
# 서버 캐시 구조 (inspection_manager 내부)
_last_result_cache = {
    "image_path":  None,   # str — 원본 이미지 경로
    "anomaly_map": None,   # np.ndarray (H, W) float32
}
```

---

## 13. 자동 검사 WebSocket 설계

> **v2.0**: `time.sleep(3) + st.rerun()` Streamlit 루프 → FastAPI `WS /ws/inspection/auto` asyncio 서버 주도 루프로 전면 교체.

### 13.1 WS 메시지 프로토콜

**Client → Server:**

| 메시지 | 시점 |
|--------|------|
| `"start"` | 자동 검사 시작 요청 |
| `"stop"` | 자동 검사 중지 요청 |

**Server → Client:**

| type | 데이터 | 설명 |
|------|--------|------|
| `result` | seq, inspected_at, image_name, image_path, verdict, anomaly_score, was_reshuffled | 추론 1회 결과 |
| `defect_stopped` | — | 불량 감지로 루프 중지 |
| `stopped` | — | 사용자 요청으로 루프 중지 |
| `error` | message | 예외 발생 |

### 13.2 WebSocket 핸들러 구현 (v2.0)

```python
# api/routers/inspection.py

@router.websocket("/ws/inspection/auto")
async def ws_inspection_auto(websocket: WebSocket):
    """
    WS /ws/inspection/auto
    서버 주도 자동 검사 루프 (3초 간격, A-18).
    """
    await websocket.accept()

    if inspection_manager.active_model is None:
        await websocket.send_json({"type": "error", "message": "ERR_INSP_NO_MODEL"})
        await websocket.close()
        return

    try:
        # 클라이언트가 "start" 전송할 때까지 대기
        msg = await websocket.receive_text()
        if msg != "start":
            await websocket.close()
            return

        # 자동 검사 asyncio 루프
        stop_requested = False

        async def _listen_for_stop():
            """별도 태스크 — "stop" 메시지 수신 시 플래그 설정."""
            nonlocal stop_requested
            try:
                while True:
                    text = await websocket.receive_text()
                    if text == "stop":
                        stop_requested = True
                        break
            except WebSocketDisconnect:
                stop_requested = True

        listener = asyncio.create_task(_listen_for_stop())

        while not stop_requested:
            # 추론 (별도 스레드)
            active = inspection_manager.active_model
            image_path, gt_label, was_reshuffled = inspection_manager.sample_from_pool()

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _run_inference_sync, active, image_path, was_reshuffled
            )

            # 이력 저장
            seq = inspection_manager.seq_counter + 1
            inspection_manager.seq_counter = seq
            record = {
                "seq": seq,
                "inspected_at": generate_created_at(),
                "image_name": Path(image_path).name,
                "verdict": result["verdict"],
                "anomaly_score": round(result["anomaly_score"], 4),
            }
            inspection_manager.records.append(record)
            _cache_last_result(image_path=image_path, anomaly_map=result["anomaly_map"])

            # 결과 전송
            await websocket.send_json({
                "type": "result",
                "seq": seq,
                "inspected_at": record["inspected_at"],
                "image_name": record["image_name"],
                "image_path": image_path,
                "verdict": record["verdict"],
                "anomaly_score": record["anomaly_score"],
                "was_reshuffled": was_reshuffled,
            })

            # 불량 감지 → 루프 중지
            if record["verdict"] == "불량":
                await websocket.send_json({"type": "defect_stopped"})
                listener.cancel()
                return

            # 3초 대기 (stop 요청 오면 즉시 탈출)
            try:
                await asyncio.wait_for(
                    asyncio.shield(listener),
                    timeout=3.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # 정상 중지
        listener.cancel()
        await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
```

### 13.3 자동 검사 상태 전이 (v2.0)

```
Vision 클라이언트 상태 (inspectionStore)

초기 상태
  isAutoRunning = false, defectStopped = false

[▶ 자동 검사 시작] 클릭
  → WS /ws/inspection/auto 연결
  → send("start")
  → isAutoRunning = true

WS type: 'result' 수신마다
  → setLastResult(result), imageStamp = Date.now()
  → 판정 pill + 이미지 패널 갱신

WS type: 'defect_stopped' 수신
  → isAutoRunning = false
  → defectStopped = true
  → DefectPopup 표시

  [✅ 확인 및 재개] 클릭:
    → setDefectStopped(false) → start() (새 WS 연결)

  [🛑 검사 종료] 클릭:
    → setDefectStopped(false) (팝업 닫기만)

[⏹ 자동 검사 중지] 클릭
  → send("stop")
  → WS type: 'stopped' 수신
  → isAutoRunning = false → WS 연결 종료
  → 화면 unmount 시 onClose: send("stop") 자동 호출
```

### 13.4 타이밍 고려사항

| 항목 | 값 | 비고 |
|------|-----|------|
| 검사 간격 | 서버 asyncio.sleep(3초) | 추론 시간 포함 시 실제 간격 > 3초 가능 |
| 추론 지연 허용 | ≤ 3초 | NFR — 00_Global §6 |
| 자동 검사 타이밍 오차 | ≤ 0.5초 | A-18 기준 (추론 < 3초인 경우 달성 가능) |
| 불량 팝업 지연 | ≤ 0.5초 | defect_stopped 메시지 → DefectPopup 표시 |

---

## 14. 검사 서비스 체크리스트

### InspectionManager

- [ ] 모듈 레벨 싱글턴 1개만 생성
- [ ] `set_active_model()` — 이전 모델 GPU 해제 선행
- [ ] `reset_records()` — seq_counter, test_pool, pool_index 포함 전체 초기화
- [ ] `sample_from_pool()` — 소진 시 재셔플 후 `pool_index = 0` (A-16)
- [ ] `was_reshuffled` 플래그 반환

### POST /api/inspection/model

- [ ] `status != "completed"` 실험 → 400 에러 (R-INSP-05)
- [ ] `load_model_for_inference()` 성공 후 `set_active_model()` 호출
- [ ] threshold: 단일 계산값 반환 (method + scores → raw float)
- [ ] test_pool 구성: `_build_test_pool(dataset_path)` 호출
- [ ] GPU 경고 배너 여부: `gpu_warning` 응답 포함

### POST /api/inspection/run

- [ ] `active_model == None` → 400 ERR_INSP_NO_MODEL
- [ ] `run_executor()`로 동기 추론 비동기 래핑
- [ ] InspectionRecord를 `inspection_manager.records`에 append
- [ ] `_cache_last_result()` 호출 (이미지 서빙용)
- [ ] `was_reshuffled` 응답에 포함

### WS /ws/inspection/auto

- [ ] 연결 직후 active_model 체크 → None이면 error + close
- [ ] "start" 메시지 수신 후 루프 시작
- [ ] 불량 감지 시 defect_stopped 전송 + 루프 종료
- [ ] "stop" 메시지 수신 시 stopped 전송 + 루프 종료
- [ ] WebSocketDisconnect 시 루프 자동 종료
- [ ] 3초 대기 중 "stop" 수신 시 즉시 탈출 (asyncio.wait_for 활용)

### 모델 교체 (POST /api/inspection/model 재호출)

- [ ] 이전 모델 GPU 메모리 해제 선행 (set_active_model 내부)
- [ ] records, seq_counter, test_pool, pool_index 자동 초기화 (R-INSP-04)
- [ ] R-INSP-01: history.json 쓰기 절대 금지

---

---

# v1.x 참고 — Streamlit 서비스 레이어 (삭제 금지)

> 이하 내용은 v1.x Streamlit 기반 서비스 레이어 구현 명세입니다.
> v2.0에서는 FastAPI 라우터/서비스 레이어로 교체됐습니다.
> 삭제하지 않고 Streamlit 내부 구현 참고용으로 보존합니다.

---

### v1.x §3 — 학습 시작 서비스 (Streamlit)

```python
# tabs/tab3_training.py  (v1.x Streamlit)

def _handle_start_training(experiment_name: str) -> None:
    if st.session_state["current_run_status"] != "idle":
        st.warning("이미 학습이 진행 중입니다.")
        st.stop()

    model_config: dict = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str = st.session_state["dataset_path"]
    device_info: dict = st.session_state.get("device_info", {"device": "cpu"})

    # ... (사전 조건 검증) ...

    exp_id = generate_experiment_id(model_config["model_type"])
    created_at = generate_created_at()

    if not experiment_name.strip():
        experiment_name = f"{model_config['model_type'].upper()} {exp_id[-4:]}"

    save_config_section("experiment", {"name": experiment_name, "created_at": created_at})

    stop_event  = threading.Event()
    pause_event = threading.Event()
    result_queue = queue.Queue()

    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device_info["device"],
        stop_event=stop_event,
        pause_event=pause_event,
        result_queue=result_queue
    )
    worker.daemon = True
    worker.start()

    # session_state 갱신 (메인 스레드 전용 — ADR-04)
    st.session_state["current_run_status"]   = "running"
    st.session_state["current_exp_id"]       = exp_id
    st.session_state["_stop_event"]          = stop_event
    st.session_state["_pause_event"]         = pause_event
    st.session_state["_result_queue"]        = result_queue
    st.session_state["_worker"]              = worker
    st.session_state["_last_ckpt_path"]      = None
    st.session_state["_progress"] = {
        "step": 0, "total": _get_total_steps(model_config),
        "loss": None, "elapsed": 0.0
    }
    st.session_state["_log_lines"]   = []
    st.session_state["_loss_history"] = []

    st.rerun()   # running UI로 전환
```

---

### v1.x §6 — 완료/중단 후처리 (Streamlit session_state 기반)

```
[completed 메시지 수신] — v1.x
  → session_state["_last_result"] = {"level": "success", "text": "..."}
  → _reset_run_state()  (current_run_status = "idle")
  → st.rerun() → _show_last_result()에서 st.success() 표시

[stopped 메시지 수신] — v1.x
  → session_state["_last_result"] = {"level": "warning", "text": MSG["TRAIN_STOPPED"]}
  → _reset_run_state()
```

---

### v1.x §7 — 탭5 이상 감지 서비스 (Streamlit 동기 블로킹)

```python
# tabs/tab5_anomaly_map.py  (v1.x Streamlit — 동기 블로킹)

# 캐시 MISS 처리
with st.spinner("모델 로드 및 전체 이미지 추론 중..."):
    model = load_model_for_inference(exp_id, model_path, model_config, device)
    image_paths = _collect_test_image_paths(exp_record["dataset_path"])

    # st.progress() 동기 블로킹
    progress_bar = st.progress(0, text="추론 중...")
    anomaly_maps = []
    for i, img_path in enumerate(image_paths):
        _, tensor = apply_preprocessing(img_path, preprocessing_config)
        tensor = tensor.unsqueeze(0).to(device)
        anomaly_map = run_inference(model, tensor)
        anomaly_maps.append(anomaly_map)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            progress_bar.progress((i + 1) / total, text=f"추론 중... {i+1}/{total}")
    progress_bar.empty()

    set_anomaly_map_cache(exp_id, {
        "anomaly_maps": np.stack(anomaly_maps, axis=0).astype(np.float32),
        "image_paths":  image_paths
    })
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
```

---

### v1.x §8 — 단일 워커 보장 (Streamlit)

```python
# v1.x: session_state 기반
def _render_idle_ui() -> None:
    if st.session_state["current_run_status"] != "idle":
        # 버튼이 아예 렌더링되지 않음 (R-UI-02)
        return
    if st.button("학습 시작", type="primary"):
        _handle_start_training(experiment_name)
```

---

### v1.x §12 — 비전검사 추론 서비스 (Streamlit st.cache_resource)

```python
# inspection/utils/insp_session_init.py  (v1.x Streamlit)

@st.cache_resource
def _load_insp_model(model_path: str, model_type: str, device: str):
    """(model_path, model_type, device) 조합이 동일하면 캐시 반환."""
    import yaml
    with open(f"{model_path}/configs.yaml") as f:
        configs = yaml.safe_load(f)
    model_config = configs.get("model", {})
    model_config["model_type"] = model_type
    return load_model_for_inference("insp", model_path, model_config, device)
```

---

### v1.x §13 — 자동 검사 루프 (time.sleep + st.rerun)

```python
# inspection/tabs/insp_tab1_realtime.py  (v1.x Streamlit)

def render() -> None:
    # ...
    if st.session_state["insp_auto_active"] and not st.session_state["insp_defect_popup"]:
        _run_single_inspection()   # 검사 1회 실행
        import time
        time.sleep(3)              # 3초 대기
        st.rerun()                 # 다음 검사 사이클 진입
```

**v1.x session_state ↔ v2.0 서버 상태 대조:**

| v1.x session_state 키 | v2.0 서버 상태 |
|-----------------------|----------------|
| `current_run_status` | `training_manager.status` |
| `_stop_event` | `training_manager.stop_event` |
| `_result_queue` | `training_manager.result_queue` |
| `insp_active_model` | `inspection_manager.active_model.metadata` |
| `insp_records` | `inspection_manager.records` |
| `insp_auto_active` | `inspectionStore.isAutoRunning` (클라이언트) |
| `insp_defect_popup` | `inspectionStore.defectStopped` (클라이언트) |
| `insp_test_pool` | `inspection_manager.test_pool` |
| `insp_pool_index` | `inspection_manager.pool_index` |

---

*다음: [08_AI_ML_Integration.md](./08_AI_ML_Integration.md) — EfficientAD / PatchCore 학습 루프, run_inference() 구현*

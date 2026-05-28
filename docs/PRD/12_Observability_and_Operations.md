# 12. Observability and Operations

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md) §7 Observability Standards
> **선행 문서**: [07_Backend_Service_Design.md](./07_Backend_Service_Design.md), [11_Non_Functional_Requirements.md](./11_Non_Functional_Requirements.md)
> **버전**: v1.1
> **작성일**: 2026-05-09
> **수정일**: 2026-05-26
> **중요**: 00절 §7의 로그 포맷·필수 이벤트·알림 조건을 기준으로, 구현에 필요한 로그 모듈 명세·파일 관리 정책·운영 확인 절차를 확정한다.

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Log Module Specification](#b-log-module-specification)
- [C. Log File Management](#c-log-file-management)
- [D. UI Notification Specification](#d-ui-notification-specification)
- [E. Training Progress Observability](#e-training-progress-observability)
- [F. Operations Runbook](#f-operations-runbook)

---

## A. Objective & Scope

### A.1 이 문서의 목적

로컬 실행 환경에서 운영자(= 사용자)가 시스템 상태를 파악하고 문제를 진단할 수 있도록:

1. **로그 모듈** (`utils/logger.py`) 인터페이스와 포맷 확정
2. **로그 파일 관리 정책** (보관 기간, 크기 제한, 정리 방법)
3. **UI 알림** 표시 조건과 메시지 확정 (00절 §7.4 확장)
4. **학습 진행 관찰** (Progress bar, Loss 곡선, 실시간 로그) 구현 명세
5. **운영 런북** (일반적인 문제 진단 절차)

### A.2 이 문서에서 다루지 않는 항목

| 항목 | 다루는 문서 |
|------|-----------|
| 보안 감사 로그 | 10_Security_and_Compliance.md |
| 테스트 로그·커버리지 보고 | 13_QA_and_Testing_Strategy.md |
| 배포 로그·CI/CD | 14_Deployment_and_Release_Plan.md |

---

## B. Log Module Specification

### B.1 `utils/logger.py` 인터페이스

```python
# utils/logger.py

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal, Any

KST = timezone(timedelta(hours=9))

LogLevel = Literal["INFO", "WARNING", "ERROR"]

def _now_kst() -> str:
    return datetime.now(tz=KST).isoformat(timespec="milliseconds")

def _build_entry(
    level: LogLevel,
    event: str,
    message: str,
    experiment_id: str | None = None,
    tab: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict:
    return {
        "timestamp": _now_kst(),
        "level": level,
        "experiment_id": experiment_id,
        "tab": tab,
        "event": event,
        "message": message,
        "data": data or {},
    }

def log_info(event: str, message: str, **kwargs) -> None:
    _write(_build_entry("INFO", event, message, **kwargs))

def log_warning(event: str, message: str, **kwargs) -> None:
    _write(_build_entry("WARNING", event, message, **kwargs))

def log_error(event: str, message: str, **kwargs) -> None:
    _write(_build_entry("ERROR", event, message, **kwargs))

# 기본 출력: Python logging (콘솔)
# 실험별 파일 출력: get_log_writer() 사용
_py_logger = logging.getLogger("smart_qc")
_py_logger.setLevel(logging.DEBUG)
if not _py_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _py_logger.addHandler(handler)

def _write(entry: dict) -> None:
    _py_logger.log(
        {"INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}[entry["level"]],
        json.dumps(entry, ensure_ascii=False),
    )
```

### B.2 실험별 로그 라이터

```python
# utils/logger.py (계속)

class ExperimentLogWriter:
    """실험 진행 중 ./logs/{experiment_id}.log 에 라인 단위로 기록"""

    def __init__(self, experiment_id: str):
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{experiment_id}.log"
        self._f = open(self._path, "a", encoding="utf-8", buffering=1)  # 라인 버퍼

    def write(self, message: str) -> None:
        ts = _now_kst()
        self._f.write(f"{ts}\t{message}\n")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __del__(self):
        self.close()


def get_log_writer(experiment_id: str) -> ExperimentLogWriter:
    """TrainingWorker.__init__()에서 1회 호출. 반환된 writer를 self._log_writer에 저장."""
    return ExperimentLogWriter(experiment_id)
```

**사용 예시 (TrainingWorker)**:
```python
class TrainingWorker(threading.Thread):
    def __init__(self, experiment_id, ...):
        ...
        self._log_writer = get_log_writer(experiment_id)

    def run(self):
        try:
            self._log_writer.write(f"[시작] 실험: {self.experiment_id}")
            # 학습 루프 ...
        finally:
            self._log_writer.close()
```

### B.3 로그 포맷 확정

**콘솔 출력** (JSON 1줄):
```json
{"timestamp":"2026-05-09T14:00:23.456+09:00","level":"INFO","experiment_id":"efficientad_20260509_140023_7f3a","tab":"tab3","event":"training_started","message":"학습이 시작되었습니다.","data":{"model_type":"efficientad","train_steps":70000}}
```

**파일 출력** (`./logs/{exp_id}.log`):
```
2026-05-09T14:00:23.456+09:00	[시작] 실험: efficientad_20260509_140023_7f3a
2026-05-09T14:00:25.100+09:00	[Step 1000/70000] Loss: 0.0521 | 경과: 2.1s
2026-05-09T14:01:05.200+09:00	[Step 2000/70000] Loss: 0.0412 | 경과: 62.3s
2026-05-09T14:20:00.000+09:00	[완료] 총 소요: 1177s | AUC: 0.971
```

---

### B.4 필수 로그 이벤트 구현 명세

00절 §7.2 기준 — 각 이벤트의 호출 위치와 `data` 필드:

| 이벤트명 | 호출 위치 | `data` 필드 |
|---------|-----------|------------|
| `dataset_validated` | `tab1._validate_dataset()` 성공 시 | `{"train_good_count": int, "defect_classes": list}` |
| `dataset_validation_failed` | `tab1._validate_dataset()` 실패 시 | `{"error_code": str, "path": str}` |
| `preprocessing_config_saved` | `tab2._save_preprocessing()` | `{"method": str, "image_size": int}` |
| `model_config_saved` | `tab2._save_model_config()` | `{"model_type": str, "device": str}` |
| `training_started` | `tab3_training.py._handle_start_training()` | `{"experiment_id": str, "model_type": str, "train_steps": int}` |
| `training_step` | `TrainingWorker` 내 매 N step | `{"step": int, "total_steps": int, "loss": float, "elapsed_s": float}` |
| `training_completed` | `tab3_training.py._handle_terminal()` completed 분기 | `{"experiment_id": str, "duration_s": int, "auc": float}` |
| `training_stopped` | `tab3_training.py._handle_terminal()` stopped 분기 | `{"experiment_id": str, "step": int}` |
| `training_failed` | `tab3_training.py._handle_terminal()` error 분기 | `{"experiment_id": str, "traceback": str}` |
| `model_saved` | `tab4_history.py._save_model()` | `{"experiment_id": str, "path": str, "size_mb": float}` |
| `experiment_deleted` | `tab4_history.py._delete_experiment()` | `{"experiment_id": str}` |

`training_step` 이벤트 주기 (00절 §9 A-08):
- EfficientAD: 매 100 step마다 → `queue.put({"type": "progress", "step": N, "loss": float, ...})`
- PatchCore: 에포크 단위

---

## C. Log File Management

### C.1 파일 위치 및 명명 규칙

```
./logs/
├── efficientad_20260509_140023_7f3a.log   # 완료된 실험
├── patchcore_20260509_150512_a1b2.log      # 완료된 실험
└── ...
```

- 파일명: `{experiment_id}.log`
- 인코딩: UTF-8
- 줄 구분: `\n` (POSIX)

### C.2 보관 정책

| 항목 | 정책 |
|------|------|
| 보관 기간 | 무제한 (로컬 스토리지 — 사용자 수동 삭제) |
| 최대 크기 | 단일 파일 상한 없음 (70k step 기준 약 2~5 MB) |
| 실험 삭제 시 | 탭4 [실험 삭제] 클릭 시 해당 `{exp_id}.log` 도 함께 삭제 |
| 앱 재시작 시 | `.tmp` 파일 탐색 후 삭제 (09절 §E.2 참조) |

### C.3 실험 삭제 시 로그 파일 처리

```python
# tab4_history.py._delete_experiment() 내부
def _delete_experiment(experiment_id: str) -> None:
    # history.json에서 제거
    history = load_history()
    history = [r for r in history if r["experiment_id"] != experiment_id]
    save_history(history)

    # 모델 파일 삭제
    model_dir = Path(f"./models/{experiment_id}")
    if model_dir.exists():
        shutil.rmtree(model_dir)

    # 로그 파일 삭제
    log_file = Path(f"./logs/{experiment_id}.log")
    if log_file.exists():
        log_file.unlink()

    # 결과 이미지 삭제
    result_dir = Path(f"./results/{experiment_id}")
    if result_dir.exists():
        shutil.rmtree(result_dir)

    log_warning(
        "experiment_deleted",
        f"실험 {experiment_id} 삭제됨",
        tab="tab4",
        experiment_id=experiment_id,
    )
```

---

## D. UI Notification Specification

00절 §7.4 기준 확장 — 각 알림의 Streamlit 함수, 조건, 메시지 문자열 확정.

### D.1 알림 조건 및 메시지

| 조건 | Streamlit 함수 | 메시지 |
|------|---------------|--------|
| 학습 완료 | `st.success()` | `f"학습이 완료되었습니다. 소요 시간: {m}분 {s}초"` |
| 학습 중단 | `st.warning()` | `MSG["TRAIN_STOPPED"]` |
| 학습 오류 | `st.error()` | `"학습 중 오류가 발생했습니다. logs/ 폴더에서 로그를 확인해 주세요."` |
| 모델 저장 완료 | `st.success()` | `f"저장 완료: {path} ({size_mb:.1f} MB)"` |
| 폴더 구조 오류 | `st.error()` | `MSG["INVALID_FOLDER"]` |
| Grayscale 감지 | `st.info()` | `MSG["GRAYSCALE_DETECT"]` |
| 디스크 공간 부족 | `st.warning()` | `f"디스크 여유 공간이 {free_mb:.0f} MB로 부족합니다. 500 MB 이상 확보 후 저장해 주세요."` |
| ImageNet penalty 디렉터리 없음 | `st.error()` (탭3 시작 버튼 클릭 시) 또는 Queue `error` 메시지 (TrainingWorker 내부 검증 실패 시) | `"EfficientAD 학습에 필요한 ImageNet penalty 데이터가 없습니다. ./dataset/imagenet_penalty/ 를 확인해 주세요."` |
| 데이터셋 없음 | `st.warning()` | `MSG["NO_DATASET"]` |
| 전처리 설정 없음 | `st.warning()` | `MSG["NO_PREPROCESSING"]` |
| 모델 설정 없음 | `st.warning()` | `MSG["NO_MODEL_CONFIG"]` |
| 실험 없음 | `st.info()` | `MSG["NO_EXPERIMENTS"]` |
| 실험 미선택 | `st.info()` | `MSG["NO_SELECTED_EXP"]` |
| 비교 실험 10개 초과 | `st.warning()` | `"최대 10개 실험까지 비교할 수 있습니다."` |

### D.2 소요 시간 포맷 함수

```python
def format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    if m == 0:
        return f"{s}초"
    return f"{m}분 {s}초"
```

### D.3 알림 표시 위치 규칙

- `st.success()`, `st.error()`, `st.warning()`, `st.info()` 는 해당 **탭의 최상단 콘텐츠 영역** 첫 번째 줄에 렌더링
- 사이드바나 다른 탭에 알림을 렌더링하지 않는다
- 알림은 next rerun까지 표시 (Streamlit 기본 동작 유지)

---

## E. Training Progress Observability

탭3 학습 중 사용자에게 표시되는 3가지 관찰 수단.

### E.1 Progress Bar

```python
# tab3 학습 중 rerun 주기마다 업데이트
progress = st.session_state.get("current_step", 0) / st.session_state["model_config"]["params"]["train_steps"]
st.progress(min(progress, 1.0), text=f"학습 진행률: {progress*100:.1f}%")
```

- `current_step`: `queue` 메시지 `progress` 타입에서 업데이트
- EfficientAD: step 단위 (0 ~ train_steps)
- PatchCore: epoch 단위 (에포크 수로 환산)

### E.2 Loss 곡선

```python
# session_state["loss_history"] = list[dict] — {"step": int, "loss": float}
import plotly.graph_objects as go

if st.session_state.get("loss_history"):
    df = pd.DataFrame(st.session_state["loss_history"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["step"], y=df["loss"], mode="lines", name="Loss"))
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Loss",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
```

`loss_history` 업데이트 조건:
- EfficientAD: `progress` 메시지에 `loss` 필드가 있을 때만 append (매 100 step)
- PatchCore: epoch 완료 메시지마다 append

### E.3 실시간 로그

```python
# session_state["log_buffer"] = list[str] — 최신 100줄 (11절 §G.3)
log_text = "\n".join(st.session_state.get("log_buffer", []))
st.text_area(
    "학습 로그",
    value=log_text,
    height=300,
    disabled=True,
    key="training_log_display",
)
```

`log_buffer` 업데이트: `queue` 메시지 `log` 타입 수신 시 append + 100줄 초과 시 앞부분 제거.

### E.4 Queue 메시지 타입별 처리 요약

| 타입 | 처리 | 참조 |
|------|------|------|
| `progress` | `current_step` 업데이트, `loss_history` append | E.1, E.2 |
| `log` | `log_buffer` append | E.3 |
| `completed` | `_handle_terminal()` → 실험 저장, anomaly_map 캐시 | 06절 §C.3, Z.4, Z.6 |
| `error` | `_handle_terminal()` → `st.error()` 표시 | 06절 §C.3 |
| `stopped` | `_handle_terminal()` → 중단 레코드 저장 | 06절 §C.3, Z.3 |

---

## F. Operations Runbook

### F.1 학습이 멈춰 보이는 경우

**증상**: Progress bar가 오랫동안 움직이지 않음.

**진단**:
1. `./logs/{exp_id}.log` 마지막 라인 확인 — 최근 타임스탬프 체크
2. 현재 step이 로그에 기록되고 있으면 학습 중 (UI rerun 주기 문제)
3. 로그에 기록이 없으면 실제 정지

**조치**:
- 브라우저 새로고침 (Streamlit rerun 재시작)
- 로그 파일에 `ERROR` 레벨 항목 있으면 해당 오류 확인
- GPU OOM: `torch.cuda.OutOfMemoryError` → `batch_size` 줄이고 재시작

---

### F.2 학습 완료 후 탭4에서 실험이 안 보이는 경우

**진단**:
1. `./experiments/history.json` 파일 존재 확인
2. 파일 열어 최신 레코드의 `status` 확인
3. `session_state["experiments"]` 가 최신 history와 동기화됐는지 확인

**조치**:
- 탭4 재진입 (탭 클릭) — `load_history()` 재호출 트리거
- `history.json` 이 존재하지 않으면: `experiments/` 디렉터리 내 `.tmp` 파일 확인 (원자적 쓰기 실패)

---

### F.3 모델 파일이 없는데 history에는 completed로 표시되는 경우

**진단**:
1. `./models/{exp_id}/` 디렉터리 존재 확인
2. `model_state_dict.pth` 파일 존재 확인
3. `history.json` 의 `model_path` 필드 확인

**조치**:
- 파일이 없으면 해당 실험은 탭5에서 추론 불가 — UI에 "(파일 없음)" 표시
- 불완전한 레코드: 탭4에서 [삭제] 후 재학습

---

### F.4 ImageNet penalty 오류 (EfficientAD)

**증상**: 탭3 [학습 시작] 클릭 시 즉시 오류 메시지.

**진단**:
```bash
# 디렉터리 내 이미지 파일 수 확인
find ./dataset/imagenet_penalty -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l
```

**조치**:
- 0이면 ImageNet 이미지를 `./dataset/imagenet_penalty/` 에 배치
- 09절 §F.3 참조

---

### F.5 Streamlit 포트 충돌

**증상**: `streamlit run app.py` 실행 시 "Address already in use" 오류.

**조치**:
```bash
# 기존 프로세스 확인
# Windows
netstat -ano | findstr :8501

# Linux/macOS
lsof -i :8501

# 다른 포트로 실행
streamlit run app.py --server.port 8502
```

---

### F.6 로그 파일 용량 정리

**주기적 정리** (수동):
```bash
# 30일 이상 된 로그 파일 삭제 (Linux)
find ./logs -name "*.log" -mtime +30 -delete

# 모든 로그 삭제 (주의)
rm ./logs/*.log
```

> 탭4에서 실험을 삭제하면 해당 로그 파일도 자동 삭제됨 (C.3 참조).

---

---

## G. 비전검사 대시보드 로그 이벤트 (v1.1)

### G.1 비전검사 필수 로그 이벤트

00절 §7 기준 — 비전검사 대시보드에서 발생하는 7개 로그 이벤트 구현 명세:

| 이벤트명 | 레벨 | 호출 위치 | `data` 필드 |
|---------|------|---------|------------|
| `insp_model_applied` | INFO | `insp_tab3_model._apply_model()` 성공 시 | `{"experiment_id": str, "model_type": str}` |
| `insp_inspection_started_manual` | INFO | `insp_tab1_realtime._run_single_inspection()` 수동 버튼 클릭 시 | `{"image_name": str}` |
| `insp_inspection_started_auto` | INFO | `insp_tab1_realtime._start_auto_inspection()` 자동 검사 시작 시 | `{"interval_s": float}` |
| `insp_inspection_stopped_auto` | INFO | `insp_tab1_realtime._stop_auto_inspection()` 자동 검사 중지 시 | `{"total_inspected": int}` |
| `insp_defect_detected` | WARNING | 판정 결과가 NG(불량)인 경우 | `{"anomaly_score": float, "threshold": float, "image_name": str}` |
| `insp_pool_reshuffled` | INFO | `test_sampler.sample_from_pool()` 풀 소진 후 재섞기 시 | `{"pool_size": int}` |

> 참고: `insp_inspection_result_recorded` 이벤트는 매 검사마다 발생하므로 필요 시 선택적으로 추가할 수 있으나, 기본 구현에서는 `insp_defect_detected`(WARNING)만 필수로 기록한다.

### G.2 비전검사 이벤트 호출 예시

```python
# inspection/tabs/insp_tab1_realtime.py

from utils.logger import log_info, log_warning

def _run_single_inspection(image_name: str) -> dict:
    log_info(
        "insp_inspection_started_manual",
        f"수동 검사 시작: {image_name}",
        tab="insp_tab1",
        data={"image_name": image_name},
    )

    # 추론 수행 ...
    result = _infer(image_name)

    if result["verdict"] == "NG":
        log_warning(
            "insp_defect_detected",
            f"불량 감지: {image_name} (score={result['anomaly_score']:.4f})",
            tab="insp_tab1",
            data={
                "anomaly_score": result["anomaly_score"],
                "threshold": st.session_state["insp_threshold"],
                "image_name": image_name,
            },
        )
    return result


def _start_auto_inspection(interval_s: float = 3.0) -> None:
    log_info(
        "insp_inspection_started_auto",
        "자동 검사 시작",
        tab="insp_tab1",
        data={"interval_s": interval_s},
    )


def _stop_auto_inspection() -> None:
    log_info(
        "insp_inspection_stopped_auto",
        "자동 검사 중지",
        tab="insp_tab1",
        data={"total_inspected": st.session_state.get("insp_seq", 0)},
    )
```

```python
# inspection/tabs/insp_tab3_model.py

def _apply_model(exp: dict) -> None:
    # ... 모델 적용 로직 ...
    log_info(
        "insp_model_applied",
        f"모델 적용 완료: {exp['experiment_id']}",
        tab="insp_tab3",
        data={
            "experiment_id": exp["experiment_id"],
            "model_type": exp.get("model_type", "unknown"),
        },
    )
```

```python
# inspection/utils/test_sampler.py

def sample_from_pool(pool: list, used: set) -> tuple[str, set]:
    remaining = [p for p in pool if p not in used]
    if not remaining:
        # 풀 소진 — 재섞기
        log_info(
            "insp_pool_reshuffled",
            "테스트 풀 소진 — 재섞기",
            tab="insp_tab1",
            data={"pool_size": len(pool)},
        )
        used = set()
        remaining = pool[:]
    selected = random.choice(remaining)
    used.add(selected)
    return selected, used
```

### G.3 비전검사 UI 알림 조건

| 조건 | Streamlit 함수 | 메시지 |
|------|---------------|--------|
| 적용된 모델 없음 | `st.warning()` | `INSP_MSG["NO_MODEL"]` |
| 완료된 실험 없음 | `st.info()` | `INSP_MSG["NO_COMPLETED_EXP"]` |
| 불량 감지 (팝업) | `st.error()` + 자동검사 중지 | `INSP_MSG["DEFECT_DETECTED"]` |

---

*다음 문서*: [10_Security_and_Compliance.md](./10_Security_and_Compliance.md)

# 12. Observability and Operations

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md) §7 Observability Standards
> **선행 문서**: [07_Backend_Service_Design.md](./07_Backend_Service_Design.md), [11_Non_Functional_Requirements.md](./11_Non_Functional_Requirements.md)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **수정일**: 2026-06-11
> **중요**: 00절 §7의 로그 포맷·필수 이벤트·알림 조건을 기준으로, 구현에 필요한 로그 모듈 명세·파일 관리 정책·운영 확인 절차를 확정한다.

---

## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 최초 작성 |
| v1.1 | 2026-05-26 | 비전검사 대시보드 로그 이벤트 추가 (G절) |
| v2.0 | 2026-06-11 | B.4 이벤트 호출 위치를 FastAPI route handler로 갱신; C.3 FastAPI handler 코드로 교체; D절 Streamlit 알림 함수 → React/API 처리 방법으로 전면 교체; E절 WebSocket 스트림 기반으로 전면 교체 (구 Streamlit 내용 → E.7 v1.x 참고); F.1/F.2/F.5 FastAPI/WS/React dev server 기준으로 갱신; G절 FastAPI API 핸들러 기준으로 교체 |

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Log Module Specification](#b-log-module-specification)
- [C. Log File Management](#c-log-file-management)
- [D. UI Notification Specification](#d-ui-notification-specification)
- [E. Training Progress Observability](#e-training-progress-observability)
- [F. Operations Runbook](#f-operations-runbook)
- [G. Vision 검사 로그 이벤트](#g-vision-검사-로그-이벤트-v20)

---

## A. Objective & Scope

### A.1 이 문서의 목적

로컬 실행 환경에서 운영자(= 사용자)가 시스템 상태를 파악하고 문제를 진단할 수 있도록:

1. **로그 모듈** (`utils/logger.py`) 인터페이스와 포맷 확정
2. **로그 파일 관리 정책** (보관 기간, 크기 제한, 정리 방법)
3. **UI 알림** 표시 조건과 메시지 확정 (00절 §7.4 확장) — v2.0: React/API 기준
4. **학습 진행 관찰** (Progress bar, Loss 곡선, 실시간 로그) — v2.0: `/ws/training` WebSocket 스트림 기준
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

> **v2.0 비고**: `tab` 파라미터는 v1.x에서 Streamlit 탭명을 전달하던 필드였다. v2.0에서는 FastAPI route 경로(`"api/training/start"`, `"ws/inspection/auto"` 등)를 전달한다. 필드명은 하위 호환을 위해 유지.

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
        self._log_writer = None  # lazy init — get_log_writer()로 초기화

    def run(self):
        self._log_writer = get_log_writer(self.experiment_id)
        try:
            self._log_writer.write(f"[시작] 실험: {self.experiment_id}")
            # 학습 루프 ...
        finally:
            self._log_writer.close()
```

### B.3 로그 포맷 확정

**콘솔 출력** (JSON 1줄):
```json
{"timestamp":"2026-05-09T14:00:23.456+09:00","level":"INFO","experiment_id":"efficientad_20260509_140023_7f3a","tab":"api/training/start","event":"training_started","message":"학습이 시작되었습니다.","data":{"model_type":"efficientad","train_steps":70000}}
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

00절 §7.2 기준 — 각 이벤트의 호출 위치(v2.0: FastAPI 핸들러)와 `data` 필드:

| 이벤트명 | 호출 위치 | `data` 필드 |
|---------|-----------|------------|
| `dataset_validated` | `POST /api/dataset/validate` 핸들러 — 검증 성공 시 | `{"train_good_count": int, "defect_classes": list}` |
| `dataset_validation_failed` | `POST /api/dataset/validate` 핸들러 — 검증 실패 시 | `{"error_code": str, "path": str}` |
| `preprocessing_config_saved` | `POST /api/config` 핸들러 — preprocessing 섹션 저장 시 | `{"method": str, "image_size": int}` |
| `model_config_saved` | `POST /api/config` 핸들러 — model config 섹션 저장 시 | `{"model_type": str, "device": str}` |
| `training_started` | `POST /api/training/start` 핸들러 | `{"experiment_id": str, "model_type": str, "train_steps": int}` |
| `training_step` | `TrainingWorker` 내 매 N step — `result_queue.put({"type": "progress", ...})` 전후 | `{"step": int, "total_steps": int, "loss": float, "elapsed_s": float}` |
| `training_completed` | FastAPI WS 핸들러 (`/ws/training`) — `completed` 메시지 처리 분기 | `{"experiment_id": str, "duration_s": int, "auc": float}` |
| `training_stopped` | FastAPI WS 핸들러 (`/ws/training`) — `stopped` 메시지 처리 분기 | `{"experiment_id": str, "step": int}` |
| `training_failed` | FastAPI WS 핸들러 (`/ws/training`) — `error` 메시지 처리 분기 | `{"experiment_id": str, "traceback": str}` |
| `model_saved` | `POST /api/experiments/{id}/save` 핸들러 | `{"experiment_id": str, "path": str, "size_mb": float}` |
| `experiment_deleted` | `DELETE /api/experiments/{id}` 핸들러 | `{"experiment_id": str}` |

`training_step` 이벤트 주기 (00절 §9 A-08):
- EfficientAD: 매 100 step마다 → `result_queue.put({"type": "progress", "step": N, "loss": float, ...})`
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
| 실험 삭제 시 | Explorer `/experiments` 화면에서 [삭제] 클릭 시 해당 `{exp_id}.log`도 함께 삭제 |
| 서버 재시작 시 | `.tmp` 파일 탐색 후 삭제 (09절 §E.2 참조) |

### C.3 실험 삭제 시 로그 파일 처리

```python
# api/routers/experiments.py — DELETE /api/experiments/{experiment_id} 핸들러
import shutil
from pathlib import Path
from utils.storage import load_history, save_history
from utils.logger import log_warning

@router.delete("/api/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
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
        tab="api/experiments/delete",
        experiment_id=experiment_id,
    )

    return {"status": "deleted", "experiment_id": experiment_id}
```

---

## D. UI Notification Specification

> **v2.0 변경**: v1.x의 "Streamlit 함수" 컬럼이 "React/API 처리 방법"으로 교체됐다.  
> 알림의 발생 위치: FastAPI 응답(오류 코드/경고 필드) 또는 WebSocket 메시지 → React 컴포넌트에서 렌더링.

### D.1 알림 조건 및 처리 방법

| 조건 | React/API 처리 방법 | 메시지 |
|------|---------------------|--------|
| 학습 완료 | WS `completed` 메시지 → `trainingStore.setStatus("completed")` → 성공 토스트/배너 | `f"학습이 완료되었습니다. 소요 시간: {m}분 {s}초"` |
| 학습 중단 | WS `stopped` 메시지 → `trainingStore.setStatus("stopped")` → 경고 배너 | "학습이 중단되었습니다." |
| 학습 오류 | WS `error` 메시지 → `trainingStore.setStatus("error")` → 오류 배너 | "학습 중 오류가 발생했습니다. logs/ 폴더에서 로그를 확인해 주세요." |
| 모델 저장 완료 | `POST /api/experiments/{id}/save` 200 응답 → 성공 토스트 | `f"저장 완료: {path} ({size_mb:.1f} MB)"` |
| 폴더 구조 오류 | `POST /api/dataset/validate` → `error_code` 필드 포함 응답 → Explorer 탭1 오류 메시지 | `MSG["INVALID_FOLDER"]` |
| Grayscale 감지 | `POST /api/dataset/validate` → `is_grayscale: true` 필드 → Explorer 탭1 정보 배너 | `MSG["GRAYSCALE_DETECT"]` |
| 디스크 공간 부족 | API 응답 `disk_warning: true` → 경고 배너 (차단 아닌 경고) | `f"디스크 여유 공간이 {free_mb:.0f} MB로 부족합니다. {warn_mb} MB 이상 확보 후 저장해 주세요."` |
| ImageNet penalty 없음 | `POST /api/training/start` 400 응답 → 오류 토스트 | "EfficientAD 학습에 필요한 ImageNet penalty 데이터가 없습니다. ./dataset/imagenet_penalty/ 를 확인해 주세요." |
| 데이터셋 없음 | `datasetStore.datasetPath === null` → D.1 화면 가드 컴포넌트 | `MSG["NO_DATASET"]` |
| 전처리 설정 없음 | `configStore.preprocessingConfig === null` → 화면 가드 컴포넌트 | `MSG["NO_PREPROCESSING"]` |
| 모델 설정 없음 | `configStore.modelConfig === null` → 화면 가드 컴포넌트 | `MSG["NO_MODEL_CONFIG"]` |
| 실험 없음 | `GET /api/experiments` 빈 배열 응답 → 안내 UI | `MSG["NO_EXPERIMENTS"]` |
| 실험 미선택 | `experimentsStore.selectedExperimentId === null` → 화면 가드 컴포넌트 | `MSG["NO_SELECTED_EXP"]` |
| 비교 실험 10개 초과 | Explorer 탭4 내 max check → 인라인 경고 메시지 | "최대 10개 실험까지 비교할 수 있습니다." |

### D.2 소요 시간 포맷 함수

```python
# utils/logger.py 또는 utils/format.py — 서버 측 메시지 생성 시 사용
def format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    if m == 0:
        return f"{s}초"
    return f"{m}분 {s}초"
```

```typescript
// Explorer/Vision 공통 유틸 — 클라이언트 측 시간 포맷
export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m === 0 ? `${s}초` : `${m}분 ${s}초`;
}
```

### D.3 알림 표시 위치 규칙 (v2.0)

- **가드 UI** (화면 진입 차단): 해당 페이지 컴포넌트 최상단에 조건부 렌더링, 콘텐츠 미렌더링
- **인라인 알림** (오류·경고 메시지): 관련 입력/버튼 바로 아래 렌더링
- **토스트 알림** (학습 완료·저장 완료 등 일시적 알림): 화면 오른쪽 하단 고정 위치
- **배너 알림** (학습 중단·오류 등 지속적 알림): 해당 화면 콘텐츠 최상단, 다음 탐색 시까지 유지
- **모달** (불량 감지 팝업): 화면 전체 오버레이 (Vision 자동 검사 중단과 연동)

---

## E. Training Progress Observability

> **v2.0 변경**: Streamlit `st.progress()` / `st.plotly_chart()` / `st.text_area()` 기반 구현이  
> `/ws/training` WebSocket 스트림 + React 컴포넌트 기반으로 전면 교체됐다.  
> Streamlit 기반 구 구현은 [E.7 v1.x 참고](#e7-v1x-참고--streamlit-기반-학습-진행-표시-이력-참조-전용)로 이동.

### E.1 WebSocket 스트림 아키텍처 (v2.0)

학습 진행 정보의 흐름:

```
TrainingWorker (Python thread)
  ↓  result_queue.put({"type": "progress"|"log"|"stage"|"completed"|..., ...})
FastAPI WS 핸들러 (/ws/training)
  ↓  await websocket.send_json(msg)
Browser WebSocket (ws://localhost:8000/ws/training)
  ↓  onmessage 이벤트
useTrainingWs.ts (Explorer hooks/)
  ↓  trainingStore dispatch
React 컴포넌트
   ├── ProgressSection.tsx  (진행률 바 + Loss 차트)
   ├── StageIndicator.tsx   (현재 학습 단계명)
   └── QueuePanel.tsx       (배치 학습 큐 진행)
```

**서버 측 WS 핸들러 개요**:
```python
# api/ws/training_ws.py (개념)
@app.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                msg = result_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
            await websocket.send_json(msg)
            if msg.get("type") in ("completed", "stopped", "error"):
                break  # 터미널 메시지 이후 연결 유지 or 종료
    except WebSocketDisconnect:
        pass
```

### E.2 Progress Bar (v2.0)

**데이터 소스**: `trainingStore.progress.currentStep` / `trainingStore.progress.totalSteps`

```typescript
// components/training/ProgressSection.tsx 패턴
import { useTrainingStore } from '@/store/trainingStore';

const { currentStep, totalSteps, elapsedS } = useTrainingStore((s) => s.progress);
const ratio = totalSteps > 0 ? currentStep / totalSteps : 0;

return (
  <div>
    <div className="flex justify-between text-sm">
      <span>Step {currentStep.toLocaleString()} / {totalSteps.toLocaleString()}</span>
      <span>{(ratio * 100).toFixed(1)}%</span>
      <span>경과: {formatDuration(elapsedS)}</span>
    </div>
    <div className="w-full bg-gray-200 rounded h-2">
      <div className="bg-blue-500 h-2 rounded" style={{ width: `${ratio * 100}%` }} />
    </div>
  </div>
);
```

- EfficientAD: `progress` 메시지의 `step` / `total_steps` 필드 사용 (매 100 step)
- PatchCore: epoch 단위 메시지 사용

### E.3 Loss 곡선 (v2.0)

**데이터 소스**: `trainingStore.lossHistory` — `Array<{ step: number; loss: number }>`

```typescript
// components/training/ProgressSection.tsx 내 Recharts 기반 차트
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useTrainingStore } from '@/store/trainingStore';

const lossHistory = useTrainingStore((s) => s.lossHistory);

return (
  <ResponsiveContainer width="100%" height={300}>
    <LineChart data={lossHistory}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom" }} />
      <YAxis label={{ value: "Loss", angle: -90, position: "insideLeft" }} />
      <Tooltip formatter={(v: number) => v.toFixed(4)} />
      <Line type="monotone" dataKey="loss" dot={false} stroke="#3b82f6" strokeWidth={1.5} />
    </LineChart>
  </ResponsiveContainer>
);
```

`lossHistory` 업데이트 조건:
- EfficientAD: `progress` 메시지에 `loss` 필드가 있을 때만 append (매 100 step)
- PatchCore: epoch 완료 메시지마다 append

### E.4 실시간 로그 (v2.0)

**데이터 소스**: `trainingStore.logs` — `string[]` (최신 100줄, 11절 §G.3)

```typescript
// components/training/ProgressSection.tsx 내 로그 표시
import { useTrainingStore } from '@/store/trainingStore';
import { useEffect, useRef } from 'react';

const logs = useTrainingStore((s) => s.logs);
const textareaRef = useRef<HTMLTextAreaElement>(null);

// 새 로그 수신 시 스크롤 하단 유지
useEffect(() => {
  if (textareaRef.current) {
    textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
  }
}, [logs]);

return (
  <textarea
    ref={textareaRef}
    readOnly
    className="w-full h-48 font-mono text-xs bg-gray-900 text-gray-100 p-2 rounded resize-none"
    value={logs.join('\n')}
  />
);
```

`logs` 업데이트: WS `log` 타입 메시지 수신 시 `trainingStore.addLog(message)` → 100줄 초과 시 앞부분 제거 (11절 §G.3 참조).

### E.5 WebSocket 연결 관리

**훅 생명주기** (`useTrainingWs.ts`):

```typescript
// hooks/useTrainingWs.ts 패턴
import { useEffect, useRef, useCallback } from 'react';
import { useTrainingStore } from '@/store/trainingStore';

const WS_URL = `ws://${import.meta.env.VITE_API_HOST ?? 'localhost:8000'}/ws/training`;
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 3_000;

export function useTrainingWs() {
  const wsRef = useRef<WebSocket | null>(null);
  const retryCount = useRef(0);
  const dispatch = useTrainingStore((s) => s.dispatch);

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      dispatch(msg);  // trainingStore에서 타입별 처리
    };

    ws.onclose = () => {
      if (retryCount.current < MAX_RETRIES) {
        retryCount.current += 1;
        setTimeout(connect, RETRY_DELAY_MS);
      }
    };
  }, [dispatch]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);
}
```

**재연결 정책**: 연결 해제 시 3초 후 자동 재연결, 최대 5회 시도. 학습 중 브라우저 새로고침 시 재연결 후 서버 측 현재 상태로 동기화.

### E.6 WS 메시지 타입별 처리 요약 (v2.0)

| 타입 | `useTrainingWs.ts` / `trainingStore` 처리 | 참조 |
|------|------------------------------------------|------|
| `progress` | `trainingStore.setProgress(step, totalSteps, loss, elapsedS)` + `addLossPoint()` | E.2, E.3 |
| `log` | `trainingStore.addLog(message)` | E.4, 11절 §G.3 |
| `stage` | `trainingStore.setStage(stageName)` → `StageIndicator.tsx` 업데이트 | Explorer README |
| `completed` | `trainingStore.setStatus("completed")` + 완료 토스트 표시 → `/experiments` 이동 제안 | 06절 §C.3 |
| `error` | `trainingStore.setStatus("error")` + 오류 배너 표시 | 06절 §C.3 |
| `stopped` | `trainingStore.setStatus("stopped")` + 중단 배너 표시 | 06절 §C.3 |

### E.7 [v1.x 참고] Streamlit 기반 학습 진행 표시 (이력 참조 전용)

> **이 절은 이력 참조 전용이다. 현행 구현에는 적용되지 않는다.**  
> Streamlit을 개발자 디버그 도구로 사용하는 경우 참고할 수 있다.

**v1.x Progress Bar**:
```python
# [v1.x 참고 — 현재 비사용]
progress = st.session_state.get("current_step", 0) / st.session_state["model_config"]["params"]["train_steps"]
st.progress(min(progress, 1.0), text=f"학습 진행률: {progress*100:.1f}%")
```

**v1.x Loss 곡선** (Plotly):
```python
# [v1.x 참고 — 현재 비사용]
import plotly.graph_objects as go
if st.session_state.get("loss_history"):
    df = pd.DataFrame(st.session_state["loss_history"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["step"], y=df["loss"], mode="lines", name="Loss"))
    st.plotly_chart(fig, use_container_width=True)
```

**v1.x 실시간 로그**:
```python
# [v1.x 참고 — 현재 비사용]
log_text = "\n".join(st.session_state.get("log_buffer", []))
st.text_area("학습 로그", value=log_text, height=300, disabled=True, key="training_log_display")
```

**v1.x Queue 메시지 처리** (session_state 기반):

| 타입 | 처리 |
|------|------|
| `progress` | `st.session_state["current_step"]` 업데이트, `loss_history` append |
| `log` | `st.session_state["log_buffer"]` append |
| `completed` | `_handle_terminal()` → 실험 저장, anomaly_map 캐시 |
| `error` | `_handle_terminal()` → `st.error()` 표시 |
| `stopped` | `_handle_terminal()` → 중단 레코드 저장 |

---

## F. Operations Runbook

### F.1 학습이 멈춰 보이는 경우

**증상**: Explorer `/training` 화면의 Progress bar가 오랫동안 움직이지 않음.

**진단**:
1. `./logs/{exp_id}.log` 마지막 라인 확인 — 최근 타임스탬프 체크
2. FastAPI 서버 콘솔 로그 확인 — `result_queue` 처리 중인지 확인
3. Explorer 브라우저 DevTools > Network > WS 탭 — `/ws/training` 연결 상태 확인 (열려 있는지, 메시지가 오는지)

**조치**:
- WS가 끊어진 경우 (`CLOSED` 상태): 브라우저 새로고침 → `useTrainingWs.ts` 재마운트 → 자동 재연결
- WS는 연결 중인데 메시지 없음: FastAPI 서버 콘솔에서 `TrainingWorker` 스레드 살아있는지 확인
- 서버 콘솔 정상, 로그 파일 업데이트 중이면: WS 전송 경로 문제 → FastAPI 서버 재시작
- GPU OOM: FastAPI 콘솔에 `torch.cuda.OutOfMemoryError` → `batch_size` 줄이고 재시작

---

### F.2 학습 완료 후 실험 히스토리에서 실험이 안 보이는 경우

**진단**:
1. `./experiments/history.json` 파일 존재 확인
2. `curl http://localhost:8000/api/experiments` 응답에 해당 실험 포함 여부 확인
3. Explorer `/experiments` 화면 새로고침 (F5 또는 페이지 재진입)

**조치**:
- `GET /api/experiments` 응답에 있는데 UI에 없음: Explorer `experimentsStore` 캐시 문제 → 페이지 재진입으로 재조회
- API 응답에도 없음:
  - `history.json` 존재하지 않으면: `experiments/` 디렉터리 내 `.tmp` 파일 확인 (원자적 쓰기 실패)
  - `history.json` 있으나 레코드 없음: WS `completed` 메시지 처리 경로 확인 (`save_history()` 호출 여부)
- `model_path` 존재하지 않는 레코드: Explorer에서 "(파일 없음)" 표시 확인 → `/experiments` 화면에서 [삭제] 후 재학습

---

### F.3 모델 파일이 없는데 history에는 completed로 표시되는 경우

**진단**:
1. `./models/{exp_id}/` 디렉터리 존재 확인
2. `model_state_dict.pth` 파일 존재 확인
3. `history.json` 의 `model_path` 필드 확인

**조치**:
- 파일이 없으면 해당 실험은 `/anomaly-map` 화면에서 추론 불가 — Explorer UI에서 "(파일 없음)" 배지 표시
- 불완전한 레코드: `/experiments` 화면에서 [삭제] 후 재학습

---

### F.4 ImageNet penalty 오류 (EfficientAD)

**증상**: Explorer `/training` 화면에서 [학습 시작] 클릭 시 즉시 오류 메시지 (`POST /api/training/start` 400 응답).

**진단**:
```bash
# Windows PowerShell — 디렉터리 내 이미지 파일 수 확인
(Get-ChildItem -Path ".\dataset\imagenet_penalty" -Recurse -Include "*.JPEG","*.jpg","*.png").Count

# Linux/macOS
find ./dataset/imagenet_penalty -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l
```

**조치**:
- 0이면 ImageNet 이미지를 `./dataset/imagenet_penalty/` 에 배치
- 09절 §F.3 참조

---

### F.5 포트 충돌 (v2.0 — FastAPI + React dev server)

**FastAPI :8000 충돌**:

```powershell
# Windows — :8000 포트 사용 프로세스 확인
netstat -ano | findstr :8000

# 프로세스 종료 (PID 확인 후)
taskkill /PID <PID> /F

# 다른 포트로 실행
uvicorn api.main:app --reload --port 8001
# → Explorer .env: VITE_API_BASE_URL=http://localhost:8001 수정 필요
```

```bash
# Linux/macOS
lsof -i :8000
kill -9 <PID>
```

**React dev server :5173 충돌 (Explorer 또는 Vision 동시 실행 시)**:

```bash
# Vision을 대체 포트로 실행
cd smart-qc-vision
npx vite --port 5174
# → Vision의 CORS 오리진 추가 필요: localhost:5174 (10절 §C.1 참조)
```

---

### F.6 로그 파일 용량 정리

**주기적 정리** (수동):
```powershell
# Windows — 30일 이상 된 로그 삭제
Get-ChildItem -Path ".\logs" -Filter "*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force

# 모든 로그 삭제 (주의)
Remove-Item ".\logs\*.log" -Force
```

```bash
# Linux/macOS
find ./logs -name "*.log" -mtime +30 -delete
```

> Explorer `/experiments` 화면에서 실험을 삭제하면 해당 로그 파일도 자동 삭제됨 (C.3 참조).

---

## G. Vision 검사 로그 이벤트 (v2.0)

> **v2.0 변경**: v1.1의 `inspection/tabs/insp_tab*.py` Streamlit 기반 코드가  
> FastAPI API 핸들러 및 Vision React hook 기준으로 교체됐다.

### G.1 Vision 검사 필수 로그 이벤트

00절 §7 기준 — Vision 검사에서 발생하는 로그 이벤트 구현 명세 (v2.0: FastAPI route 기준):

| 이벤트명 | 레벨 | 호출 위치 | `data` 필드 |
|---------|------|---------|------------|
| `insp_model_applied` | INFO | `POST /api/inspection/model` 핸들러 — 모델 적용 성공 시 | `{"experiment_id": str, "model_type": str}` |
| `insp_inspection_started_manual` | INFO | `POST /api/inspection/run` 핸들러 — 수동 검사 요청 수신 시 | `{"image_name": str}` |
| `insp_inspection_started_auto` | INFO | `WS /ws/inspection/auto` 핸들러 — 연결 수립 직후 | `{"interval_s": float}` |
| `insp_inspection_stopped_auto` | INFO | `WS /ws/inspection/auto` 핸들러 — `WebSocketDisconnect` 수신 시 | `{"total_inspected": int}` |
| `insp_defect_detected` | WARNING | `POST /api/inspection/run` 또는 `WS /ws/inspection/auto` — 판정 결과가 NG(불량)인 경우 | `{"anomaly_score": float, "threshold": float, "image_name": str}` |
| `insp_pool_reshuffled` | INFO | `test_sampler.sample_from_pool()` 풀 소진 후 재섞기 시 | `{"pool_size": int}` |

> 참고: `insp_inspection_result_recorded` 이벤트는 매 검사마다 발생하므로 필요 시 선택적으로 추가할 수 있으나, 기본 구현에서는 `insp_defect_detected`(WARNING)만 필수로 기록한다.

### G.2 Vision 검사 이벤트 호출 예시 (v2.0)

```python
# api/routers/inspection.py — FastAPI 핸들러

from utils.logger import log_info, log_warning

@router.post("/api/inspection/run")
async def run_manual_inspection(request: InspectionRunRequest):
    log_info(
        "insp_inspection_started_manual",
        f"수동 검사 시작: {request.image_name}",
        tab="api/inspection/run",
        data={"image_name": request.image_name},
    )

    result = await _run_inspection_async(request.image_name)

    if result["verdict"] == "NG":
        log_warning(
            "insp_defect_detected",
            f"불량 감지: {request.image_name} (score={result['anomaly_score']:.4f})",
            tab="api/inspection/run",
            data={
                "anomaly_score": result["anomaly_score"],
                "threshold": result["threshold"],
                "image_name": request.image_name,
            },
        )
    return result


@router.post("/api/inspection/model")
async def apply_model(request: ApplyModelRequest):
    # ... 모델 적용 로직 ...
    log_info(
        "insp_model_applied",
        f"모델 적용 완료: {request.experiment_id}",
        tab="api/inspection/model",
        data={
            "experiment_id": request.experiment_id,
            "model_type": request.model_type,
        },
    )
    return {"status": "applied"}
```

```python
# api/ws/inspection_ws.py — WebSocket 핸들러

@app.websocket("/ws/inspection/auto")
async def ws_auto_inspection(websocket: WebSocket):
    await websocket.accept()
    log_info(
        "insp_inspection_started_auto",
        "자동 검사 시작",
        tab="ws/inspection/auto",
        data={"interval_s": 3.0},
    )
    count = 0
    try:
        while True:
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_inspection_sync
            )
            count += 1
            if result["verdict"] == "NG":
                log_warning(
                    "insp_defect_detected",
                    f"불량 감지 (score={result['anomaly_score']:.4f})",
                    tab="ws/inspection/auto",
                    data={
                        "anomaly_score": result["anomaly_score"],
                        "threshold": result["threshold"],
                        "image_name": result.get("image_name", ""),
                    },
                )
            await websocket.send_json(result)
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        log_info(
            "insp_inspection_stopped_auto",
            "자동 검사 중지 — 클라이언트 연결 해제",
            tab="ws/inspection/auto",
            data={"total_inspected": count},
        )
```

```python
# utils/test_sampler.py

def sample_from_pool(pool: list, used: set) -> tuple[str, set]:
    remaining = [p for p in pool if p not in used]
    if not remaining:
        log_info(
            "insp_pool_reshuffled",
            "테스트 풀 소진 — 재섞기",
            data={"pool_size": len(pool)},
        )
        used = set()
        remaining = pool[:]
    selected = random.choice(remaining)
    used.add(selected)
    return selected, used
```

### G.3 Vision 검사 UI 알림 조건 (v2.0)

| 조건 | React/API 처리 방법 | 메시지 |
|------|---------------------|--------|
| 적용된 모델 없음 | `NoModelGuard` 컴포넌트 렌더링 (Vision `components/layout/`) — 콘텐츠 미렌더링 | "모델을 먼저 적용해 주세요. (탭3 모델 교체)" |
| 완료된 실험 없음 | `GET /api/models` 빈 배열 → 안내 UI 표시 | "학습 완료된 실험이 없습니다. Explorer에서 학습을 완료해 주세요." |
| 불량 감지 (팝업) | WS 메시지 `verdict=="NG"` → `inspectionStore.showDefectPopup = true` → React 모달 + `ws.close()` 자동 호출 | "불량이 감지되었습니다. 자동 검사를 중지합니다." |

**Vision `useAutoInspection.ts` 내 불량 처리 패턴**:
```typescript
// hooks/useAutoInspection.ts
ws.onmessage = (event) => {
  const msg: WsMessage = JSON.parse(event.data);
  inspectionStore.getState().setLastResult(msg);

  if (msg.verdict === 'NG') {
    inspectionStore.getState().setShowDefectPopup(true);
    ws.close();  // 자동 검사 중지 — 서버 측 WebSocketDisconnect 트리거
  }
};
```

---

*다음 문서*: [13_QA_and_Testing_Strategy.md](./13_QA_and_Testing_Strategy.md)

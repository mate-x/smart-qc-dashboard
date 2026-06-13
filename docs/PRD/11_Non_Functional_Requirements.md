# 11. Non-Functional Requirements

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md) §6 Global NFR
> **선행 문서**: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **수정일**: 2026-06-11
> **중요**: 00절 §6의 NFR 테이블을 기준으로, 각 항목에 대한 **측정 방법·합격 기준·위반 시 처리**를 확정한다. 구현 시 이 문서와 00절 §6을 함께 참조한다.

---

## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 최초 작성 |
| v1.1 | 2026-05-26 | 비전검사 대시보드 NFR 추가 (B.5~B.7, H 요약표 갱신) |
| v2.0 | 2026-06-11 | B.1 React UI 화면 전환 기준으로 교체; B.8 WebSocket 지연 NFR 신규 추가; B.5~B.7 WS asyncio 루프 기준으로 재작성; C.3 FastAPI 재시작 기준 갱신; D.1 화면 가드 + Zustand 기준으로 교체; D.2/D.4 React 컴포넌트·API response 기준으로 교체; E.1 datasetStore 기준 갱신; G.2 서버 측 LRU 캐시 기준; G.3 WS 스트림 + trainingStore.logs 기준; H 요약표 NFR-P-06/07/08 추가 |

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Performance Requirements](#b-performance-requirements)
- [C. Reliability Requirements](#c-reliability-requirements)
- [D. Usability Requirements](#d-usability-requirements)
- [E. Data Integrity Requirements](#e-data-integrity-requirements)
- [F. Reproducibility Requirements](#f-reproducibility-requirements)
- [G. Resource Management Requirements](#g-resource-management-requirements)
- [H. NFR 합격 기준 요약표](#h-nfr-합격-기준-요약표)

---

## A. Objective & Scope

### A.1 이 문서의 목적

00절 §6에서 테이블 형식으로 선언된 NFR 항목을 다음 세 축으로 구체화한다:

1. **측정 방법**: 어떻게 해당 요구사항이 충족됐는지 확인하는가
2. **합격 기준**: 숫자 또는 명확한 조건으로 표현된 PASS/FAIL 기준
3. **위반 시 처리**: 합격 기준 미충족 시 취해야 할 조치

이 문서는 성능·신뢰성·사용성·무결성·재현성·자원 관리 영역을 다룬다.  
보안 관련 NFR은 `10_Security_and_Compliance.md`, 관찰 가능성은 `12_Observability_and_Operations.md`에서 별도 다룬다.

---

## B. Performance Requirements

### B.1 UI 응답성 (v2.0 — React 기준)

**요구사항 (00절 §6)**: 학습 중 UI 블로킹 없음. 화면 전환 응답 < 1초.

**설계 보장 메커니즘**:
- 학습은 `threading.Thread` + `queue.Queue`로 백그라운드 분리 (00절 §9 A-02)
- 학습 진행 상태는 WebSocket(`/ws/training`)으로 Push — React UI는 WS 이벤트 수신만 처리
- Explorer 화면 전환은 `react-router-dom`의 `navigate()` 호출 — Python 코드 실행 없음
- API 호출 중 로딩 상태는 Zustand store의 `isLoading` 플래그로 관리, React Suspense·skeleton으로 표시

**측정 방법**:
1. 학습 실행 중 Explorer 탭1~탭5 화면을 순서대로 전환
2. 각 `navigate()` 호출 시 브라우저 렌더링 완료까지 시간 측정 (Chrome DevTools Performance)
3. API 호출이 포함된 화면(탭4 `/experiments` 등)은 응답 완료 ~ 첫 콘텐츠 표시까지 측정

**합격 기준**:

| 측정 항목 | 기준 |
|----------|------|
| react-router 화면 전환 (API 호출 없음) | ≤ 200ms |
| 화면 진입 후 첫 API 응답 + 렌더링 | ≤ 1,000ms |
| 학습 실행 중 다른 화면으로 전환 시 UI 블로킹 | 없음 (≤ 1,000ms) |
| 로딩 스피너 / 스켈레톤 미표시로 빈 화면 유지 | 허용 불가 |

**위반 시 처리**:
- 특정 화면 전환 > 1,000ms: 해당 화면 `useEffect`에서 실행 중인 동기 연산 확인
- API 응답 대기 중 빈 화면: `isLoading` 상태에서 skeleton 컴포넌트 표시 확인

---

### B.2 학습 성능 — EfficientAD

**요구사항 (00절 §6)**: g4dn.xlarge (Tesla T4, 16GB VRAM)에서 EfficientAD-medium 70,000 steps **20분 이내**.

**측정 방법**:
1. MVTec AD Screw 데이터셋 기준
2. `model_size=medium`, `train_steps=70000`, 기본 파라미터
3. `experiment.duration_seconds` 필드 값 사용 (07절 §B.4 확정)

**합격 기준**:
- `duration_seconds` ≤ 1,200 (= 20분)

**위반 시 처리**:
- `penalty_batch_size` 축소 (8 → 4) 검토
- `image_size` 축소 (256 → 224) 검토
- 위 조치 후 재측정

---

### B.3 학습 성능 — PatchCore

**요구사항 (00절 §6)**: g4dn.xlarge에서 PatchCore (coreset_sampling_ratio=0.1) **10분 이내**.

**측정 방법**:
1. MVTec AD Screw 데이터셋 기준
2. `backbone=wide_resnet50_2`, `coreset_sampling_ratio=0.1`, 기본 파라미터
3. `experiment.duration_seconds` 필드 값 사용

**합격 기준**:
- `duration_seconds` ≤ 600 (= 10분)

**위반 시 처리**:
- `coreset_sampling_ratio` 축소 (0.1 → 0.05) 검토
- `max_train` 축소 검토
- backbone을 `resnet50`으로 변경 검토

---

### B.4 추론 응답 시간

**요구사항**: 단일 이미지 추론 (anomaly map 생성) 결과 표시까지 사용자 대기 시간 합리적 범위.

**설계 보장 메커니즘**:
- Explorer 탭5 Anomaly Map 빌드: `POST /api/anomaly-map/{expId}/build` → 비동기 job + `GET /api/anomaly-map/job/{jobId}` 1초 폴링
- 이미 빌드된 캐시가 서버 메모리에 존재하면 `GET /api/anomaly-map/{expId}/status` 즉시 완료 응답
- `MAX_ANOMALY_MAP_CACHE = 3`으로 반복 추론 회피 (G.2 참조)

**합격 기준**:
- 모델 로드 포함 첫 Anomaly Map 빌드 완료: ≤ 30초 (GPU 환경, image_size=256)
- 캐시 히트 시 빌드 job 즉시 완료 응답: ≤ 1초

**위반 시 처리**:
- 로드 시간 초과: `load_model_for_inference()` 내 불필요한 초기화 제거
- 캐시 miss가 잦으면 `MAX_ANOMALY_MAP_CACHE` 상향 검토 (서버 메모리 여유 확인 후)

---

## C. Reliability Requirements

### C.1 파일 저장 원자성

**요구사항 (00절 §8 R-ATOMIC-01)**: 모든 파일 쓰기는 `.tmp` → `os.replace()` 패턴 사용.

**설계 보장 메커니즘** (05절 §3.2):
- `history.json` 쓰기: `history.tmp` → `os.replace()`
- `model_state_dict.pth`: `model_state_dict.pth.tmp` → `os.replace()`
- `configs.yaml`: 직접 쓰기 허용 (텍스트 파일, 실패 시 YAML 파싱 오류로 즉시 감지)

**합격 기준**:
- 저장 중 프로세스 강제 종료 시 `.tmp` 파일만 남고 완성 파일은 손상 없음
- 재시작 후 `.tmp` 파일 존재 시 자동 삭제 (앱 시작 시 cleanup)

**위반 시 처리**:
- `.tmp` 파일이 잔존하는 경우: 앱 시작 시 `glob("**/*.tmp")`로 탐색 후 삭제

---

### C.2 학습 중단 처리

**요구사항**: Explorer 학습 화면에서 [학습 중지] 클릭 시 학습 스레드가 안전하게 종료되고 `status="중단"` 레코드가 history.json에 기록되어야 한다.

**설계 보장 메커니즘** (06절 §B.4 R-RACE-01~04):
- `POST /api/training/stop` → `stop_event.set()` → 학습 루프 내 주기적 `if stop_event.is_set(): break`
- "first terminal message wins" 규칙 — 중복 terminal 메시지 무시
- `stopped` 메시지 수신 후 `_build_experiment_record(status="중단")` → `save_history()`
- WebSocket으로 `{"type": "stopped", "step": step}` 전송 → Explorer UI 업데이트

**합격 기준**:
- [학습 중지] 클릭 후 ≤ 10초 내 학습 스레드 완전 종료 (마지막 step 완료 시간 포함)
- history.json에 `status == "중단"` 레코드 존재
- `model_path`, `configs_path`, `metrics` 모두 `null` (00절 §1.1 R-05)

**위반 시 처리**:
- 10초 초과 시: `stop_event` 체크 주기 확인 (매 step 또는 매 배치)
- 레코드 미기록 시: `_handle_terminal()` 내 `save_history()` 호출 경로 추적

---

### C.3 서버 재시작 후 복구

**요구사항**: FastAPI 서버 재시작 후 이전 실험 히스토리가 Explorer 실험 히스토리 화면(`/experiments`)에 복원되어야 한다.

**설계 보장 메커니즘**:
- `history.json` 파일 기반 영속 (05절 §3.1) — FastAPI 프로세스 재시작과 무관하게 유지
- `GET /api/experiments` 호출 시 항상 `history.json`에서 최신 데이터 로딩
- Explorer `/experiments` 화면 진입 시 `experimentsApi.getExperiments()` 자동 호출

**합격 기준**:
- 서버 재시작 후 Explorer `/experiments` 화면 진입 시 이전 실험 목록 전체 표시
- `completed` 상태 실험의 `model_path` 파일 존재 확인 (R-04 참조무결성)

**위반 시 처리**:
- `GET /api/experiments` 응답이 빈 목록: FastAPI 서버의 `load_history()` 호출 시점 확인
- `model_path` 존재하지 않는 레코드: Explorer UI에서 "(파일 없음)" 배지 표시 + 추론 비활성화

---

## D. Usability Requirements

### D.1 화면 가드 (v2.0 — React route 기준)

**요구사항 (00절 §3.2)**: 선행 화면 미완료 시 다음 화면 접근 차단 + 안내 메시지.

**설계 보장 메커니즘**:
- 각 페이지 컴포넌트 상단에서 Zustand store 상태 확인 → 조건 미충족 시 가드 UI 렌더링 (콘텐츠 미렌더링)
- react-router의 route 수준 리다이렉트가 아닌, **컴포넌트 수준 가드** 패턴 사용

**합격 기준**: 아래 조건 전부 충족

| 조건 | 기대 동작 |
|------|-----------|
| `datasetStore.datasetPath === null` + `/config` 화면 진입 | 가드 UI 표시 ("데이터셋을 먼저 설정해 주세요") + 콘텐츠 미렌더링 |
| `datasetStore.datasetPath === null` 또는 `configStore.preprocessingConfig === null` 또는 `configStore.modelConfig === null` + `/training` 화면 진입 | 가드 UI 표시 ("전처리 및 모델 설정을 먼저 완료해 주세요") + 콘텐츠 미렌더링 |
| `experimentsStore` 빈 목록 + `/experiments` 화면 진입 | 안내 메시지 표시 ("완료된 실험이 없습니다") |
| `experimentsStore.selectedExperimentId === null` + `/anomaly-map` 화면 진입 | 가드 UI 표시 ("실험 히스토리에서 실험을 먼저 선택해 주세요") |

**측정 방법**: 각 조건을 수동으로 재현하여 가드 UI 표시 및 콘텐츠 미렌더링 확인.

---

### D.2 UI 언어 및 표기 일관성

**요구사항 (00절 §8 R-UI-01)**: 모든 라벨·버튼·안내 메시지 한국어. 기술 용어는 한국어+영문 병기.

**합격 기준**:
- 모든 `<button>`, `<label>`, `<h1>~<h4>` 텍스트가 한국어
- 모델명 등 고유명사: `EfficientAD`, `PatchCore` (영문 유지)
- 수치 단위: 한국어 병기 (예: "학습 단계 수 (Train Steps)")

---

### D.3 숨김 처리

**요구사항 (00절 §8 R-UI-02)**: 비선택 파라미터 UI는 DOM 미렌더링 (`if` 분기). `disabled` prop 금지.

**합격 기준**:
- EfficientAD 선택 시 PatchCore 전용 파라미터 UI 요소 (`PatchCoreParams.tsx`)가 DOM에 존재하지 않음
- `<input disabled>` 형태 코드가 없음 (React `disabled={true}` prop 형태 포함)

---

### D.4 디스크 공간 경고

**요구사항 (00절 §6)**: 모델 저장 전 여유 공간이 모델 타입별 기준 미만이면 경고 메시지.

PatchCore는 `register_buffer`로 memory_bank가 `.pth`에 포함되어 파일 크기가 크므로 기준을 높게 설정한다.

| 모델 타입 | 경고 기준 | 근거 |
|----------|----------|------|
| EfficientAD | 500 MB | backbone만 저장 (~200~400MB) |
| PatchCore | 1024 MB | backbone + memory_bank 포함 (~600~1000MB) |

**설계 보장 메커니즘**:
- FastAPI 서버 측에서 디스크 여유 공간 확인 후 API 응답에 `disk_warning` 필드 포함
- React UI는 `disk_warning === true` 시 토스트/배너 표시

```python
# api/routers/experiments.py (또는 유사) — 서버 측 경고 생성
import shutil

def build_disk_warning(model_type: str) -> dict:
    usage = shutil.disk_usage("./models")
    free_mb = usage.free / (1024 * 1024)
    warn_mb = 1024 if model_type == "patchcore" else 500
    return {
        "disk_warning": free_mb < warn_mb,
        "free_mb": round(free_mb),
        "warn_mb": warn_mb,
    }
```

**합격 기준**:
- `free_mb < warn_mb` 조건에서 API 응답에 `"disk_warning": true` 포함
- Explorer UI에서 경고 배너/토스트가 렌더링되어야 함
- 경고 상태에서도 사용자가 강제 저장 가능해야 함 (차단 아닌 경고)

---

## E. Data Integrity Requirements

### E.1 MVTec AD 폴더 구조 검증

**요구사항 (00절 §6)**: 폴더 구조 검증 실패 시 Config 화면(`/config`) 진입 완전 차단.

**검증 항목** (01절 §B.2 기준):

| 조건 | 오류 코드 |
|------|----------|
| `{dataset_root}/train/good/` 미존재 | `ERR_INVALID_FOLDER_STRUCTURE` |
| `train/good/` 내 지원 포맷 이미지 0개 | `ERR_NO_VALID_IMAGES` |
| `{dataset_root}/test/` 미존재 | `ERR_INVALID_FOLDER_STRUCTURE` |

**합격 기준**:
- 위 조건 중 하나라도 해당 시 `datasetStore.datasetPath = null` 유지
- D.1 화면 가드가 `datasetStore.datasetPath === null` 체크로 `/config` 화면 차단

**위반 판단**: 검증 실패 경로에서 Config 화면이 렌더링되면 즉시 버그 처리.

---

### E.2 이미지 처리 무결성

**요구사항 (00절 §6)**: Resize+Padding 후 가로·세로 비율 유지 오차 < 1픽셀.

**측정 방법**:
```python
from utils.image_utils import resize_with_padding
import numpy as np

# 원본 비율: 4:3 이미지 (400×300)
img = np.zeros((300, 400, 3), dtype=np.uint8)
result = resize_with_padding(img, target_size=256)

# 원본 비율 계산
orig_ratio = 400 / 300  # ≈ 1.333
# 리사이즈 후 실제 이미지 영역 비율 계산 (패딩 제외)
h_content = int(256 / orig_ratio)  # ≈ 192
ratio_error = abs(256 * (300/400) - h_content)  # 픽셀 단위

assert ratio_error < 1.0, f"비율 오차 {ratio_error:.3f} >= 1픽셀"
```

**합격 기준**: `ratio_error < 1.0` (픽셀)

---

### E.3 채널 처리

**요구사항 (00절 §6)**: Grayscale 이미지 자동 RGB 변환, 3채널 보장.

**합격 기준**:
```python
from PIL import Image
from utils.image_utils import ensure_rgb
import numpy as np

gray_img = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))  # L mode
rgb_arr = ensure_rgb(gray_img)
assert rgb_arr.shape == (256, 256, 3), f"채널 수 오류: {rgb_arr.shape}"
assert np.all(rgb_arr[:, :, 0] == rgb_arr[:, :, 1] == rgb_arr[:, :, 2]), "R=G=B 조건 위반"
```

---

### E.4 참조 무결성 (00절 §2 R-01~R-05)

| 규칙 | 합격 기준 | 검증 시점 |
|------|----------|-----------|
| R-01 | `experimentsStore.selectedExperimentId` 조회 시 실험 목록에 해당 ID 존재 | `/anomaly-map` 화면 진입 시 |
| R-02 | `modelConfig.image_size == preprocessingConfig.image_size` | `/config` 화면에서 설정 저장 시 자동 동기화 |
| R-04 | `status=="completed"` 이고 `model_path` NOT NULL인 경우 파일 존재 | `/experiments` 화면 로딩 시 |
| R-05 | `status=="중단"` 레코드의 `metrics`, `model_path`, `configs_path` 모두 `null` | 중단 처리 시 |

---

## F. Reproducibility Requirements

### F.1 랜덤 시드 고정

**요구사항 (00절 §6, §8 R-SEED-01)**: 동일 `random_seed` + 동일 하이퍼파라미터 시 동일 결과.

**구현 명세**:
```python
# utils/reproducibility.py
import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 결정론적 알고리즘 강제 (성능 일부 희생)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

`set_global_seed()`는 `TrainingWorker.run()` 진입 직후 호출한다.

**합격 기준**:
- 동일 데이터셋, 동일 `model_config` (포함 `random_seed`), 동일 환경에서 2회 실행 시:
  - `metrics.auc` 차이 ≤ 0.001
  - `metrics.anomaly_scores` 배열 요소별 절대 오차 ≤ 0.0001

**제약사항**:
- `torch.backends.cudnn.benchmark = True`가 설정된 경우 재현성 보장 불가
- 멀티스레드 데이터로더 (`num_workers > 0`)가 OS 스케줄링 영향으로 비결정론적일 수 있음 — `num_workers=0` 권장

---

### F.2 실험 설정 기록

**요구사항**: 각 실험의 `model_config`와 `preprocessing_config`가 `configs.yaml`에 완전히 기록되어야 한다.

**합격 기준**:
- `save_completed_experiment()` Stage 2 완료 후 `{exp_id}/configs.yaml` 존재
- 해당 파일 로드 시 원본 `model_config`와 동일한 값 복원 가능
- `json.dumps(original) == json.dumps(loaded_from_yaml)` (정렬 기준 동일)

---

## G. Resource Management Requirements

### G.1 GPU 메모리 해제

**요구사항**: Anomaly Map 빌드 완료 후 GPU 메모리를 즉시 해제해야 한다.

**구현 명세** (07절 §C.3):
```python
# FastAPI ML 레이어 — 추론 완료 직후
del model
torch.cuda.empty_cache()
```

**합격 기준**:
- 추론 전 VRAM 사용량 (A)과 추론 후 VRAM 사용량 (B): `B - A < 100 MB` (잔여 캐시 허용)
- `nvidia-smi` 또는 `torch.cuda.memory_allocated()`로 측정

---

### G.2 Anomaly Map 캐시 상한

**요구사항 (05절 §4.2)**: `MAX_ANOMALY_MAP_CACHE = 3` — 최대 3개 실험의 anomaly_maps만 **서버 메모리**에 보관.

**설계 보장 메커니즘**:
- `set_anomaly_map_cache(experiment_id, data)` — LRU 3-entry 인메모리 캐시에 추가
- 4번째 추가 시 `cached_at` 가장 오래된 항목 자동 삭제

**합격 기준**:
- 4번째 `set_anomaly_map_cache()` 호출 시 `cached_at` 가장 오래된 항목 자동 삭제
- 호출 후 캐시에 보관된 experiment_id 수 ≤ 3

**검증 방법**:
```python
# 서버 측 단위 테스트
from utils.cache import set_anomaly_map_cache, get_anomaly_map_cache

for i in range(4):
    set_anomaly_map_cache(f"exp_{i}", {"data": i})

# exp_0 이 LRU 삭제됐는지 확인
assert get_anomaly_map_cache("exp_0") is None
assert get_anomaly_map_cache("exp_3") is not None
```

---

### G.3 학습 로그 버퍼 (UI)

**요구사항 (00절 §9 A-09)**: Explorer 학습 화면(`/training`) 로그 표시는 최신 100줄만 유지. 파일에는 전량 저장.

**설계 보장 메커니즘**:
- WebSocket(`/ws/training`)으로 log 이벤트 수신 → `trainingStore.logs` 배열 업데이트
- `trainingStore`의 `addLog()` 액션에서 최신 100줄 유지 처리

```typescript
// store/trainingStore.ts
interface TrainingStore {
  logs: string[];
  addLog: (line: string) => void;
}

// addLog 구현 (Zustand)
addLog: (line) => set((state) => {
  const next = [...state.logs, line];
  return { logs: next.length > 100 ? next.slice(-100) : next };
}),
```

**합격 기준**:
- `trainingStore.logs.length <= 100` at all times
- `./logs/{exp_id}.log` 파일 라인 수 = 실제 로그 이벤트 수 (전량, 100줄 제한 없음)

---

### G.4 비교 차트 최대 실험 수

**요구사항 (00절 §9 A-13)**: Explorer 실험 히스토리 화면(`/experiments`)의 비교 차트에서 최대 10개 실험 동시 비교.

**합격 기준**:
- 11개 이상 선택 시 경고 메시지 표시 ("최대 10개 실험까지 비교할 수 있습니다.")
- 11번째 이상 항목 체크박스 선택이 차단되거나 무시되어야 함

---

---

## B.5 검사 추론 응답 시간 (v2.0)

**요구사항 (00절 §6, NFR-INSP-01)**: 단일 이미지 검사 (추론 + 결과 표시) ≤ 3초.

**설계 보장 메커니즘**:
- `POST /api/inspection/run` → FastAPI 서버에서 추론 실행 → 결과 JSON 반환
- FastAPI 프로세스 수준에서 모델을 인메모리 캐시로 유지 (A-19) — 매 요청마다 모델 로드 없음
- `torch.no_grad()` 컨텍스트 내 단일 이미지 추론
- Vision의 `useManualInspection` hook이 응답 수신 즉시 `inspectionStore` 업데이트 → React 리렌더

**측정 방법**:
1. 모델 적용 완료 상태에서 수동 검사 버튼 클릭
2. 버튼 클릭 시각 ~ Vision 검사 결과 화면(판정 카드 + 이미지 패널) 표시까지 시간 측정 (Chrome DevTools Network 탭에서 `POST /api/inspection/run` 응답 시간 + React 리렌더 시간 합산)

**합격 기준**:
- 모델 캐시 히트 상태에서 수동 검사 요청 → 결과 표시 ≤ 3,000ms
- GPU (CUDA) 환경 기준; CPU 환경은 ≤ 10,000ms 허용

**위반 시 처리**:
- API 응답 > 3초: `image_size` 축소 검토, FastAPI 서버의 모델 캐시 적중 여부 확인
- React 리렌더 지연: `inspectionStore` 업데이트 후 리렌더 경로 최적화 검토

---

## B.6 자동 검사 타이밍 정확도 (v2.0)

**요구사항 (00절 §6, NFR-INSP-02)**: 자동 검사 실제 간격 = 설정값 3초 ± 0.5초.

**설계 보장 메커니즘**:
- FastAPI 서버의 `/ws/inspection/auto` WebSocket 핸들러에서 `asyncio.sleep(3)` + 추론 반복 루프
- 실제 간격 = 추론 시간 + 3초 (B.5 기준 추론 ≤ 3초이므로 최대 6초)
- 서버 측 간격 제어이므로 클라이언트(React) 폴링 오버헤드 없음

```python
# FastAPI WebSocket 핸들러 — 자동 검사 asyncio 루프 (개념)
@app.websocket("/ws/inspection/auto")
async def ws_auto_inspection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_single_inspection  # 동기 추론을 스레드풀에서 실행
            )
            await websocket.send_json(result)
            await asyncio.sleep(3)           # 3초 대기 후 다음 검사
    except WebSocketDisconnect:
        pass  # 클라이언트 연결 해제 시 루프 종료
```

**측정 방법**:
1. 자동 검사 시작 후 연속 10회 검사
2. 각 검사의 `inspected_at` 타임스탬프 간격 측정 (`GET /api/inspection/records` 응답 기준)

**합격 기준**:
- 각 검사 간격 = 추론 시간 + 3초, 타이밍 오차 ≤ 0.5초 (A-18)
- 연속 10회 평균 간격이 3초 ~ 6초 범위 내

**위반 시 처리**:
- 간격이 6초를 크게 초과: 추론 시간 단축 (image_size 축소, backbone 변경)
- 간격이 불규칙: asyncio 이벤트 루프 블로킹 여부 확인 (`run_in_executor` 미사용 시 블로킹 발생)

---

## B.7 불량 팝업 표시 지연 (v2.0)

**요구사항 (00절 §6, NFR-INSP-03)**: 불량 감지 → 팝업 표시 ≤ 0.5초.

**설계 보장 메커니즘**:
- FastAPI 서버가 불량 판정 즉시 `{"type": "result", "verdict": "defect", ...}` WS 메시지 전송
- Vision의 `useAutoInspection.ts` hook이 메시지 수신 즉시 `inspectionStore.showDefectPopup = true` 업데이트
- React는 Zustand 상태 변경 → 동기적 리렌더 → 팝업 DOM 표시 (별도 rerun 사이클 없음)
- 자동 검사 중지(`ws.close()`)는 팝업 표시 직후 호출

**측정 방법**:
1. 자동 검사 중 불량 이미지 노출
2. WS 메시지 수신 타임스탬프 (`performance.now()` in `useAutoInspection.ts`) ~ 팝업 DOM 표시까지 시간 측정 (Chrome DevTools Performance)

**합격 기준**:
- WS 메시지 수신 ~ 팝업 DOM 표시 ≤ 500ms
- 불량 감지 후 자동 검사 루프가 다음 검사를 시작하지 않음 (ws.close() 확인)

**위반 시 처리**:
- 팝업 렌더링 지연: `inspectionStore.showDefectPopup` 업데이트 경로 추적, Zustand selector 최적화 확인
- 자동 검사 중지 미동작: `ws.close()` 호출이 WS 메시지 핸들러 내부에서 즉시 실행되는지 확인

---

## B.8 WebSocket 이벤트-UI 반영 지연 (v2.0 신규)

**요구사항**: WebSocket 이벤트 발생 시점부터 React UI에 반영되기까지 지연이 허용 기준 이내여야 한다.

**대상 WebSocket**:
1. `/ws/training` — Explorer 학습 화면: 학습 진행 이벤트 (step, loss, stage, log)
2. `/ws/inspection/auto` — Vision 실시간 검사 화면: 자동 검사 결과 이벤트

**설계 보장 메커니즘**:
- Explorer: `useTrainingWs.ts` → WS `onmessage` 핸들러에서 `trainingStore` 즉시 업데이트 → React 동기 리렌더
- Vision: `useAutoInspection.ts` → WS `onmessage` 핸들러에서 `inspectionStore` 즉시 업데이트 → React 동기 리렌더
- 두 경우 모두 Zustand store는 메인 스레드에서 동기적으로 업데이트 — 별도 큐잉 없음

**합격 기준**:

| WebSocket | 이벤트 유형 | UI 반영 허용 지연 |
|-----------|------------|-----------------|
| `/ws/training` | 진행 메시지 (step, loss) | ≤ 1,000ms |
| `/ws/training` | 터미널 메시지 (`completed` / `stopped`) | ≤ 500ms |
| `/ws/inspection/auto` | 검사 결과 (양품/불량 판정) | ≤ 500ms |
| `/ws/inspection/auto` | 불량 팝업 트리거 | ≤ 200ms |

**측정 방법**:
```typescript
// useTrainingWs.ts 또는 useAutoInspection.ts 내 측정 포인트
ws.onmessage = (event) => {
  const receiveTime = performance.now();
  const data = JSON.parse(event.data);

  // store 업데이트
  trainingStore.getState().dispatch(data);

  // 리렌더 완료 시점 측정 (React flushSync 또는 requestAnimationFrame 후)
  requestAnimationFrame(() => {
    const renderTime = performance.now();
    console.debug(`WS→UI 지연: ${(renderTime - receiveTime).toFixed(1)}ms`);
  });
};
```

**위반 시 처리**:
- UI 반영 지연 > 기준: Zustand store 업데이트 → 리렌더 경로에서 불필요한 재계산 확인 (`useMemo`, `useCallback` 적용 여부)
- 서버 측 WS 전송 지연: `result_queue.get()` → `asyncio` 이벤트 루프 처리 시간 확인 (Python 측)
- 진행 메시지가 UI에 묶음으로 표시되는 경우: 서버에서 이벤트를 일괄 버퍼링하는지 확인 (즉시 전송 필요)

---

## H. NFR 합격 기준 요약표

| ID | 항목 | 합격 기준 | 문서 위치 |
|----|------|----------|----------|
| NFR-P-01 | UI 화면 전환 응답 (react-router) | ≤ 200ms (API 미포함), ≤ 1,000ms (API 포함) | B.1 |
| NFR-P-02 | EfficientAD 학습 시간 | ≤ 1,200s (g4dn.xlarge, 70k steps) | B.2 |
| NFR-P-03 | PatchCore 학습 시간 | ≤ 600s (g4dn.xlarge, coreset 10%) | B.3 |
| NFR-P-04 | 첫 Anomaly Map 빌드 | ≤ 30s (GPU, image_size=256) | B.4 |
| NFR-P-05 | Anomaly Map 캐시 히트 | ≤ 1s (빌드 job 즉시 완료 응답) | B.4 |
| **NFR-P-06** | **API 응답 (간단한 GET)** | **≤ 1,000ms (설정·실험 목록 조회)** | **B.1** |
| **NFR-P-07** | **WS 이벤트 → UI 반영 (/ws/training)** | **≤ 1,000ms (진행), ≤ 500ms (터미널)** | **B.8** |
| **NFR-P-08** | **WS 이벤트 → UI 반영 (/ws/inspection/auto)** | **≤ 500ms (결과), ≤ 200ms (불량 팝업)** | **B.8** |
| NFR-R-01 | 파일 원자성 | 부분 파일 없음 (강제 종료 후) | C.1 |
| NFR-R-02 | 중단 처리 | ≤ 10s 종료 + history 기록 | C.2 |
| NFR-R-03 | 서버 재시작 복구 | `/experiments` 화면에 전체 히스토리 표시 | C.3 |
| NFR-U-01 | 화면 가드 | 4개 조건 전부 (Zustand store 기반) | D.1 |
| NFR-U-02 | UI 언어 | 모든 라벨 한국어 | D.2 |
| NFR-U-03 | 디스크 경고 | API `disk_warning: true` + React 배너 | D.4 |
| NFR-D-01 | 폴더 구조 검증 | 실패 시 `/config` 화면 차단 | E.1 |
| NFR-D-02 | 비율 보존 | 오차 < 1픽셀 | E.2 |
| NFR-D-03 | 채널 보장 | 3채널 출력 | E.3 |
| NFR-F-01 | 재현성 | AUC 오차 ≤ 0.001 | F.1 |
| NFR-F-02 | 설정 기록 | configs.yaml 완전 복원 | F.2 |
| NFR-G-01 | GPU 메모리 해제 | 잔여 < 100MB | G.1 |
| NFR-G-02 | 캐시 상한 | 서버 측 LRU ≤ 3개 유지 | G.2 |
| NFR-G-03 | 로그 버퍼 | `trainingStore.logs.length` ≤ 100 | G.3 |
| **NFR-INSP-01** | **검사 추론 응답** | **≤ 3,000ms (GPU, 모델 캐시 히트)** | **B.5** |
| **NFR-INSP-02** | **자동 검사 타이밍 오차** | **≤ 0.5초 (asyncio 3초 간격 기준)** | **B.6** |
| **NFR-INSP-03** | **불량 팝업 표시 지연** | **≤ 500ms (WS 수신 → DOM 표시)** | **B.7** |

---

*다음 문서*: [12_Observability_and_Operations.md](./12_Observability_and_Operations.md)

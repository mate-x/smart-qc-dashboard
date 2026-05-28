# 11. Non-Functional Requirements

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md) §6 Global NFR
> **선행 문서**: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md)
> **버전**: v1.1
> **작성일**: 2026-05-09
> **수정일**: 2026-05-26 — v1.1: 비전검사 대시보드 NFR 추가 (B.5~B.7, H 요약표 갱신)
> **중요**: 00절 §6의 NFR 테이블을 기준으로, 각 항목에 대한 **측정 방법·합격 기준·위반 시 처리**를 확정한다. 구현 시 이 문서와 00절 §6을 함께 참조한다.

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

### B.1 UI 응답성

**요구사항 (00절 §6)**: 학습 중 UI 블로킹 없음. 탭 전환 응답 < 1초.

**설계 보장 메커니즘**:
- 학습은 `threading.Thread` + `queue.Queue` 로 백그라운드 분리 (00절 §9 A-02)
- 메인 스레드는 UI rerun 전담 — `st.rerun()` 1.0초 주기 (06절 §C.3 R-THREAD-05)
- 탭 전환은 Streamlit 자체 처리 — Python 코드 실행 없음

**측정 방법**:
1. 학습 실행 중 탭1~탭6을 순서대로 클릭
2. 각 탭 클릭 시 브라우저 렌더링 완료까지 시간 측정 (Chrome DevTools Network 탭)

**합격 기준**:
- 학습 실행 중 탭 전환 시 UI 응답 ≤ 1,000ms (JavaScript load 포함)
- `st.spinner()` 없이 메인 탭 콘텐츠가 렌더링되어야 함

**위반 시 처리**:
- 원인: tab에서 무거운 동기 연산 실행 여부 확인
- 수정: 해당 연산을 `@st.cache_data` 또는 백그라운드 스레드로 이동

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
- 탭6 추론 시 모델 재로드 필요 (05절 캐시 없는 경우)
- 최대 3개 anomaly_maps 캐시로 반복 추론 회피 (05절 §4.2)

**합격 기준**:
- 모델 로드 포함 첫 추론: ≤ 30초 (GPU 환경, image_size=256)
- 캐시 히트 시 재추론 없이 즉시 표시: ≤ 1초

**위반 시 처리**:
- 로드 시간 초과: `load_model_for_inference()` 내 불필요한 초기화 제거
- 모델 캐시 miss가 잦으면 `MAX_ANOMALY_MAP_CACHE` 상향 검토 (메모리 여유 확인 후)

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
- `.tmp` 파일이 잔존하는 경우: 앱 시작 시 `glob("**/*.tmp")` 로 탐색 후 삭제

---

### C.2 학습 중단 처리

**요구사항**: [학습 중지] 클릭 시 학습 스레드가 안전하게 종료되고 `status="중단"` 레코드가 history.json에 기록되어야 한다.

**설계 보장 메커니즘** (06절 §B.4 R-RACE-01~04):
- `stop_event.set()` → 학습 루프 내 주기적 `if stop_event.is_set(): break`
- "first terminal message wins" 규칙 — 중복 terminal 메시지 무시
- `stopped` 메시지 수신 후 `_build_experiment_record(status="중단")` → `save_history()`

**합격 기준**:
- [학습 중지] 클릭 후 ≤ 10초 내 학습 스레드 완전 종료 (마지막 step 완료 시간 포함)
- history.json에 `status == "중단"` 레코드 존재
- `model_path`, `configs_path`, `metrics` 모두 `null` (00절 §1.1 R-05)

**위반 시 처리**:
- 10초 초과 시: `stop_event` 체크 주기 확인 (매 step 또는 매 배치)
- 레코드 미기록 시: `_handle_terminal()` 내 `save_history()` 호출 경로 추적

---

### C.3 앱 재시작 후 복구

**요구사항**: Streamlit 앱 재시작 후 이전 실험 히스토리가 복원되어야 한다.

**설계 보장 메커니즘**:
- `history.json` 파일 기반 영속 (05절 §3.1)
- `app.py` 시작 시 `load_history()` → `session_state["experiments"]` 로딩
- `session_state`는 재시작 시 초기화되므로, 탭5 진입 시 `history.json` 재로드

**합격 기준**:
- 재시작 후 탭5에서 이전 실험 목록 전체 표시
- `completed` 상태 실험의 `model_path` 파일 존재 확인 (R-04 참조무결성)

**위반 시 처리**:
- `load_history()` 호출 위치 확인 (탭5 진입 시 반드시 호출)
- `model_path` 존재하지 않는 레코드: UI에서 "(파일 없음)" 표시 + 추론 비활성화

---

## D. Usability Requirements

### D.1 탭 진입 가드

**요구사항 (00절 §3.2)**: 선행 탭 미완료 시 다음 탭 진입 차단 + 안내 메시지.

**합격 기준**: 아래 조건 전부 충족

| 조건 | 기대 동작 |
|------|-----------|
| `dataset_path is None` + 탭2 진입 | `st.warning(MSG["NO_DATASET"])` + 탭 콘텐츠 미렌더링 |
| `preprocessing_config is None` + 탭3 진입 | `st.warning(MSG["NO_PREPROCESSING"])` + 탭 콘텐츠 미렌더링 |
| `model_config is None` + 탭4 진입 | `st.warning(MSG["NO_MODEL_CONFIG"])` + 탭 콘텐츠 미렌더링 |
| `experiments` 빈 dict + 탭5 진입 | `st.info(MSG["NO_EXPERIMENTS"])` |
| `selected_experiment_id is None` + 탭6 진입 | `st.info(MSG["NO_SELECTED_EXP"])` |

**측정 방법**: 각 조건을 수동으로 재현하여 안내 메시지 표시 및 콘텐츠 미렌더링 확인.

---

### D.2 UI 언어 및 표기 일관성

**요구사항 (00절 §8 R-UI-01)**: 모든 라벨·버튼·안내 메시지 한국어. 기술 용어는 한국어+영문 병기.

**합격 기준**:
- 모든 `st.button()`, `st.label_visibility`, `st.header()` 텍스트가 한국어
- 모델명 등 고유명사: `EfficientAD`, `PatchCore` (영문 유지)
- 수치 단위: 한국어 병기 (예: "학습 단계 수 (Train Steps)")

---

### D.3 숨김 처리

**요구사항 (00절 §8 R-UI-02)**: 비선택 파라미터 UI는 DOM 미렌더링 (`if` 분기). `disabled=True` 금지.

**합격 기준**:
- EfficientAD 선택 시 PatchCore 전용 파라미터 UI 요소가 DOM에 존재하지 않음
- `st.text_input(disabled=True)` 형태 코드가 없음

---

### D.4 디스크 공간 경고

**요구사항 (00절 §6)**: 모델 저장 전 여유 공간 < 500 MB 시 경고 메시지.

**합격 기준**:
```python
import shutil
usage = shutil.disk_usage("./models")
if usage.free < 500 * 1024 * 1024:
    st.warning(f"디스크 여유 공간이 {usage.free // (1024*1024):.0f} MB로 부족합니다. 500 MB 이상 확보 후 저장해 주세요.")
```
- 위 조건에서 `st.warning()` 가 렌더링되어야 함
- 경고 상태에서도 사용자가 강제 저장 가능해야 함 (차단 아닌 경고)

---

## E. Data Integrity Requirements

### E.1 MVTec AD 폴더 구조 검증

**요구사항 (00절 §6)**: 폴더 구조 검증 실패 시 탭4 진입 완전 차단.

**검증 항목** (01절 §B.2 기준):

| 조건 | 오류 코드 |
|------|----------|
| `{dataset_root}/train/good/` 미존재 | `ERR_INVALID_FOLDER_STRUCTURE` |
| `train/good/` 내 지원 포맷 이미지 0개 | `ERR_NO_VALID_IMAGES` |
| `{dataset_root}/test/` 미존재 | `ERR_INVALID_FOLDER_STRUCTURE` |

**합격 기준**:
- 위 조건 중 하나라도 해당 시 `session_state["dataset_path"] = None` 유지
- 탭4 진입 가드가 `dataset_path is None` 체크로 차단

**위반 판단**: 검증 실패 경로에서 탭4가 렌더링되면 즉시 버그 처리.

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
| R-01 | `selected_experiment_id` 조회 시 `experiments` dict에 키 존재 | 탭6 진입 시 |
| R-02 | `model_config.image_size == preprocessing_config.image_size` | 탭3 저장 시 자동 동기화 |
| R-04 | `status=="completed"` 이고 `model_path` NOT NULL인 경우 파일 존재 | 탭5 로딩 시 |
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

**요구사항**: 탭6 추론 완료 후 GPU 메모리를 즉시 해제해야 한다.

**구현 명세** (07절 §C.3):
```python
# tab6 추론 파이프라인 — 추론 완료 직후
del model
torch.cuda.empty_cache()
```

**합격 기준**:
- 추론 전 VRAM 사용량 (A)과 추론 후 VRAM 사용량 (B): `B - A < 100 MB` (잔여 캐시 허용)
- `nvidia-smi` 또는 `torch.cuda.memory_allocated()` 로 측정

---

### G.2 Anomaly Map 캐시 상한

**요구사항 (05절 §4.2)**: `MAX_ANOMALY_MAP_CACHE = 3` — 최대 3개 실험의 anomaly_maps만 session_state에 보관.

**합격 기준**:
- 4번째 `set_anomaly_map_cache()` 호출 시 `cached_at` 가장 오래된 항목 자동 삭제
- 호출 후 `len([k for k in st.session_state if k.startswith("_anomaly_maps_")]) <= 3`

---

### G.3 학습 로그 버퍼 (UI)

**요구사항 (00절 §9 A-09)**: UI 로그 텍스트박스는 최신 100줄만 유지. 파일에는 전량 저장.

**구현 명세**:
```python
# tab4 로그 표시 — UI 버퍼
log_lines: list[str] = st.session_state.get("log_buffer", [])
log_lines.append(new_line)
if len(log_lines) > 100:
    log_lines = log_lines[-100:]  # 최신 100줄만 유지
st.session_state["log_buffer"] = log_lines
st.text_area("학습 로그", value="\n".join(log_lines), height=300)
```

**합격 기준**:
- `len(st.session_state["log_buffer"]) <= 100` at all times
- `./logs/{exp_id}.log` 파일 라인 수 = 실제 로그 이벤트 수 (전량)

---

### G.4 비교 차트 최대 실험 수

**요구사항 (00절 §9 A-13)**: 탭5 비교 차트에서 최대 10개 실험 동시 비교.

**합격 기준**:
- 11개 이상 선택 시 `st.warning("최대 10개 실험까지 비교할 수 있습니다.")`
- 11번째 이상 항목 선택이 차단되거나 무시되어야 함

---

---

## B.5 검사 추론 응답 시간 (v1.1)

**요구사항 (00절 §6, NFR-INSP-01)**: 단일 이미지 검사 (추론 + 결과 표시) ≤ 3초.

**설계 보장 메커니즘**:
- `st.cache_resource`로 모델을 캐시하여 매 검사마다 모델 로드 회피 (A-19)
- `torch.no_grad()` 컨텍스트 내 단일 이미지 추론 (B.10.4 패턴)
- anomaly_score = `np.max(anomaly_map)` — 단순 최댓값 연산

**측정 방법**:
1. 모델 최초 로드 후 수동 검사 버튼 클릭
2. 버튼 클릭 시각 ~ 3열 결과 화면 표시까지 시간 측정 (브라우저 DevTools 또는 로그 타임스탬프)

**합격 기준**:
- 모델 캐시 히트 상태에서 검사 버튼 클릭 → 결과 표시 ≤ 3,000ms
- GPU (CUDA) 환경 기준; CPU 환경은 ≤ 10,000ms 허용

**위반 시 처리**:
- 추론 시간 > 3초: `image_size` 축소 검토, 모델 캐시 확인
- Streamlit 렌더링 지연: `st.empty()` 컨테이너 최소화

---

## B.6 자동 검사 타이밍 정확도 (v1.1)

**요구사항 (00절 §6, NFR-INSP-02)**: 자동 검사 실제 간격 = 설정값 3초 ± 0.5초.

**설계 보장 메커니즘**:
- `time.sleep(3)` + `st.rerun()` 패턴 (07절 §13.1)
- 실제 간격 = 3초 + 추론 시간 (≤3초) ≤ 6초 최대

**측정 방법**:
1. 자동 검사 시작 후 연속 10회 검사
2. 각 검사의 `inspected_at` 타임스탬프 간격 측정 (insp_records 기준)

**합격 기준**:
- 각 검사 간격 = 3초 + 추론 시간, 타이밍 오차 ≤ 0.5초 (A-18)
- 연속 10회 평균 간격이 3초 ~ 6초 범위 내

**위반 시 처리**:
- 간격이 6초를 크게 초과: 추론 시간 단축 (image_size 축소, backbone 변경)
- 간격이 불규칙: Streamlit rerun 오버헤드 측정 후 sleep 조정

---

## B.7 불량 팝업 표시 지연 (v1.1)

**요구사항 (00절 §6, NFR-INSP-03)**: 불량 감지 → 팝업 표시 ≤ 0.5초.

**설계 보장 메커니즘**:
- 불량 감지 즉시 `insp_defect_popup = True` + `st.rerun()` (07절 §13.2)
- 다음 rerun에서 팝업 렌더링 최우선 처리 (`insp_defect_popup` 체크를 render() 상단에 배치)

**측정 방법**:
1. 자동 검사 중 불량 이미지 노출
2. 검사 완료 타임스탬프 ~ 팝업 화면 표시까지 시간 측정

**합격 기준**:
- 불량 감지 후 `st.rerun()` 실행 ~ 팝업 DOM 표시 ≤ 500ms

**위반 시 처리**:
- 팝업 렌더링 우선순위 확인: `insp_defect_popup` 체크가 render() 최상단에 있는지 검증

---

## H. NFR 합격 기준 요약표

| ID | 항목 | 합격 기준 | 문서 위치 |
|----|------|----------|----------|
| NFR-P-01 | 탭 전환 응답 | ≤ 1,000ms (학습 중 포함) | B.1 |
| NFR-P-02 | EfficientAD 학습 시간 | ≤ 1,200s (g4dn.xlarge, 70k steps) | B.2 |
| NFR-P-03 | PatchCore 학습 시간 | ≤ 600s (g4dn.xlarge, coreset 10%) | B.3 |
| NFR-P-04 | 첫 추론 응답 | ≤ 30s (GPU, image_size=256) | B.4 |
| NFR-P-05 | 캐시 히트 추론 | ≤ 1s | B.4 |
| NFR-R-01 | 파일 원자성 | 부분 파일 없음 (강제 종료 후) | C.1 |
| NFR-R-02 | 중단 처리 | ≤ 10s 종료 + history 기록 | C.2 |
| NFR-R-03 | 재시작 복구 | 전체 히스토리 탭5 표시 | C.3 |
| NFR-U-01 | 탭 진입 가드 | 5개 조건 전부 | D.1 |
| NFR-U-02 | UI 언어 | 모든 라벨 한국어 | D.2 |
| NFR-U-03 | 디스크 경고 | < 500MB 시 표시 | D.4 |
| NFR-D-01 | 폴더 구조 검증 | 실패 시 탭4 차단 | E.1 |
| NFR-D-02 | 비율 보존 | 오차 < 1픽셀 | E.2 |
| NFR-D-03 | 채널 보장 | 3채널 출력 | E.3 |
| NFR-F-01 | 재현성 | AUC 오차 ≤ 0.001 | F.1 |
| NFR-F-02 | 설정 기록 | configs.yaml 완전 복원 | F.2 |
| NFR-G-01 | GPU 메모리 해제 | 잔여 < 100MB | G.1 |
| NFR-G-02 | 캐시 상한 | ≤ 3개 유지 | G.2 |
| NFR-G-03 | 로그 버퍼 | ≤ 100줄 (UI) | G.3 |
| **NFR-INSP-01** | **검사 추론 응답** | **≤ 3,000ms (GPU 캐시 히트)** | **B.5** |
| **NFR-INSP-02** | **자동 검사 타이밍 오차** | **≤ 0.5초 (3초 간격 기준)** | **B.6** |
| **NFR-INSP-03** | **불량 팝업 표시 지연** | **≤ 500ms** | **B.7** |

---

*다음 문서*: [12_Observability_and_Operations.md](./12_Observability_and_Operations.md)

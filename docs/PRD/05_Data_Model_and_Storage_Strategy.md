# 05. Data Model and Storage Strategy

> **참조 문서**: `00_Global_Context_Document.md` §1 (Core Data Model), §3 (File I/O 계약), §8 (결정 규칙)
> **버전**: v1.1
> **작성일**: 2026-05-09
> **최종수정**: 2026-05-26
> **목적**: 이 시스템의 모든 영속 데이터 및 임시 캐시에 대한 저장 전략, 읽기/쓰기 계약, 원자성 보장, rollback 정책을 구현 가능한 수준으로 명세한다. 04_System_Architecture.md의 파일시스템 레이어 설계를 구체화하고, 08_AI_ML_Integration.md에서 참조할 스토리지 계약을 선행 정의한다.

---

## 목차

1. [설계 원칙 및 ADR](#1-설계-원칙-및-adr)
2. [파일시스템 레이아웃 전체 명세](#2-파일시스템-레이아웃-전체-명세)
3. [history.json 상세 명세](#3-historyjson-상세-명세)
4. [configs.yaml 상세 명세](#4-configsyaml-상세-명세)
5. [모델 저장 디렉토리 명세](#5-모델-저장-디렉토리-명세)
6. [모델 저장 3단계 원자성 프로토콜](#6-모델-저장-3단계-원자성-프로토콜)
7. [실험 삭제 프로토콜](#7-실험-삭제-프로토콜)
8. [세션 캐시 정책 (anomaly_maps)](#8-세션-캐시-정책-anomaly_maps)
9. [EfficientAD ImageNet Penalty 데이터](#9-efficientad-imagenet-penalty-데이터)
10. [로그 파일 관리](#10-로그-파일-관리)
11. [디스크 용량 모니터링](#11-디스크-용량-모니터링)
12. [구현 체크리스트](#12-구현-체크리스트)

---

## 1. 설계 원칙 및 ADR

### ADR-DS-01: 파일시스템 기반 영속화 + MySQL 인프라 병행

| 항목 | 내용 |
|------|------|
| **결정** | 모델 가중치(`.pth`), 학습 로그(`.log`), 설정(`configs.yaml`), 실험 히스토리(`history.json`)는 파일시스템에 저장한다. MySQL 8.0은 인프라 레이어에 포함하며, 향후 구조화 데이터 저장에 활용한다. |
| **근거** | MVP 단일 사용자 환경(A-01)에서 파일시스템으로 충분하나, MySQL을 인프라에 포함해 확장성을 확보한다. |
| **트레이드오프** | 동시성 제어 불가 → 단일 사용자 가정으로 수용. 전체 히스토리 로드 시 파일 전체 읽기 필요 → 최대 실험 수 제한 없음(실험 수백 개 이하에서 성능 문제 없음). |

### ADR-DS-02: 모든 파일 쓰기는 원자적으로 수행

| 항목 | 내용 |
|------|------|
| **결정** | 단일 파일 쓰기는 `.tmp` → `rename` 패턴. 다단계 저장(모델 + 설정 + 히스토리)은 §6의 3단계 프로토콜을 따른다. |
| **근거** | 쓰기 도중 프로세스 종료 시 부분 쓰기로 인한 데이터 손상 방지. `os.replace()` (POSIX rename)는 원자적 보장. |

### ADR-DS-03: 세션 캐시와 영속 스토리지의 엄격한 경계

| 항목 | 내용 |
|------|------|
| **결정** | `st.session_state`는 임시 캐시 전용. 영속이 필요한 데이터는 반드시 파일에 저장한다. `session_state`에만 있는 데이터는 페이지 새로고침 시 소실될 수 있음을 전제로 설계한다. |
| **근거** | Streamlit 세션 모델 제약 (A-01). 탭 전환 시 session_state는 유지되지만, 브라우저 새로고침 시 초기화됨. |

---

## 2. 파일시스템 레이아웃 전체 명세

```
{WORKDIR}/                          # Docker: /app, 로컬: 프로젝트 루트
│
├── configs.yaml                    # [읽기/쓰기] 탭2, 탭3 공유 설정 파일
│                                   # 탭4 학습 시작 시점의 스냅샷 기준
│
├── experiments/
│   └── history.json                # [읽기/쓰기] 실험 레코드 배열
│                                   # 초기 미존재 → load_history()가 [] 반환
│
├── models/
│   └── {experiment_id}/            # 실험별 디렉토리 (학습 완료 시 생성)
│       ├── model_state_dict.pth    # PyTorch state_dict
│       └── configs.yaml            # 학습 시점 설정 스냅샷 (루트 configs.yaml 복사본)
│
├── logs/
│   └── {experiment_id}.log         # 학습 로그 (학습 시작 시 생성, append 전용)
│
├── dataset/
│   └── {사용자_데이터셋}/           # [읽기 전용] Docker 볼륨 마운트 대상
│       ├── train/good/
│       ├── test/{class}/
│       └── ground_truth/{class}/
│
└── dataset/
    └── imagenet_penalty/           # [읽기 전용] EfficientAD penalty 데이터 (§9 참조)
        └── *.jpg / *.png
```

### 경로 접근 규칙

| 규칙 | 내용 |
|------|------|
| **R-PATH-01** (00_Global §8) | 모든 경로는 `pathlib.Path` 사용. 문자열 연산(`os.path.join` 포함) 금지. |
| **절대 경로 vs 상대 경로** | 파일 저장 시 `Path("./experiments/history.json")` 형식의 상대 경로 사용. Docker WORKDIR(`/app`)에서 자동 해석됨. |
| **WORKDIR 환경변수** | `WORKDIR` 환경변수가 설정된 경우 `Path(os.environ.get("WORKDIR", "."))` 를 베이스로 사용. 미설정 시 `.` (현재 디렉토리). |

---

## 3. history.json 상세 명세

### 3.1 파일 구조

```json
[
  {
    "experiment_id": "efficientad_20260508_140023_7f3a",
    "name": "EfficientAD medium 기본 실험",
    "status": "completed",
    "created_at": "2026-05-08T14:00:23+09:00",
    "model_type": "efficientad",
    "preprocessing_method": "clahe",
    "preprocessing_params": { "clip_limit": 2.0 },
    "model_params": { ... },
    "metrics": { ... },
    "threshold_method": "percentile",
    "threshold_value": 95.0,
    "model_path": "./models/efficientad_20260508_140023_7f3a/",
    "configs_path": "./models/efficientad_20260508_140023_7f3a/configs.yaml",
    "duration_seconds": 1177,
    "dataset_path": "./dataset/screw",
    "image_size": 256
  }
]
```

> 전체 필드 명세는 `00_Global_Context_Document.md §1.1` 참조.

### 3.2 읽기 계약

```python
# utils/storage.py

def load_history() -> list[dict]:
    """
    반환: 실험 레코드 리스트. 파일 미존재 시 빈 리스트.
    예외: JSON 파싱 실패 시 빈 리스트 반환 + WARNING 로그 (예외 전파 금지).
    """
    path = Path("./experiments/history.json")
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        _log_warning("history_load_failed", {"path": str(path)})
        return []
```

**중요**: `load_history()`는 절대 예외를 전파하지 않는다. UI 렌더링 중단을 방지하기 위함.

### 3.3 쓰기 계약 (원자적 Append)

```python
def append_experiment(record: dict) -> None:
    """
    기존 히스토리 로드 → 레코드 append → 원자적 쓰기.
    실패 시 ERR_HISTORY_WRITE_FAILED 로깅 후 예외 재발생 (호출자가 처리).
    """
    path = Path("./experiments/history.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    records = load_history()  # 기존 데이터 보존
    records.append(record)

    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tmp.replace(path)   # 원자적 rename (R-ATOMIC-01)
    except IOError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)  # 실패한 tmp 파일 정리
        raise RuntimeError(f"ERR_HISTORY_WRITE_FAILED: {e}") from e
```

### 3.4 삭제 계약

```python
def delete_experiment_from_history(experiment_id: str) -> bool:
    """
    반환: True(삭제 성공) | False(해당 ID 없음).
    원자적 쓰기 동일하게 적용.
    """
    records = load_history()
    filtered = [r for r in records if r["experiment_id"] != experiment_id]
    if len(filtered) == len(records):
        return False  # 해당 ID 없음

    path = Path("./experiments/history.json")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return True
```

### 3.5 history.json 무결성 규칙

| 규칙 | 설명 |
|------|------|
| **중복 ID 금지** | `append_experiment()` 호출 전 동일 `experiment_id` 존재 여부 확인. 중복 시 덮어쓰지 않고 예외 발생. |
| **status="중단" 레코드** | `model_path`, `configs_path`, `metrics`는 반드시 `null`. 쓰기 시 강제 설정. |
| **배열 타입 보장** | `load_history()` 반환값이 list가 아닌 경우(파일 손상 등) 빈 리스트로 처리. |

---

## 4. configs.yaml 상세 명세

### 4.1 파일 위치 및 역할

| 파일 | 위치 | 역할 |
|------|------|------|
| **공유 설정 파일** | `./configs.yaml` | 탭2(전처리), 탭3(모델) 파라미터 편집용. 탭4 학습 시작 시 이 파일을 읽어 실험 스냅샷 생성. |
| **실험 스냅샷** | `./models/{exp_id}/configs.yaml` | 학습 시점의 설정 고정 사본. 탭6 모델 재로드 시 사용. 이후 변경 불가. |

### 4.2 읽기 계약

```python
# utils/config_manager.py

def load_config(path: str | Path = "./configs.yaml") -> dict:
    """
    반환: 설정 dict. 파일 미존재 시 빈 dict.
    예외: YAML 파싱 실패 시 빈 dict 반환 + WARNING 로그 (예외 전파 금지).
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        _log_warning("config_load_failed", {"path": str(p)})
        return {}
```

### 4.3 섹션 업데이트 계약

```python
def save_config_section(
    section: str,
    data: dict,
    path: str | Path = "./configs.yaml"
) -> None:
    """
    기존 파일의 다른 섹션을 보존하면서 지정 섹션만 업데이트.
    원자적 쓰기 적용 (tmpfile → rename).

    Args:
        section: "experiment" | "preprocessing" | "model"
        data:    해당 섹션 dict
        path:    대상 파일 경로 (기본값: 루트 configs.yaml)
                 실험 스냅샷 저장 시 "./models/{exp_id}/configs.yaml" 전달

    Raises:
        RuntimeError: 쓰기 실패 시
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(p)
    config[section] = data

    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        tmp.replace(p)
    except IOError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ERR_CONFIG_WRITE_FAILED: {e}") from e
```

**핵심 계약**: `path` 파라미터를 통해 루트 `./configs.yaml`과 실험별 스냅샷 `./models/{exp_id}/configs.yaml` 양쪽에 동일 함수 재사용. 탭별 Write 권한은 아래 표 참조.

### 4.4 탭별 configs.yaml 접근 권한

| 탭 | section | 접근 종류 | 대상 파일 |
|----|---------|-----------|-----------|
| 탭2 (전처리 설정) | `"preprocessing"` | Write | `./configs.yaml` |
| 탭3 (모델 파라미터) | `"model"` | Write | `./configs.yaml` |
| 탭4 (학습 시작) | `"experiment"` + 전체 | Read + 스냅샷 Write | `./configs.yaml` (Read), `./models/{exp_id}/configs.yaml` (Write) |
| 탭6 (이상 영역 시각화) | `"preprocessing"`, `"model"` | Read 전용 | `./models/{exp_id}/configs.yaml` |

---

## 5. 모델 저장 디렉토리 명세

### 5.1 디렉토리 구조

```
./models/
└── {experiment_id}/                # 예: efficientad_20260508_140023_7f3a/
    ├── model_state_dict.pth        # PyTorch state_dict
    └── configs.yaml                # 학습 시점 설정 스냅샷
```

### 5.2 각 파일 명세

#### model_state_dict.pth

| 항목 | 내용 |
|------|------|
| **저장 방식** | `torch.save(model.state_dict(), path)` |
| **로드 방식** | `model.load_state_dict(torch.load(path, map_location=device))` |
| **EfficientAD 크기** | ≈ 200~400 MB (model_size에 따라 다름) |
| **PatchCore 크기** | ≈ 400~800 MB (backbone 및 coreset 크기에 따라 다름) |

> `torch.save(model, path)` 형식 금지. `state_dict()`만 저장. 모델 클래스 import 없이도 로드 가능하도록 보장.

#### configs.yaml (실험 스냅샷)

| 항목 | 내용 |
|------|------|
| **저장 시점** | 학습 완료 후, `model_state_dict.pth` 저장 성공 직후 |
| **저장 내용** | 루트 `./configs.yaml` 전체 내용 + `experiment.name`, `experiment.created_at` 추가 |
| **불변 원칙** | 저장 후 수정 금지. 탭6에서 재로드 시 이 파일의 파라미터를 사용하여 모델 재현. |

### 5.3 디렉토리 생성 규칙

```python
def prepare_model_dir(experiment_id: str) -> Path:
    """
    반환: 생성된 모델 디렉토리 Path.
    이미 존재하면 예외 발생 (중복 experiment_id 방지).
    """
    model_dir = Path(f"./models/{experiment_id}")
    if model_dir.exists():
        raise RuntimeError(f"모델 디렉토리가 이미 존재합니다: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=False)
    return model_dir
```

---

## 6. 모델 저장 3단계 원자성 프로토콜

> 이 절은 08_AI_ML_Integration.md §8 (학습 완료 후 메인 스레드 처리)와 직접 연동된다.
> **실행 주체**: 메인 스레드 (백그라운드 스레드가 아님 — ADR-SA-04 참조).

### 6.1 3단계 순서

```
[단계 1] model_state_dict.pth 저장
    → ./models/{exp_id}/model_state_dict.pth
    → 실패 시: 디렉토리 삭제 → history 미기록 → ERR_MODEL_SAVE_FAILED

[단계 2] configs.yaml 스냅샷 저장
    → ./models/{exp_id}/configs.yaml
    → 실패 시: model_state_dict.pth + 디렉토리 삭제 → history 미기록 → ERR_MODEL_SAVE_FAILED

[단계 3] history.json append
    → experiment record (status="completed", model_path 설정)
    → 실패 시: 2단계까지 성공한 파일 보존 (孤立 디렉토리) + ERR_HISTORY_WRITE_FAILED
             → UI에 경고: "모델 파일은 저장되었으나 히스토리 기록에 실패했습니다.
                          {model_path}에서 수동으로 확인하세요."
```

**3단계의 부분 실패 처리가 다른 이유**: `model_state_dict.pth`는 수백 MB의 가치 있는 데이터이므로, 히스토리 기록만 실패한 경우 파일 삭제보다 보존을 우선한다.

### 6.2 전체 구현 코드

```python
# utils/storage.py

def save_completed_experiment(
    experiment_id: str,
    model,                  # EfficientAd | Patchcore 인스턴스
    experiment_record: dict # status="completed" 레코드 (model_path, configs_path 포함)
) -> None:
    """
    학습 완료 후 메인 스레드에서 호출하는 3단계 원자성 저장.

    Raises:
        RuntimeError: 단계 1 또는 단계 2 실패 시 (파일 정리 후 raise)
        RuntimeError: 단계 3 실패 시 (파일 보존 후 raise — 호출자가 UI 경고 표시)
    """
    model_dir = prepare_model_dir(experiment_id)
    pth_path = model_dir / "model_state_dict.pth"
    cfg_path = model_dir / "configs.yaml"

    # --- 단계 1: model_state_dict.pth 저장 ---
    try:
        torch.save(model.state_dict(), pth_path)
    except Exception as e:
        _cleanup_dir(model_dir)
        raise RuntimeError(f"ERR_MODEL_SAVE_FAILED (단계1): {e}") from e

    # --- 단계 2: configs.yaml 스냅샷 저장 ---
    try:
        root_config = load_config("./configs.yaml")
        root_config.setdefault("experiment", {})
        root_config["experiment"]["name"] = experiment_record["name"]
        root_config["experiment"]["created_at"] = experiment_record["created_at"]
        save_config_section("experiment", root_config["experiment"], cfg_path)
    except Exception as e:
        _cleanup_dir(model_dir)
        raise RuntimeError(f"ERR_MODEL_SAVE_FAILED (단계2): {e}") from e

    # --- 단계 3: history.json append ---
    # 이 시점부터 모델 파일 삭제 안 함
    experiment_record["model_path"] = str(model_dir) + "/"
    experiment_record["configs_path"] = str(cfg_path)
    try:
        append_experiment(experiment_record)
    except RuntimeError as e:
        raise RuntimeError(
            f"ERR_HISTORY_WRITE_FAILED: 모델 저장 성공, 히스토리 기록 실패. "
            f"model_path={model_dir}"
        ) from e


def _cleanup_dir(path: Path) -> None:
    """실패 시 디렉토리 전체 삭제."""
    import shutil
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
```

### 6.3 호출자 (tab4_training.py) 처리 패턴

```python
# tabs/tab4_training.py — 메인 스레드의 완료 처리 부분

try:
    save_completed_experiment(exp_id, model, record)
    st.session_state["experiments"][exp_id] = record
    st.success(f"학습이 완료되었습니다. 소요 시간: {mins}분 {secs}초")

except RuntimeError as e:
    error_msg = str(e)
    if "ERR_HISTORY_WRITE_FAILED" in error_msg:
        st.warning(f"모델 파일은 저장되었으나 히스토리 기록에 실패했습니다. {error_msg}")
    else:
        st.error(f"모델 저장에 실패했습니다. 디스크 공간을 확인해 주세요. {error_msg}")
```

---

## 7. 실험 삭제 프로토콜

> 탭5에서 [실험 삭제] 버튼 클릭 시 실행.

### 7.1 삭제 순서

```
[단계 1] history.json에서 레코드 제거 (delete_experiment_from_history)
[단계 2] ./models/{exp_id}/ 디렉토리 삭제 (model_path가 NOT NULL인 경우만)
[단계 3] ./logs/{exp_id}.log 파일 삭제 (존재하는 경우만)
[단계 4] session_state.experiments 에서 제거
[단계 5] selected_experiment_id가 삭제된 ID이면 None으로 초기화
```

### 7.2 삭제 구현

```python
# utils/storage.py

def delete_experiment(experiment_id: str, model_path: str | None = None) -> None:
    """
    실험 레코드 및 관련 파일 삭제.
    model_path: experiment_record["model_path"] 값. None이면 파일 삭제 생략.
    """
    import shutil

    # 단계 1: history.json
    delete_experiment_from_history(experiment_id)

    # 단계 2: 모델 디렉토리
    if model_path:
        model_dir = Path(model_path)
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    # 단계 3: 로그 파일
    log_path = Path(f"./logs/{experiment_id}.log")
    log_path.unlink(missing_ok=True)
```

### 7.3 삭제 시 참조 무결성

| 조건 | 처리 |
|------|------|
| 삭제 대상이 `selected_experiment_id`와 동일 | 탭5에서 삭제 후 `st.session_state["selected_experiment_id"] = None` |
| 삭제 대상의 anomaly_maps 캐시가 session_state에 존재 | `st.session_state.pop(f"_anomaly_maps_{exp_id}", None)` |
| `status == "중단"` 레코드 삭제 | model_path가 None이므로 파일 삭제 단계 건너뜀 |

---

## 8. 세션 캐시 정책 (anomaly_maps)

### 8.1 캐시 키 및 저장 구조

```python
# session_state 캐시 키 형식
cache_key = f"_anomaly_maps_{experiment_id}"

# 저장 값 구조
st.session_state[cache_key] = {
    "anomaly_maps": np.ndarray,   # shape: (N, H, W), dtype: float32
    "image_paths":  list[str],    # 테스트 이미지 절대 경로 목록
    "cached_at":    float,        # time.time() 저장 시점
}
```

**캐시 키 앞에 `_` prefix를 붙이는 이유**: `session_state` 초기화 스키마(§3.1)에 정의된 공식 키와 구분하기 위함.

### 8.2 크기 추정

| 조건 | 크기 |
|------|------|
| 테스트 이미지 100장, image_size=256 | `100 × 256 × 256 × 4 bytes ≈ 25 MB` |
| 테스트 이미지 500장, image_size=256 | `≈ 125 MB` |
| 테스트 이미지 100장, image_size=512 | `≈ 100 MB` |

### 8.3 캐시 수명(TTL) 및 무효화 조건

| 이벤트 | 처리 |
|--------|------|
| **탭6 진입** | 캐시 키 존재 확인. 있으면 재사용, 없으면 모델 로드 + 전체 추론 후 저장. |
| **다른 실험 선택** (`selected_experiment_id` 변경) | 이전 실험 캐시 삭제하지 않음. 새 키로 별도 캐시. |
| **실험 삭제** | 해당 `_anomaly_maps_{exp_id}` 키 즉시 삭제 (§7.3 참조). |
| **브라우저 새로고침** | session_state 전체 초기화 → 캐시 자동 소멸. |
| **캐시 동시 보유 상한** | **최대 3개** 실험 anomaly_map 캐시를 session_state에 유지. 초과 시 가장 오래된 `cached_at` 캐시 삭제. |

### 8.4 최대 3개 캐시 eviction 구현

```python
# utils/cache_manager.py

MAX_ANOMALY_MAP_CACHE = 3

def set_anomaly_map_cache(experiment_id: str, data: dict) -> None:
    """
    anomaly_map 캐시 저장. 3개 초과 시 가장 오래된 캐시 자동 제거.
    """
    cache_keys = [
        k for k in st.session_state
        if k.startswith("_anomaly_maps_")
    ]

    if len(cache_keys) >= MAX_ANOMALY_MAP_CACHE:
        oldest_key = min(
            cache_keys,
            key=lambda k: st.session_state[k].get("cached_at", 0)
        )
        del st.session_state[oldest_key]

    st.session_state[f"_anomaly_maps_{experiment_id}"] = {
        **data,
        "cached_at": time.time()
    }


def get_anomaly_map_cache(experiment_id: str) -> dict | None:
    """캐시 반환. 없으면 None."""
    return st.session_state.get(f"_anomaly_maps_{experiment_id}")
```

---

## 9. EfficientAD ImageNet Penalty 데이터

### 9.1 배경

EfficientAD는 학습 중 "ImageNet penalty" 손실항을 계산하기 위해 정상 이미지 외에 ImageNet 분포의 랜덤 이미지 배치를 추가로 사용한다 (Student-Teacher 과적합 방지). 이 데이터는 모델 학습 전 미리 준비되어 있어야 한다.

### 9.2 경로 규칙

```
{WORKDIR}/dataset/imagenet_penalty/
├── n01440764_0.jpg
├── n01440764_1.jpg
...
└── (최소 1,000장 이상 권장)
```

**고정 경로**: `./dataset/imagenet_penalty/` — 환경변수나 configs.yaml 설정 없이 코드에서 직접 참조.

```python
# utils/model_factory.py 또는 training_worker.py 내부

IMAGENET_PENALTY_DIR = Path("./dataset/imagenet_penalty")
```

### 9.3 존재 여부 검증

```python
def validate_imagenet_penalty_dir() -> tuple[bool, int]:
    """
    반환: (존재 여부, 이미지 수)
    학습 시작 전 탭4에서 호출하여 EfficientAD 선택 시 사전 검증.
    """
    d = IMAGENET_PENALTY_DIR
    if not d.exists():
        return False, 0
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    count = sum(1 for f in d.iterdir() if f.suffix.lower() in supported)
    return count > 0, count
```

### 9.4 탭4 검증 로직

```python
# tabs/tab4_training.py — [학습 시작] 버튼 클릭 핸들러

if model_config["model_type"] == "efficientad":
    ok, count = validate_imagenet_penalty_dir()
    if not ok:
        st.error(
            "EfficientAD 학습에 필요한 ImageNet penalty 데이터가 없습니다. "
            f"`{IMAGENET_PENALTY_DIR}` 경로에 이미지를 추가해 주세요."
        )
        st.stop()
    elif count < 1000:
        st.warning(
            f"ImageNet penalty 이미지가 {count}장입니다. "
            "1,000장 이상 권장합니다."
        )
```

### 9.5 Docker 마운트 방법

```bash
# docker-compose.yml 또는 docker run 시
docker run \
  -v /host/imagenet_penalty:/app/dataset/imagenet_penalty:ro \
  ...
```

> 09_Infrastructure_and_Cloud.md에서 docker-compose.yml 볼륨 섹션에 이 항목이 포함되어야 한다.

### 9.6 ImageNet penalty 데이터 준비 방법 (운영 가이드)

| 방법 | 설명 |
|------|------|
| **ImageNet 서브셋** | ImageNet-1k validation set 중 임의 1,000장 이상 추출 |
| **대체 데이터** | 학습 데이터셋과 무관한 자연 이미지(COCO, Open Images 등) |
| **최소 요건** | 1장 이상(기술적 동작 가능), 1,000장 이상 권장(학습 품질) |
| **금지** | 학습 데이터셋(`./dataset/{screw 등}/`)과 동일 이미지 사용 금지 |

---

## 10. 로그 파일 관리

### 10.1 파일 경로 및 생성 시점

```
./logs/{experiment_id}.log
```

- 학습 스레드 시작 직후 **즉시 생성** (append 모드).
- 학습 중단(`status == "중단"`) 또는 완료(`status == "completed"`) 후에도 **삭제하지 않는다**.
- 실험 삭제 시 함께 삭제 (§7.2 단계 3).

### 10.2 로그 포맷

`00_Global_Context_Document.md §7.3` 형식 그대로 적용:

```
{ISO8601_KST}\t[시작] 실험: {experiment_id}
{ISO8601_KST}\t[Step {step}/{total}] Loss: {loss:.4f} | 경과: {elapsed:.1f}s
{ISO8601_KST}\t[완료] 총 소요: {duration}s | AUC: {auc:.4f}
{ISO8601_KST}\t[중단] {step}번째 스텝에서 사용자 중단
{ISO8601_KST}\t[오류] {traceback_summary}
```

### 10.3 로그 쓰기 구현

```python
# utils/storage.py

def get_log_writer(experiment_id: str):
    """
    로그 파일 append 전용 writer 반환.
    학습 스레드에서 직접 호출 (메인 스레드가 아님).
    """
    log_path = Path(f"./logs/{experiment_id}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
```

**line-buffered (`buffering=1`)**: 프로세스 종료나 중단 시에도 마지막 줄이 디스크에 보존됨을 보장.

### 10.4 UI 로그 표시 규칙

- `st.text_area()` 내에 최신 **100줄**만 표시 (A-09).
- 파일에는 전량 저장. 표시 줄 수 초과분은 UI에서 절삭.

```python
def read_log_tail(experiment_id: str, n_lines: int = 100) -> str:
    log_path = Path(f"./logs/{experiment_id}.log")
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])
```

---

## 11. 디스크 용량 모니터링

### 11.1 경고 조건

`00_Global_Context_Document.md §6` 기준:

| 조건 | 처리 |
|------|------|
| 모델 저장 전 여유 공간 < 500 MB | `st.warning()` 표시 + 저장 진행 허용 (차단 아님) |
| 모델 저장 전 여유 공간 < 100 MB | `st.error()` 표시 + 저장 차단 |

### 11.2 구현

```python
# utils/storage.py

def check_disk_space(
    required_mb: float = 500.0,
    path: str = "."
) -> tuple[bool, float]:
    """
    반환: (여유 공간 충분 여부, 여유 공간 MB)
    """
    import shutil
    usage = shutil.disk_usage(path)
    free_mb = usage.free / (1024 ** 2)
    return free_mb >= required_mb, free_mb


def check_disk_before_save(model_type: str) -> None:
    """
    모델 저장 전 디스크 공간 확인. 부족 시 st 알림 (Streamlit context에서 호출).
    100 MB 미만 시 RuntimeError 발생 (저장 차단).
    500 MB 미만 시 st.warning() (저장 허용).
    """
    sufficient, free_mb = check_disk_space(required_mb=100.0)
    if not sufficient:
        raise RuntimeError(
            f"ERR_DISK_SPACE: 여유 공간 {free_mb:.0f} MB — 100 MB 미만. 저장 불가."
        )

    _, free_mb = check_disk_space(required_mb=500.0)
    if free_mb < 500.0:
        st.warning(
            f"디스크 여유 공간이 {free_mb:.0f} MB로 부족합니다. "
            "500 MB 이상 확보 후 저장해 주세요."
        )
```

---

## 12. 구현 체크리스트

> `utils/storage.py`와 `utils/config_manager.py` 구현 완료 기준.

### storage.py

- [ ] `load_history()` — 파일 미존재, JSON 파싱 실패 모두 `[]` 반환
- [ ] `append_experiment()` — `.tmp` → `rename` 원자적 쓰기, IOError 시 `.tmp` 정리
- [ ] `delete_experiment_from_history()` — 해당 ID 없으면 `False` 반환
- [ ] `save_completed_experiment()` — 3단계 원자성 프로토콜 전체 구현
- [ ] `_cleanup_dir()` — `shutil.rmtree(ignore_errors=True)`
- [ ] `delete_experiment()` — history + 모델 디렉토리 + 로그 파일 삭제
- [ ] `get_log_writer()` — append 모드, line-buffered
- [ ] `read_log_tail()` — 최신 N줄 반환
- [ ] `check_disk_space()` — `shutil.disk_usage()` 기반
- [ ] `check_disk_before_save()` — 100 MB 미만 시 예외, 500 MB 미만 시 경고

### config_manager.py

- [ ] `load_config(path)` — 파일 미존재, YAML 파싱 실패 모두 `{}` 반환
- [ ] `save_config_section(section, data, path)` — 기존 섹션 보존, `.tmp` → `rename`
- [ ] `path` 파라미터로 루트 configs.yaml과 실험 스냅샷 양쪽 대응

### cache_manager.py

- [ ] `set_anomaly_map_cache(exp_id, data)` — 최대 3개 캐시, LRU eviction
- [ ] `get_anomaly_map_cache(exp_id)` — 없으면 `None` 반환

### 기타 검증

- [ ] `validate_imagenet_penalty_dir()` — `IMAGENET_PENALTY_DIR` 경로 존재 + 이미지 수 반환
- [ ] EfficientAD 학습 시작 전 탭4에서 penalty dir 검증 호출
- [ ] 실험 삭제 시 `_anomaly_maps_{exp_id}` 캐시 session_state에서 제거
- [ ] `status == "중단"` 레코드 저장 시 `model_path`, `configs_path`, `metrics` 모두 `null`

---

*이 문서는 08_AI_ML_Integration.md §8 (학습 완료 후 처리) 및 04_System_Architecture.md §3 (파일시스템 레이어)와 연동된다.*
*다음: [06_API_Specification.md](./06_API_Specification.md)*

---

## 13. 비전검사 대시보드 데이터 저장 전략 (v1.1 신규)

### 13.1 저장 원칙

비전검사 대시보드는 **세션 메모리 전용** 전략을 사용한다. 아래 원칙을 강제한다.

| 원칙 | 내용 |
|------|------|
| **영속 없음** | `insp_records`, `insp_test_pool`, `insp_last_result` 등 모든 검사 세션 데이터는 `session_state`에만 저장. 파일/DB 쓰기 금지 (R-INSP-03) |
| **읽기 전용 참조** | `history.json`, `./models/{exp_id}/model_state_dict.pth`, `./models/{exp_id}/configs.yaml`은 읽기 전용으로 참조. `save_history()` 등 쓰기 함수 호출 금지 (R-INSP-04) |
| **초기화 시점** | 앱 재시작 시 자동 초기화. 모델 교체 시 `reset_inspection_state()` 호출. 수동 초기화 버튼 제공 |

### 13.2 inspection_record 저장 구조

```python
# session_state.insp_records: list[dict]
# 각 dict의 구조 (00_Global_Context 1.10절 스키마)

example_record = {
    "seq":           1,
    "inspected_at":  "2026-05-26T14:02:31+09:00",  # ISO 8601 KST
    "image_name":    "crack_001.png",               # basename only
    "image_path":    "/app/dataset/screw/test/crack/001.png",  # 절대경로
    "verdict":       "불량",                         # "양품" | "불량"
    "anomaly_score": 0.4873,                        # round(value, 4)
}

# anomaly_map은 insp_records에 포함하지 않음.
# insp_last_result["anomaly_map"]에만 보관 (현재 화면 표시용).
```

### 13.3 test_pool 구성 로직

```python
# inspection/utils/test_sampler.py

def build_test_pool(dataset_path: str) -> list[tuple[str, str]]:
    """
    dataset_path/test/ 스캔하여 (절대경로, verdict_label) 풀 구성.
    good/ 하위 → "양품", 그 외 디렉토리 → "불량".
    반환 전 random.shuffle() 1회 적용.
    """
    test_root = Path(dataset_path) / "test"
    pool = []
    for cls_dir in test_root.iterdir():
        if not cls_dir.is_dir():
            continue
        label = "양품" if cls_dir.name == "good" else "불량"
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                pool.append((str(img_path), label))
    random.shuffle(pool)
    return pool

def sample_next(pool: list, index: int) -> tuple[tuple[str, str], int]:
    """
    pool[index] 반환. 끝 도달 시 재셔플 후 index = 0.
    반환: (sample, next_index)
    """
    if index >= len(pool):
        random.shuffle(pool)
        index = 0
    return pool[index], index + 1
```

### 13.4 모델 교체 시 상태 초기화 프로토콜

모델 교체는 아래 순서를 반드시 준수한다 (R-INSP-05):

```
Step 1. reset_inspection_state() 호출 (insp_active_model 제외 전체 초기화)
Step 2. insp_active_model = new_experiment_record
Step 3. build_test_pool(new_experiment_record["dataset_path"]) → insp_test_pool 갱신
Step 4. st.rerun()
```

Step 1 전에 Step 2를 실행하면 안 됨. 초기화 도중 모델이 변경되면 test_pool이 잘못 구성될 수 있음.

### 13.5 CSV 내보내기 스펙

탭2 [CSV 내보내기] 버튼은 `insp_records`를 5컬럼 CSV로 변환하여 `st.download_button`으로 제공한다.

```python
# 컬럼 순서 및 헤더
CSV_COLUMNS = ["번호", "시각", "이미지명", "판정결과", "Anomaly Score"]
# 실제 키 매핑
KEY_MAP = {
    "번호":          "seq",
    "시각":          "inspected_at",
    "이미지명":      "image_name",
    "판정결과":      "verdict",
    "Anomaly Score": "anomaly_score",
}
# 파일명: inspection_history_{YYYYMMDD_HHMMSS}.csv
```

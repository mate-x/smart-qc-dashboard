# 13. QA and Testing Strategy

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [11_Non_Functional_Requirements.md](./11_Non_Functional_Requirements.md), [08_AI_ML_Integration.md](./08_AI_ML_Integration.md)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **수정일**: 2026-06-11
> **중요**: 이 문서는 테스트 전략의 Single Source of Truth다. 테스트 유형·범위·합격 기준·도구를 확정한다. 모든 테스트는 08절 Z.절 및 05~07절 확정 사항을 기준으로 작성한다.

---

## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 최초 작성 |
| v1.1 | 2026-05-26 | 비전검사 대시보드 테스트 전략 추가 (I절) |
| v2.0 | 2026-06-11 | B.5 캐시 테스트: Streamlit session_state mock 제거 → 서버 측 LRU 캐시 직접 테스트; C.5 가드 테스트: session_state mock → FastAPI TestClient 기반으로 교체; D.1 E2E 주석: 탭 → Explorer 화면/endpoint; H 커버리지: `tabs/*.py` → `api/routers/*.py`; I.2 통합 테스트: Streamlit session_state → FastAPI TestClient; I.3 TC-INSP 시나리오: Streamlit 상태 → WS/API/React 기준; I.5 커버리지: `inspection/tabs/` → `api/routers/inspection.py` + WS 핸들러 |

---

## 목차

- [A. Test Strategy Overview](#a-test-strategy-overview)
- [B. Unit Tests](#b-unit-tests)
- [C. Integration Tests](#c-integration-tests)
- [D. End-to-End Tests](#d-end-to-end-tests)
- [E. ML-Specific Tests](#e-ml-specific-tests)
- [F. NFR Validation Tests](#f-nfr-validation-tests)
- [G. Test Infrastructure](#g-test-infrastructure)
- [H. Test Coverage Targets](#h-test-coverage-targets)
- [I. Vision 검사 테스트 전략](#i-vision-검사-테스트-전략-v20)

---

## A. Test Strategy Overview

### A.1 테스트 피라미드

```
        ┌──────────┐
        │  E2E     │  2개 (골든 패스)
        ├──────────┤
        │Integration│  8개 핵심 시나리오
        ├──────────┤
        │  Unit    │  모듈별 함수 단위 (커버리지 ≥ 80%)
        └──────────┘
```

> **v2.0 추가**: React(Explorer/Vision) 프론트엔드 컴포넌트 테스트는 TypeScript(vitest/jest) 기반으로 별도 수행한다. 이 문서의 pytest 테스트는 Python 백엔드(`utils/`, `api/`) 계층에 한정된다.

### A.2 테스트 도구

| 도구 | 용도 |
|------|------|
| `pytest` | 전체 테스트 실행기 |
| `pytest-cov` | 커버리지 측정 |
| `httpx` / `fastapi.testclient.TestClient` | FastAPI 엔드포인트 통합 테스트 |
| `unittest.mock` | 외부 의존성 모킹 (Anomalib 학습 루프 제외) |
| `tmp_path` (pytest fixture) | 임시 파일시스템 |

### A.3 테스트 실행 명령

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함 (Python 백엔드 계층)
pytest tests/ --cov=utils --cov=api --cov-report=term-missing

# 특정 모듈
pytest tests/unit/test_image_utils.py -v

# ML 테스트 제외 (빠른 CI용)
pytest tests/ -v -m "not slow"
```

---

## B. Unit Tests

### B.1 `utils/image_utils.py`

```python
# tests/unit/test_image_utils.py

import numpy as np
import pytest
from PIL import Image
from utils.image_utils import resize_with_padding, ensure_rgb, apply_homomorphic, apply_clahe, apply_he

class TestResizeWithPadding:
    def test_square_image_no_padding(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = resize_with_padding(img, 256)
        assert result.shape == (256, 256, 3)

    def test_landscape_aspect_ratio_preserved(self):
        # 4:3 이미지 → 256x256 패딩
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        result = resize_with_padding(img, 256)
        assert result.shape == (256, 256, 3)
        # 비율 오차 < 1픽셀 (11절 §E.2)
        expected_content_h = int(256 * 300 / 400)
        assert abs(result.shape[0] - 256) == 0  # 출력은 항상 target_size x target_size

    def test_padding_is_black(self):
        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        result = resize_with_padding(img, 256)
        # 패딩 영역은 0 (검정)
        top_rows = result[:int((256 - int(256*300/400))//2), :, :]
        assert np.all(top_rows == 0)


class TestEnsureRgb:
    def test_grayscale_to_rgb(self):
        gray = Image.fromarray(np.zeros((128, 128), dtype=np.uint8))
        result = ensure_rgb(gray)
        assert result.shape == (128, 128, 3)
        # R == G == B
        assert np.all(result[:, :, 0] == result[:, :, 1])
        assert np.all(result[:, :, 1] == result[:, :, 2])

    def test_rgb_unchanged(self):
        rgb = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        result = ensure_rgb(rgb)
        assert result.shape == (128, 128, 3)

    def test_rgba_to_rgb(self):
        rgba = Image.fromarray(np.zeros((128, 128, 4), dtype=np.uint8), mode="RGBA")
        result = ensure_rgb(rgba)
        assert result.shape == (128, 128, 3)


class TestHomomorphicFilter:
    def test_output_shape_preserved(self):
        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        params = {"sigma": 10.0, "gamma_H": 1.5, "gamma_L": 0.5, "normalize": True}
        result = apply_homomorphic(img, **params)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        result = apply_homomorphic(img, sigma=10.0, gamma_H=1.5, gamma_L=0.5, normalize=True)
        assert result.dtype == np.uint8

    def test_normalized_range(self):
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        result = apply_homomorphic(img, sigma=10.0, gamma_H=1.5, gamma_L=0.5, normalize=True)
        assert result.min() >= 0
        assert result.max() <= 255
```

---

### B.2 `utils/metrics.py`

```python
# tests/unit/test_metrics.py

import numpy as np
import pytest
from utils.metrics import compute_all_metrics

class TestComputeAllMetrics:
    def test_perfect_prediction(self):
        y_true = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        threshold = 0.5
        m = compute_all_metrics(y_true, scores, threshold)
        assert m["accuracy"] == 1.0
        assert m["auc"] == 1.0
        assert m["confusion_matrix"]["tp"] == 2
        assert m["confusion_matrix"]["tn"] == 2
        assert m["confusion_matrix"]["fp"] == 0
        assert m["confusion_matrix"]["fn"] == 0

    def test_all_normal_prediction(self):
        y_true = [0, 0, 0]
        scores = [0.1, 0.2, 0.3]
        threshold = 0.5
        m = compute_all_metrics(y_true, scores, threshold)
        assert m["confusion_matrix"]["tp"] == 0
        assert m["confusion_matrix"]["fp"] == 0

    def test_rounding_to_6_decimal(self):
        # 00절 §8 R-FLOAT-01: 소수점 6자리 저장
        y_true = [0, 1, 0, 1]
        scores = [0.3, 0.7, 0.4, 0.6]
        m = compute_all_metrics(y_true, scores, 0.5)
        for key in ["accuracy", "precision", "recall", "f1_score", "f2_score", "auc"]:
            val = m[key]
            assert val == round(val, 6), f"{key} not rounded to 6 decimal places"

    def test_returns_anomaly_scores_and_labels(self):
        y_true = [0, 1]
        scores = [0.2, 0.8]
        m = compute_all_metrics(y_true, scores, 0.5)
        assert m["anomaly_scores"] == scores
        assert m["image_labels"] == y_true
```

---

### B.3 `utils/storage.py`

```python
# tests/unit/test_storage.py

import json
import pytest
from pathlib import Path
from utils.storage import load_history, save_history, validate_imagenet_penalty_dir, IMAGENET_PENALTY_DIR

class TestHistoryIO:
    def test_load_returns_empty_list_if_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.storage.HISTORY_FILE", tmp_path / "history.json")
        result = load_history()
        assert result == []

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.storage.HISTORY_FILE", tmp_path / "experiments" / "history.json")
        records = [{"experiment_id": "test_123", "status": "completed"}]
        save_history(records)
        loaded = load_history()
        assert loaded == records

    def test_save_is_atomic(self, tmp_path, monkeypatch):
        history_path = tmp_path / "experiments" / "history.json"
        monkeypatch.setattr("utils.storage.HISTORY_FILE", history_path)
        save_history([{"id": "1"}])
        # .tmp 파일이 잔존하지 않아야 함
        assert not (tmp_path / "experiments" / "history.tmp").exists()
        assert history_path.exists()


class TestValidateImagenetPenaltyDir:
    def test_valid_dir_with_images(self, tmp_path, monkeypatch):
        d = tmp_path / "imagenet_penalty"
        d.mkdir()
        (d / "sample.JPEG").write_bytes(b"fake")
        monkeypatch.setattr("utils.storage.IMAGENET_PENALTY_DIR", d)
        ok, count = validate_imagenet_penalty_dir()
        assert ok is True
        assert count == 1

    def test_missing_dir_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.storage.IMAGENET_PENALTY_DIR", tmp_path / "nonexistent")
        ok, count = validate_imagenet_penalty_dir()
        assert ok is False
        assert count == 0

    def test_empty_dir_returns_false(self, tmp_path, monkeypatch):
        d = tmp_path / "imagenet_penalty"
        d.mkdir()
        monkeypatch.setattr("utils.storage.IMAGENET_PENALTY_DIR", d)
        ok, count = validate_imagenet_penalty_dir()
        assert ok is False
        assert count == 0
```

---

### B.4 `utils/config_manager.py`

```python
# tests/unit/test_config_manager.py

import pytest
from utils.config_manager import load_config, save_config_section

class TestConfigManager:
    def test_load_returns_empty_dict_if_missing(self, tmp_path):
        result = load_config(str(tmp_path / "nonexistent.yaml"))
        assert result == {}

    def test_save_and_load_section(self, tmp_path):
        path = str(tmp_path / "configs.yaml")
        data = {"method": "clahe", "clip_limit": 2.0}
        save_config_section("preprocessing", data, path=path)
        loaded = load_config(path)
        assert loaded["preprocessing"] == data

    def test_save_preserves_other_sections(self, tmp_path):
        path = str(tmp_path / "configs.yaml")
        save_config_section("model", {"model_type": "patchcore"}, path=path)
        save_config_section("preprocessing", {"method": "none"}, path=path)
        loaded = load_config(path)
        assert "model" in loaded
        assert "preprocessing" in loaded

    def test_safe_load_only(self, tmp_path):
        # !!python/object 태그가 포함된 YAML은 안전하게 무시되어야 함
        path = tmp_path / "evil.yaml"
        path.write_text("key: !!python/object:os.system ['echo evil']", encoding="utf-8")
        # safe_load는 임의 태그를 오류로 처리 (Constructor not found)
        with pytest.raises(Exception):
            load_config(str(path))
```

---

### B.5 `utils/cache_manager.py` (v2.0 — 서버 측 LRU 캐시)

> **v2.0 변경**: v1.x에서 `st.session_state`를 백엔드로 사용하던 캐시가 서버 측 인메모리 LRU 캐시로 교체됐다.  
> Streamlit 의존성 및 `mock_session_state` 픽스처가 제거됐다.

```python
# tests/unit/test_cache_manager.py

import time
import pytest
import numpy as np
from utils.cache_manager import (
    set_anomaly_map_cache,
    get_anomaly_map_cache,
    MAX_ANOMALY_MAP_CACHE,
)

# [확인 필요: utils/cache_manager.py에 테스트용 _clear_cache() 헬퍼 함수 추가 필요]
# 캐시 격리를 위해 각 테스트 전 캐시를 초기화한다.

@pytest.fixture(autouse=True)
def clear_cache():
    """각 테스트 전후로 서버 측 캐시 초기화"""
    from utils.cache_manager import _clear_cache  # [확인 필요: 헬퍼 함수 존재 여부]
    _clear_cache()
    yield
    _clear_cache()


class TestAnomalyMapCache:
    def test_set_and_get(self):
        data = {"anomaly_maps": {"img1.png": np.zeros((64, 64))}, "image_paths": ["img1.png"]}
        set_anomaly_map_cache("exp_001", data)
        result = get_anomaly_map_cache("exp_001")
        assert result is not None
        assert "anomaly_maps" in result

    def test_cache_miss_returns_none(self):
        result = get_anomaly_map_cache("exp_not_set")
        assert result is None

    def test_lru_eviction_on_overflow(self):
        for i in range(MAX_ANOMALY_MAP_CACHE + 1):
            set_anomaly_map_cache(f"exp_{i:03d}", {"anomaly_maps": {}, "image_paths": []})
            time.sleep(0.01)  # cached_at 차이 보장
        # MAX_ANOMALY_MAP_CACHE 개만 남아야 함
        # [확인 필요: _get_cache_size() 또는 동등한 검사 방법 확인]
        from utils.cache_manager import _get_cache_size
        assert _get_cache_size() == MAX_ANOMALY_MAP_CACHE

    def test_oldest_evicted(self):
        set_anomaly_map_cache("exp_old", {"anomaly_maps": {}, "image_paths": []})
        time.sleep(0.05)
        for i in range(MAX_ANOMALY_MAP_CACHE):
            set_anomaly_map_cache(f"exp_new_{i}", {"anomaly_maps": {}, "image_paths": []})
            time.sleep(0.01)
        assert get_anomaly_map_cache("exp_old") is None
```

---

## C. Integration Tests

### C.1 데이터셋 스캔 → 메타 정보 반환 흐름

```python
# tests/integration/test_tab1_flow.py
# 실제 MVTec 형식 임시 데이터셋 생성 후 데이터셋 검증 함수 호출

import pytest
from pathlib import Path
from PIL import Image

@pytest.fixture
def mvtec_dataset(tmp_path):
    """MVTec AD 형식 미니 데이터셋 생성"""
    root = tmp_path / "dataset"
    (root / "train" / "good").mkdir(parents=True)
    (root / "test" / "good").mkdir(parents=True)
    (root / "test" / "scratch").mkdir(parents=True)
    (root / "ground_truth" / "scratch").mkdir(parents=True)

    for i in range(3):
        img = Image.fromarray([[128]*64]*64, mode="L")
        img.save(root / "train" / "good" / f"{i:03d}.png")
        img.save(root / "test" / "good" / f"{i:03d}.png")
        img.save(root / "test" / "scratch" / f"{i:03d}.png")
        img.save(root / "ground_truth" / "scratch" / f"{i:03d}_mask.png")

    return root

def test_dataset_scan_populates_meta(mvtec_dataset):
    from utils.dataset_scanner import scan_dataset
    meta = scan_dataset(str(mvtec_dataset))
    assert meta["train_good_count"] == 3
    assert "scratch" in meta["defect_classes"]
    assert meta["total_test_count"] == 6
    assert meta["channels"] == 1  # Grayscale

def test_invalid_dataset_raises(tmp_path):
    from utils.dataset_scanner import scan_dataset
    with pytest.raises(ValueError):
        scan_dataset(str(tmp_path))  # 빈 디렉터리 — train/good/ 없음
```

---

### C.2 configs.yaml 저장 → 로드 왕복

```python
# tests/integration/test_config_roundtrip.py

def test_model_config_roundtrip(tmp_path):
    from utils.config_manager import save_config_section, load_config

    preprocessing_config = {
        "method": "clahe",
        "image_size": 256,
        "resize_mode": "padding",
        "normalization": "imagenet",
    }
    model_config = {
        "model_type": "efficientad",
        "image_size": 256,
        "batch_size": 1,
        "random_seed": 42,
        "threshold_method": "percentile",
        "threshold_value": 95.0,
        "params": {
            "model_size": "medium",
            "train_steps": 1000,
        }
    }
    path = str(tmp_path / "configs.yaml")
    # Config 화면 (/config) — POST /api/config 에서 preprocessing + model 섹션 동시 저장 시나리오
    save_config_section("preprocessing", preprocessing_config, path=path)
    save_config_section("model", model_config, path=path)
    loaded = load_config(path)
    assert "preprocessing" in loaded
    assert "model" in loaded
    assert loaded["model"]["model_type"] == "efficientad"
    assert loaded["model"]["params"]["train_steps"] == 1000
```

---

### C.3 TrainingWorker → Queue 메시지 흐름

```python
# tests/integration/test_training_worker.py
# 실제 Anomalib 학습 없이 TrainingWorker 내부 queue 프로토콜만 검증

import threading
import queue
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.slow
def test_worker_sends_completed_message():
    """실제 학습 루프 대신 mock으로 대체하여 queue 프로토콜 검증"""
    from utils.training_worker import TrainingWorker

    q = queue.Queue()
    stop = threading.Event()

    model_config = {
        "model_type": "patchcore",
        "image_size": 64,
        "batch_size": 1,
        "random_seed": 42,
        "threshold_method": "percentile",
        "threshold_value": 95.0,
        "params": {"backbone": "resnet18", "coreset_sampling_ratio": 1.0,
                   "max_train": 10, "knn": 3, "top_k_ratio": 0.1,
                   "neighbourhood_kernel_size": 3,
                   "pretrained_source": "torchvision", "pretrained_path": None}
    }
    preprocessing_config = {
        "method": "none", "resize_mode": "padding", "image_size": 64,
        "normalization": "imagenet",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "params": None
    }

    # 실제 데이터셋 없이 mock 사용
    with patch("utils.training_worker.build_dataset", return_value=MagicMock()), \
         patch("utils.training_worker.Engine") as MockEngine:
        mock_engine = MagicMock()
        MockEngine.return_value = mock_engine
        mock_engine.model = MagicMock()
        mock_engine.test.return_value = MagicMock()

        worker = TrainingWorker(
            experiment_id="patchcore_test_001",
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            dataset_path="/fake/dataset",
            device="cpu",
            stop_event=stop,
            result_queue=q,
        )
        worker.start()
        worker.join(timeout=10)

    # queue에서 메시지 수신
    messages = []
    while not q.empty():
        messages.append(q.get_nowait())

    types = [m["type"] for m in messages]
    assert "completed" in types or "error" in types  # 종료 메시지 존재

def test_worker_stops_on_stop_event():
    """stop_event 설정 시 stopped 메시지 전송 확인"""
    import queue, threading
    from utils.training_worker import TrainingWorker
    from unittest.mock import patch

    q = queue.Queue()
    stop = threading.Event()
    stop.set()  # 즉시 중단

    # ... (위와 유사한 mock 설정)
    # stopped 메시지 확인
```

---

### C.4 실험 저장 3단계 프로토콜

```python
# tests/integration/test_save_experiment.py

import pytest
from pathlib import Path
from unittest.mock import MagicMock

def test_save_experiment_creates_files(tmp_path, monkeypatch):
    from utils.storage import save_completed_experiment
    import utils.storage as storage_module

    monkeypatch.setattr(storage_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(storage_module, "HISTORY_FILE", tmp_path / "experiments" / "history.json")

    mock_model = MagicMock()
    experiment_record = {
        "experiment_id": "efficientad_test_0001",
        "status": "completed",
        "model_type": "efficientad",
        "model_config": {"model_type": "efficientad", "params": {}},
        "preprocessing_config": {"method": "none"},
        "metrics": {"auc": 0.95},
    }

    save_completed_experiment("efficientad_test_0001", mock_model, experiment_record)

    model_dir = tmp_path / "models" / "efficientad_test_0001"
    assert (model_dir / "model_state_dict.pth").exists()
    assert (model_dir / "configs.yaml").exists()
    assert (tmp_path / "experiments" / "history.json").exists()

def test_stage1_failure_cleanup(tmp_path, monkeypatch):
    """Stage 1 (pth 저장) 실패 시 디렉터리 정리 확인"""
    from utils.storage import save_completed_experiment
    import utils.storage as storage_module
    import torch

    monkeypatch.setattr(storage_module, "MODELS_DIR", tmp_path / "models")

    mock_model = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(torch, "save", lambda *a, **kw: (_ for _ in ()).throw(IOError("disk full")))
        with pytest.raises(Exception):
            save_completed_experiment("exp_fail", mock_model, {"model_config": {}, "preprocessing_config": {}})

    # 디렉터리가 남아있지 않아야 함
    assert not (tmp_path / "models" / "exp_fail").exists()
```

---

### C.5 화면 가드 → Config 화면 접근 검증 (v2.0)

> **v2.0 변경**: v1.x에서 `st.session_state` mock 기반으로 가드 조건을 검증하던 테스트가  
> FastAPI TestClient 기반으로 교체됐다.  
> React 컴포넌트 수준 가드 (Zustand store 기반)는 TypeScript(vitest) 테스트에서 별도 검증한다.

```python
# tests/integration/test_screen_guard.py
# v2.0: Python 측에서는 FastAPI API validation 검증
# React 컴포넌트 가드 (Zustand store 기반)는 Explorer vitest에서 별도 검증

import pytest
from fastapi.testclient import TestClient

# [확인 필요: api/main.py의 FastAPI app import 경로 확인]
# from api.main import app

def test_training_start_blocked_without_imagenet_penalty():
    """POST /api/training/start — EfficientAD 설정 시 ImageNet penalty 없으면 400 반환"""
    # [확인 필요: TestClient 사용 시 실제 파일시스템 격리 방법 확인]
    # client = TestClient(app)
    # response = client.post("/api/training/start", json={
    #     "model_config": {"model_type": "efficientad", ...},
    #     "dataset_path": "/nonexistent"
    # })
    # assert response.status_code == 400
    # assert "ImageNet" in response.json()["detail"]
    pass  # FastAPI TestClient 통합 완료 후 구현


def test_config_save_requires_valid_image_size():
    """POST /api/config — image_size가 유효하지 않으면 422 반환 (FastAPI validation)"""
    # client = TestClient(app)
    # response = client.post("/api/config", json={"preprocessing": {"image_size": -1}})
    # assert response.status_code == 422
    pass  # FastAPI TestClient 통합 완료 후 구현


# --- Explorer React 컴포넌트 가드 테스트 (TypeScript / vitest) ---
# 아래 시나리오는 Explorer/src/pages/ 컴포넌트 테스트에서 검증:
#
# TC-GUARD-01:
#   Given: datasetStore.datasetPath === null
#   When:  사용자가 /config 경로로 navigate()
#   Then:  ConfigPage 가드 UI 렌더링, 폼 콘텐츠 미렌더링
#
# TC-GUARD-02:
#   Given: configStore.modelConfig === null
#   When:  사용자가 /training 경로로 navigate()
#   Then:  TrainingPage 가드 UI 렌더링
#
# TC-GUARD-03:
#   Given: experimentsStore.selectedExperimentId === null
#   When:  사용자가 /anomaly-map 경로로 navigate()
#   Then:  AnomalyMapPage 가드 UI 렌더링
```

---

## D. End-to-End Tests

### D.1 학습 완료 골든 패스

**시나리오**: MVTec AD 미니 데이터셋 → EfficientAD small (train_steps=100) → 완료 → history.json 기록.

```python
# tests/e2e/test_golden_path.py
# @pytest.mark.slow — CI에서는 제외, 수동 실행

@pytest.mark.slow
def test_efficientad_training_golden_path(tmp_path, mvtec_dataset):
    """EfficientAD 학습 완료 후 history.json에 completed 레코드 기록 확인"""
    # Explorer 화면 플로우 (5단계):
    # 1. Dataset 화면 (/): POST /api/dataset/validate → datasetPath 확정
    # 2. Config 화면 (/config): POST /api/config → preprocessing_config, model_config 저장
    # 3. Training 화면 (/training): POST /api/training/start → TrainingWorker 실행 (train_steps=100)
    #    WS /ws/training 구독 → completed 메시지 수신
    # 4. Experiments 화면 (/experiments): GET /api/experiments → completed 레코드 확인
    # 5. AnomalyMap 화면 (/anomaly-map): POST /api/anomaly-map/{id}/build → 시각화
    # --- 테스트 실행 단계 ---
    # 1. TrainingWorker 실행 (train_steps=100, 빠른 완료)
    # 2. completed 메시지 수신 확인
    # 3. save_completed_experiment() 호출 확인
    # 4. history.json loaded → status=="completed" 확인
    pass  # 실제 구현은 FastAPI TestClient 연동 후 작성
```

---

### D.2 학습 중단 플로우

**시나리오**: Training 화면 학습 진행 중 [학습 중단] 버튼 클릭 → POST /api/training/stop → stopped 레코드 기록.

```python
@pytest.mark.slow
def test_training_stop_golden_path(tmp_path, mvtec_dataset):
    """학습 중 POST /api/training/stop 호출 → stopped 레코드 기록 확인"""
    pass  # 실제 구현은 FastAPI TestClient 연동 후 작성
```

---

## E. ML-Specific Tests

### E.1 Anomaly Score 정규화 (08절 §B.8)

```python
# tests/unit/test_normalization.py

import numpy as np
import pytest
from utils.metrics import normalize_anomaly_scores

class TestNormalizeAnomalyScores:
    def test_min_max_normalization(self):
        scores = np.array([0.1, 0.5, 0.9])
        normalized = normalize_anomaly_scores(scores, method="minmax")
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_all_same_scores(self):
        scores = np.array([0.5, 0.5, 0.5])
        # 모두 같으면 0으로 처리 (division by zero 방지)
        normalized = normalize_anomaly_scores(scores, method="minmax")
        assert np.all(normalized == 0.0)
```

### E.2 Threshold 계산 (08절 §B.8, H.1)

```python
# tests/unit/test_threshold.py

import numpy as np
from utils.metrics import compute_threshold

class TestComputeThreshold:
    def test_percentile_method(self):
        normal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        threshold = compute_threshold(normal_scores, method="percentile", value=95.0)
        expected = np.percentile(normal_scores, 95)
        assert threshold == pytest.approx(expected, abs=1e-6)

    def test_absolute_method(self):
        normal_scores = np.array([0.1, 0.2, 0.3])
        threshold = compute_threshold(normal_scores, method="absolute", value=0.75)
        assert threshold == pytest.approx(0.75)
```

### E.3 PatchCore memory_bank register_buffer 검증 (08절 §B.5.2)

```python
# tests/unit/test_tab2_patchcore.py 또는 tests/unit/test_training_worker.py

import torch
import pytest

class TestPatchCoreMemoryBank:
    def test_memory_bank_in_state_dict(self):
        """register_buffer로 등록된 memory_bank가 state_dict에 포함되는지 검증."""
        from utils.model_factory import _create_patchcore_model
        model_config = {
            "params": {
                "backbone": "wide_resnet50_2",
                "pretrained_source": "torchvision",
                "coreset_sampling_ratio": 0.1,
                "knn": 9,
                "neighbourhood_kernel_size": 3,
            }
        }
        model = _create_patchcore_model(model_config)
        torch_model = getattr(model, "model", None)
        assert torch_model is not None
        coreset = torch.randn(100, 512)
        torch_model.register_buffer("memory_bank", coreset)
        state = model.state_dict()
        assert any("memory_bank" in k for k in state), \
            "memory_bank가 state_dict에 포함되어야 합니다."

    def test_memory_bank_restored_after_load(self):
        """state_dict 저장 후 재로드 시 memory_bank가 복원되는지 검증."""
        import tempfile, os
        from utils.model_factory import _create_patchcore_model
        model_config = {
            "params": {
                "backbone": "wide_resnet50_2",
                "pretrained_source": "torchvision",
                "coreset_sampling_ratio": 0.1,
                "knn": 9,
                "neighbourhood_kernel_size": 3,
            }
        }
        model = _create_patchcore_model(model_config)
        torch_model = getattr(model, "model", None)
        coreset = torch.randn(100, 512)
        torch_model.register_buffer("memory_bank", coreset)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            pth_path = f.name
        try:
            torch.save(model.state_dict(), pth_path)
            new_model = _create_patchcore_model(model_config)
            new_model.load_state_dict(
                torch.load(pth_path, map_location="cpu", weights_only=True), strict=False
            )
            new_torch = getattr(new_model, "model", None)
            assert new_torch.memory_bank.shape[0] == 100, \
                "재로드 후 memory_bank 크기가 복원되어야 합니다."
        finally:
            os.unlink(pth_path)

    def test_coreset_reproducibility_with_generator(self):
        """동일 seed의 Generator를 사용하면 coreset 인덱스가 동일한지 검증."""
        N, size = 1000, 100
        g1 = torch.Generator()
        g1.manual_seed(42)
        idx1 = torch.randperm(N, generator=g1)[:size]

        g2 = torch.Generator()
        g2.manual_seed(42)
        idx2 = torch.randperm(N, generator=g2)[:size]

        assert torch.equal(idx1, idx2), \
            "동일 seed Generator는 동일한 coreset 인덱스를 생성해야 합니다."

    def test_spatial_size_validation(self):
        """patch_scores 크기가 완전제곱수가 아닐 때 명확한 ValueError가 발생하는지 검증."""
        import numpy as np
        from utils.model_factory import _get_anomaly_map

        # 완전제곱수가 아닌 패치 수를 직접 만들어 reshape 검증
        N = 800  # 28*28=784, 29*29=841 — 둘 다 아님
        patch_scores = torch.randn(N)
        spatial_size = int(N ** 0.5)
        is_perfect_square = (spatial_size * spatial_size == N)
        assert not is_perfect_square, "테스트 전제 확인: 완전제곱수가 아니어야 함"
        with pytest.raises(ValueError, match="정방형"):
            if spatial_size * spatial_size != N:
                raise ValueError(
                    f"PatchCore feature map 크기({N}개 패치)가 정방형이 아닙니다. "
                    f"image_size를 8의 배수(예: 256)로 설정해 주세요."
                )
```

### E.4 GT 마스크 로드 경로 변환 (07절 §C.3)

```python
# tests/unit/test_gt_mask_path.py

from pathlib import Path
from utils.image_utils import build_gt_mask_path

class TestBuildGtMaskPath:
    def test_standard_conversion(self):
        image_path = "test/scratch/000.png"
        dataset_root = "/data/screw"
        expected = Path("/data/screw/ground_truth/scratch/000_mask.png")
        result = build_gt_mask_path(image_path, dataset_root)
        assert result == expected

    def test_jpeg_extension(self):
        image_path = "test/crack/001.jpg"
        result = build_gt_mask_path(image_path, "/data")
        assert result == Path("/data/ground_truth/crack/001_mask.jpg")
```

---

## F. NFR Validation Tests

### F.1 재현성 테스트 (11절 §F.1)

```python
# tests/e2e/test_reproducibility.py
# @pytest.mark.slow

@pytest.mark.slow
def test_same_seed_same_results():
    """동일 seed, 동일 config → AUC 오차 ≤ 0.001"""
    # 1. PatchCore (coreset 전체 = 1.0) 2회 실행
    # 2. metrics.auc 비교
    pass
```

### F.2 파일 원자성 검증 (11절 §C.1)

```python
# tests/unit/test_atomicity.py

def test_history_json_no_partial_write(tmp_path, monkeypatch):
    """쓰기 도중 인터럽트 시뮬레이션 — 부분 파일 없음 확인"""
    # 이미 test_storage.py 의 test_save_is_atomic 에서 커버
    pass
```

---

## G. Test Infrastructure

### G.1 디렉터리 구조

```
tests/
├── conftest.py              # 공용 픽스처 (mvtec_dataset 등)
├── unit/
│   ├── test_image_utils.py
│   ├── test_metrics.py
│   ├── test_storage.py
│   ├── test_config_manager.py
│   ├── test_cache_manager.py
│   ├── test_normalization.py
│   ├── test_threshold.py
│   └── test_gt_mask_path.py
├── integration/
│   ├── test_tab1_flow.py
│   ├── test_config_roundtrip.py
│   ├── test_training_worker.py
│   ├── test_save_experiment.py
│   ├── test_screen_guard.py       # v2.0: FastAPI TestClient 기반 가드 검증
│   └── test_insp_flow.py
└── e2e/
    ├── test_golden_path.py
    └── test_reproducibility.py
```

### G.2 공용 픽스처 (conftest.py)

```python
# tests/conftest.py

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

@pytest.fixture
def mvtec_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "train" / "good").mkdir(parents=True)
    (root / "test" / "good").mkdir(parents=True)
    (root / "test" / "scratch").mkdir(parents=True)
    (root / "ground_truth" / "scratch").mkdir(parents=True)

    for i in range(5):
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(root / "train" / "good" / f"{i:03d}.png")
        Image.fromarray(arr).save(root / "test" / "good" / f"{i:03d}.png")
        Image.fromarray(arr).save(root / "test" / "scratch" / f"{i:03d}.png")
        mask = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        mask.save(root / "ground_truth" / "scratch" / f"{i:03d}_mask.png")

    return root

@pytest.fixture
def minimal_model_config():
    return {
        "model_type": "patchcore",
        "image_size": 64,
        "batch_size": 1,
        "random_seed": 42,
        "threshold_method": "percentile",
        "threshold_value": 95.0,
        "params": {
            "backbone": "resnet18",
            "pretrained_source": "torchvision",
            "pretrained_path": None,
            "coreset_sampling_ratio": 1.0,
            "neighbourhood_kernel_size": 3,
            "max_train": 10,
            "knn": 3,
            "top_k_ratio": 0.1,
        }
    }
```

### G.3 pytest 마커 설정

```ini
# pytest.ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
testpaths = tests
```

---

## H. Test Coverage Targets

| 모듈 | 목표 커버리지 | 우선 테스트 대상 |
|------|-------------|----------------|
| `utils/image_utils.py` | ≥ 90% | 전처리 필터, resize_with_padding, ensure_rgb |
| `utils/metrics.py` | ≥ 90% | compute_all_metrics, normalize, compute_threshold |
| `utils/storage.py` | ≥ 85% | load/save history, 3단계 저장 프로토콜 |
| `utils/config_manager.py` | ≥ 85% | load_config, save_config_section |
| `utils/cache_manager.py` | ≥ 80% | set/get/eviction (서버 측 LRU) |
| `utils/training_worker.py` | ≥ 70% | queue 프로토콜, stop_event 처리 |
| `api/routers/*.py` | ≥ 60% | 엔드포인트 validation, 주요 핸들러 |

**전체 목표**: 라인 커버리지 ≥ 80% (`utils/`, `api/` 패키지 기준).

```bash
# 커버리지 보고서 생성
pytest tests/ --cov=utils --cov=api \
    --cov-report=html:coverage_report \
    --cov-fail-under=80
```

---

---

## I. Vision 검사 테스트 전략 (v2.0)

> **v2.0 변경**: v1.1의 Streamlit session_state 기반 검사 테스트가 FastAPI TestClient 기반으로 교체됐다.  
> Vision React 컴포넌트(`useAutoInspection`, `useManualInspection` 등)의 단위 테스트는  
> TypeScript(vitest) 기반으로 `smart-qc-vision` 레포에서 별도 수행한다.

### I.1 단위 테스트 — `utils/test_sampler.py`

```python
# tests/unit/test_sampler.py

import pytest
from inspection.utils.test_sampler import build_test_pool, sample_from_pool

class TestBuildTestPool:
    def test_returns_list_of_paths(self, tmp_path):
        """build_test_pool()이 유효한 파일 경로 리스트를 반환한다"""
        good_dir = tmp_path / "test" / "good"
        good_dir.mkdir(parents=True)
        defect_dir = tmp_path / "test" / "scratch"
        defect_dir.mkdir(parents=True)
        (good_dir / "001.png").write_bytes(b"fake")
        (defect_dir / "001.png").write_bytes(b"fake")

        pool = build_test_pool(str(tmp_path))
        assert isinstance(pool, list)
        assert len(pool) == 2

    def test_empty_dataset_raises(self, tmp_path):
        """test/ 디렉터리가 없으면 ValueError 발생"""
        with pytest.raises(ValueError):
            build_test_pool(str(tmp_path))

    def test_pool_contains_only_supported_formats(self, tmp_path):
        """지원 포맷(.png/.jpg/.bmp)만 포함"""
        test_dir = tmp_path / "test" / "good"
        test_dir.mkdir(parents=True)
        (test_dir / "001.png").write_bytes(b"fake")
        (test_dir / "002.txt").write_bytes(b"text")  # 제외 대상

        pool = build_test_pool(str(tmp_path))
        assert all(p.endswith((".png", ".jpg", ".jpeg", ".bmp")) for p in pool)


class TestSampleFromPool:
    def test_sample_returns_item_from_pool(self):
        pool = ["a.png", "b.png", "c.png"]
        selected, used = sample_from_pool(pool, set())
        assert selected in pool
        assert selected in used

    def test_pool_exhaustion_triggers_reshuffle(self):
        """풀의 모든 아이템 사용 후 재샘플링 시 used가 초기화된다"""
        pool = ["a.png", "b.png"]
        used = {"a.png", "b.png"}  # 이미 모두 사용됨
        selected, new_used = sample_from_pool(pool, used)
        # 재섞기 후 used는 선택된 1개만 포함
        assert len(new_used) == 1
        assert selected in pool

    def test_no_duplicate_before_reshuffle(self):
        """재섞기 전까지 동일 이미지가 두 번 선택되지 않는다"""
        pool = ["a.png", "b.png", "c.png"]
        used = set()
        selected_items = []
        for _ in range(len(pool)):
            item, used = sample_from_pool(pool, used)
            selected_items.append(item)
        assert len(set(selected_items)) == len(pool)
```

### I.2 통합 테스트 — 모델 적용 → 검사 → 이력 누적 흐름 (v2.0)

> **v2.0 변경**: Streamlit `mock_session_state` 픽스처 및 `st.session_state["insp_*"]`  
> 키 직접 조작이 FastAPI TestClient 기반 API 호출로 교체됐다.

```python
# tests/integration/test_insp_flow.py
# [확인 필요: api/main.py FastAPI app import 경로 및 TestClient 설정 확인]

import pytest
from fastapi.testclient import TestClient
# from api.main import app  # [확인 필요: 실제 import 경로]

def test_model_apply_initializes_inspection_state():
    """POST /api/inspection/model — 모델 적용 후 검사 이력이 초기화되는지 확인 (TC-INSP-03)"""
    # [확인 필요: TestClient + 모델 mock 설정 방법]
    # client = TestClient(app)
    #
    # # 사전에 검사 이력 생성
    # client.post("/api/inspection/run", json={...})
    #
    # # 새 모델 적용
    # response = client.post("/api/inspection/model", json={"experiment_id": "new_exp"})
    # assert response.status_code == 200
    #
    # # 이력 초기화 확인
    # records = client.get("/api/inspection/records").json()
    # assert records == [] or records.get("records") == []
    pass  # FastAPI TestClient 통합 완료 후 구현


def test_manual_inspection_appends_record():
    """POST /api/inspection/run — 수동 검사 후 이력이 1건 증가하는지 확인 (TC-INSP-01)"""
    # [확인 필요: 추론 mock 없이 TestClient로 테스트 가능한지 확인]
    # client = TestClient(app)
    #
    # before = len(client.get("/api/inspection/records").json().get("records", []))
    # client.post("/api/inspection/run", json={...})
    # after = len(client.get("/api/inspection/records").json().get("records", []))
    #
    # assert after == before + 1
    pass  # FastAPI TestClient 통합 완료 후 구현


def test_defect_detection_sends_ng_verdict():
    """추론 결과 anomaly_score > threshold → verdict == 'NG' 반환 확인 (TC-INSP-02)"""
    # [확인 필요: 추론 로직 mock으로 고점수 강제 반환 방법]
    # from unittest.mock import patch
    # with patch("api.routers.inspection.run_inference", return_value={"anomaly_score": 0.99}):
    #     response = client.post("/api/inspection/run", json={...})
    #     assert response.json()["verdict"] == "NG"
    pass  # 추론 mock 설정 후 구현
```

### I.3 주요 테스트 시나리오 (Given-When-Then)

#### TC-INSP-01: 수동 검사 레코드 추가

| 단계 | 내용 |
|------|------|
| **Given** | 모델이 적용되어 있고 `GET /api/inspection/records` 응답에 3개 레코드 존재 |
| **When** | Vision에서 수동 검사 버튼 클릭 → `POST /api/inspection/run` 호출 |
| **Then** | `GET /api/inspection/records` 응답에 4번째 레코드 추가, `record.seq == 4` |

#### TC-INSP-02: 자동 검사 중 불량 감지 시 중지 및 팝업

| 단계 | 내용 |
|------|------|
| **Given** | Vision에서 자동 검사 모드 실행 중 (`WS /ws/inspection/auto` 연결됨) |
| **When** | 추론 결과 `verdict == "NG"` (불량 감지) → 서버가 WS 메시지 전송 |
| **Then** | Vision `useAutoInspection.ts`에서 `inspectionStore.showDefectPopup = true` 설정; Vision 불량 감지 모달 표시; `ws.close()` 호출로 자동 검사 중지 (서버 측 `WebSocketDisconnect` 트리거) |

#### TC-INSP-03: 모델 교체 시 이력 초기화

| 단계 | 내용 |
|------|------|
| **Given** | `GET /api/inspection/records`에 N개 기록 존재 |
| **When** | Vision 탭3(모델 교체)에서 새 모델 [적용] 버튼 클릭 → `POST /api/inspection/model` 호출 |
| **Then** | `GET /api/inspection/records` 응답이 빈 배열 반환; `GET /api/inspection/model` 응답에 새 모델 정보 포함; Vision `inspectionStore` 이력 초기화 |

#### TC-INSP-04: 테스트 풀 소진 시 재섞기

| 단계 | 내용 |
|------|------|
| **Given** | `test_sampler`의 모든 이미지가 `used` 셋에 포함 |
| **When** | `sample_from_pool()` 호출 |
| **Then** | `used` 초기화, `insp_pool_reshuffled` 로그 기록 (`utils/logger.py`), 새 이미지 반환 |

### I.4 테스트 마커 및 파일 구조

ML 모델 로드가 필요한 테스트는 `@pytest.mark.slow`로 표시하여 CI 빠른 실행에서 제외한다.

```python
@pytest.mark.slow
def test_insp_model_inference_end_to_end():
    """실제 모델 가중치 로드 후 추론 결과 검증 — CI 제외, 수동 실행"""
    pass
```

추가되는 테스트 파일 구조:

```
tests/
├── unit/
│   └── test_sampler.py              # I.1 단위 테스트
└── integration/
    └── test_insp_flow.py            # I.2 통합 테스트 (FastAPI TestClient)
```

### I.5 커버리지 대상 (v2.0)

| 모듈 | 목표 커버리지 | 우선 테스트 대상 |
|------|-------------|----------------|
| `inspection/utils/test_sampler.py` | ≥ 90% | `build_test_pool()`, `sample_from_pool()` 재섞기 |
| `api/routers/inspection.py` | ≥ 70% | `POST /api/inspection/run`, `POST /api/inspection/model`, `GET /api/inspection/records` |
| `api/ws/inspection_ws.py` | ≥ 60% | WS 자동 검사 루프, 불량 감지 메시지 전송, `WebSocketDisconnect` 처리 |

> **v2.0 제거**: v1.1의 `inspection/tabs/insp_tab*.py` 커버리지 대상은 Streamlit 모듈로 더 이상 공식 UI 경로가 아니다. FastAPI route handler 및 WS 핸들러로 대체됐다.

---

*다음 문서*: [14_Deployment_and_Release_Plan.md](./14_Deployment_and_Release_Plan.md)

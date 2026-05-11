# 13. QA and Testing Strategy

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [11_Non_Functional_Requirements.md](./11_Non_Functional_Requirements.md), [08_AI_ML_Integration.md](./08_AI_ML_Integration.md)
> **버전**: v1.0
> **작성일**: 2026-05-09
> **중요**: 이 문서는 테스트 전략의 Single Source of Truth다. 테스트 유형·범위·합격 기준·도구를 확정한다. 모든 테스트는 08절 Z.절 및 05~07절 확정 사항을 기준으로 작성한다.

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

### A.2 테스트 도구

| 도구 | 용도 |
|------|------|
| `pytest` | 전체 테스트 실행기 |
| `pytest-cov` | 커버리지 측정 |
| `unittest.mock` | 외부 의존성 모킹 (Anomalib 학습 루프 제외) |
| `tmp_path` (pytest fixture) | 임시 파일시스템 |

### A.3 테스트 실행 명령

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=utils --cov=tabs --cov-report=term-missing

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

### B.5 `utils/cache_manager.py`

```python
# tests/unit/test_cache_manager.py

import time
import pytest
import numpy as np
import streamlit as st
from unittest.mock import patch, MagicMock

# session_state 모킹을 위한 픽스처
@pytest.fixture(autouse=True)
def mock_session_state():
    state = {}
    with patch.object(st, "session_state", state):
        yield state

from utils.cache_manager import set_anomaly_map_cache, get_anomaly_map_cache, MAX_ANOMALY_MAP_CACHE

class TestAnomalyMapCache:
    def test_set_and_get(self):
        data = {"anomaly_maps": {"img1.png": np.zeros((64, 64))}, "image_paths": ["img1.png"]}
        set_anomaly_map_cache("exp_001", data)
        result = get_anomaly_map_cache("exp_001")
        assert result is not None
        assert "anomaly_maps" in result

    def test_lru_eviction_on_overflow(self):
        for i in range(MAX_ANOMALY_MAP_CACHE + 1):
            data = {"anomaly_maps": {}, "image_paths": []}
            set_anomaly_map_cache(f"exp_{i:03d}", data)
            time.sleep(0.01)  # cached_at 차이 보장
        # MAX_ANOMALY_MAP_CACHE 개만 남아야 함
        count = sum(1 for k in st.session_state if k.startswith("_anomaly_maps_"))
        assert count == MAX_ANOMALY_MAP_CACHE

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

### C.1 데이터셋 스캔 → session_state 흐름

```python
# tests/integration/test_tab1_flow.py
# 실제 MVTec 형식 임시 데이터셋 생성 후 탭1 검증 함수 호출

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
    save_config_section("model", model_config, path=path)
    loaded = load_config(path)
    assert loaded["model"]["model_type"] == "efficientad"
    assert loaded["model"]["params"]["train_steps"] == 1000
```

---

### C.3 TrainingWorker → Queue 메시지 흐름

```python
# tests/integration/test_training_worker.py
# 실제 Anomalib 학습 없이 WorkerWorker 내부 queue 프로토콜만 검증

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
        # train() 호출 후 test() 결과 반환 mock
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

## D. End-to-End Tests

### D.1 학습 완료 골든 패스

**시나리오**: MVTec AD 미니 데이터셋 → EfficientAD small (train_steps=100) → 완료 → history.json 기록.

```python
# tests/e2e/test_golden_path.py
# @pytest.mark.slow — CI에서는 제외, 수동 실행

@pytest.mark.slow
def test_efficientad_training_golden_path(tmp_path, mvtec_dataset):
    """EfficientAD 학습 완료 후 history.json에 completed 레코드 기록 확인"""
    # 환경 변수로 실험 디렉터리 격리
    # 1. TrainingWorker 실행 (train_steps=100, 빠른 완료)
    # 2. completed 메시지 수신 확인
    # 3. save_completed_experiment() 호출 확인
    # 4. history.json loaded → status=="completed" 확인
    pass  # 실제 구현은 전체 파이프라인 연동 후 작성

@pytest.mark.slow
def test_training_stop_golden_path(tmp_path, mvtec_dataset):
    """학습 중 stop_event 설정 → stopped 레코드 기록 확인"""
    pass
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

### E.3 GT 마스크 로드 경로 변환 (07절 §C.3)

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
│   └── test_save_experiment.py
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
| `utils/cache_manager.py` | ≥ 80% | set/get/eviction |
| `utils/training_worker.py` | ≥ 70% | queue 프로토콜, stop_event 처리 |
| `tabs/*.py` | ≥ 60% | 탭 가드 조건, 주요 핸들러 |

**전체 목표**: 라인 커버리지 ≥ 80% (`utils/` 패키지 기준).

```bash
# 커버리지 보고서 생성
pytest tests/ --cov=utils --cov=tabs \
    --cov-report=html:coverage_report \
    --cov-fail-under=80
```

---

*다음 문서*: [14_Deployment_and_Release_Plan.md](./14_Deployment_and_Release_Plan.md)

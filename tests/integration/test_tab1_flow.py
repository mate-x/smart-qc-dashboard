from __future__ import annotations

import pytest

from utils.dataset_scanner import scan_dataset
from utils.path_validator import validate_dataset_path


class TestTab1DatasetScanFlow:
    """Tab 1: 데이터셋 경로 입력 → 검증 → 스캔 통합 플로우."""

    def test_validate_then_scan(self, mvtec_dataset):
        validated = validate_dataset_path(str(mvtec_dataset))
        meta = scan_dataset(str(validated))
        assert meta["train_good_count"] == 5
        assert "scratch" in meta["defect_classes"]
        assert meta["total_test_count"] == 10

    def test_scan_returns_gt_counts(self, mvtec_dataset):
        meta = scan_dataset(str(mvtec_dataset))
        assert "scratch" in meta["gt_counts"]
        assert meta["gt_counts"]["scratch"] == 5

    def test_invalid_path_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="경로가 존재하지 않습니다"):
            validate_dataset_path(str(tmp_path / "nonexistent"))

    def test_non_mvtec_dir_raises_value_error(self, tmp_path):
        empty_dir = tmp_path / "not_mvtec"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="MVTec AD 구조 미충족"):
            scan_dataset(str(empty_dir))

    def test_scan_channels_detected(self, mvtec_dataset):
        meta = scan_dataset(str(mvtec_dataset))
        assert meta["channels"] in (1, 3)

    def test_scan_supported_formats_detected(self, mvtec_dataset):
        meta = scan_dataset(str(mvtec_dataset))
        assert ".png" in meta["supported_formats"]

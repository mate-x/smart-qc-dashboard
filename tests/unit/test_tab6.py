from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tabs.tab6_anomaly_map import (
    _build_csv_bytes,
    _build_table_rows,
    _build_zip_bytes,
    _classify,
    _overlay_binary_mask,
)


class TestClassify:
    def test_tp(self):
        assert _classify(label=1, score=0.8, threshold=0.5) == "TP"

    def test_fn(self):
        assert _classify(label=1, score=0.3, threshold=0.5) == "FN"

    def test_fp(self):
        assert _classify(label=0, score=0.8, threshold=0.5) == "FP"

    def test_tn(self):
        assert _classify(label=0, score=0.3, threshold=0.5) == "TN"

    def test_score_equal_threshold_is_ng(self):
        """score == threshold → NG → FP for normal image."""
        assert _classify(label=0, score=0.5, threshold=0.5) == "FP"

    def test_score_just_below_threshold_is_ok(self):
        assert _classify(label=1, score=0.499, threshold=0.5) == "FN"


class TestBuildTableRows:
    def _make(self, paths, scores, labels, threshold):
        return _build_table_rows(paths, scores, labels, threshold)

    def test_basic_row_count(self):
        rows = self._make(
            ["dataset/test/good/a.jpg", "dataset/test/crack/b.jpg"],
            [0.3, 0.8],
            [0, 1],
            threshold=0.5,
        )
        assert len(rows) == 2

    def test_row_fields_present(self):
        rows = self._make(["test/good/a.jpg"], [0.4], [0], threshold=0.5)
        row = rows[0]
        assert "이미지명" in row
        assert "결함 유형" in row
        assert "Anomaly Score" in row
        assert "판정" in row
        assert "GT 일치" in row
        assert "오분류" in row
        assert "_path_idx" in row
        assert "_path" in row

    def test_classification_assigned_correctly(self):
        rows = self._make(
            ["t/good/a.jpg", "t/crack/b.jpg", "t/good/c.jpg", "t/crack/d.jpg"],
            [0.2, 0.8, 0.9, 0.1],  # TN, TP, FP, FN
            [0, 1, 0, 1],
            threshold=0.5,
        )
        assert rows[0]["오분류"] == "TN"
        assert rows[1]["오분류"] == "TP"
        assert rows[2]["오분류"] == "FP"
        assert rows[3]["오분류"] == "FN"

    def test_gt_match_matches_classification(self):
        rows = self._make(
            ["t/good/a.jpg", "t/crack/b.jpg"],
            [0.2, 0.8],
            [0, 1],
            threshold=0.5,
        )
        # TN and TP → GT 일치 = True
        assert rows[0]["GT 일치"] is True
        assert rows[1]["GT 일치"] is True

    def test_gt_mismatch(self):
        rows = self._make(["t/good/a.jpg"], [0.9], [0], threshold=0.5)
        # FP → GT 일치 = False
        assert rows[0]["GT 일치"] is False

    def test_path_idx_preserved(self):
        rows = self._make(
            ["t/good/x.jpg", "t/crack/y.jpg"],
            [0.1, 0.9],
            [0, 1],
            threshold=0.5,
        )
        assert rows[0]["_path_idx"] == 0
        assert rows[1]["_path_idx"] == 1

    def test_truncates_to_shortest_list(self):
        """입력 리스트 길이 불일치 시 짧은 쪽 기준."""
        rows = self._make(
            ["t/a/x.jpg", "t/b/y.jpg", "t/c/z.jpg"],
            [0.1, 0.2],  # shorter
            [0, 1, 0],
            threshold=0.5,
        )
        assert len(rows) == 2

    def test_판정_ok_ng(self):
        rows = self._make(
            ["t/good/a.jpg", "t/crack/b.jpg"],
            [0.3, 0.7],
            [0, 1],
            threshold=0.5,
        )
        assert rows[0]["판정"] == "OK"
        assert rows[1]["판정"] == "NG"

    def test_결함유형_extracted_from_path(self):
        rows = self._make(["dataset/test/scratches/img.png"], [0.1], [1], threshold=0.5)
        assert rows[0]["결함 유형"] == "scratches"
        assert rows[0]["이미지명"] == "img.png"


class TestOverlayBinaryMask:
    def _make_heatmap(self, size=(64, 64)) -> Image.Image:
        arr = np.zeros((*size, 3), dtype=np.uint8)
        arr[:, :, 0] = 128  # some red channel
        return Image.fromarray(arr, mode="RGB")

    def test_returns_pil_image(self):
        heatmap = self._make_heatmap()
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        result = _overlay_binary_mask(heatmap, mask)
        assert isinstance(result, Image.Image)

    def test_output_size_matches_input(self):
        heatmap = self._make_heatmap((80, 60))
        mask = np.zeros((80, 60), dtype=np.uint8)
        result = _overlay_binary_mask(heatmap, mask)
        assert result.size == heatmap.size

    def test_empty_mask_does_not_crash(self):
        heatmap = self._make_heatmap()
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = _overlay_binary_mask(heatmap, mask)
        assert result is not None

    def test_contour_drawn_on_nonzero_mask(self):
        """윤곽선 픽셀은 빨간색(255,0,0)으로 그려진다."""
        heatmap = Image.new("RGB", (64, 64), (0, 0, 0))
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 1
        result = _overlay_binary_mask(heatmap, mask)
        arr = np.array(result)
        # 윤곽선 인근 픽셀 중 빨간색 채널이 255인 픽셀이 존재해야 함
        assert (arr[:, :, 0] == 255).any()


class TestBuildCsvBytes:
    def _make_df(self):
        rows = _build_table_rows(
            ["t/good/a.jpg", "t/crack/b.jpg"],
            [0.2, 0.8],
            [0, 1],
            threshold=0.5,
        )
        return pd.DataFrame(rows)

    def test_returns_bytes(self):
        df = self._make_df()
        result = _build_csv_bytes(df)
        assert isinstance(result, bytes)

    def test_utf8_bom_header(self):
        """UTF-8 BOM(EF BB BF)으로 시작해야 Excel에서 한글 깨짐 없음."""
        df = self._make_df()
        result = _build_csv_bytes(df)
        assert result[:3] == b"\xef\xbb\xbf"

    def test_contains_expected_columns(self):
        df = self._make_df()
        result = _build_csv_bytes(df)
        text = result.decode("utf-8-sig")
        assert "이미지명" in text
        assert "Anomaly Score" in text
        assert "판정" in text
        assert "오분류" in text

    def test_private_columns_excluded(self):
        df = self._make_df()
        result = _build_csv_bytes(df)
        text = result.decode("utf-8-sig")
        assert "_path" not in text
        assert "_path_idx" not in text

    def test_row_count_matches(self):
        df = self._make_df()
        result = _build_csv_bytes(df)
        lines = result.decode("utf-8-sig").strip().splitlines()
        # header + 2 data rows
        assert len(lines) == 3


class TestBuildZipBytes:
    def _make_df(self, tmp_path):
        good_dir = tmp_path / "test" / "good"
        good_dir.mkdir(parents=True)
        img = Image.new("RGB", (32, 32), (100, 100, 100))
        img_path = good_dir / "a.jpg"
        img.save(str(img_path))

        rows = _build_table_rows(
            [str(img_path)],
            [0.3],
            [0],
            threshold=0.5,
        )
        return pd.DataFrame(rows)

    def test_returns_bytes(self, tmp_path):
        df = self._make_df(tmp_path)
        maps = np.zeros((1, 32, 32), dtype=np.float32)
        result = _build_zip_bytes(df, maps, {"dataset_path": str(tmp_path)}, 0.5, "exp1")
        assert isinstance(result, bytes)

    def test_valid_zip_structure(self, tmp_path):
        df = self._make_df(tmp_path)
        maps = np.zeros((1, 32, 32), dtype=np.float32)
        result = _build_zip_bytes(df, maps, {"dataset_path": str(tmp_path)}, 0.5, "exp1")
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = zf.namelist()
        assert len(names) == 1
        assert names[0].endswith("_anomaly.png")
        assert names[0].startswith("exp1_")

    def test_filename_includes_exp_id_and_stem(self, tmp_path):
        df = self._make_df(tmp_path)
        maps = np.zeros((1, 32, 32), dtype=np.float32)
        result = _build_zip_bytes(df, maps, {"dataset_path": str(tmp_path)}, 0.5, "myexp")
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            name = zf.namelist()[0]
        assert "myexp" in name
        assert "a" in name  # stem of a.jpg

    def test_missing_image_skipped_gracefully(self, tmp_path):
        """존재하지 않는 이미지 경로는 건너뛰고 빈 ZIP 반환."""
        rows = _build_table_rows(
            [str(tmp_path / "nonexistent.jpg")],
            [0.3],
            [0],
            threshold=0.5,
        )
        df = pd.DataFrame(rows)
        maps = np.zeros((1, 32, 32), dtype=np.float32)
        result = _build_zip_bytes(df, maps, {"dataset_path": str(tmp_path)}, 0.5, "exp1")
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            assert len(zf.namelist()) == 0

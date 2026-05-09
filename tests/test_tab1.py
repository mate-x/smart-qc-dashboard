"""
탭1 순수 함수 단위 테스트.
Streamlit context가 필요 없는 _build_dataset_meta, _build_tree_text,
_build_count_table 를 직접 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from tabs.tab1_data_folder import (
    _build_count_table,
    _build_dataset_meta,
    _build_tree_text,
)


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture()
def mvtec_dir(tmp_path: Path) -> Path:
    """MVTec AD 형식 미니 데이터셋 구조 생성."""
    dummy_rgb = Image.new("RGB", (64, 64), "white")
    dummy_gray = Image.new("L", (64, 64), 128)

    # train/good/
    train_good = tmp_path / "train" / "good"
    train_good.mkdir(parents=True)
    for i in range(5):
        dummy_rgb.save(train_good / f"{i:03d}.png")

    # test/good/, test/crack/, test/scratch/
    for cls_name, count, is_gray in [("good", 3, False), ("crack", 2, False), ("scratch", 1, True)]:
        cls_dir = tmp_path / "test" / cls_name
        cls_dir.mkdir(parents=True)
        src = dummy_gray if is_gray else dummy_rgb
        for i in range(count):
            src.save(cls_dir / f"{i:03d}.png")

    # ground_truth/crack/, ground_truth/scratch/
    for cls_name, count in [("crack", 2), ("scratch", 1)]:
        gt_dir = tmp_path / "ground_truth" / cls_name
        gt_dir.mkdir(parents=True)
        for i in range(count):
            dummy_rgb.save(gt_dir / f"{i:03d}_mask.png")

    return tmp_path


@pytest.fixture()
def sample_meta(mvtec_dir: Path) -> dict:
    return _build_dataset_meta(mvtec_dir)


# ---------------------------------------------------------------------------
# _build_dataset_meta 테스트
# ---------------------------------------------------------------------------

class TestBuildDatasetMeta:
    def test_train_good_count(self, sample_meta: dict) -> None:
        assert sample_meta["train_good_count"] == 5

    def test_test_counts(self, sample_meta: dict) -> None:
        assert sample_meta["test_counts"]["good"] == 3
        assert sample_meta["test_counts"]["crack"] == 2
        assert sample_meta["test_counts"]["scratch"] == 1

    def test_total_test_count(self, sample_meta: dict) -> None:
        assert sample_meta["total_test_count"] == 6  # 3+2+1

    def test_gt_counts(self, sample_meta: dict) -> None:
        assert sample_meta["gt_counts"]["crack"] == 2
        assert sample_meta["gt_counts"]["scratch"] == 1

    def test_defect_classes_includes_good(self, sample_meta: dict) -> None:
        # FR-T1-02: good 포함
        assert "good" in sample_meta["defect_classes"]

    def test_defect_classes_sorted(self, sample_meta: dict) -> None:
        assert sample_meta["defect_classes"] == sorted(sample_meta["defect_classes"])

    def test_channels_rgb(self, sample_meta: dict) -> None:
        # train/good/ 이미지가 RGB이므로 3채널
        assert sample_meta["channels"] == 3

    def test_channels_grayscale(self, tmp_path: Path) -> None:
        # Grayscale 이미지 전용 데이터셋
        train_good = tmp_path / "train" / "good"
        train_good.mkdir(parents=True)
        (tmp_path / "test" / "good").mkdir(parents=True)
        Image.new("L", (64, 64)).save(train_good / "000.png")
        meta = _build_dataset_meta(tmp_path)
        assert meta["channels"] == 1

    def test_has_invalid_files_false(self, sample_meta: dict) -> None:
        # 지원 포맷(.png)만 존재
        assert sample_meta["has_invalid_files"] is False

    def test_has_invalid_files_true(self, mvtec_dir: Path) -> None:
        # 지원 외 이미지 파일(.tiff)은 invalid로 집계
        (mvtec_dir / "test" / "good" / "bad.tiff").write_bytes(b"\x00")
        meta = _build_dataset_meta(mvtec_dir)
        assert meta["has_invalid_files"] is True
        assert meta["_invalid_file_count"] >= 1

    def test_hidden_file_not_invalid(self, mvtec_dir: Path) -> None:
        # 숨김 파일(.DS_Store 등)은 invalid 카운트에서 제외
        (mvtec_dir / "test" / "good" / ".DS_Store").write_bytes(b"")
        meta = _build_dataset_meta(mvtec_dir)
        assert meta["has_invalid_files"] is False
        assert meta["_invalid_file_count"] == 0

    def test_non_image_ext_not_invalid(self, mvtec_dir: Path) -> None:
        # .txt, .json 같은 메타 파일은 invalid 카운트에서 제외
        (mvtec_dir / "test" / "README.txt").write_text("readme")
        (mvtec_dir / "train" / "meta.json").write_text("{}")
        meta = _build_dataset_meta(mvtec_dir)
        assert meta["has_invalid_files"] is False
        assert meta["_invalid_file_count"] == 0

    def test_unsupported_image_ext_is_invalid(self, mvtec_dir: Path) -> None:
        # .gif, .webp 같이 이미지처럼 생겼지만 미지원 포맷은 invalid
        (mvtec_dir / "test" / "good" / "extra.gif").write_bytes(b"\x00")
        (mvtec_dir / "test" / "crack" / "extra.webp").write_bytes(b"\x00")
        meta = _build_dataset_meta(mvtec_dir)
        assert meta["has_invalid_files"] is True
        assert meta["_invalid_file_count"] == 2

    def test_supported_formats(self, sample_meta: dict) -> None:
        assert ".png" in sample_meta["supported_formats"]

    def test_no_gt_dir(self, tmp_path: Path) -> None:
        # ground_truth/ 없는 경우 gt_counts 빈 dict
        train_good = tmp_path / "train" / "good"
        train_good.mkdir(parents=True)
        (tmp_path / "test" / "good").mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(train_good / "000.png")
        meta = _build_dataset_meta(tmp_path)
        assert meta["gt_counts"] == {}


# ---------------------------------------------------------------------------
# _build_tree_text 테스트
# ---------------------------------------------------------------------------

class TestBuildTreeText:
    def test_root_name(self, mvtec_dir: Path, sample_meta: dict) -> None:
        tree = _build_tree_text(mvtec_dir, sample_meta)
        assert f"📂 {mvtec_dir.name}/" in tree

    def test_contains_train_section(self, mvtec_dir: Path, sample_meta: dict) -> None:
        tree = _build_tree_text(mvtec_dir, sample_meta)
        assert "📂 train/" in tree
        assert f"📂 good/ ({sample_meta['train_good_count']}장)" in tree

    def test_contains_test_section(self, mvtec_dir: Path, sample_meta: dict) -> None:
        tree = _build_tree_text(mvtec_dir, sample_meta)
        assert "📂 test/" in tree
        assert "📂 crack/" in tree

    def test_contains_gt_section(self, mvtec_dir: Path, sample_meta: dict) -> None:
        tree = _build_tree_text(mvtec_dir, sample_meta)
        assert "📂 ground_truth/" in tree

    def test_max_depth_three(self, mvtec_dir: Path, sample_meta: dict) -> None:
        # 각 줄의 들여쓰기 최대 4칸(2단계 = "    ")
        tree = _build_tree_text(mvtec_dir, sample_meta)
        for line in tree.splitlines():
            leading = len(line) - len(line.lstrip())
            assert leading <= 4, f"들여쓰기 4칸 초과: {repr(line)}"


# ---------------------------------------------------------------------------
# _build_count_table 테스트
# ---------------------------------------------------------------------------

class TestBuildCountTable:
    def test_has_total_row(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        assert df.iloc[-1]["클래스"] == "합계"

    def test_total_train(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        assert df.iloc[-1]["학습(train)"] == 5  # train_good_count

    def test_total_test(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        assert df.iloc[-1]["테스트(test)"] == 6  # 3+2+1

    def test_good_row_has_train_count(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        good_row = df[df["클래스"] == "good"].iloc[0]
        assert good_row["학습(train)"] == 5

    def test_defect_rows_no_train(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        crack_row = df[df["클래스"] == "crack"].iloc[0]
        assert crack_row["학습(train)"] == 0

    def test_gt_mask_count(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        crack_row = df[df["클래스"] == "crack"].iloc[0]
        assert crack_row["GT 마스크"] == 2

    def test_columns(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        assert list(df.columns) == ["클래스", "학습(train)", "테스트(test)", "GT 마스크"]

    def test_good_not_duplicated(self, sample_meta: dict) -> None:
        df = _build_count_table(sample_meta)
        # 합계 행 제외하고 good은 정확히 1행
        data_rows = df[df["클래스"] != "합계"]
        assert (data_rows["클래스"] == "good").sum() == 1


# ---------------------------------------------------------------------------
# apply_filter Grayscale 입력 처리 (image_utils 보완)
# ---------------------------------------------------------------------------

class TestApplyFilterGrayscale:
    """apply_filter가 Grayscale(L) 입력을 crash 없이 처리하는지 검증."""

    def test_grayscale_with_he(self) -> None:
        from utils.image_utils import apply_filter
        gray = Image.new("L", (64, 64), 128)
        result = apply_filter(gray, "he", None)
        assert result.mode == "RGB"
        assert result.size == (64, 64)

    def test_grayscale_with_clahe(self) -> None:
        from utils.image_utils import apply_filter
        gray = Image.new("L", (64, 64), 128)
        result = apply_filter(gray, "clahe", {"clip_limit": 2.0})
        assert result.mode == "RGB"

    def test_grayscale_with_homomorphic(self) -> None:
        from utils.image_utils import apply_filter
        gray = Image.new("L", (64, 64), 128)
        result = apply_filter(gray, "homomorphic", None)
        assert result.mode == "RGB"

    def test_grayscale_with_none_method(self) -> None:
        from utils.image_utils import apply_filter
        gray = Image.new("L", (64, 64), 128)
        result = apply_filter(gray, "none", None)
        # "none"은 변환 없이 그대로 반환
        assert result.mode == "L"

    def test_rgba_input_converted(self) -> None:
        from utils.image_utils import apply_filter
        rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        result = apply_filter(rgba, "he", None)
        assert result.mode == "RGB"

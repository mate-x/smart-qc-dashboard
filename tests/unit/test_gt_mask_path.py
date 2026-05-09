from __future__ import annotations

from pathlib import Path

import pytest

from utils.image_utils import build_gt_mask_path


class TestBuildGtMaskPath:
    def test_basic_scratch_class(self):
        result = build_gt_mask_path(
            image_path="/data/dataset/test/scratch/000.png",
            dataset_root="/data/dataset",
        )
        expected = Path("/data/dataset/ground_truth/scratch/000_mask.png")
        assert result == expected

    def test_bent_class(self):
        result = build_gt_mask_path(
            image_path="/ds/test/bent/001.png",
            dataset_root="/ds",
        )
        expected = Path("/ds/ground_truth/bent/001_mask.png")
        assert result == expected

    def test_extension_preserved(self):
        result = build_gt_mask_path(
            image_path="/ds/test/crack/img.jpg",
            dataset_root="/ds",
        )
        assert result.suffix == ".jpg"
        assert result.name == "img_mask.jpg"

    def test_returns_path_object(self):
        result = build_gt_mask_path(
            image_path="test/good/000.png",
            dataset_root=".",
        )
        assert isinstance(result, Path)

    def test_stem_with_multiple_underscores(self):
        result = build_gt_mask_path(
            image_path="/ds/test/scratch/my_file_001.png",
            dataset_root="/ds",
        )
        assert result.name == "my_file_001_mask.png"

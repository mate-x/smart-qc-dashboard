from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from utils.image_utils import (
    apply_clahe,
    apply_he,
    apply_homomorphic,
    ensure_rgb,
    resize_with_padding,
)


@pytest.fixture
def rgb_array() -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


@pytest.fixture
def rgb_image(rgb_array) -> Image.Image:
    return Image.fromarray(rgb_array)


class TestResizeWithPadding:
    def test_numpy_input_returns_correct_shape(self, rgb_array):
        result = resize_with_padding(rgb_array, 128)
        assert result.shape == (128, 128, 3)

    def test_pil_input_returns_correct_shape(self, rgb_image):
        result = resize_with_padding(rgb_image, 256)
        assert result.shape == (256, 256, 3)

    def test_returns_numpy_array(self, rgb_array):
        result = resize_with_padding(rgb_array, 64)
        assert isinstance(result, np.ndarray)

    def test_non_square_input_preserved_ratio_within_canvas(self):
        arr = np.zeros((32, 64, 3), dtype=np.uint8)
        result = resize_with_padding(arr, 128)
        assert result.shape == (128, 128, 3)


class TestEnsureRgb:
    def test_rgb_image_unchanged_shape(self, rgb_image):
        result = ensure_rgb(rgb_image)
        assert result.shape == (64, 64, 3)
        assert isinstance(result, np.ndarray)

    def test_grayscale_to_rgb(self):
        gray = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
        result = ensure_rgb(gray)
        assert result.shape == (32, 32, 3)

    def test_rgba_to_rgb(self):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba = Image.fromarray(arr, mode="RGBA")
        result = ensure_rgb(rgba)
        assert result.shape == (16, 16, 3)


class TestApplyHe:
    def test_output_shape_preserved(self, rgb_array):
        result = apply_he(rgb_array)
        assert result.shape == rgb_array.shape

    def test_output_dtype_uint8(self, rgb_array):
        result = apply_he(rgb_array)
        assert result.dtype == np.uint8


class TestApplyClahe:
    def test_output_shape_preserved(self, rgb_array):
        result = apply_clahe(rgb_array)
        assert result.shape == rgb_array.shape

    def test_custom_clip_limit(self, rgb_array):
        result = apply_clahe(rgb_array, clip_limit=4.0)
        assert result.shape == rgb_array.shape

    def test_output_dtype_uint8(self, rgb_array):
        result = apply_clahe(rgb_array)
        assert result.dtype == np.uint8


class TestApplyHomomorphic:
    def test_output_shape_preserved(self, rgb_array):
        result = apply_homomorphic(rgb_array)
        assert result.shape == rgb_array.shape

    def test_output_dtype_uint8(self, rgb_array):
        result = apply_homomorphic(rgb_array)
        assert result.dtype == np.uint8

    def test_cutoff_param_accepted(self, rgb_array):
        result = apply_homomorphic(rgb_array, cutoff=20.0)
        assert result.shape == rgb_array.shape

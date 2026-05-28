"""
탭2 순수 함수 단위 테스트.
Streamlit context가 필요 없는 _build_config, _parse_float_list, _snap_image_size
와 매핑 상수를 직접 검증한다.
"""

from __future__ import annotations

import pytest

from tabs.tab2_config import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _METHOD_MAP,
    _REVERSE_METHOD_MAP,
    _build_preprocessing_config as _build_config,
    _parse_float_list,
    _snap_image_size,
)


# ---------------------------------------------------------------------------
# _METHOD_MAP / _REVERSE_METHOD_MAP 테스트
# ---------------------------------------------------------------------------

class TestMethodMapping:
    def test_all_labels_present(self) -> None:
        for label in ["없음", "Homomorphic", "HE", "CLAHE"]:
            assert label in _METHOD_MAP

    def test_none_mapping(self) -> None:
        assert _METHOD_MAP["없음"] == "none"

    def test_homomorphic_mapping(self) -> None:
        assert _METHOD_MAP["Homomorphic"] == "homomorphic"

    def test_he_mapping(self) -> None:
        assert _METHOD_MAP["HE"] == "he"

    def test_clahe_mapping(self) -> None:
        assert _METHOD_MAP["CLAHE"] == "clahe"

    def test_reverse_map_roundtrip(self) -> None:
        for label, method in _METHOD_MAP.items():
            assert _REVERSE_METHOD_MAP[method] == label

    def test_reverse_map_covers_all_methods(self) -> None:
        for method in ["none", "homomorphic", "he", "clahe"]:
            assert method in _REVERSE_METHOD_MAP


# ---------------------------------------------------------------------------
# _build_config 테스트
# ---------------------------------------------------------------------------

class TestBuildConfig:
    # ── 스키마 구조 ──────────────────────────────────────────────────────────

    def test_schema_keys(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert set(config.keys()) == {
            "method", "resize_mode", "image_size", "normalization", "mean", "std", "params"
        }

    def test_resize_mode_always_padding(self) -> None:
        for method in ["none", "homomorphic", "he", "clahe"]:
            config = _build_config(method, None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
            assert config["resize_mode"] == "padding"

    # ── method ───────────────────────────────────────────────────────────────

    def test_method_none(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["method"] == "none"

    def test_method_homomorphic(self) -> None:
        params = {"sigma": 10.0, "gamma_H": 1.5, "gamma_L": 0.5, "normalize": True}
        config = _build_config("homomorphic", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["method"] == "homomorphic"

    def test_method_he(self) -> None:
        config = _build_config("he", {}, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["method"] == "he"

    def test_method_clahe(self) -> None:
        params = {"clip_limit": 2.0}
        config = _build_config("clahe", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["method"] == "clahe"

    # ── image_size ───────────────────────────────────────────────────────────

    def test_image_size_256(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["image_size"] == 256

    def test_image_size_512(self) -> None:
        config = _build_config("none", None, 512, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["image_size"] == 512

    def test_image_size_32(self) -> None:
        config = _build_config("none", None, 32, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["image_size"] == 32

    # ── normalization ────────────────────────────────────────────────────────

    def test_imagenet_normalization_label(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["normalization"] == "imagenet"

    def test_custom_normalization_label(self) -> None:
        config = _build_config("none", None, 256, "커스텀", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert config["normalization"] == "custom"

    def test_imagenet_mean_values(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["mean"] == [0.485, 0.456, 0.406]

    def test_imagenet_std_values(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["std"] == [0.229, 0.224, 0.225]

    def test_custom_mean_stored(self) -> None:
        mean = [0.1, 0.2, 0.3]
        config = _build_config("none", None, 256, "커스텀", mean, [0.5, 0.5, 0.5])
        assert config["mean"] == [0.1, 0.2, 0.3]

    def test_custom_std_stored(self) -> None:
        std = [0.4, 0.5, 0.6]
        config = _build_config("none", None, 256, "커스텀", [0.5, 0.5, 0.5], std)
        assert config["std"] == [0.4, 0.5, 0.6]

    # ── params ───────────────────────────────────────────────────────────────

    def test_none_params_stored_as_none(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["params"] is None

    def test_empty_dict_params_becomes_none(self) -> None:
        # HE는 파라미터 없음 → {} → None으로 저장
        config = _build_config("he", {}, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["params"] is None

    def test_homomorphic_params_stored(self) -> None:
        params = {
            "sigma": 5.0,
            "gamma_H": 2.0,
            "gamma_L": 0.3,
            "normalize": False,
        }
        config = _build_config("homomorphic", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["params"] == params

    def test_homomorphic_normalize_false_preserved(self) -> None:
        params = {"sigma": 10.0, "gamma_H": 1.5, "gamma_L": 0.5, "normalize": False}
        config = _build_config("homomorphic", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["params"]["normalize"] is False

    def test_clahe_params_stored(self) -> None:
        params = {"clip_limit": 4.0}
        config = _build_config("clahe", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["params"] == {"clip_limit": 4.0}

    def test_clahe_clip_limit_value(self) -> None:
        params = {"clip_limit": 8.5}
        config = _build_config("clahe", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["params"]["clip_limit"] == 8.5

    # ── 복합 시나리오 ────────────────────────────────────────────────────────

    def test_full_homomorphic_config(self) -> None:
        params = {"sigma": 10.0, "gamma_H": 1.5, "gamma_L": 0.5, "normalize": True}
        config = _build_config("homomorphic", params, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        assert config["method"] == "homomorphic"
        assert config["resize_mode"] == "padding"
        assert config["image_size"] == 256
        assert config["normalization"] == "imagenet"
        assert config["params"]["sigma"] == 10.0
        assert config["params"]["normalize"] is True

    def test_full_custom_norm_config(self) -> None:
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        config = _build_config("clahe", {"clip_limit": 2.0}, 128, "커스텀", mean, std)
        assert config["normalization"] == "custom"
        assert config["mean"] == [0.5, 0.5, 0.5]
        assert config["std"] == [0.25, 0.25, 0.25]
        assert config["image_size"] == 128

    # ── 리스트 복사 (aliasing 방지) ──────────────────────────────────────────

    def test_returned_mean_is_copy(self) -> None:
        mean = [0.5, 0.5, 0.5]
        config = _build_config("none", None, 256, "커스텀", mean, [0.1, 0.1, 0.1])
        config["mean"][0] = 99.0
        assert mean[0] == 0.5  # 원본 미변경

    def test_returned_std_is_copy(self) -> None:
        std = [0.1, 0.2, 0.3]
        config = _build_config("none", None, 256, "커스텀", [0.5, 0.5, 0.5], std)
        config["std"][0] = 99.0
        assert std[0] == 0.1  # 원본 미변경

    def test_imagenet_mean_not_aliased(self) -> None:
        config = _build_config("none", None, 256, "ImageNet", _IMAGENET_MEAN, _IMAGENET_STD)
        config["mean"][0] = 99.0
        assert _IMAGENET_MEAN[0] == pytest.approx(0.485)  # 모듈 상수 미변경


# ---------------------------------------------------------------------------
# _parse_float_list 테스트
# ---------------------------------------------------------------------------

class TestParseFloatList:
    def test_valid_three_elements(self) -> None:
        assert _parse_float_list("0.5,0.5,0.5") == [0.5, 0.5, 0.5]

    def test_valid_with_spaces(self) -> None:
        assert _parse_float_list(" 0.1 , 0.2 , 0.3 ") == [0.1, 0.2, 0.3]

    def test_valid_integers(self) -> None:
        assert _parse_float_list("1,2,3") == [1.0, 2.0, 3.0]

    def test_valid_one_element(self) -> None:
        result = _parse_float_list("0.5")
        assert result == [0.5]  # 길이 검사는 호출자 책임

    def test_valid_four_elements(self) -> None:
        result = _parse_float_list("0.1,0.2,0.3,0.4")
        assert result == [0.1, 0.2, 0.3, 0.4]  # 길이 검사는 호출자 책임

    def test_empty_string_returns_none(self) -> None:
        assert _parse_float_list("") is None

    def test_non_numeric_returns_none(self) -> None:
        assert _parse_float_list("a,b,c") is None

    def test_mixed_valid_invalid_returns_none(self) -> None:
        assert _parse_float_list("0.5,abc,0.5") is None

    def test_empty_slot_returns_none(self) -> None:
        assert _parse_float_list("0.5,,0.5") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _parse_float_list("   ") is None

    def test_imagenet_values_roundtrip(self) -> None:
        text = ",".join(str(v) for v in _IMAGENET_MEAN)
        assert _parse_float_list(text) == pytest.approx(_IMAGENET_MEAN)


# ---------------------------------------------------------------------------
# _snap_image_size 테스트
# ---------------------------------------------------------------------------

class TestSnapImageSize:
    def test_exact_multiple_256(self) -> None:
        assert _snap_image_size(256) == 256

    def test_exact_multiple_32(self) -> None:
        assert _snap_image_size(32) == 32

    def test_exact_multiple_1024(self) -> None:
        assert _snap_image_size(1024) == 1024

    def test_round_down(self) -> None:
        # 100 → round(100/32)=round(3.125)=3, 3*32=96
        assert _snap_image_size(100) == 96

    def test_round_down_200(self) -> None:
        # 200 → round(200/32)=round(6.25)=6, 6*32=192
        assert _snap_image_size(200) == 192

    def test_below_min_clamped(self) -> None:
        # 0 → 0*32=0 → clamp to 32
        assert _snap_image_size(0) == 32

    def test_banker_round_half_to_even_low(self) -> None:
        # 16 → round(0.5)=0 (banker: 0 is even) → 0 → clamp to 32
        assert _snap_image_size(16) == 32

    def test_banker_round_half_to_even_high(self) -> None:
        # 48 → round(1.5)=2 (banker: 2 is even) → 2*32=64
        assert _snap_image_size(48) == 64

    def test_above_max_clamped(self) -> None:
        # 2048 → 64*32=2048 → clamp to 1024
        assert _snap_image_size(2048) == 1024

    def test_just_above_max_clamped(self) -> None:
        # 1500 → round(46.875)=47, 47*32=1504 → clamp to 1024
        assert _snap_image_size(1500) == 1024

from __future__ import annotations

import pytest

from utils.config_manager import (
    ConfigLoadError,
    get_model_config,
    get_preprocessing_config,
    load_config,
    save_config_section,
)


@pytest.fixture
def config_path(tmp_path):
    return str(tmp_path / "configs.yaml")


class TestLoadConfig:
    def test_returns_empty_dict_for_missing_file(self, config_path):
        assert load_config(config_path) == {}

    def test_invalid_yaml_raises_config_load_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigLoadError):
            load_config(str(bad))


class TestSaveConfigSection:
    def test_save_and_load_roundtrip(self, config_path):
        data = {"image_size": 256, "method": "clahe"}
        save_config_section("preprocessing", data, config_path)
        loaded = load_config(config_path)
        assert loaded["preprocessing"] == data

    def test_multiple_sections_preserved(self, config_path):
        save_config_section("preprocessing", {"image_size": 64}, config_path)
        save_config_section("model", {"model_type": "patchcore"}, config_path)
        loaded = load_config(config_path)
        assert "preprocessing" in loaded
        assert "model" in loaded

    def test_atomic_write_no_tmp_after_save(self, tmp_path, config_path):
        save_config_section("model", {"key": "val"}, config_path)
        tmp = (tmp_path / "configs.yaml").with_suffix(".tmp")
        assert not tmp.exists()

    def test_overwrite_section(self, config_path):
        save_config_section("preprocessing", {"image_size": 64}, config_path)
        save_config_section("preprocessing", {"image_size": 128}, config_path)
        loaded = load_config(config_path)
        assert loaded["preprocessing"]["image_size"] == 128


class TestGetConfigHelpers:
    def test_get_preprocessing_config(self, config_path):
        expected = {"method": "he", "image_size": 256}
        save_config_section("preprocessing", expected, config_path)
        assert get_preprocessing_config(config_path) == expected

    def test_get_preprocessing_config_returns_none_if_missing(self, config_path):
        assert get_preprocessing_config(config_path) is None

    def test_get_model_config(self, config_path):
        expected = {"model_type": "efficientad"}
        save_config_section("model", expected, config_path)
        assert get_model_config(config_path) == expected

    def test_get_model_config_returns_none_if_missing(self, config_path):
        assert get_model_config(config_path) is None

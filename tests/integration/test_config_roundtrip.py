from __future__ import annotations

import pytest

from utils.config_manager import (
    get_model_config,
    get_preprocessing_config,
    load_config,
    save_config_section,
)


@pytest.fixture
def cfg(tmp_path):
    return str(tmp_path / "configs.yaml")


class TestPreprocessingConfigRoundtrip:
    def test_full_roundtrip(self, cfg, minimal_preprocessing_config):
        save_config_section("preprocessing", minimal_preprocessing_config, cfg)
        result = get_preprocessing_config(cfg)
        assert result == minimal_preprocessing_config

    def test_image_size_preserved(self, cfg):
        save_config_section("preprocessing", {"image_size": 224, "method": "clahe"}, cfg)
        assert get_preprocessing_config(cfg)["image_size"] == 224

    def test_nested_params_preserved(self, cfg):
        data = {"method": "homomorphic", "params": {"sigma": 15.0, "gamma_H": 2.0}}
        save_config_section("preprocessing", data, cfg)
        loaded = get_preprocessing_config(cfg)
        assert loaded["params"]["sigma"] == 15.0


class TestModelConfigRoundtrip:
    def test_full_roundtrip(self, cfg, minimal_model_config):
        save_config_section("model", minimal_model_config, cfg)
        result = get_model_config(cfg)
        assert result == minimal_model_config

    def test_both_sections_coexist(self, cfg, minimal_preprocessing_config, minimal_model_config):
        save_config_section("preprocessing", minimal_preprocessing_config, cfg)
        save_config_section("model", minimal_model_config, cfg)
        full = load_config(cfg)
        assert "preprocessing" in full
        assert "model" in full

    def test_update_model_does_not_clobber_preprocessing(self, cfg):
        save_config_section("preprocessing", {"method": "he"}, cfg)
        save_config_section("model", {"model_type": "patchcore"}, cfg)
        save_config_section("model", {"model_type": "efficientad"}, cfg)
        assert get_preprocessing_config(cfg) == {"method": "he"}
        assert get_model_config(cfg)["model_type"] == "efficientad"

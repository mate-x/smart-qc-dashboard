"""
탭2 EfficientAD 파트 단위 테스트

PRD 03_Functional_Requirements.md FR-T2-03, FR-T2-04 관련.
순수 함수만 테스트 (Streamlit context 불필요).
"""

from __future__ import annotations

import pytest

from tabs.tab2_config import (
    build_efficientad_params,
    build_model_config,
    compute_threshold_ratio,
    _apply_efficientad_widgets,
)

# ─────────────────────────────────────────────
# 상수 — PRD §1.4 EfficientAD model_params 필수 키
# ─────────────────────────────────────────────

EFFICIENTAD_REQUIRED_KEYS = {
    "model_size",
    "train_steps",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "out_channels",
    "padding",
    "ae_loss_weight",
    "autoencoder_lr",
    "autoencoder_weight_decay",
    "lr_decay_epochs",
    "lr_decay_factor",
    "scheduler",
    "use_imagenet_penalty",
    "penalty_batch_size",
}


def _default_ead_params(**overrides) -> dict:
    defaults = dict(
        model_size="medium",
        train_steps=70_000,
        optimizer="adam",
        learning_rate=1e-4,
        weight_decay=1e-4,
        out_channels=384,
        padding=False,
        ae_loss_weight=0.5,
        autoencoder_lr=1e-4,
        autoencoder_weight_decay=1e-5,
        lr_decay_epochs=50_000,
        lr_decay_factor=0.1,
        scheduler="StepLR",
        use_imagenet_penalty=False,
        penalty_batch_size=8,
    )
    defaults.update(overrides)
    return build_efficientad_params(**defaults)


# ─────────────────────────────────────────────
# build_efficientad_params — PRD §1.4 스키마 검증
# ─────────────────────────────────────────────

class TestBuildEfficientadParams:
    def test_required_keys_present(self):
        params = _default_ead_params()
        assert EFFICIENTAD_REQUIRED_KEYS == set(params.keys())

    def test_default_model_size_medium(self):
        params = _default_ead_params()
        assert params["model_size"] == "medium"

    @pytest.mark.parametrize("size", ["small", "medium"])
    def test_valid_model_sizes(self, size):
        params = _default_ead_params(model_size=size)
        assert params["model_size"] == size

    def test_train_steps_is_int(self):
        params = _default_ead_params(train_steps=70_000)
        assert isinstance(params["train_steps"], int)
        assert params["train_steps"] == 70_000

    @pytest.mark.parametrize("opt", ["adam", "adamw", "sgd"])
    def test_valid_optimizers(self, opt):
        params = _default_ead_params(optimizer=opt)
        assert params["optimizer"] == opt

    def test_learning_rate_is_float(self):
        params = _default_ead_params(learning_rate=1e-4)
        assert isinstance(params["learning_rate"], float)
        assert params["learning_rate"] == pytest.approx(1e-4)

    def test_weight_decay_is_float(self):
        params = _default_ead_params(weight_decay=1e-4)
        assert isinstance(params["weight_decay"], float)

    def test_out_channels_is_int(self):
        for ch in (128, 256, 384, 512):
            params = _default_ead_params(out_channels=ch)
            assert isinstance(params["out_channels"], int)
            assert params["out_channels"] == ch

    def test_padding_is_bool(self):
        for pad in (True, False):
            params = _default_ead_params(padding=pad)
            assert isinstance(params["padding"], bool)
            assert params["padding"] == pad

    def test_ae_loss_weight_is_float(self):
        params = _default_ead_params(ae_loss_weight=0.5)
        assert isinstance(params["ae_loss_weight"], float)
        assert params["ae_loss_weight"] == pytest.approx(0.5)

    def test_ae_loss_weight_zero(self):
        params = _default_ead_params(ae_loss_weight=0.0)
        assert params["ae_loss_weight"] == pytest.approx(0.0)

    def test_ae_loss_weight_one(self):
        params = _default_ead_params(ae_loss_weight=1.0)
        assert params["ae_loss_weight"] == pytest.approx(1.0)

    def test_autoencoder_lr_is_float(self):
        params = _default_ead_params(autoencoder_lr=1e-4)
        assert isinstance(params["autoencoder_lr"], float)

    def test_autoencoder_weight_decay_is_float(self):
        params = _default_ead_params(autoencoder_weight_decay=1e-5)
        assert isinstance(params["autoencoder_weight_decay"], float)
        assert params["autoencoder_weight_decay"] == pytest.approx(1e-5)

    def test_lr_decay_epochs_is_int(self):
        params = _default_ead_params(lr_decay_epochs=50_000)
        assert isinstance(params["lr_decay_epochs"], int)
        assert params["lr_decay_epochs"] == 50_000

    def test_lr_decay_factor_is_float(self):
        params = _default_ead_params(lr_decay_factor=0.1)
        assert isinstance(params["lr_decay_factor"], float)
        assert params["lr_decay_factor"] == pytest.approx(0.1)

    @pytest.mark.parametrize("sched", ["StepLR", "CosineAnnealingLR"])
    def test_valid_schedulers(self, sched):
        params = _default_ead_params(scheduler=sched)
        assert params["scheduler"] == sched

    def test_use_imagenet_penalty_is_bool(self):
        for val in (True, False):
            params = _default_ead_params(use_imagenet_penalty=val)
            assert isinstance(params["use_imagenet_penalty"], bool)
            assert params["use_imagenet_penalty"] is val

    def test_penalty_batch_size_is_int(self):
        params = _default_ead_params(penalty_batch_size=8)
        assert isinstance(params["penalty_batch_size"], int)
        assert params["penalty_batch_size"] == 8

    def test_prd_defaults(self):
        """PRD §1.4 기본값 일괄 검증."""
        params = _default_ead_params()
        assert params["model_size"] == "medium"
        assert params["train_steps"] == 70_000
        assert params["optimizer"] == "adam"
        assert params["learning_rate"] == pytest.approx(1e-4)
        assert params["weight_decay"] == pytest.approx(1e-4)
        assert params["out_channels"] == 384
        assert params["padding"] is False
        assert params["ae_loss_weight"] == pytest.approx(0.5)
        assert params["autoencoder_lr"] == pytest.approx(1e-4)
        assert params["autoencoder_weight_decay"] == pytest.approx(1e-5)
        assert params["lr_decay_epochs"] == 50_000
        assert params["lr_decay_factor"] == pytest.approx(0.1)
        assert params["scheduler"] == "StepLR"
        assert params["use_imagenet_penalty"] is False
        assert params["penalty_batch_size"] == 8

    def test_tc_fr_t2_03_ae_0_73(self):
        """PRD TC-FR-T2-03: ae=0.73 → ae_loss_weight 저장 검증."""
        params = _default_ead_params(ae_loss_weight=0.73)
        assert params["ae_loss_weight"] == pytest.approx(0.73)


# ─────────────────────────────────────────────
# _apply_efficientad_widgets — 위젯 프리필 키 매핑 검증
# ─────────────────────────────────────────────

class TestApplyEfficientadWidgets:
    """configs.yaml 불러오기 후 session_state 위젯 키 설정 검증."""

    def _run(self, params: dict) -> dict:
        """session_state를 일반 dict로 대체하여 Streamlit 없이 테스트."""
        import tabs.tab2_config as mod
        fake_ss: dict = {}
        original_ss = getattr(mod.st, "session_state", None)
        mod.st.session_state = fake_ss  # type: ignore[attr-defined]
        try:
            _apply_efficientad_widgets(params)
        finally:
            if original_ss is not None:
                mod.st.session_state = original_ss
        return fake_ss

    def test_model_size_key_set(self):
        ss = self._run({"model_size": "small"})
        assert ss["tab2_ead_model_size"] == "small"

    def test_train_steps_key_set(self):
        ss = self._run({"train_steps": 50_000})
        assert ss["tab2_ead_train_steps"] == 50_000

    def test_optimizer_key_set(self):
        ss = self._run({"optimizer": "adamw"})
        assert ss["tab2_ead_optimizer"] == "adamw"

    def test_learning_rate_key_set(self):
        ss = self._run({"learning_rate": 2e-4})
        assert ss["tab2_ead_lr"] == pytest.approx(2e-4)

    def test_weight_decay_key_set(self):
        ss = self._run({"weight_decay": 1e-3})
        assert ss["tab2_ead_wd"] == pytest.approx(1e-3)

    def test_out_channels_key_set(self):
        ss = self._run({"out_channels": 256})
        assert ss["tab2_ead_out_channels"] == 256

    def test_padding_key_set_true(self):
        ss = self._run({"padding": True})
        assert ss["tab2_ead_padding"] is True

    def test_padding_key_set_false(self):
        ss = self._run({"padding": False})
        assert ss["tab2_ead_padding"] is False

    def test_ae_loss_weight_key_set(self):
        ss = self._run({"ae_loss_weight": 0.7})
        assert ss["tab2_ead_ae_weight"] == pytest.approx(0.7)

    def test_autoencoder_lr_key_set(self):
        ss = self._run({"autoencoder_lr": 5e-4})
        assert ss["tab2_ead_ae_lr"] == pytest.approx(5e-4)

    def test_autoencoder_weight_decay_key_set(self):
        ss = self._run({"autoencoder_weight_decay": 1e-5})
        assert ss["tab2_ead_ae_wd"] == pytest.approx(1e-5)

    def test_lr_decay_epochs_key_set(self):
        ss = self._run({"lr_decay_epochs": 40_000})
        assert ss["tab2_ead_decay_ep"] == 40_000

    def test_lr_decay_factor_key_set(self):
        ss = self._run({"lr_decay_factor": 0.5})
        assert ss["tab2_ead_decay_f"] == pytest.approx(0.5)

    def test_scheduler_key_set(self):
        ss = self._run({"scheduler": "CosineAnnealingLR"})
        assert ss["tab2_ead_sched"] == "CosineAnnealingLR"

    def test_use_imagenet_penalty_key_set(self):
        ss = self._run({"use_imagenet_penalty": True})
        assert ss["tab2_ead_use_penalty"] is True

    def test_penalty_batch_size_key_set(self):
        ss = self._run({"penalty_batch_size": 16})
        assert ss["tab2_ead_pen_bs"] == 16

    def test_empty_params_no_error(self):
        """파라미터 없이 호출해도 예외 없음."""
        self._run({})

    def test_empty_params_no_keys_set(self):
        """파라미터 없으면 위젯 키가 하나도 설정되지 않는다."""
        ss = self._run({})
        assert ss == {}

    def test_partial_params_only_present_keys_set(self):
        """일부 파라미터만 있을 때 해당 키만 설정된다."""
        ss = self._run({"model_size": "small", "train_steps": 30_000})
        assert "tab2_ead_model_size" in ss
        assert "tab2_ead_train_steps" in ss
        assert "tab2_ead_optimizer" not in ss
        assert "tab2_ead_lr" not in ss


# ─────────────────────────────────────────────
# build_model_config — EfficientAD params 사용 시 model_config 구조 검증
# ─────────────────────────────────────────────

class TestBuildModelConfigEfficientad:
    """build_model_config — EfficientAD params 조합 시 00_Global_Context 1.7절 스키마 검증."""

    _MODEL_CONFIG_KEYS = {
        "model_type", "image_size", "batch_size", "random_seed",
        "threshold_method", "threshold_value", "params",
    }

    def _make(self, **overrides) -> dict:
        kwargs = dict(
            model_type="efficientad",
            image_size=256,
            batch_size=16,
            random_seed=42,
            threshold_method="percentile",
            threshold_value=95.0,
            params=_default_ead_params(),
        )
        kwargs.update(overrides)
        return build_model_config(**kwargs)

    def test_required_keys(self):
        cfg = self._make()
        assert self._MODEL_CONFIG_KEYS == set(cfg.keys())

    def test_model_type_is_efficientad(self):
        cfg = self._make()
        assert cfg["model_type"] == "efficientad"

    def test_params_contains_efficientad_keys(self):
        cfg = self._make()
        assert EFFICIENTAD_REQUIRED_KEYS == set(cfg["params"].keys())

    def test_image_size_is_int(self):
        cfg = self._make(image_size=320)
        assert isinstance(cfg["image_size"], int)
        assert cfg["image_size"] == 320

    def test_batch_size_is_int(self):
        cfg = self._make(batch_size=32)
        assert isinstance(cfg["batch_size"], int)
        assert cfg["batch_size"] == 32

    def test_random_seed_is_int(self):
        cfg = self._make(random_seed=0)
        assert isinstance(cfg["random_seed"], int)
        assert cfg["random_seed"] == 0

    @pytest.mark.parametrize("method", ["percentile", "absolute"])
    def test_threshold_methods(self, method):
        cfg = self._make(threshold_method=method)
        assert cfg["threshold_method"] == method

    def test_threshold_value_is_float(self):
        cfg = self._make(threshold_value=90.0)
        assert isinstance(cfg["threshold_value"], float)
        assert cfg["threshold_value"] == pytest.approx(90.0)

    def test_params_is_dict(self):
        cfg = self._make()
        assert isinstance(cfg["params"], dict)

    def test_prd_defaults_roundtrip(self):
        """PRD 기본값으로 build_model_config → params 필드 일치."""
        cfg = self._make()
        p = cfg["params"]
        assert p["model_size"] == "medium"
        assert p["train_steps"] == 70_000
        assert p["out_channels"] == 384
        assert p["padding"] is False
        assert p["ae_loss_weight"] == pytest.approx(0.5)


# ─────────────────────────────────────────────
# compute_threshold_ratio — FR-T2-10 검증
# ─────────────────────────────────────────────

class TestComputeThresholdRatio:
    """compute_threshold_ratio — FR-T2-10 정상/결함 비율 계산 검증."""

    def test_percentile_95(self):
        normal, defect = compute_threshold_ratio("percentile", 95.0)
        assert normal == pytest.approx(0.95)
        assert defect == pytest.approx(0.05)

    def test_percentile_0(self):
        normal, defect = compute_threshold_ratio("percentile", 0.0)
        assert normal == pytest.approx(0.0)
        assert defect == pytest.approx(1.0)

    def test_percentile_100(self):
        normal, defect = compute_threshold_ratio("percentile", 100.0)
        assert normal == pytest.approx(1.0)
        assert defect == pytest.approx(0.0)

    def test_percentile_sum_equals_one(self):
        for pct in (10.0, 30.0, 50.0, 70.0, 90.0, 95.0, 99.0):
            normal, defect = compute_threshold_ratio("percentile", pct)
            assert abs(normal + defect - 1.0) < 1e-6, f"sum != 1.0 for pct={pct}"

    def test_percentile_returns_floats(self):
        normal, defect = compute_threshold_ratio("percentile", 80.0)
        assert isinstance(normal, float)
        assert isinstance(defect, float)

    def test_absolute_returns_none_none(self):
        normal, defect = compute_threshold_ratio("absolute", 0.5)
        assert normal is None
        assert defect is None

    def test_absolute_any_value_returns_none(self):
        for val in (0.0, 0.1, 0.5, 0.9, 1.0):
            n, d = compute_threshold_ratio("absolute", val)
            assert (n, d) == (None, None), f"expected (None, None) for val={val}"

    def test_tc_fr_t2_10_percentile_50(self):
        """PRD TC-FR-T2-10: percentile=50 → normal=0.5, defect=0.5."""
        normal, defect = compute_threshold_ratio("percentile", 50.0)
        assert normal == pytest.approx(0.5)
        assert defect == pytest.approx(0.5)

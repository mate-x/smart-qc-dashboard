"""
탭3 PatchCore 파트 단위 테스트

PRD 03_Functional_Requirements.md H.2 TC-FR-T3-03 포함.
순수 함수만 테스트 (Streamlit context 불필요).
"""

from __future__ import annotations

import pytest

from tabs.tab3_model_params import (
    build_model_config,
    build_patchcore_params,
    compute_st_loss_weight,
    compute_threshold_ratio,
    _apply_patchcore_widgets,
)


# ─────────────────────────────────────────────
# compute_st_loss_weight — ae/st weight 자동 보정 (R-03)
# ─────────────────────────────────────────────

class TestComputeStLossWeight:
    """TC-FR-T3-03: ae_loss_weight 변경 시 st_loss_weight 자동 보정."""

    def test_default_half(self):
        assert compute_st_loss_weight(0.5) == 0.5

    def test_sum_equals_one(self):
        for ae in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            st_w = compute_st_loss_weight(ae)
            assert abs(ae + st_w - 1.0) < 1e-6, f"sum != 1.0 for ae={ae}"

    def test_tc_fr_t3_03_0_73(self):
        """PRD TC-FR-T3-03: ae=0.73 → st=0.27."""
        ae = 0.73
        st_w = compute_st_loss_weight(ae)
        assert st_w == round(1.0 - 0.73, 6)
        assert abs(ae + st_w - 1.0) < 1e-6

    def test_edge_zero(self):
        assert compute_st_loss_weight(0.0) == 1.0

    def test_edge_one(self):
        assert compute_st_loss_weight(1.0) == 0.0

    def test_rounded_to_6_decimals(self):
        val = compute_st_loss_weight(0.333333)
        assert val == round(val, 6)


# ─────────────────────────────────────────────
# build_patchcore_params — 00_Global_Context 1.4절 스키마 검증
# ─────────────────────────────────────────────

PATCHCORE_REQUIRED_KEYS = {
    "backbone",
    "pretrained_source",
    "pretrained_path",
    "coreset_sampling_ratio",
    "neighbourhood_kernel_size",
    "max_train",
    "knn",
    "top_k_ratio",
}


def _default_patchcore_params(**overrides) -> dict:
    defaults = dict(
        backbone="wide_resnet50_2",
        pretrained_source="torchvision",
        pretrained_path=None,
        coreset_sampling_ratio=0.1,
        neighbourhood_kernel_size=3,
        max_train=1000,
        knn=9,
        top_k_ratio=0.1,
    )
    defaults.update(overrides)
    return build_patchcore_params(**defaults)


class TestBuildPatchcoreParams:
    def test_required_keys_present(self):
        params = _default_patchcore_params()
        assert PATCHCORE_REQUIRED_KEYS == set(params.keys())

    def test_default_backbone(self):
        params = _default_patchcore_params()
        assert params["backbone"] == "wide_resnet50_2"

    def test_pretrained_source_torchvision(self):
        params = _default_patchcore_params(pretrained_source="torchvision")
        assert params["pretrained_source"] == "torchvision"
        assert params["pretrained_path"] is None

    def test_pretrained_source_local(self):
        params = _default_patchcore_params(
            pretrained_source="local",
            pretrained_path="/models/resnet18.pth",
        )
        assert params["pretrained_source"] == "local"
        assert params["pretrained_path"] == "/models/resnet18.pth"

    def test_coreset_sampling_ratio_is_float(self):
        params = _default_patchcore_params(coreset_sampling_ratio=0.2)
        assert isinstance(params["coreset_sampling_ratio"], float)
        assert params["coreset_sampling_ratio"] == 0.2

    def test_neighbourhood_kernel_size_is_int(self):
        for ks in (1, 3, 5, 7, 9):
            params = _default_patchcore_params(neighbourhood_kernel_size=ks)
            assert isinstance(params["neighbourhood_kernel_size"], int)
            assert params["neighbourhood_kernel_size"] == ks

    def test_max_train_is_int(self):
        params = _default_patchcore_params(max_train=500)
        assert isinstance(params["max_train"], int)
        assert params["max_train"] == 500

    def test_knn_is_int(self):
        params = _default_patchcore_params(knn=5)
        assert isinstance(params["knn"], int)
        assert params["knn"] == 5

    def test_top_k_ratio_is_float(self):
        params = _default_patchcore_params(top_k_ratio=0.2)
        assert isinstance(params["top_k_ratio"], float)
        assert params["top_k_ratio"] == 0.2

    @pytest.mark.parametrize("backbone", ["wide_resnet50_2", "resnet18", "resnet50"])
    def test_valid_backbones(self, backbone):
        params = _default_patchcore_params(backbone=backbone)
        assert params["backbone"] == backbone

    def test_coreset_ratio_min(self):
        params = _default_patchcore_params(coreset_sampling_ratio=0.01)
        assert params["coreset_sampling_ratio"] == pytest.approx(0.01)

    def test_coreset_ratio_max(self):
        params = _default_patchcore_params(coreset_sampling_ratio=1.0)
        assert params["coreset_sampling_ratio"] == pytest.approx(1.0)


# ─────────────────────────────────────────────
# build_model_config — 00_Global_Context 1.7절 스키마 검증
# ─────────────────────────────────────────────

MODEL_CONFIG_REQUIRED_KEYS = {
    "model_type",
    "image_size",
    "batch_size",
    "random_seed",
    "threshold_method",
    "threshold_value",
    "params",
}


class TestBuildModelConfig:
    def _make(self, **overrides) -> dict:
        defaults = dict(
            model_type="patchcore",
            image_size=256,
            batch_size=16,
            random_seed=42,
            threshold_method="percentile",
            threshold_value=95.0,
            params=_default_patchcore_params(),
        )
        defaults.update(overrides)
        return build_model_config(**defaults)

    def test_required_keys_present(self):
        cfg = self._make()
        assert MODEL_CONFIG_REQUIRED_KEYS == set(cfg.keys())

    def test_model_type_patchcore(self):
        cfg = self._make(model_type="patchcore")
        assert cfg["model_type"] == "patchcore"

    def test_model_type_efficientad(self):
        cfg = self._make(model_type="efficientad")
        assert cfg["model_type"] == "efficientad"

    def test_image_size_is_int(self):
        cfg = self._make(image_size=256)
        assert isinstance(cfg["image_size"], int)
        assert cfg["image_size"] == 256

    def test_batch_size_is_int(self):
        cfg = self._make(batch_size=32)
        assert isinstance(cfg["batch_size"], int)

    def test_random_seed_is_int(self):
        cfg = self._make(random_seed=42)
        assert isinstance(cfg["random_seed"], int)

    def test_threshold_method_percentile(self):
        cfg = self._make(threshold_method="percentile", threshold_value=95.0)
        assert cfg["threshold_method"] == "percentile"
        assert cfg["threshold_value"] == pytest.approx(95.0)

    def test_threshold_method_absolute(self):
        cfg = self._make(threshold_method="absolute", threshold_value=0.5)
        assert cfg["threshold_method"] == "absolute"
        assert cfg["threshold_value"] == pytest.approx(0.5)

    def test_threshold_value_is_float(self):
        cfg = self._make(threshold_value=90.0)
        assert isinstance(cfg["threshold_value"], float)

    def test_params_is_dict(self):
        cfg = self._make()
        assert isinstance(cfg["params"], dict)


# ─────────────────────────────────────────────
# _apply_patchcore_widgets — 위젯 프리필 키 매핑 검증
# ─────────────────────────────────────────────

class TestApplyPatchcoreWidgets:
    """configs.yaml 불러오기 후 session_state 위젯 키 설정 검증."""

    def _run(self, params: dict) -> dict:
        """session_state를 일반 dict로 대체하여 Streamlit 없이 테스트."""
        import tabs.tab3_model_params as mod
        fake_ss: dict = {}
        original_ss = getattr(mod.st, "session_state", None)
        mod.st.session_state = fake_ss  # type: ignore[attr-defined]
        try:
            _apply_patchcore_widgets(params)
        finally:
            if original_ss is not None:
                mod.st.session_state = original_ss
        return fake_ss

    def test_backbone_key_set(self):
        ss = self._run({"backbone": "resnet18"})
        assert ss["pc_backbone"] == "resnet18"

    def test_pretrained_source_torchvision(self):
        ss = self._run({"pretrained_source": "torchvision"})
        assert ss["pc_pretrained_label"] == "torchvision"

    def test_pretrained_source_local(self):
        ss = self._run({"pretrained_source": "local"})
        assert ss["pc_pretrained_label"] == "로컬 경로"

    def test_pretrained_path_set(self):
        ss = self._run({"pretrained_source": "local", "pretrained_path": "/w/r18.pth"})
        assert ss["pc_pretrained_path"] == "/w/r18.pth"

    def test_coreset_ratio_set(self):
        ss = self._run({"coreset_sampling_ratio": 0.3})
        assert ss["pc_coreset"] == pytest.approx(0.3)

    def test_neighbourhood_kernel_odd_values(self):
        for ks in (1, 3, 5, 7, 9):
            ss = self._run({"neighbourhood_kernel_size": ks})
            assert ss["pc_kernel"] == ks

    def test_neighbourhood_kernel_even_skipped(self):
        """짝수 값은 select_slider options에 없으므로 키를 설정하지 않는다."""
        ss = self._run({"neighbourhood_kernel_size": 4})
        assert "pc_kernel" not in ss

    def test_max_train_set(self):
        ss = self._run({"max_train": 2000})
        assert ss["pc_max_train"] == 2000

    def test_knn_set(self):
        ss = self._run({"knn": 15})
        assert ss["pc_knn"] == 15

    def test_top_k_ratio_set(self):
        ss = self._run({"top_k_ratio": 0.25})
        assert ss["pc_top_k"] == pytest.approx(0.25)

    def test_empty_params_no_error(self):
        """파라미터 없이 호출해도 예외 없음."""
        self._run({})


# ─────────────────────────────────────────────
# compute_threshold_ratio — FR-T3-10 정상/결함 비율 근사치
# ─────────────────────────────────────────────

class TestComputeThresholdRatio:
    """FR-T3-10 (S): Threshold 기준 정상/결함 비율 실시간 표시 순수 함수 검증."""

    def test_percentile_95_returns_0_95(self):
        normal, defect = compute_threshold_ratio("percentile", 95.0)
        assert normal == pytest.approx(0.95)
        assert defect == pytest.approx(0.05)

    def test_percentile_sum_equals_one(self):
        for pct in (0.0, 10.0, 50.0, 90.0, 95.0, 100.0):
            normal, defect = compute_threshold_ratio("percentile", pct)
            assert normal is not None and defect is not None
            assert abs(normal + defect - 1.0) < 1e-6, f"sum != 1.0 for pct={pct}"

    def test_percentile_0_all_defect(self):
        normal, defect = compute_threshold_ratio("percentile", 0.0)
        assert normal == pytest.approx(0.0)
        assert defect == pytest.approx(1.0)

    def test_percentile_100_all_normal(self):
        normal, defect = compute_threshold_ratio("percentile", 100.0)
        assert normal == pytest.approx(1.0)
        assert defect == pytest.approx(0.0)

    def test_absolute_returns_none_none(self):
        normal, defect = compute_threshold_ratio("absolute", 0.5)
        assert normal is None
        assert defect is None

    def test_absolute_any_value_returns_none(self):
        for val in (0.0, 0.1, 0.5, 0.9, 1.0):
            normal, defect = compute_threshold_ratio("absolute", val)
            assert (normal, defect) == (None, None)

    def test_return_values_are_float(self):
        normal, defect = compute_threshold_ratio("percentile", 80.0)
        assert isinstance(normal, float)
        assert isinstance(defect, float)

    def test_rounded_to_6_decimals(self):
        normal, defect = compute_threshold_ratio("percentile", 33.3)
        assert normal == round(normal, 6)
        assert defect == round(defect, 6)

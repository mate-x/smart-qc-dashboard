from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def pandas_python_string_backend():
    """pandas 3 + PyArrow 환경에서 torch 사용 테스트 이후 string_arrow access
    violation 방지. string_storage를 "python"으로 임시 고정한다."""
    old = pd.options.mode.string_storage
    pd.options.mode.string_storage = "python"
    yield
    pd.options.mode.string_storage = old


from tabs.tab4_history import (
    _BACKBONE_ABBREV,
    _METRIC_MAP,
    _build_table_df,
    _confusion_matrix_fig,
    _param_summary,
    _roc_curve_fig,
    _score_dist_fig,
)


# ── _param_summary ────────────────────────────────────────────────────────────

class TestParamSummary:
    def _efficientad_record(self, **overrides) -> dict:
        params = {
            "model_size": "medium",
            "train_steps": 70000,
            "optimizer": "adam",
        }
        params.update(overrides)
        return {"model_type": "efficientad", "model_params": params}

    def _patchcore_record(self, **overrides) -> dict:
        params = {
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
        }
        params.update(overrides)
        return {"model_type": "patchcore", "model_params": params}

    def test_efficientad_default_format(self):
        r = self._efficientad_record()
        assert _param_summary(r) == "medium/70k/adam"

    def test_efficientad_steps_converted_to_k(self):
        r = self._efficientad_record(train_steps=50000)
        assert _param_summary(r) == "medium/50k/adam"

    def test_efficientad_small_model_adamw(self):
        r = self._efficientad_record(model_size="small", optimizer="adamw")
        assert _param_summary(r) == "small/70k/adamw"

    def test_efficientad_uses_model_params_not_model_config(self):
        """model_config 키가 있어도 model_params 를 사용해야 한다 (R-ENUM-01 §1.1)."""
        r = {
            "model_type": "efficientad",
            "model_params": {"model_size": "medium", "train_steps": 70000, "optimizer": "adam"},
            "model_config": {"model_size": "small", "train_steps": 1000, "optimizer": "sgd"},
        }
        assert _param_summary(r) == "medium/70k/adam"

    def test_patchcore_wrn50_abbreviation(self):
        r = self._patchcore_record()
        assert _param_summary(r) == "wrn50/0.1"

    def test_patchcore_resnet50_abbreviation(self):
        r = self._patchcore_record(backbone="resnet50")
        assert _param_summary(r) == "r50/0.1"

    def test_patchcore_resnet18_abbreviation(self):
        r = self._patchcore_record(backbone="resnet18")
        assert _param_summary(r) == "r18/0.1"

    def test_patchcore_unknown_backbone_kept_verbatim(self):
        r = self._patchcore_record(backbone="custom_backbone")
        assert "custom_backbone" in _param_summary(r)

    def test_patchcore_coreset_ratio_in_output(self):
        r = self._patchcore_record(coreset_sampling_ratio=0.05)
        assert "0.05" in _param_summary(r)

    def test_unknown_model_type_returns_empty(self):
        r = {"model_type": "unknown", "model_params": {}}
        assert _param_summary(r) == ""

    def test_missing_model_params_returns_question_marks(self):
        """model_params 키 자체가 없어도 예외 없이 처리."""
        r = {"model_type": "efficientad"}
        result = _param_summary(r)
        assert "?" in result

    def test_model_type_lowercase_only(self):
        """R-ENUM-01: 대문자 model_type은 매칭되지 않아야 한다."""
        r_upper = {"model_type": "EfficientAD", "model_params": {"train_steps": 70000}}
        assert _param_summary(r_upper) == ""

        r_mixed = {"model_type": "PatchCore", "model_params": {"backbone": "wide_resnet50_2"}}
        assert _param_summary(r_mixed) == ""


# ── _build_table_df ──────────────────────────────────────────────────────────

class TestBuildTableDf:
    def _completed_record(self, exp_id: str = "exp_001") -> dict:
        return {
            "experiment_id": exp_id,
            "name": f"Test {exp_id}",
            "model_type": "efficientad",
            "model_params": {"model_size": "medium", "train_steps": 70000, "optimizer": "adam"},
            "created_at": "2026-05-10T12:00:00+09:00",
            "status": "completed",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.90,
                "recall": 0.88,
                "f1_score": 0.89,
                "f2_score": 0.885,
                "auc": 0.97,
            },
        }

    def _stopped_record(self, exp_id: str = "exp_002") -> dict:
        return {
            "experiment_id": exp_id,
            "name": f"Stopped {exp_id}",
            "model_type": "patchcore",
            "model_params": {},
            "created_at": "2026-05-10T11:00:00+09:00",
            "status": "중단",
            "metrics": None,
        }

    def test_returns_dataframe_with_correct_columns(self):
        import pandas as pd
        df = _build_table_df([self._completed_record()])
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"실험명", "모델", "파라미터 요약", "Accuracy", "Precision",
                         "Recall", "F1", "F2", "AUC", "실행 시각", "상태"}
        assert expected_cols.issubset(set(df.columns))

    def test_completed_record_shows_metrics(self):
        df = _build_table_df([self._completed_record()])
        assert df.iloc[0]["Accuracy"] == "0.9500"
        assert df.iloc[0]["AUC"] == "0.9700"

    def test_stopped_record_shows_dash(self):
        df = _build_table_df([self._stopped_record()])
        assert df.iloc[0]["Accuracy"] == "—"
        assert df.iloc[0]["상태"] == "중단"

    def test_multiple_records_preserves_order(self):
        records = [self._completed_record("e1"), self._stopped_record("e2")]
        df = _build_table_df(records)
        assert len(df) == 2
        assert df.iloc[0]["상태"] == "completed"
        assert df.iloc[1]["상태"] == "중단"

    def test_created_at_formatted_without_t(self):
        df = _build_table_df([self._completed_record()])
        assert "T" not in df.iloc[0]["실행 시각"]


# ── _confusion_matrix_fig ─────────────────────────────────────────────────────

class TestConfusionMatrixFig:
    def test_returns_figure(self):
        import plotly.graph_objects as go
        metrics = {"confusion_matrix": {"tn": 40, "fp": 2, "fn": 3, "tp": 15}}
        fig = _confusion_matrix_fig(metrics)
        assert isinstance(fig, go.Figure)

    def test_missing_confusion_matrix_does_not_raise(self):
        import plotly.graph_objects as go
        fig = _confusion_matrix_fig({})
        assert isinstance(fig, go.Figure)

    def test_cell_labels_present(self):
        metrics = {"confusion_matrix": {"tn": 40, "fp": 2, "fn": 3, "tp": 15}}
        fig = _confusion_matrix_fig(metrics)
        texts = fig.data[0].text
        flat = [cell for row in texts for cell in row]
        assert any("TN" in t for t in flat)
        assert any("TP" in t for t in flat)
        assert any("FP" in t for t in flat)
        assert any("FN" in t for t in flat)


# ── _roc_curve_fig ────────────────────────────────────────────────────────────

class TestRocCurveFig:
    def _metrics_with_scores(self) -> dict:
        # 이진 분류 예시: 0=정상, 1=결함
        return {
            "anomaly_scores": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            "image_labels":   [0,   0,   1,   1,   0,   1],
        }

    def test_returns_figure(self):
        import plotly.graph_objects as go
        fig = _roc_curve_fig(self._metrics_with_scores())
        assert isinstance(fig, go.Figure)

    def test_auc_in_legend_name(self):
        fig = _roc_curve_fig(self._metrics_with_scores())
        names = [t.name for t in fig.data]
        assert any("AUC" in n for n in names)

    def test_empty_metrics_does_not_raise(self):
        import plotly.graph_objects as go
        fig = _roc_curve_fig({})
        assert isinstance(fig, go.Figure)

    def test_single_class_does_not_raise(self):
        """레이블이 1종류뿐이면 ROC 계산 불가 — 빈 그래프만 반환해야 함."""
        import plotly.graph_objects as go
        metrics = {"anomaly_scores": [0.1, 0.2, 0.3], "image_labels": [0, 0, 0]}
        fig = _roc_curve_fig(metrics)
        assert isinstance(fig, go.Figure)


# ── _score_dist_fig ───────────────────────────────────────────────────────────

class TestScoreDistFig:
    def _metrics(self) -> dict:
        return {
            "anomaly_scores": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            "image_labels":   [0,   0,   1,   1,   0,   1],
        }

    def test_returns_figure(self):
        import plotly.graph_objects as go
        fig = _score_dist_fig(self._metrics())
        assert isinstance(fig, go.Figure)

    def test_threshold_vline_added_when_provided(self):
        fig = _score_dist_fig(self._metrics(), threshold_value=0.5)
        shapes = fig.layout.shapes
        assert len(shapes) >= 1, "threshold 수직선(vline)이 추가돼야 합니다"

    def test_no_vline_when_threshold_is_none(self):
        fig = _score_dist_fig(self._metrics(), threshold_value=None)
        shapes = fig.layout.shapes
        assert len(shapes) == 0, "threshold=None이면 수직선이 없어야 합니다"

    def test_threshold_not_read_from_metrics_dict(self):
        """metrics 에 'threshold' 키가 있어도 무시하고 파라미터를 사용해야 한다."""
        metrics = {**self._metrics(), "threshold": 0.9}
        fig_with_param = _score_dist_fig(metrics, threshold_value=0.5)
        fig_no_param = _score_dist_fig(metrics, threshold_value=None)
        # threshold_value=None 이면 vline 없음
        assert len(fig_no_param.layout.shapes) == 0
        # threshold_value=0.5 이면 vline 있음
        assert len(fig_with_param.layout.shapes) >= 1

    def test_normal_and_defect_traces(self):
        fig = _score_dist_fig(self._metrics())
        names = [t.name for t in fig.data]
        assert "정상" in names
        assert "결함" in names

    def test_empty_metrics_does_not_raise(self):
        import plotly.graph_objects as go
        fig = _score_dist_fig({})
        assert isinstance(fig, go.Figure)


# ── _METRIC_MAP / _BACKBONE_ABBREV 상수 검증 ─────────────────────────────────

class TestConstants:
    def test_metric_map_keys(self):
        assert set(_METRIC_MAP.keys()) == {"Accuracy", "Precision", "Recall", "F1", "F2"}

    def test_metric_map_values_match_record_keys(self):
        assert _METRIC_MAP["Accuracy"] == "accuracy"
        assert _METRIC_MAP["F1"] == "f1_score"
        assert _METRIC_MAP["F2"] == "f2_score"

    def test_backbone_abbrev_covers_prd_values(self):
        assert _BACKBONE_ABBREV["wide_resnet50_2"] == "wrn50"
        assert _BACKBONE_ABBREV["resnet50"] == "r50"
        assert _BACKBONE_ABBREV["resnet18"] == "r18"

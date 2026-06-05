"""
tests/inspection/test_tab2_history.py

FR-INSP-T2-01: 이력 테이블 빌드 — 컬럼, 정렬, 이모지 판정, 빈 DataFrame
FR-INSP-T2-02: KPI — 총검사/양품/불량/불량률, 빈 기록 시 "-"
FR-INSP-T2-03: CSV 빌드 — 헤더, 행 수, BOM, 이모지 없는 판정
FR-INSP-T2-05: 시뮬레이션 시간 계산 유틸리티
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_records(n_good: int = 2, n_bad: int = 1) -> list[dict]:
    records = []
    seq = 1
    for _ in range(n_good):
        records.append(
            {
                "seq": seq,
                "inspected_at": f"2026-05-27 10:00:{seq:02d}",
                "image_name": f"good_{seq:03d}.png",
                "image_path": f"/data/good_{seq:03d}.png",
                "verdict": "양품",
                "anomaly_score": 0.1 + seq * 0.01,
            }
        )
        seq += 1
    for _ in range(n_bad):
        records.append(
            {
                "seq": seq,
                "inspected_at": f"2026-05-27 10:00:{seq:02d}",
                "image_name": f"bad_{seq:03d}.png",
                "image_path": f"/data/bad_{seq:03d}.png",
                "verdict": "불량",
                "anomaly_score": 0.85,
            }
        )
        seq += 1
    return records


# ── _build_dataframe ──────────────────────────────────────────────────────────


class TestBuildDataframe:
    def _call(self, records, filter_opt="전체"):
        from inspection.tabs.insp_tab2_history import _build_dataframe

        return _build_dataframe(records, filter_opt)

    def test_empty_records_returns_empty_df(self):
        df = self._call([])
        assert df.empty
        assert list(df.columns) == ["번호", "시각", "이미지명", "판정결과", "Anomaly Score"]

    def test_columns_match_spec(self):
        df = self._call(_make_records())
        assert list(df.columns) == ["번호", "시각", "이미지명", "판정결과", "Anomaly Score"]

    def test_seq_sorted_descending(self):
        df = self._call(_make_records(n_good=3, n_bad=1))
        seqs = df["번호"].tolist()
        assert seqs == sorted(seqs, reverse=True)

    def test_verdict_good_has_emoji(self):
        df = self._call(_make_records(n_good=1, n_bad=0))
        assert df.iloc[0]["판정결과"] == "🟢 양품"

    def test_verdict_bad_has_emoji(self):
        df = self._call(_make_records(n_good=0, n_bad=1))
        assert df.iloc[0]["판정결과"] == "🔴 불량"

    def test_anomaly_score_formatted_4dp(self):
        records = _make_records(n_good=1, n_bad=0)
        records[0]["anomaly_score"] = 0.123456
        df = self._call(records)
        assert df.iloc[0]["Anomaly Score"] == "0.1235"

    def test_filter_yangpum_only(self):
        df = self._call(_make_records(n_good=2, n_bad=1), filter_opt="양품만")
        assert all(v == "🟢 양품" for v in df["판정결과"])
        assert len(df) == 2

    def test_filter_bulyang_only(self):
        df = self._call(_make_records(n_good=2, n_bad=2), filter_opt="불량만")
        assert all(v == "🔴 불량" for v in df["판정결과"])
        assert len(df) == 2

    def test_filter_all_returns_all_rows(self):
        records = _make_records(n_good=2, n_bad=1)
        df = self._call(records, filter_opt="전체")
        assert len(df) == 3


# ── _build_csv ────────────────────────────────────────────────────────────────


class TestBuildCsv:
    def _call(self, records):
        from inspection.tabs.insp_tab2_history import _build_csv

        return _build_csv(records)

    def test_empty_records_returns_header_only(self):
        csv_bytes = self._call([])
        text = csv_bytes.decode("utf-8-sig")
        lines = [l for l in text.splitlines() if l.strip()]
        assert len(lines) == 1
        assert "번호" in lines[0]

    def test_row_count_matches_records(self):
        records = _make_records(n_good=2, n_bad=1)
        csv_bytes = self._call(records)
        text = csv_bytes.decode("utf-8-sig")
        data_lines = [l for l in text.splitlines() if l.strip()]
        assert len(data_lines) == 1 + len(records)  # header + rows

    def test_verdict_in_csv_has_no_emoji(self):
        records = _make_records(n_good=1, n_bad=1)
        csv_bytes = self._call(records)
        text = csv_bytes.decode("utf-8-sig")
        assert "🟢" not in text
        assert "🔴" not in text
        assert "양품" in text
        assert "불량" in text

    def test_bom_present(self):
        records = _make_records(n_good=1)
        csv_bytes = self._call(records)
        assert csv_bytes[:3] == b"\xef\xbb\xbf"

    def test_columns_order(self):
        records = _make_records(n_good=1)
        csv_bytes = self._call(records)
        header = csv_bytes.decode("utf-8-sig").splitlines()[0]
        expected_cols = ["번호", "시각", "이미지명", "판정결과", "Anomaly Score"]
        for col in expected_cols:
            assert col in header


# ── KPI helpers (unit) ────────────────────────────────────────────────────────


class TestKpiValues:
    """KPI 계산 로직은 _render_kpi 내부에 인라인이므로 같은 로직을 직접 검증."""

    def _compute(self, records):
        total = len(records)
        good = sum(1 for r in records if r.get("verdict") == "양품")
        bad = total - good
        rate = f"{bad / total * 100:.1f}%" if total > 0 else "-"
        return total, good, bad, rate

    def test_empty_records(self):
        total, good, bad, rate = self._compute([])
        assert total == 0
        assert good == 0
        assert bad == 0
        assert rate == "-"

    def test_all_good(self):
        total, good, bad, rate = self._compute(_make_records(n_good=3, n_bad=0))
        assert total == 3
        assert good == 3
        assert bad == 0
        assert rate == "0.0%"

    def test_all_bad(self):
        total, good, bad, rate = self._compute(_make_records(n_good=0, n_bad=4))
        assert total == 4
        assert good == 0
        assert bad == 4
        assert rate == "100.0%"

    def test_mixed(self):
        total, good, bad, rate = self._compute(_make_records(n_good=3, n_bad=1))
        assert total == 4
        assert bad == 1
        assert rate == "25.0%"


# ── TestBuildGroupTableDf ─────────────────────────────────────────────────────


class TestBuildGroupTableDf:
    """_build_group_table_df() — 시간 범위 테이블 DataFrame 검증 (FR-INSP-T2-05)."""

    def _call(self, total_seqs: int, unit: int):
        from inspection.tabs.insp_tab2_history import _build_group_table_df
        return _build_group_table_df(total_seqs, unit)

    def test_empty_when_no_records(self):
        df = self._call(0, 20)
        assert len(df) == 0
        assert "시간 범위" in df.columns

    def test_single_column_only(self):
        """테이블은 '시간 범위' 단일 컬럼만 가진다."""
        df = self._call(10, 20)
        assert list(df.columns) == ["시간 범위"]

    def test_one_group_when_total_less_than_unit(self):
        """total_seqs < unit → 미완성 그룹 1개."""
        df = self._call(15, 20)
        assert len(df) == 1

    def test_one_group_when_total_equals_unit(self):
        """total_seqs == unit → 완성 그룹 1개."""
        df = self._call(20, 20)
        assert len(df) == 1

    def test_two_groups_when_total_exceeds_unit(self):
        """total_seqs = unit + 1 → 그룹 2개."""
        df = self._call(21, 20)
        assert len(df) == 2

    def test_group_count_unit20(self):
        """unit=20, total=60 → 3그룹."""
        df = self._call(60, 20)
        assert len(df) == 3

    def test_group_count_unit100(self):
        """unit=100, total=250 → 3그룹."""
        df = self._call(250, 100)
        assert len(df) == 3

    def test_first_label_unit20(self):
        """단위=20, 첫 번째 레이블 → '2026-06-24 14:00~14:01'."""
        df = self._call(20, 20)
        assert df.iloc[0]["시간 범위"] == "2026-06-24 14:00~14:01"

    def test_second_label_unit20(self):
        """단위=20, 두 번째 레이블 → '2026-06-24 14:01~14:02'."""
        df = self._call(21, 20)
        assert df.iloc[1]["시간 범위"] == "2026-06-24 14:01~14:02"

    def test_first_label_unit100(self):
        """단위=100, 첫 번째 레이블 → '2026-06-24 14:00~14:05'."""
        df = self._call(100, 100)
        assert df.iloc[0]["시간 범위"] == "2026-06-24 14:00~14:05"

    def test_labels_are_strings(self):
        """모든 레이블이 문자열이어야 한다."""
        df = self._call(50, 20)
        assert all(isinstance(v, str) for v in df["시간 범위"])

    def test_all_labels_contain_tilde(self):
        """모든 레이블에 '~' 구분자가 포함된다."""
        df = self._call(50, 20)
        assert all("~" in v for v in df["시간 범위"])

    def test_row_count_consistent_with_sim_utils(self):
        """행 수가 get_sim_group_count와 일치한다."""
        from inspection.utils.insp_session_init import get_sim_group_count
        for unit in (20, 40, 100):
            for total in (0, 1, 19, 20, 21, 99, 100, 101):
                df = self._call(total, unit)
                assert len(df) == get_sim_group_count(total, unit)

    def test_unit_change_changes_label_count(self):
        """같은 total_seqs라도 단위 변경 시 그룹 수가 달라진다."""
        df20  = self._call(100, 20)
        df100 = self._call(100, 100)
        assert len(df20) == 5   # 100 / 20 = 5그룹
        assert len(df100) == 1  # 100 / 100 = 1그룹


# ── TestSimTimeUtils ──────────────────────────────────────────────────────────


class TestSimTimeUtils:
    """시뮬레이션 시간 계산 유틸리티 검증 (FR-INSP-T2-05).

    고정 시작: 2026-06-24 14:00:00
    검사 간격: 3초/건
    """

    _BASE = datetime(2026, 6, 24, 14, 0, 0)

    def setup_method(self):
        from inspection.utils.insp_session_init import (
            get_sim_timestamp,
            get_sim_group_index,
            get_sim_group_label,
            get_sim_group_count,
            get_sim_all_group_labels,
        )
        self.get_sim_timestamp       = get_sim_timestamp
        self.get_sim_group_index     = get_sim_group_index
        self.get_sim_group_label     = get_sim_group_label
        self.get_sim_group_count     = get_sim_group_count
        self.get_sim_all_group_labels = get_sim_all_group_labels

    # ── get_sim_timestamp ────────────────────────────────────────────────────

    def test_seq1_is_base_time(self):
        """seq=1 → 2026-06-24 14:00:00."""
        assert self.get_sim_timestamp(1) == self._BASE

    def test_seq2_is_3_seconds_later(self):
        """seq=2 → base + 3s."""
        assert self.get_sim_timestamp(2) == self._BASE + timedelta(seconds=3)

    def test_seq_20_unit_20_boundary(self):
        """seq=20 → base + 57s (그룹 0 마지막 검사)."""
        assert self.get_sim_timestamp(20) == self._BASE + timedelta(seconds=57)

    def test_seq_21_starts_new_minute(self):
        """seq=21 → base + 60s = 14:01:00 (unit=20 기준 그룹 1 시작)."""
        assert self.get_sim_timestamp(21) == self._BASE + timedelta(seconds=60)

    def test_seq_101_with_unit_100(self):
        """seq=101 → base + 300s = 14:05:00 (unit=100 기준 그룹 1 시작)."""
        assert self.get_sim_timestamp(101) == self._BASE + timedelta(seconds=300)

    def test_returns_datetime(self):
        assert isinstance(self.get_sim_timestamp(1), datetime)

    # ── get_sim_group_index ──────────────────────────────────────────────────

    def test_group_index_seq1_unit20(self):
        assert self.get_sim_group_index(1, 20) == 0

    def test_group_index_seq20_unit20(self):
        assert self.get_sim_group_index(20, 20) == 0

    def test_group_index_seq21_unit20(self):
        assert self.get_sim_group_index(21, 20) == 1

    def test_group_index_seq40_unit20(self):
        assert self.get_sim_group_index(40, 20) == 1

    def test_group_index_seq41_unit20(self):
        assert self.get_sim_group_index(41, 20) == 2

    def test_group_index_seq1_unit100(self):
        assert self.get_sim_group_index(1, 100) == 0

    def test_group_index_seq100_unit100(self):
        assert self.get_sim_group_index(100, 100) == 0

    def test_group_index_seq101_unit100(self):
        assert self.get_sim_group_index(101, 100) == 1

    # ── get_sim_group_label ──────────────────────────────────────────────────

    def test_group_label_idx0_unit20(self):
        """단위=20 → 1분 간격: 그룹 0 → '2026-06-24 14:00~14:01'."""
        assert self.get_sim_group_label(0, 20) == "2026-06-24 14:00~14:01"

    def test_group_label_idx1_unit20(self):
        """단위=20 → 그룹 1 → '2026-06-24 14:01~14:02'."""
        assert self.get_sim_group_label(1, 20) == "2026-06-24 14:01~14:02"

    def test_group_label_idx0_unit40(self):
        """단위=40 → 2분 간격: 그룹 0 → '2026-06-24 14:00~14:02'."""
        assert self.get_sim_group_label(0, 40) == "2026-06-24 14:00~14:02"

    def test_group_label_idx0_unit100(self):
        """단위=100 → 5분 간격: 그룹 0 → '2026-06-24 14:00~14:05'."""
        assert self.get_sim_group_label(0, 100) == "2026-06-24 14:00~14:05"

    def test_group_label_idx1_unit100(self):
        """단위=100 → 그룹 1 → '2026-06-24 14:05~14:10'."""
        assert self.get_sim_group_label(1, 100) == "2026-06-24 14:05~14:10"

    def test_group_label_idx2_unit100(self):
        """단위=100 → 그룹 2 → '2026-06-24 14:10~14:15'."""
        assert self.get_sim_group_label(2, 100) == "2026-06-24 14:10~14:15"

    def test_group_label_format_date_prefix(self):
        """레이블에 날짜 '2026-06-24' 접두사가 포함된다."""
        label = self.get_sim_group_label(0, 20)
        assert label.startswith("2026-06-24")

    def test_group_label_format_tilde_separator(self):
        """시작~종료 사이에 '~' 구분자가 사용된다."""
        label = self.get_sim_group_label(0, 20)
        assert "~" in label

    # ── get_sim_group_count ──────────────────────────────────────────────────

    def test_group_count_zero_seqs(self):
        assert self.get_sim_group_count(0, 20) == 0

    def test_group_count_one_seq(self):
        assert self.get_sim_group_count(1, 20) == 1

    def test_group_count_exactly_one_unit(self):
        """total_seqs == unit → 그룹 1개 (완전한 그룹)."""
        assert self.get_sim_group_count(20, 20) == 1

    def test_group_count_one_over_unit(self):
        """total_seqs = unit + 1 → 그룹 2개 (마지막 미완성 그룹 포함)."""
        assert self.get_sim_group_count(21, 20) == 2

    def test_group_count_two_full_units(self):
        assert self.get_sim_group_count(40, 20) == 2

    def test_group_count_100_unit_with_partial(self):
        """total_seqs=150, unit=100 → 그룹 2개."""
        assert self.get_sim_group_count(150, 100) == 2

    # ── get_sim_all_group_labels ─────────────────────────────────────────────

    def test_all_labels_empty_when_no_seqs(self):
        labels = self.get_sim_all_group_labels(0, 20)
        assert labels == []

    def test_all_labels_one_group(self):
        labels = self.get_sim_all_group_labels(15, 20)
        assert len(labels) == 1
        assert labels[0] == "2026-06-24 14:00~14:01"

    def test_all_labels_two_groups_unit20(self):
        labels = self.get_sim_all_group_labels(25, 20)
        assert len(labels) == 2
        assert labels[0] == "2026-06-24 14:00~14:01"
        assert labels[1] == "2026-06-24 14:01~14:02"

    def test_all_labels_three_groups_unit100(self):
        """unit=100, total=250 → 3그룹."""
        labels = self.get_sim_all_group_labels(250, 100)
        assert len(labels) == 3
        assert labels[0] == "2026-06-24 14:00~14:05"
        assert labels[1] == "2026-06-24 14:05~14:10"
        assert labels[2] == "2026-06-24 14:10~14:15"

    def test_all_labels_count_matches_group_count(self):
        for unit in (20, 40, 100):
            for total in (0, 1, 15, 20, 21, 99, 100, 101):
                labels = self.get_sim_all_group_labels(total, unit)
                assert len(labels) == self.get_sim_group_count(total, unit)


# ── TestGetGroupRecords ────────────────────────────────────────────────────────


class TestGetGroupRecords:
    """_get_group_records() — 그룹별 기록 필터링 검증 (FR-INSP-T2-06)."""

    def _call(self, records, group_idx, unit):
        from inspection.tabs.insp_tab2_history import _get_group_records
        return _get_group_records(records, group_idx, unit)

    def _make_records(self, n: int) -> list[dict]:
        return [
            {"seq": i, "anomaly_score": 0.1 * (i % 10), "verdict": "양품"}
            for i in range(1, n + 1)
        ]

    def test_empty_returns_empty(self):
        assert self._call([], 0, 20) == []

    def test_first_group_unit20_has_20_records(self):
        records = self._make_records(40)
        result = self._call(records, 0, 20)
        assert len(result) == 20

    def test_first_group_seq_range_unit20(self):
        """그룹 0, unit=20 → seq 1~20."""
        records = self._make_records(40)
        result = self._call(records, 0, 20)
        seqs = [r["seq"] for r in result]
        assert all(1 <= s <= 20 for s in seqs)

    def test_second_group_seq_range_unit20(self):
        """그룹 1, unit=20 → seq 21~40."""
        records = self._make_records(40)
        result = self._call(records, 1, 20)
        seqs = [r["seq"] for r in result]
        assert all(21 <= s <= 40 for s in seqs)

    def test_partial_last_group(self):
        """25개 기록, unit=20 → 그룹 1은 5개(seq 21~25)."""
        records = self._make_records(25)
        result = self._call(records, 1, 20)
        assert len(result) == 5

    def test_nonexistent_group_returns_empty(self):
        """10개 기록, unit=20 → 그룹 2는 없음."""
        records = self._make_records(10)
        result = self._call(records, 2, 20)
        assert result == []

    def test_unit100_first_group(self):
        """unit=100, 150개 기록 → 그룹 0은 100개."""
        records = self._make_records(150)
        result = self._call(records, 0, 100)
        assert len(result) == 100

    def test_total_records_preserved_across_groups(self):
        """모든 그룹의 기록 수 합 = 전체 기록 수."""
        records = self._make_records(45)
        unit = 20
        from inspection.utils.insp_session_init import get_sim_group_count
        total_cnt = get_sim_group_count(len(records), unit)
        all_results = [self._call(records, g, unit) for g in range(total_cnt)]
        assert sum(len(r) for r in all_results) == len(records)


# ── TestBuildHistogramFig ──────────────────────────────────────────────────────


class TestBuildHistogramFig:
    """_build_histogram_fig() — 히스토그램 Plotly 피겨 검증 (FR-INSP-T2-06)."""

    def _call(self, records, threshold=0.5, label="테스트", unit=20):
        from inspection.tabs.insp_tab2_history import _build_histogram_fig
        return _build_histogram_fig(records, threshold, label, unit)

    def _make_records(self, normal_scores, defect_scores, threshold=0.5):
        recs = []
        seq = 1
        for s in normal_scores:
            recs.append({"seq": seq, "anomaly_score": s, "verdict": "양품"})
            seq += 1
        for s in defect_scores:
            recs.append({"seq": seq, "anomaly_score": s, "verdict": "불량"})
            seq += 1
        return recs

    def test_returns_plotly_figure(self):
        """go.Figure 인스턴스를 반환해야 한다."""
        import plotly.graph_objects as go
        records = self._make_records([0.2, 0.3], [0.7, 0.8])
        fig = self._call(records)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces_for_mixed_data(self):
        """정상 + 불량 데이터가 있으면 trace 2개(각 histogram 1개)."""
        records = self._make_records([0.2, 0.3], [0.7, 0.8])
        fig = self._call(records)
        assert len(fig.data) == 2

    def test_normal_only_has_one_trace(self):
        """정상만 있으면 trace 1개."""
        records = self._make_records([0.1, 0.2, 0.3], [])
        fig = self._call(records)
        assert len(fig.data) == 1

    def test_defect_only_has_one_trace(self):
        """불량만 있으면 trace 1개."""
        records = self._make_records([], [0.7, 0.8, 0.9])
        fig = self._call(records)
        assert len(fig.data) == 1

    def test_normal_trace_is_blue(self):
        """정상 trace는 파란색 계열."""
        records = self._make_records([0.2], [])
        fig = self._call(records)
        color = fig.data[0].marker.color
        assert "#4e9af1" in str(color)

    def test_defect_trace_is_red(self):
        """불량 trace는 빨간색 계열."""
        records = self._make_records([], [0.8])
        fig = self._call(records)
        color = fig.data[0].marker.color
        assert "#e05555" in str(color)

    def test_x_axis_range_minus_0_2_to_1_2(self):
        """x축 범위가 [-0.2, 1.2]이어야 한다 (여유 공간 포함)."""
        records = self._make_records([0.2], [0.8])
        fig = self._call(records)
        x_range = fig.layout.xaxis.range
        assert tuple(x_range) == pytest.approx((-0.2, 1.2))

    def test_y_axis_starts_at_0(self):
        """y축이 0부터 시작해야 한다."""
        records = self._make_records([0.2], [0.8])
        fig = self._call(records)
        assert fig.layout.yaxis.range[0] == 0

    def test_y_axis_minimum_10(self):
        """y축 최대값은 데이터가 적어도 최소 10 이상이어야 한다."""
        records = self._make_records([0.2], [])  # 1개 데이터
        fig = self._call(records)
        assert fig.layout.yaxis.range[1] >= 10

    def test_title_contains_group_label(self):
        """차트 제목에 그룹 레이블이 포함된다."""
        label = "2026-06-24 14:00~14:01"
        records = self._make_records([0.2], [0.8])
        fig = _build_histogram_fig_direct(records, 0.5, label, 20)
        assert label in fig.layout.title.text

    def test_threshold_vline_present(self):
        """threshold 수직선(shape 또는 annotation)이 존재해야 한다."""
        records = self._make_records([0.2, 0.3], [0.7])
        fig = self._call(records, threshold=0.5)
        shapes = fig.layout.shapes or []
        has_vline = any(
            getattr(s, "type", None) == "line" or
            getattr(s, "x0", None) is not None
            for s in shapes
        )
        # add_vline creates a layout.shapes entry
        # At minimum, a threshold annotation should be present
        annotations = fig.layout.annotations or []
        has_threshold = (
            has_vline or
            any("thr" in str(getattr(a, "text", "")) for a in annotations)
        )
        assert has_threshold

    def test_barmode_is_overlay(self):
        """barmode가 'overlay'이어야 정상/불량이 겹쳐 표시된다."""
        records = self._make_records([0.2, 0.3], [0.7, 0.8])
        fig = self._call(records)
        assert fig.layout.barmode == "overlay"


def _build_histogram_fig_direct(records, threshold, label, unit):
    from inspection.tabs.insp_tab2_history import _build_histogram_fig
    return _build_histogram_fig(records, threshold, label, unit)


# ── TestBuildScatterFig ────────────────────────────────────────────────────────


class TestBuildScatterFig:
    """_build_scatter_fig() — Anomaly Score 산점도 Plotly 피겨 검증 (FR-INSP-T2-07)."""

    def _call(self, records, threshold=0.5, label="테스트", unit=20):
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        return _build_scatter_fig(records, threshold, label, unit)

    def _make_records(self, scores, start_seq=1, threshold=0.5):
        return [
            {"seq": start_seq + i, "anomaly_score": s,
             "verdict": "불량" if s >= threshold else "양품"}
            for i, s in enumerate(scores)
        ]

    # ── 기본 구조 ──────────────────────────────────────────────────────────────

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        records = self._make_records([0.3, 0.7])
        fig = self._call(records)
        assert isinstance(fig, go.Figure)

    def test_has_exactly_one_trace(self):
        """단일 scatter trace로 정상/불량 모두 표현한다."""
        records = self._make_records([0.2, 0.8, 0.3, 0.9])
        fig = self._call(records)
        assert len(fig.data) == 1

    def test_mode_is_lines_and_markers(self):
        """mode="lines+markers" 이어야 점과 선이 함께 표시된다."""
        records = self._make_records([0.3, 0.7])
        fig = self._call(records)
        assert "lines" in str(fig.data[0].mode)
        assert "markers" in str(fig.data[0].mode)

    # ── 축 범위 ────────────────────────────────────────────────────────────────

    def test_y_axis_range_minus_0_2_to_1_2(self):
        """y축 범위가 [-0.2, 1.2]이어야 한다 (여유 공간 포함)."""
        records = self._make_records([0.3, 0.7])
        fig = self._call(records)
        assert tuple(fig.layout.yaxis.range) == pytest.approx((-0.2, 1.2))

    def test_x_axis_range_unit20_is_0_to_60(self):
        """unit=20 → x축 범위 [0, 60] (20×3=60sec)."""
        records = self._make_records([0.3, 0.7])
        fig = self._call(records, unit=20)
        assert tuple(fig.layout.xaxis.range) == (0, 60)

    def test_x_axis_range_unit40_is_0_to_120(self):
        """unit=40 → x축 범위 [0, 120] (40×3=120sec)."""
        records = self._make_records([0.3, 0.7])
        fig = self._call(records, unit=40)
        assert tuple(fig.layout.xaxis.range) == (0, 120)

    def test_x_axis_range_unit100_is_0_to_300(self):
        """unit=100 → x축 범위 [0, 300] (100×3=300sec)."""
        records = self._make_records([0.3, 0.7])
        fig = self._call(records, unit=100)
        assert tuple(fig.layout.xaxis.range) == (0, 300)

    # ── x 좌표 계산 ────────────────────────────────────────────────────────────

    def test_x_positions_first_group_unit20_in_seconds(self):
        """seq=1~5, unit=20 → x positions 0,3,6,9,12 (초 단위)."""
        records = self._make_records([0.3] * 5, start_seq=1)
        fig = self._call(records, unit=20)
        x_values = list(fig.data[0].x)
        assert x_values == [0, 3, 6, 9, 12]

    def test_x_positions_second_group_unit20_same_as_first(self):
        """seq=21~25, unit=20 → x positions 0,3,6,9,12 (그룹 내 상대 시간)."""
        records = self._make_records([0.3] * 5, start_seq=21)
        fig = self._call(records, unit=20)
        x_values = list(fig.data[0].x)
        assert x_values == [0, 3, 6, 9, 12]

    def test_x_positions_full_group_unit20_ends_at_57(self):
        """seq=1~20, unit=20 → 마지막 x=57 (= (20-1)×3), 범위는 0~60."""
        records = self._make_records([0.3] * 20, start_seq=1)
        fig = self._call(records, unit=20)
        x_values = list(fig.data[0].x)
        assert x_values == list(range(0, 60, 3))   # [0, 3, 6, ..., 57]

    # ── 색상 ───────────────────────────────────────────────────────────────────

    def test_normal_markers_are_blue(self):
        """score < threshold → 파란 점 (#4e9af1)."""
        records = self._make_records([0.1, 0.2, 0.3])
        fig = self._call(records, threshold=0.5)
        colors = list(fig.data[0].marker.color)
        assert all("#4e9af1" in c for c in colors)

    def test_defect_markers_are_red(self):
        """score >= threshold → 빨간 점 (#e05555)."""
        records = self._make_records([0.6, 0.7, 0.9])
        fig = self._call(records, threshold=0.5)
        colors = list(fig.data[0].marker.color)
        assert all("#e05555" in c for c in colors)

    def test_mixed_colors_in_correct_order(self):
        """정상 후 불량 순서 → 파란색 먼저, 빨간색 다음."""
        records = self._make_records([0.2, 0.8])  # normal, defect
        fig = self._call(records, threshold=0.5)
        colors = list(fig.data[0].marker.color)
        assert "#4e9af1" in colors[0]
        assert "#e05555" in colors[1]

    def test_threshold_at_exact_value_is_defect(self):
        """score == threshold → 불량 (>=)."""
        records = self._make_records([0.5])
        fig = self._call(records, threshold=0.5)
        colors = list(fig.data[0].marker.color)
        assert "#e05555" in colors[0]

    # ── threshold 선 ──────────────────────────────────────────────────────────

    def test_threshold_hline_present(self):
        """threshold 수평선(shape 또는 annotation)이 존재해야 한다."""
        records = self._make_records([0.3, 0.7])
        fig = self._call(records, threshold=0.5)
        shapes = fig.layout.shapes or []
        annotations = fig.layout.annotations or []
        has_threshold = (
            len(shapes) > 0 or
            any("thr" in str(getattr(a, "text", "")) for a in annotations)
        )
        assert has_threshold

    # ── 제목 ───────────────────────────────────────────────────────────────────

    def test_title_contains_group_label(self):
        """차트 제목에 그룹 레이블이 포함된다."""
        label = "2026-06-24 14:00~14:01"
        records = self._make_records([0.3, 0.7])
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        fig = _build_scatter_fig(records, 0.5, label, 20)
        assert label in fig.layout.title.text

    def test_y_values_match_anomaly_scores(self):
        """y 좌표가 anomaly_score와 일치해야 한다."""
        scores = [0.1, 0.5, 0.9]
        records = self._make_records(scores)
        fig = self._call(records)
        y_values = list(fig.data[0].y)
        assert y_values == pytest.approx(scores)

    # ── tick 검증 (unit별) ────────────────────────────────────────────────────

    def test_x_tickvals_unit20(self):
        """unit=20 → tickvals [0, 12, 24, 36, 48, 60]."""
        fig = self._call_with_unit(unit=20)
        assert list(fig.layout.xaxis.tickvals) == [0, 12, 24, 36, 48, 60]

    def test_x_tickvals_unit40(self):
        """unit=40 → tickvals [0, 24, 48, 72, 96, 120]."""
        fig = self._call_with_unit(unit=40)
        assert list(fig.layout.xaxis.tickvals) == [0, 24, 48, 72, 96, 120]

    def test_x_tickvals_unit100(self):
        """unit=100 → tickvals [0, 60, 120, 180, 240, 300]."""
        fig = self._call_with_unit(unit=100)
        assert list(fig.layout.xaxis.tickvals) == [0, 60, 120, 180, 240, 300]

    def test_x_title_is_seconds(self):
        """x축 제목이 '시간 (sec)'이어야 한다."""
        fig = self._call_with_unit(unit=20)
        assert fig.layout.xaxis.title.text == "시간 (sec)"

    def test_x_axis_6_ticks_for_all_units(self):
        """모든 단위에서 tick이 정확히 6개여야 한다."""
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        r = self._make_records([0.3, 0.7])
        for unit in (20, 40, 100):
            fig = _build_scatter_fig(r, 0.5, "test", unit)
            assert len(fig.layout.xaxis.tickvals) == 6, f"unit={unit}: tick 6개 필요"

    def _call_with_unit(self, unit: int):
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        r = self._make_records([0.3, 0.7])
        return _build_scatter_fig(r, 0.5, "test", unit)


# ── TestInspChartIntegration ───────────────────────────────────────────────────


class TestInspChartIntegration:
    """
    검사 차트 통합 검증 (FR-INSP-T2-05~07).

    시나리오 1: 자동 검사 20건 진행 후 unit=20 기준 그룹 1개 확정
    시나리오 2: 단위 전환(20→40→100) 시 그룹 수·레이블 재계산
    시나리오 3: 그룹 선택 시 올바른 기록 필터링 + 차트 데이터 검증
    """

    _THRESHOLD = 0.5

    def _make_records(self, n: int, defect_seqs: set | None = None) -> list[dict]:
        """n건의 검사 기록 생성. defect_seqs에 포함된 seq는 불량(score=0.7)."""
        defect_seqs = defect_seqs or set()
        return [
            {
                "seq":           seq,
                "inspected_at":  f"2026-06-24 14:00:{seq:02d}",
                "image_name":    f"img_{seq:04d}.png",
                "image_path":    (
                    f"/data/test/scratch/img_{seq:04d}.png"
                    if seq in defect_seqs
                    else f"/data/test/good/img_{seq:04d}.png"
                ),
                "verdict":       "불량" if seq in defect_seqs else "양품",
                "anomaly_score": 0.7 if seq in defect_seqs else 0.2,
            }
            for seq in range(1, n + 1)
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _group_count(total: int, unit: int) -> int:
        from inspection.utils.insp_session_init import get_sim_group_count
        return get_sim_group_count(total, unit)

    @staticmethod
    def _group_label(idx: int, unit: int) -> str:
        from inspection.utils.insp_session_init import get_sim_group_label
        return get_sim_group_label(idx, unit)

    @staticmethod
    def _group_recs(records: list, idx: int, unit: int) -> list:
        from inspection.tabs.insp_tab2_history import _get_group_records
        return _get_group_records(records, idx, unit)

    @staticmethod
    def _group_df(total: int, unit: int):
        from inspection.tabs.insp_tab2_history import _build_group_table_df
        return _build_group_table_df(total, unit)

    # ── 시나리오 1: 20건 진행 후 그룹 확정 ───────────────────────────────────

    def test_20_inspections_unit20_one_complete_group(self):
        """20건 완료 후 unit=20 → 정확히 1개 완성 그룹."""
        records = self._make_records(20)
        assert self._group_count(len(records), 20) == 1

    def test_20_inspections_group0_contains_all_20_records(self):
        """그룹 0에 정확히 20개 기록이 속한다."""
        records = self._make_records(20)
        g0 = self._group_recs(records, 0, 20)
        assert len(g0) == 20

    def test_20_inspections_group0_seq_range_1_to_20(self):
        """그룹 0의 seq는 1~20이어야 한다."""
        records = self._make_records(20)
        g0 = self._group_recs(records, 0, 20)
        seqs = sorted(r["seq"] for r in g0)
        assert seqs == list(range(1, 21))

    def test_20_inspections_group_label_is_correct(self):
        """20건, unit=20 → 그룹 0 레이블 '2026-06-24 14:00~14:01'."""
        assert self._group_label(0, 20) == "2026-06-24 14:00~14:01"

    def test_20_inspections_table_has_1_row(self):
        """그룹 테이블 행 수가 1이어야 한다."""
        df = self._group_df(20, 20)
        assert len(df) == 1

    def test_19_inspections_still_one_partial_group(self):
        """19건(미완성) → 그룹 1개(partial)."""
        assert self._group_count(19, 20) == 1

    def test_21st_inspection_creates_second_partial_group(self):
        """21번째 검사 → 그룹 2개 (1완성 + 1미완성)."""
        records = self._make_records(21)
        assert self._group_count(len(records), 20) == 2
        g1 = self._group_recs(records, 1, 20)
        assert len(g1) == 1  # seq=21만 속함

    # ── 시나리오 2: 단위 전환 재계산 ─────────────────────────────────────────

    def test_unit_switch_20_records_group_counts(self):
        """20건 기록, 단위별 그룹 수 재계산."""
        n = 20
        assert self._group_count(n, 20)  == 1   # 완성 1개
        assert self._group_count(n, 40)  == 1   # 미완성 1개 (20/40)
        assert self._group_count(n, 100) == 1   # 미완성 1개 (20/100)

    def test_unit_switch_labels_differ(self):
        """단위별 그룹 0 레이블이 서로 달라야 한다."""
        label_20  = self._group_label(0, 20)
        label_40  = self._group_label(0, 40)
        label_100 = self._group_label(0, 100)
        assert label_20  == "2026-06-24 14:00~14:01"
        assert label_40  == "2026-06-24 14:00~14:02"
        assert label_100 == "2026-06-24 14:00~14:05"
        assert len({label_20, label_40, label_100}) == 3  # 모두 다름

    def test_unit_switch_40_records_unit20_vs_unit40(self):
        """40건, unit=20→40 전환 시 그룹 0의 기록 수가 달라진다."""
        records = self._make_records(40)
        g0_u20 = self._group_recs(records, 0, 20)
        g0_u40 = self._group_recs(records, 0, 40)
        assert len(g0_u20) == 20  # unit=20 그룹 0: seq 1~20
        assert len(g0_u40) == 40  # unit=40 그룹 0: seq 1~40

    def test_unit_switch_60_records_group_counts(self):
        """60건, 단위별 그룹 수가 정확하다."""
        n = 60
        assert self._group_count(n, 20) == 3  # 60/20=3 완성
        assert self._group_count(n, 40) == 2  # 40완성+20미완성
        assert self._group_count(n, 100) == 1 # 60/100=0.6 → 1 미완성

    def test_unit_switch_group_table_row_count_changes(self):
        """단위 변경 시 테이블 행 수가 달라진다."""
        n = 60
        df20  = self._group_df(n, 20)
        df40  = self._group_df(n, 40)
        df100 = self._group_df(n, 100)
        assert len(df20)  == 3
        assert len(df40)  == 2
        assert len(df100) == 1

    def test_all_records_covered_after_unit_switch(self):
        """단위 전환 후에도 모든 기록이 어느 그룹에 속한다."""
        records = self._make_records(45)
        for unit in (20, 40, 100):
            n_groups = self._group_count(len(records), unit)
            total_in_groups = sum(
                len(self._group_recs(records, g, unit))
                for g in range(n_groups)
            )
            assert total_in_groups == len(records), (
                f"unit={unit}: {total_in_groups} != {len(records)}"
            )

    # ── 시나리오 3: 그룹 선택 후 차트 데이터 검증 ─────────────────────────────

    def test_group_selection_correct_records_45_inspections(self):
        """45건, unit=20 → 3개 그룹의 기록 수가 정확하다."""
        records = self._make_records(45)
        g0 = self._group_recs(records, 0, 20)
        g1 = self._group_recs(records, 1, 20)
        g2 = self._group_recs(records, 2, 20)
        assert len(g0) == 20
        assert len(g1) == 20
        assert len(g2) == 5

    def test_group1_scatter_x_positions_in_seconds(self):
        """그룹 1(seq 21~40)의 x positions가 0,3,6,...,57 (초 단위, 그룹 내 상대 시간)."""
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        records = self._make_records(40)
        g1 = self._group_recs(records, 1, 20)
        fig = _build_scatter_fig(g1, self._THRESHOLD, self._group_label(1, 20), 20)
        assert list(fig.data[0].x) == list(range(0, 60, 3))  # 0,3,6,...,57

    def test_partial_group_scatter_x_in_seconds_axis_0_to_60(self):
        """미완성 그룹(5건): x=0,3,6,9,12 (초), x축 0~60 고정 (unit=20)."""
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        records = self._make_records(45)
        g2 = self._group_recs(records, 2, 20)
        fig = _build_scatter_fig(g2, self._THRESHOLD, self._group_label(2, 20), 20)
        assert list(fig.data[0].x) == [0, 3, 6, 9, 12]
        assert tuple(fig.layout.xaxis.range) == (0, 60)  # 축은 0~unit×3 고정

    def test_histogram_normal_defect_split(self):
        """seq 1~10 양품 / 11~20 불량 → 히스토그램 trace 2개(정상+불량)."""
        from inspection.tabs.insp_tab2_history import _build_histogram_fig
        records = self._make_records(20, defect_seqs=set(range(11, 21)))
        g0 = self._group_recs(records, 0, 20)
        fig = _build_histogram_fig(
            g0, self._THRESHOLD, self._group_label(0, 20), 20
        )
        assert len(fig.data) == 2
        trace_names = {t.name for t in fig.data}
        assert "정상" in trace_names
        assert "불량" in trace_names

    def test_scatter_colors_match_verdicts(self):
        """seq 1~5 양품(파랑), 6~10 불량(빨강) → 색상 순서 정확."""
        from inspection.tabs.insp_tab2_history import _build_scatter_fig
        records = self._make_records(10, defect_seqs={6, 7, 8, 9, 10})
        g0 = self._group_recs(records, 0, 20)
        fig = _build_scatter_fig(
            g0, self._THRESHOLD, self._group_label(0, 20), 20
        )
        colors = list(fig.data[0].marker.color)
        for c in colors[:5]:
            assert "#4e9af1" in c, f"정상은 파란색이어야 함: {c}"
        for c in colors[5:]:
            assert "#e05555" in c, f"불량은 빨간색이어야 함: {c}"

    def test_group2_histogram_uses_only_group2_records(self):
        """그룹 2의 히스토그램은 그룹 2 기록(seq 41~45)만 사용한다."""
        from inspection.tabs.insp_tab2_history import _build_histogram_fig
        records = self._make_records(45)
        g2 = self._group_recs(records, 2, 20)
        fig = _build_histogram_fig(
            g2, self._THRESHOLD, self._group_label(2, 20), 20
        )
        # 5건, score=0.2 모두 정상 → trace 1개
        assert len(fig.data) == 1
        assert fig.data[0].name == "정상"

    def test_group_table_df_labels_continuous_across_groups(self):
        """연속된 그룹 레이블이 시간순으로 정렬된다."""
        df = self._group_df(60, 20)
        labels = list(df["시간 범위"])
        assert labels[0] == "2026-06-24 14:00~14:01"
        assert labels[1] == "2026-06-24 14:01~14:02"
        assert labels[2] == "2026-06-24 14:02~14:03"
        # 시간 순서 확인: 각 레이블이 이전 레이블보다 늦어야 함
        for i in range(1, len(labels)):
            start_prev = labels[i - 1].split(" ")[1].split("~")[0]
            start_curr = labels[i].split(" ")[1].split("~")[0]
            assert start_curr > start_prev


# ── TestNormalizeAnomalyScore ──────────────────────────────────────────────────


class TestNormalizeAnomalyScore:
    """normalize_anomaly_score() — anomaly score [0,1] 정규화 검증."""

    def _call(self, raw, s_min, s_max):
        from inspection.utils.insp_session_init import normalize_anomaly_score
        return normalize_anomaly_score(raw, s_min, s_max)

    # ── 기본 정규화 ────────────────────────────────────────────────────────────

    def test_midpoint_normalizes_to_0_5(self):
        """min=0, max=100, score=50 → 0.5."""
        assert self._call(50.0, 0.0, 100.0) == 0.5

    def test_min_value_normalizes_to_0(self):
        """score == score_min → 0.0."""
        assert self._call(10.0, 10.0, 90.0) == 0.0

    def test_max_value_normalizes_to_1(self):
        """score == score_max → 1.0."""
        assert self._call(90.0, 10.0, 90.0) == 1.0

    # ── clip 동작 ──────────────────────────────────────────────────────────────

    def test_score_below_min_clips_to_0(self):
        """score < score_min → 0.0 (학습보다 낮은 점수)."""
        assert self._call(5.0, 10.0, 90.0) == 0.0

    def test_score_above_max_clips_to_1(self):
        """score > score_max → 1.0 (학습보다 훨씬 높은 이상값)."""
        assert self._call(200.0, 10.0, 90.0) == 1.0

    # ── 실제 PatchCore 스케일 (63~64 범위) ────────────────────────────────────

    def test_patchcore_raw_score_range(self):
        """실제 PatchCore 스케일(63~64)을 0~1로 정규화."""
        score_min, score_max = 62.5, 65.0
        normalized = self._call(63.75, score_min, score_max)
        # (63.75 - 62.5) / (65.0 - 62.5) = 1.25 / 2.5 = 0.5
        assert abs(normalized - 0.5) < 1e-6

    def test_threshold_normalizes_consistently(self):
        """threshold도 같은 min/max로 정규화하면 판정 결과가 동일해야 한다."""
        score_min, score_max = 62.0, 66.0

        raw_threshold = 63.5
        raw_score_normal = 62.8  # 정상 (< threshold)
        raw_score_defect = 64.1  # 불량 (>= threshold)

        from inspection.utils.insp_session_init import normalize_anomaly_score
        thr_norm   = normalize_anomaly_score(raw_threshold,    score_min, score_max)
        norm_ok    = normalize_anomaly_score(raw_score_normal, score_min, score_max)
        norm_ng    = normalize_anomaly_score(raw_score_defect, score_min, score_max)

        # 정규화 전후 판정 결과 일치
        assert (raw_score_normal >= raw_threshold) == (norm_ok >= thr_norm)
        assert (raw_score_defect >= raw_threshold) == (norm_ng >= thr_norm)

    # ── 엣지 케이스 ────────────────────────────────────────────────────────────

    def test_min_equals_max_returns_zero(self):
        """score_min == score_max → 0.0 (division by zero 방어)."""
        assert self._call(50.0, 50.0, 50.0) == 0.0

    def test_returns_float(self):
        """반환값이 float 타입이어야 한다."""
        assert isinstance(self._call(5.0, 0.0, 10.0), float)

    def test_output_range_is_0_to_1(self):
        """어떤 입력이든 출력은 항상 [0, 1] 내에 있어야 한다."""
        from inspection.utils.insp_session_init import normalize_anomaly_score
        test_cases = [
            (0.0, 10.0, 90.0),
            (100.0, 10.0, 90.0),
            (50.0, 10.0, 90.0),
            (-999.0, 10.0, 90.0),
            (9999.0, 10.0, 90.0),
        ]
        for raw, s_min, s_max in test_cases:
            result = normalize_anomaly_score(raw, s_min, s_max)
            assert 0.0 <= result <= 1.0, f"raw={raw}, min={s_min}, max={s_max} → {result}"


# ── TestChartRenderingBug ──────────────────────────────────────────────────────


class TestChartRenderingBug:
    """#02 버그 테스트: 그룹 선택 후 히스토그램·산점도 렌더링 확인.

    재현 시나리오:
      1. insp_chart_selected_group = None (초기 상태)
      2. _render_group_table()가 0으로 설정
      3. _render_histogram() / _render_scatter()가 차트를 그려야 함
    """

    _THRESHOLD = 0.5

    def _make_records(self, n: int = 20) -> list[dict]:
        return [
            {"seq": i, "anomaly_score": 0.2 if i % 2 else 0.7,
             "verdict": "양품" if i % 2 else "불량",
             "inspected_at": f"2026-06-24 14:00:{i:02d}",
             "image_name": f"img_{i}.png",
             "image_path": f"/data/img_{i}.png"}
            for i in range(1, n + 1)
        ]

    # ── _render_group_table 세션 상태 갱신 검증 ───────────────────────────────

    def test_render_group_table_sets_selected_group_when_none(self):
        """selected_group이 None일 때 _render_group_table()이 0으로 설정한다."""
        from unittest.mock import patch, MagicMock
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import _render_group_table

        state = {"insp_chart_selected_group": None}
        mock_event = MagicMock()
        mock_event.selection.rows = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "dataframe", return_value=mock_event), \
             patch.object(_st, "caption"):
            _render_group_table(20, 20)

        assert state["insp_chart_selected_group"] == 0, (
            f"Expected 0, got {state['insp_chart_selected_group']}"
        )

    def test_render_group_table_updates_on_row_click(self):
        """행 클릭 이벤트가 selected_group을 올바르게 업데이트한다."""
        from unittest.mock import patch, MagicMock
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import _render_group_table

        # 45건: 3그룹(0,1,2). 사용자가 그룹 1 클릭
        state = {"insp_chart_selected_group": 0}
        mock_event = MagicMock()
        mock_event.selection.rows = [1]  # 행 1 = 그룹 1

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "dataframe", return_value=mock_event), \
             patch.object(_st, "caption"):
            _render_group_table(45, 20)

        assert state["insp_chart_selected_group"] == 1

    # ── _render_histogram 렌더링 조건 검증 ────────────────────────────────────

    def test_render_histogram_draws_chart_when_group_selected(self):
        """records 있고 selected_group=0이면 caption이 아닌 차트를 그린다."""
        from unittest.mock import patch, MagicMock
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import _render_histogram

        records = self._make_records(20)
        state = {
            "insp_chart_selected_group": 0,
            "insp_active_model": {"threshold": self._THRESHOLD},
        }
        captured_charts = []
        captured_captions = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "plotly_chart", side_effect=lambda f, **k: captured_charts.append(f)), \
             patch.object(_st, "caption", side_effect=lambda t: captured_captions.append(t)):
            _render_histogram(records, 20)

        assert len(captured_charts) == 1, (
            f"Expected 1 chart, got 0 charts. Captions shown: {captured_captions}"
        )
        assert len(captured_captions) == 0

    def test_render_histogram_shows_caption_when_selected_group_none(self):
        """selected_group=None이면 caption을 보여야 한다 (초기 상태 방어)."""
        from unittest.mock import patch
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import _render_histogram

        records = self._make_records(20)
        state = {
            "insp_chart_selected_group": None,  # 아직 설정 안 됨
            "insp_active_model": {"threshold": self._THRESHOLD},
        }
        captured_charts = []
        captured_captions = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "plotly_chart", side_effect=lambda f, **k: captured_charts.append(f)), \
             patch.object(_st, "caption", side_effect=lambda t: captured_captions.append(t)):
            _render_histogram(records, 20)

        assert len(captured_charts) == 0
        assert len(captured_captions) == 1

    # ── _render_scatter 렌더링 조건 검증 ──────────────────────────────────────

    def test_render_scatter_draws_chart_when_group_selected(self):
        """records 있고 selected_group=0이면 caption이 아닌 차트를 그린다."""
        from unittest.mock import patch
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import _render_scatter

        records = self._make_records(20)
        state = {
            "insp_chart_selected_group": 0,
            "insp_active_model": {"threshold": self._THRESHOLD},
        }
        captured_charts = []
        captured_captions = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "plotly_chart", side_effect=lambda f, **k: captured_charts.append(f)), \
             patch.object(_st, "caption", side_effect=lambda t: captured_captions.append(t)):
            _render_scatter(records, 20)

        assert len(captured_charts) == 1, (
            f"Expected 1 chart, got 0. Captions: {captured_captions}"
        )

    def test_render_scatter_shows_caption_when_no_records(self):
        """records=[] 이면 chart가 아닌 caption을 보여야 한다."""
        from unittest.mock import patch
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import _render_scatter

        state = {
            "insp_chart_selected_group": 0,
            "insp_active_model": {"threshold": self._THRESHOLD},
        }
        captured_charts = []
        captured_captions = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "plotly_chart", side_effect=lambda f, **k: captured_charts.append(f)), \
             patch.object(_st, "caption", side_effect=lambda t: captured_captions.append(t)):
            _render_scatter([], 20)

        assert len(captured_charts) == 0
        assert len(captured_captions) == 1

    # ── 단위 버튼 클릭 후 렌더 흐름 검증 ─────────────────────────────────────

    def test_after_unit_button_click_charts_render(self):
        """단위 버튼 클릭(selected_group=None) 후 rerun에서 차트가 그려져야 한다.

        흐름:
          1. 버튼 클릭 → insp_chart_selected_group = None 으로 리셋
          2. rerun → _render_group_table()이 selected_group = 0 설정
          3. _render_histogram()이 차트를 그려야 함
        """
        from unittest.mock import patch, MagicMock
        import streamlit as _st
        from inspection.tabs.insp_tab2_history import (
            _render_group_table, _render_histogram
        )

        records = self._make_records(20)

        # rerun 후 상태: 버튼 클릭으로 None이 되었음
        state = {
            "insp_chart_selected_group": None,
            "insp_active_model": {"threshold": self._THRESHOLD},
        }

        # Step 1: _render_group_table()이 selected_group을 0으로 복원
        mock_event = MagicMock()
        mock_event.selection.rows = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "dataframe", return_value=mock_event), \
             patch.object(_st, "caption"):
            _render_group_table(20, 20)

        # selected_group이 0으로 복원되어야 함
        assert state["insp_chart_selected_group"] == 0, (
            "단위 버튼 클릭 후 _render_group_table()이 selected_group=0을 복원해야 함"
        )

        # Step 2: _render_histogram()이 선택된 그룹의 차트를 그려야 함
        captured_charts = []
        captured_captions = []

        with patch.object(_st, "session_state", state), \
             patch.object(_st, "plotly_chart", side_effect=lambda f, **k: captured_charts.append(f)), \
             patch.object(_st, "caption", side_effect=lambda t: captured_captions.append(t)):
            _render_histogram(records, 20)

        assert len(captured_charts) == 1, (
            f"단위 변경 후 히스토그램이 그려져야 함. Captions: {captured_captions}"
        )
        assert len(captured_captions) == 0

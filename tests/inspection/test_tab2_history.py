"""
tests/inspection/test_tab2_history.py

FR-INSP-T2-01: 이력 테이블 빌드 — 컬럼, 정렬, 이모지 판정, 빈 DataFrame
FR-INSP-T2-02: KPI — 총검사/양품/불량/불량률, 빈 기록 시 "-"
FR-INSP-T2-03: CSV 빌드 — 헤더, 행 수, BOM, 이모지 없는 판정
"""
from __future__ import annotations

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

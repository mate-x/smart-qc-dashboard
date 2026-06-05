"""
inspection/utils/insp_session_init.py

책임:
  - 검사 세션 초기화 + st.cache_resource 기반 모델 캐시.
  - 실시간 통계 차트용 시뮬레이션 시간 계산 유틸리티 (FR-INSP-T2-05).

금지: 학습 관련 session_state 키 직접 수정, history.json 쓰기 (R-INSP-04).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from utils.model_factory import load_model_for_inference

# 00_Global_Context §1.11 INSPECTION_SESSION_SCHEMA
INSPECTION_SESSION_SCHEMA: dict = {
    # 대시보드 라우팅
    "active_dashboard":  "explorer",   # "explorer" | "inspection"

    # 적용 모델
    "insp_active_model": None,         # dict | None — history.json experiment 레코드 전체
                                       # None이면 탭1 검사 버튼 비활성화

    # 검사 이력
    "insp_records":      [],           # list[dict] — inspection_record 배열 (1.10절)
    "insp_seq_counter":  0,            # int — 다음 seq 값

    # 자동 검사 상태
    "insp_auto_active":  False,        # bool — 자동 검사(3초마다) 실행 중 여부

    # 마지막 추론 결과 (탭1 화면 유지용)
    "insp_last_result":      None,     # dict | None — {seq, inspected_at, image_name, image_path, verdict, anomaly_score}
    "insp_last_anomaly_map": None,     # np.ndarray | None — 직전 이상 맵 (H, W) float32

    # 팝업 제어
    "insp_defect_popup": False,        # bool — 불량 감지 팝업 표시 여부

    # 테스트 이미지 풀
    "insp_test_pool":    [],           # list[tuple[str, str]] — (절대경로, "양품"|"불량")
    "insp_pool_index":   0,            # int — 현재 샘플링 위치
}


@st.cache_resource
def _load_insp_model(model_path: str, model_type: str, device: str) -> object:
    """
    (model_path, model_type, device) 조합이 동일하면 캐시 반환.
    모델 교체 시 _load_insp_model.clear() 호출 필수 (A-19).

    model_config는 model_path/configs.yaml에서 읽는다.
    Raises: RuntimeError("ERR_INSP_MODEL_LOAD_FAILED: ...")
    """
    import yaml
    from pathlib import Path

    configs_path = Path(model_path) / "configs.yaml"
    try:
        with open(configs_path, encoding="utf-8") as f:
            configs = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ERR_INSP_MODEL_LOAD_FAILED: configs.yaml 없음 — {configs_path}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"ERR_INSP_MODEL_LOAD_FAILED: configs.yaml 파싱 실패 — {e}"
        ) from e

    model_config = configs.get("model", {})
    model_config["model_type"] = model_type

    try:
        return load_model_for_inference(
            exp_id="insp",
            model_path=model_path,
            model_config=model_config,
            device=device,
        )
    except RuntimeError as e:
        raise RuntimeError(f"ERR_INSP_MODEL_LOAD_FAILED: {e}") from e


def get_insp_model() -> object | None:
    """
    insp_active_model 기준으로 _load_insp_model() 호출.
    insp_active_model is None 이면 None 반환 (예외 전파 없음).
    """
    active = st.session_state.get("insp_active_model")
    if active is None:
        return None

    device = (st.session_state.get("device_info") or {}).get("device", "cpu")
    return _load_insp_model(
        model_path=active["model_path"],
        model_type=active["model_type"],
        device=device,
    )


def init_inspection_session() -> None:
    """
    INSPECTION_SESSION_SCHEMA 기준 session_state 초기화.
    멱등성: 이미 존재하는 키는 덮어쓰지 않는다 (00_Global §1.11).
    """
    for key, default in INSPECTION_SESSION_SCHEMA.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_inspection_state() -> None:
    """
    모델 교체 또는 이력 초기화 시 호출. insp_active_model은 유지 (R-INSP-05).
    호출 전에 _load_insp_model.clear()를 먼저 호출해야 한다 (A-19).
    """
    st.session_state.insp_records      = []
    st.session_state.insp_seq_counter  = 0
    st.session_state.insp_auto_active  = False
    st.session_state.insp_last_result      = None
    st.session_state.insp_last_anomaly_map = None
    st.session_state.insp_defect_popup     = False
    st.session_state.insp_test_pool    = []
    st.session_state.insp_pool_index   = 0


# ---------------------------------------------------------------------------
# Anomaly Score 정규화 (학습 데이터 기준 Min-Max)
# ---------------------------------------------------------------------------

def normalize_anomaly_score(
    raw_score: float,
    score_min: float,
    score_max: float,
) -> float:
    """
    anomaly score를 학습 테스트셋의 min/max 기준으로 [0, 1] 범위로 정규화.

    score_min == score_max(모든 학습 점수가 동일): 0.0 반환.
    범위 초과(학습보다 훨씬 높은/낮은 score): clip하여 0.0~1.0 유지.
    """
    if score_max <= score_min:
        return 0.0
    normalized = (raw_score - score_min) / (score_max - score_min)
    return float(max(0.0, min(1.0, normalized)))


# ---------------------------------------------------------------------------
# 시뮬레이션 시간 계산 유틸리티 (FR-INSP-T2-05)
# ---------------------------------------------------------------------------
# 차트 레이블 표시 전용 — 실제 inspected_at 필드와 무관 (Q-Final-2 A안)

_SIM_BASE              = datetime(2026, 6, 24, 14, 0, 0)
_SIM_INTERVAL_SECONDS  = 3  # 검사 1건 = 3초


def get_sim_timestamp(seq: int) -> datetime:
    """
    seq 번호(1-indexed)에 대응하는 시뮬레이션 타임스탬프 반환.

    규칙: 고정 시작 2026-06-24 14:00:00, 검사 간격 3초.
    seq=1 → 14:00:00 / seq=2 → 14:00:03 / seq=21 → 14:01:00
    """
    return _SIM_BASE + timedelta(seconds=(seq - 1) * _SIM_INTERVAL_SECONDS)


def get_sim_group_index(seq: int, unit: int) -> int:
    """
    seq 번호(1-indexed)가 속하는 그룹 인덱스(0-indexed) 반환.

    예: unit=20 → seq 1~20 → 0, seq 21~40 → 1
    """
    return (seq - 1) // unit


def get_sim_group_label(group_idx: int, unit: int) -> str:
    """
    그룹 인덱스(0-indexed)와 단위로 시간 범위 레이블 반환.

    형식: "YYYY-MM-DD HH:MM~HH:MM"

    예:
      group_idx=0, unit=20  → "2026-06-24 14:00~14:01"  (20×3=60s=1분)
      group_idx=1, unit=20  → "2026-06-24 14:01~14:02"
      group_idx=0, unit=40  → "2026-06-24 14:00~14:02"  (40×3=120s=2분)
      group_idx=0, unit=100 → "2026-06-24 14:00~14:05"  (100×3=300s=5분)
      group_idx=1, unit=100 → "2026-06-24 14:05~14:10"
    """
    start_dt = _SIM_BASE + timedelta(
        seconds=group_idx * unit * _SIM_INTERVAL_SECONDS
    )
    end_dt = _SIM_BASE + timedelta(
        seconds=(group_idx + 1) * unit * _SIM_INTERVAL_SECONDS
    )
    date_str  = start_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%H:%M")
    end_str   = end_dt.strftime("%H:%M")
    return f"{date_str} {start_str}~{end_str}"


def get_sim_group_count(total_seqs: int, unit: int) -> int:
    """
    전체 seq 수와 단위로 총 그룹 수 반환 (미완성 마지막 그룹 포함).

    예: total_seqs=0  → 0
        total_seqs=1  → 1 (그룹 0: seq 1)
        total_seqs=20 → 1 (그룹 0: seq 1~20)
        total_seqs=21 → 2 (그룹 0: 1~20, 그룹 1: 21~진행중)
    """
    if total_seqs <= 0:
        return 0
    return (total_seqs - 1) // unit + 1


def get_sim_all_group_labels(total_seqs: int, unit: int) -> list[str]:
    """
    현재까지 생성된 모든 그룹의 시간 범위 레이블 목록 반환.

    예: total_seqs=25, unit=20
        → ["2026-06-24 14:00~14:01", "2026-06-24 14:01~14:02"]
    """
    return [
        get_sim_group_label(i, unit)
        for i in range(get_sim_group_count(total_seqs, unit))
    ]

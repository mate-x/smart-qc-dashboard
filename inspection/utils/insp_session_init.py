"""
inspection/utils/insp_session_init.py

책임: 검사 세션 초기화 + st.cache_resource 기반 모델 캐시.
금지: 학습 관련 session_state 키 직접 수정, history.json 쓰기 (R-INSP-04).
"""
from __future__ import annotations

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

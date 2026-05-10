from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.cache_manager import get_anomaly_map_cache, set_anomaly_map_cache
from utils.image_utils import (
    anomaly_map_to_heatmap,
    build_gt_mask_path,
    create_triplet_image,
    load_image,
)
from utils.messages import MSG
from utils.metrics import compute_threshold


def render() -> None:
    st.header("탭6. 이상 영역 시각화")
    if not _guard():
        return

    exp_id: str = st.session_state["selected_experiment_id"]
    experiments: dict = st.session_state.get("experiments", {})
    exp = experiments.get(exp_id)

    if exp is None:
        st.error(f"실험 '{exp_id}'을 찾을 수 없습니다. 탭5에서 다시 선택해 주세요.")
        return

    if exp.get("status") == "중단":
        st.warning("중단된 실험은 이상 영역 시각화를 지원하지 않습니다.")
        return

    metrics = exp.get("metrics")
    if not metrics or not metrics.get("anomaly_scores"):
        st.warning("실험에 지표 정보가 없습니다.")
        return

    cache = _ensure_anomaly_map_cache(exp_id, exp)
    if cache is None:
        return

    _render_ui(exp_id, exp, metrics, cache)


def _guard() -> bool:
    if st.session_state.get("selected_experiment_id") is None:
        st.info(MSG["NO_SELECTED_EXP"])
        return False
    return True


def _ensure_anomaly_map_cache(exp_id: str, exp: dict) -> dict | None:
    """캐시 확인. MISS이면 모델 로드 + 재추론 후 캐시 저장."""
    cache = get_anomaly_map_cache(exp_id)
    if cache is not None:
        return cache

    model_path = exp.get("model_path")
    if not model_path:
        st.warning("모델 파일 경로가 없습니다. 저장에 실패한 실험입니다.")
        return None

    with st.spinner("Anomaly Map을 생성하는 중입니다..."):
        try:
            cache = _load_and_cache_anomaly_maps(exp_id, exp)
        except Exception as e:
            st.error(f"추론 중 오류가 발생했습니다: {e}")
            return None

    return cache


def _load_and_cache_anomaly_maps(exp_id: str, exp: dict) -> dict:
    """모델 로드 → 테스트셋 순차 추론 → 캐시 저장 후 반환."""
    import time

    from utils.model_factory import load_model_for_inference, run_inference
    from utils.mvtec_dataset import MVTecDataset

    image_size = exp.get("image_size", 256)
    model_config = {
        "model_type": exp["model_type"],
        "image_size": image_size,
        "batch_size": 16,
        "random_seed": 42,
        "params": exp.get("model_params", {}),
        "threshold_method": exp.get("threshold_method", "percentile"),
        "threshold_value": exp.get("threshold_value", 95.0),
    }
    preprocessing_config = {
        "method": exp.get("preprocessing_method", "none"),
        "params": exp.get("preprocessing_params"),
        "image_size": image_size,
    }

    device_str = (st.session_state.get("device_info") or {}).get("device", "cpu")
    model = load_model_for_inference(exp_id, exp["model_path"], model_config, device_str)

    test_ds = MVTecDataset(exp["dataset_path"], "test", preprocessing_config)

    image_paths: list[str] = []
    maps_list: list[np.ndarray] = []
    for i in range(len(test_ds)):
        item = test_ds[i]
        amap = run_inference(model, item["image"])
        image_paths.append(item["image_path"])
        maps_list.append(amap)

    maps_array = np.stack(maps_list, axis=0)  # (N, H, W)
    data = {"anomaly_maps": maps_array, "image_paths": image_paths}
    set_anomaly_map_cache(exp_id, data)
    return {**data, "cached_at": time.time()}


# ── 순수 함수 (테스트 가능) ────────────────────────────────────────────────────

def _classify(label: int, score: float, threshold: float) -> str:
    """이미지 하나의 FP/FN/TN/TP 분류."""
    pred = 1 if score >= threshold else 0
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    if label == 0 and pred == 0:
        return "TN"
    return "TP"


def _build_table_rows(
    image_paths: list[str],
    anomaly_scores: list[float],
    image_labels: list[int],
    threshold: float,
) -> list[dict]:
    """테이블 행 데이터 생성. _path_idx는 캐시 내 원본 인덱스."""
    n = min(len(image_paths), len(anomaly_scores), len(image_labels))
    rows = []
    for i in range(n):
        score = anomaly_scores[i]
        label = image_labels[i]
        cls = _classify(label, score, threshold)
        rows.append({
            "이미지명": Path(image_paths[i]).name,
            "결함 유형": Path(image_paths[i]).parent.name,
            "Anomaly Score": round(score, 6),
            "판정": "NG" if score >= threshold else "OK",
            "GT 일치": (cls in ("TN", "TP")),
            "오분류": cls,
            "_path_idx": i,
            "_path": image_paths[i],
        })
    return rows


def _overlay_binary_mask(heatmap: Image.Image, binary_mask: np.ndarray) -> Image.Image:
    """히트맵 위에 이진 마스크 윤곽선(빨간색)을 오버레이한 이미지를 반환."""
    arr = np.array(heatmap.convert("RGB"))
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (255, 0, 0), 2)
    return Image.fromarray(arr)


def _build_csv_bytes(df_display: pd.DataFrame) -> bytes:
    """테이블 데이터를 UTF-8 BOM CSV 바이트로 반환 (Excel 한글 호환)."""
    export_cols = ["이미지명", "결함 유형", "Anomaly Score", "판정", "GT 일치", "오분류"]
    available = [c for c in export_cols if c in df_display.columns]
    return df_display[available].to_csv(index=False).encode("utf-8-sig")


def _build_zip_bytes(
    df_display: pd.DataFrame,
    anomaly_maps: np.ndarray,
    exp: dict,
    threshold: float,
    exp_id: str,
) -> bytes:
    """필터된 행의 triplet PNG를 모두 묶어 ZIP 바이트로 반환."""
    dataset_path = exp.get("dataset_path", "")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for _, row in df_display.iterrows():
            path = str(row["_path"])
            cache_idx = int(row["_path_idx"])
            stem = Path(path).stem

            try:
                original = load_image(path)
            except Exception:
                continue

            gt_mask_pil: Image.Image | None = None
            gt_path = build_gt_mask_path(path, dataset_path)
            if gt_path.exists():
                try:
                    gt_mask_pil = Image.open(str(gt_path)).convert("RGB")
                except Exception:
                    gt_mask_pil = None

            amap = anomaly_maps[cache_idx]
            heatmap = anomaly_map_to_heatmap(amap)

            gt_mask_gray: np.ndarray | None = None
            if gt_mask_pil is not None:
                gt_arr = np.array(gt_mask_pil.convert("L"))
                gt_mask_gray = (gt_arr > 127).astype(np.uint8)

            if gt_mask_gray is not None and gt_mask_gray.any():
                heatmap_out = _overlay_binary_mask(heatmap, gt_mask_gray)
            else:
                heatmap_out = heatmap

            triplet = create_triplet_image(original, gt_mask_pil, heatmap_out)
            img_buf = io.BytesIO()
            triplet.save(img_buf, format="PNG")
            zf.writestr(f"{exp_id}_{stem}_anomaly.png", img_buf.getvalue())

    buf.seek(0)
    return buf.getvalue()


# ── UI 렌더링 ─────────────────────────────────────────────────────────────────

def _render_ui(exp_id: str, exp: dict, metrics: dict, cache: dict) -> None:
    anomaly_scores: list[float] = metrics["anomaly_scores"]
    image_labels: list[int] = metrics["image_labels"]
    image_paths: list[str] = cache["image_paths"]
    anomaly_maps: np.ndarray = cache["anomaly_maps"]

    max_score = float(max(anomaly_scores)) if anomaly_scores else 1.0
    slider_max = max_score * 1.2

    # 기본 threshold 초기화 (탭 첫 진입 또는 실험 전환 시)
    if st.session_state.get("anomaly_map_threshold") is None:
        normal_scores = [s for s, lbl in zip(anomaly_scores, image_labels) if lbl == 0]
        if normal_scores:
            default_thr = compute_threshold(
                np.array(normal_scores, dtype=np.float32),
                exp.get("threshold_method", "percentile"),
                float(exp.get("threshold_value", 95.0)),
            )
        else:
            default_thr = max_score * 0.5
        st.session_state["anomaly_map_threshold"] = round(float(default_thr), 6)

    # FR-T6-06 (S): 결함 유형 필터
    all_classes = sorted({Path(p).parent.name for p in image_paths})
    selected_class = st.selectbox(
        "결함 유형 필터",
        ["전체"] + all_classes,
        key="tab6_class_filter",
    )

    # FR-T6-03: Threshold 슬라이더
    current_thr = float(st.session_state["anomaly_map_threshold"])
    current_thr = max(0.0, min(current_thr, slider_max))
    threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=float(slider_max),
        value=current_thr,
        step=0.001,
        format="%.3f",
    )
    st.session_state["anomaly_map_threshold"] = threshold

    # 행 데이터 + 필터 적용
    rows = _build_table_rows(image_paths, anomaly_scores, image_labels, threshold)
    df_full = pd.DataFrame(rows)

    if selected_class != "전체":
        df_display = df_full[df_full["결함 유형"] == selected_class].reset_index(drop=True)
    else:
        df_display = df_full.reset_index(drop=True)

    scores_view = df_display["Anomaly Score"].tolist()

    # FR-T6-07 (S): Score 요약
    if scores_view:
        col1, col2 = st.columns(2)
        col1.metric("최대 Anomaly Score", f"{max(scores_view):.4f}")
        col2.metric("평균 Anomaly Score", f"{sum(scores_view) / len(scores_view):.4f}")

    # FR-T6-08 (S): FP/FN/TN/TP 카운트
    cls_counts = df_display["오분류"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", int(cls_counts.get("TP", 0)))
    c2.metric("FP", int(cls_counts.get("FP", 0)))
    c3.metric("TN", int(cls_counts.get("TN", 0)))
    c4.metric("FN", int(cls_counts.get("FN", 0)))

    # FR-T6-02: 이미지 목록 테이블 — GT 일치를 ✓/✗ 기호로 표시
    df_table = df_display.copy()
    df_table["GT 일치"] = df_table["GT 일치"].map({True: "✓", False: "✗"})

    display_cols = ["이미지명", "결함 유형", "Anomaly Score", "판정", "GT 일치", "오분류"]
    event = st.dataframe(
        df_table[display_cols],
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="tab6_image_table",
    )

    selected_rows = event.selection.rows if event else []
    if selected_rows:
        sel_idx = selected_rows[0]
        if 0 <= sel_idx < len(df_display):
            row = df_display.iloc[sel_idx]
            cache_idx = int(row["_path_idx"])
            _render_triplet(
                path=str(row["_path"]),
                anomaly_map=anomaly_maps[cache_idx],
                dataset_path=exp.get("dataset_path", ""),
                threshold=threshold,
                exp_id=exp_id,
            )

    st.divider()

    # 내보내기 버튼
    col_csv, col_zip = st.columns(2)

    with col_csv:
        csv_bytes = _build_csv_bytes(df_display)
        st.download_button(
            label="CSV 내보내기",
            data=csv_bytes,
            file_name=f"{exp_id}_results.csv",
            mime="text/csv",
            key="tab6_csv_export",
        )

    with col_zip:
        zip_key = f"tab6_zip_{exp_id}_{selected_class}_{threshold:.3f}"
        if st.button("ZIP 준비", key="tab6_zip_prepare"):
            with st.spinner("ZIP 파일 생성 중..."):
                st.session_state["tab6_zip_bytes"] = _build_zip_bytes(
                    df_display, anomaly_maps, exp, threshold, exp_id
                )
                st.session_state["tab6_zip_key"] = zip_key

        if st.session_state.get("tab6_zip_key") == zip_key and st.session_state.get("tab6_zip_bytes"):
            st.download_button(
                label="ZIP 다운로드",
                data=st.session_state["tab6_zip_bytes"],
                file_name=f"{exp_id}_anomaly_maps.zip",
                mime="application/zip",
                key="tab6_zip_download",
            )


def _render_triplet(
    path: str,
    anomaly_map: np.ndarray,
    dataset_path: str,
    threshold: float,
    exp_id: str,
) -> None:
    """FR-T6-04: 원본 / GT 마스크 / Heatmap 3-패널 시각화 + FR-T6-05: PNG 다운로드."""
    st.subheader("이미지 상세 시각화")

    try:
        original = load_image(path)
    except Exception as e:
        st.warning(f"이미지를 불러올 수 없습니다: {e}")
        return

    gt_mask_pil: Image.Image | None = None
    gt_path = build_gt_mask_path(path, dataset_path)
    if gt_path.exists():
        try:
            gt_mask_pil = Image.open(str(gt_path)).convert("RGB")
        except Exception:
            gt_mask_pil = None

    heatmap = anomaly_map_to_heatmap(anomaly_map)

    # GT 마스크가 있으면 이진 윤곽선 오버레이 적용
    gt_mask_gray: np.ndarray | None = None
    if gt_mask_pil is not None:
        gt_arr = np.array(gt_mask_pil.convert("L"))
        gt_mask_gray = (gt_arr > 127).astype(np.uint8)

    if gt_mask_gray is not None and gt_mask_gray.any():
        heatmap_display = _overlay_binary_mask(heatmap, gt_mask_gray)
    else:
        heatmap_display = heatmap

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("원본 이미지")
        st.image(original, use_container_width=True)
    with col2:
        st.caption("GT 마스크")
        if gt_mask_pil is not None:
            st.image(gt_mask_pil, use_container_width=True)
        else:
            blank = Image.new("RGB", original.size, (40, 40, 40))
            st.image(blank, use_container_width=True)
            st.caption("GT 마스크 없음")
    with col3:
        st.caption("Anomaly Heatmap (윤곽선 오버레이)")
        st.image(heatmap_display, use_container_width=True)

    # Score / Threshold / 판정 요약 (PRD §07 §7.5)
    score = float(np.max(anomaly_map))
    verdict = "NG" if score >= threshold else "OK"
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomaly Score", f"{score:.4f}")
    m2.metric("Threshold", f"{threshold:.4f}")
    m3.metric("판정", verdict)

    # FR-T6-05: PNG 다운로드
    triplet = create_triplet_image(original, gt_mask_pil, heatmap_display)
    buf = io.BytesIO()
    triplet.save(buf, format="PNG")
    buf.seek(0)
    stem = Path(path).stem
    st.download_button(
        label="PNG 다운로드",
        data=buf.getvalue(),
        file_name=f"{exp_id}_{stem}_anomaly.png",
        mime="image/png",
        key=f"tab6_dl_{stem}",
    )

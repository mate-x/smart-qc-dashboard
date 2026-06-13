"""
inspection/utils/test_sampler.py

책임: test_pool 구성(build) + 순서 관리(sample).
금지: st.session_state 직접 수정 (sample_from_pool 제외).

A-17 레이블 규칙:
  dataset_path/test/good/ 하위 이미지 → "양품"
  그 외 폴더 (결함 클래스)            → "불량"
"""
from __future__ import annotations

import random
from pathlib import Path

import streamlit as st

from utils.logger import log_info

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_test_pool(
    dataset_path: str,
    background_method: str = "none",
) -> list[tuple[str, str]]:
    """
    dataset_path/test/ 하위 이미지 스캔 → (절대경로, gt_label) 리스트.
    background_method in ("sam2", "sam3") 이면 {dataset_name}_{method}/test/ 를 우선 사용.
    해당 폴더가 없으면 dataset_path/test/ 로 fallback.
    gt_label: "양품" (good/) | "불량" (기타 클래스) — A-17 레이블 규칙.

    반환 전 random.shuffle() 1회 적용.

    Raises:
        FileNotFoundError: 사용할 test/ 디렉토리가 존재하지 않을 때.
    Returns:
        list[tuple[str, str]] — 이미지 없으면 빈 리스트.
    """
    root = Path(dataset_path)

    if background_method in ("sam2", "sam3"):
        _lower = root.parent / f"{root.name}_{background_method}"
        _upper = root.parent / f"{root.name}_{background_method.upper()}"
        bg_root = _lower if _lower.is_dir() else _upper
        candidate = bg_root / "test"
        test_root = candidate if candidate.is_dir() else root / "test"
    else:
        test_root = root / "test"

    if not test_root.exists():
        raise FileNotFoundError(
            f"테스트 디렉토리가 없습니다: {test_root}"
        )

    pool: list[tuple[str, str]] = []
    for cls_dir in sorted(test_root.iterdir()):   # sorted: 재현 가능한 초기 순서
        if not cls_dir.is_dir():
            continue
        label = "양품" if cls_dir.name == "good" else "불량"
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS and img_path.exists():
                pool.append((str(img_path.resolve()), label))

    random.shuffle(pool)
    return pool


def sample_from_pool() -> tuple[str, str, bool]:
    """
    session_state.insp_test_pool 에서 insp_pool_index 위치 샘플 반환 후 index 증가.
    pool 소진(index >= len(pool)) 시 재셔플 + index 리셋 (A-16 pool 소진 정책).

    Raises:
        RuntimeError("ERR_INSP_TEST_POOL_EMPTY"): pool이 비어 있을 때.
    Returns:
        (image_path: str, gt_label: str, was_reshuffled: bool)
        was_reshuffled: 이번 호출에서 pool 소진 후 재셔플이 발생했으면 True.
    """
    pool: list[tuple[str, str]] = st.session_state["insp_test_pool"]
    index: int = st.session_state["insp_pool_index"]

    if not pool:
        raise RuntimeError(
            "ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지가 없습니다. "
            "데이터셋 경로를 확인하거나 탭3에서 모델을 재선택해 주세요."
        )

    was_reshuffled = False
    if index >= len(pool):
        random.shuffle(pool)
        st.session_state["insp_test_pool"] = pool
        index = 0
        was_reshuffled = True
        log_info(
            "insp_pool_reshuffled",
            "테스트 풀 소진 — 재셔플",
            tab="insp_tab1",
            data={"pool_size": len(pool)},
        )

    sample = pool[index]
    st.session_state["insp_pool_index"] = index + 1
    return sample[0], sample[1], was_reshuffled

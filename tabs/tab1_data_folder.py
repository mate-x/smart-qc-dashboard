from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from utils.image_utils import SUPPORTED_FORMATS
from utils.messages import ERR, MSG

# 데이터셋 디렉토리에서 invalid 파일 카운트 시 무시할 확장자.
# 숨김 파일(.DS_Store 등)은 파일명 앞 '.' 체크로 별도 처리.
_NON_IMAGE_EXTS = {
    ".txt", ".json", ".yaml", ".yml", ".csv", ".log", ".md",
    ".xml", ".ini", ".cfg", ".toml", ".zip", ".tar", ".gz",
    ".py", ".sh", ".bat", ".exe", ".db", ".npy", ".npz",
}


def render() -> None:
    st.header("탭1. 데이터 폴더 구조")
    _render_path_input()

    dataset_meta = st.session_state.get("dataset_meta")
    if dataset_meta is None:
        return

    # FR-T1-06: Grayscale 감지 안내
    if dataset_meta["channels"] == 1:
        st.info(MSG["GRAYSCALE_DETECT"])

    # FR-T1-07: 지원 포맷 외 파일 경고 (S)
    if dataset_meta["has_invalid_files"]:
        n = dataset_meta.get("_invalid_file_count", 0)
        st.warning(f"지원하지 않는 파일 {n}개가 발견되었습니다. 학습에서 제외됩니다.")

    # FR-T1-03: 폴더 구조 트리
    st.subheader("폴더 구조")
    tree = _build_tree_text(Path(st.session_state["dataset_path"]), dataset_meta)
    st.code(tree, language=None)

    # FR-T1-04: 클래스별 이미지 수 테이블
    st.subheader("클래스별 이미지 수")
    st.dataframe(_build_count_table(dataset_meta), use_container_width=True)

    # FR-T1-05: 대표 썸네일
    _render_thumbnails(Path(st.session_state["dataset_path"]), dataset_meta)


# ---------------------------------------------------------------------------
# FR-T1-01: 경로 입력 UI
# ---------------------------------------------------------------------------

def _render_path_input() -> None:
    path_value = st.text_input(
        label="데이터셋 경로 (Dataset Path)",
        key="input_dataset_path",
        placeholder="예: C:/datasets/mvtec/screw",
        help="MVTec AD 형식의 데이터셋 루트 경로를 입력하세요.",
    )
    if st.button("경로 확인", type="primary", key="_tab1_validate_btn"):
        _validate_and_load((path_value or "").strip())


# ---------------------------------------------------------------------------
# FR-T1-01: 4단계 검증 + FR-T1-08: 상세 오류 메시지
# ---------------------------------------------------------------------------

def _validate_and_load(path_str: str) -> None:
    if not path_str:
        st.error(ERR["ERR_DATASET_NOT_FOUND"])
        _clear_dataset_state()
        return

    root = Path(path_str)

    # Step 1: 경로 존재 여부
    if not root.exists():
        st.error(ERR["ERR_DATASET_NOT_FOUND"])
        _clear_dataset_state()
        return

    # Step 2: train/good/ 존재 여부 (FR-T1-08 상세 메시지)
    train_good = root / "train" / "good"
    if not train_good.is_dir():
        st.error(f"누락된 폴더: `train/good/` — {MSG['INVALID_FOLDER']}")
        _clear_dataset_state()
        return

    # Step 3: test/ 존재 여부 (FR-T1-08 상세 메시지)
    test_dir = root / "test"
    if not test_dir.is_dir():
        st.error(f"누락된 폴더: `test/` — {MSG['INVALID_FOLDER']}")
        _clear_dataset_state()
        return

    # Step 4: train/good/ 이미지 최소 1개 이상
    has_train_image = any(
        f.suffix.lower() in SUPPORTED_FORMATS for f in train_good.iterdir()
    )
    if not has_train_image:
        st.error("train/good/ 에 유효한 이미지가 없습니다.")
        _clear_dataset_state()
        return

    # Step 5: 검증 통과 → dataset_meta 구성 및 저장
    meta = _build_dataset_meta(root)
    _handle_path_change(str(root))
    st.session_state["dataset_path"] = str(root)
    st.session_state["dataset_meta"] = meta
    st.success("데이터셋 구조 검증 완료.")


# ---------------------------------------------------------------------------
# FR-T1-02: dataset_meta 구성 (00_Global_Context §1.5 스키마)
# ---------------------------------------------------------------------------

def _build_dataset_meta(root: Path) -> dict:
    train_good = root / "train" / "good"
    train_images = sorted(
        f for f in train_good.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
    )

    # 채널 감지: 첫 번째 train/good 이미지 기준
    channels = 3
    if train_images:
        try:
            with Image.open(train_images[0]) as img:
                channels = 1 if img.mode == "L" else 3
        except Exception:
            channels = 3

    # test/ 하위 디렉토리 스캔 (good 포함 — FR-T1-02)
    test_dir = root / "test"
    test_counts: dict[str, int] = {}
    defect_classes: list[str] = []

    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        count = sum(
            1 for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
        )
        test_counts[cls_dir.name] = count
        defect_classes.append(cls_dir.name)  # good 포함

    total_test_count = sum(test_counts.values())

    # ground_truth/ 하위 디렉토리 스캔 (선택적)
    gt_dir = root / "ground_truth"
    gt_counts: dict[str, int] = {}
    if gt_dir.is_dir():
        for cls_dir in sorted(gt_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            count = sum(
                1 for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
            )
            gt_counts[cls_dir.name] = count

    # 포맷 목록 + 지원 외 파일 감지
    # · 숨김 파일(이름이 '.'으로 시작, 예: .DS_Store)은 제외
    # · 메타/설정 파일(_NON_IMAGE_EXTS)은 제외
    # · 위 조건을 제외하고 SUPPORTED_FORMATS에 없는 파일만 invalid로 집계
    found_formats: set[str] = set()
    invalid_count = 0
    for scan_dir in [root / "train", test_dir, gt_dir]:
        if not scan_dir.exists():
            continue
        for f in scan_dir.rglob("*"):
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            ext = f.suffix.lower()
            if ext in SUPPORTED_FORMATS:
                found_formats.add(ext)
            elif ext and ext not in _NON_IMAGE_EXTS:
                invalid_count += 1

    return {
        "dataset_path": str(root),
        "train_good_count": len(train_images),
        "test_counts": test_counts,
        "gt_counts": gt_counts,
        "total_test_count": total_test_count,
        "channels": channels,
        "defect_classes": defect_classes,
        "supported_formats": sorted(found_formats),
        "has_invalid_files": invalid_count > 0,
        "_invalid_file_count": invalid_count,  # UI 표시용 (스키마 외 필드)
    }


# ---------------------------------------------------------------------------
# FR-T1-03: 폴더 트리 (최대 3단계)
# ---------------------------------------------------------------------------

def _build_tree_text(root: Path, meta: dict) -> str:
    lines: list[str] = [f"📂 {root.name}/"]

    train_dir = root / "train"
    if train_dir.is_dir():
        lines.append("  📂 train/")
        good_dir = train_dir / "good"
        if good_dir.is_dir():
            lines.append(f"    📂 good/ ({meta['train_good_count']}장)")

    test_dir = root / "test"
    if test_dir.is_dir():
        lines.append("  📂 test/")
        for cls_name, count in sorted(meta["test_counts"].items()):
            lines.append(f"    📂 {cls_name}/ ({count}장)")

    gt_dir = root / "ground_truth"
    if gt_dir.is_dir():
        lines.append("  📂 ground_truth/")
        for cls_name, count in sorted(meta["gt_counts"].items()):
            lines.append(f"    📂 {cls_name}/ ({count}장)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FR-T1-04: 클래스별 이미지 수 테이블
# ---------------------------------------------------------------------------

def _build_count_table(meta: dict) -> pd.DataFrame:
    rows: list[dict] = []

    # good 행: 학습/테스트 모두 표시
    rows.append({
        "클래스": "good",
        "학습(train)": meta["train_good_count"],
        "테스트(test)": meta["test_counts"].get("good", 0),
        "GT 마스크": meta["gt_counts"].get("good", 0),
    })

    # 결함 클래스 행 (good 제외)
    for cls in meta["defect_classes"]:
        if cls == "good":
            continue
        rows.append({
            "클래스": cls,
            "학습(train)": 0,
            "테스트(test)": meta["test_counts"].get(cls, 0),
            "GT 마스크": meta["gt_counts"].get(cls, 0),
        })

    df = pd.DataFrame(rows)
    total = pd.DataFrame([{
        "클래스": "합계",
        "학습(train)": int(df["학습(train)"].sum()),
        "테스트(test)": int(df["테스트(test)"].sum()),
        "GT 마스크": int(df["GT 마스크"].sum()),
    }])
    return pd.concat([df, total], ignore_index=True)


# ---------------------------------------------------------------------------
# FR-T1-05: 클래스 대표 썸네일 (최대 4열)
# ---------------------------------------------------------------------------

def _render_thumbnails(root: Path, meta: dict) -> None:
    classes = meta["defect_classes"]
    if not classes:
        return

    test_dir = root / "test"
    st.subheader("클래스 대표 이미지")

    for row_start in range(0, len(classes), 4):
        row_classes = classes[row_start : row_start + 4]
        cols = st.columns(len(row_classes))
        for col, cls_name in zip(cols, row_classes):
            cls_dir = test_dir / cls_name
            if not cls_dir.is_dir():
                continue
            images = sorted(
                f for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
            )
            if not images:
                col.write(f"{cls_name}: 이미지 없음")
                continue
            try:
                img = Image.open(images[0]).convert("RGB")
                img = img.resize((150, 150), Image.LANCZOS)
                col.image(img, caption=cls_name, width=150)
            except Exception:
                col.warning(f"{cls_name}: 이미지 로드 실패")


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _handle_path_change(new_path: str) -> None:
    """새 경로가 기존 dataset_path와 다를 때 하위 session_state 초기화."""
    if new_path != st.session_state.get("dataset_path"):
        st.session_state["preprocessing_config"] = None
        st.session_state["model_config"] = None
        st.session_state["device_info"] = None
        # experiments, selected_experiment_id는 유지 (이전 실험 보존)


def _clear_dataset_state() -> None:
    st.session_state["dataset_path"] = None
    st.session_state["dataset_meta"] = None

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from utils.dataset_converter import count_images, detect_ok_ng_dirs
from utils.image_utils import SUPPORTED_FORMATS
from utils.messages import ERR, MSG

# 데이터셋 디렉토리에서 invalid 파일 카운트 시 무시할 확장자.
# 숨김 파일(.DS_Store 등)은 파일명 앞 '.' 체크로 별도 처리.
_NON_IMAGE_EXTS = {
    ".txt", ".json", ".yaml", ".yml", ".csv", ".log", ".md",
    ".xml", ".ini", ".cfg", ".toml", ".zip", ".tar", ".gz",
    ".py", ".sh", ".bat", ".exe", ".db", ".npy", ".npz",
}

# OK/NG 자동 분할 비율 (train:test = 8:2)
_DEFAULT_TRAIN_RATIO = 0.8


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

    # OK/NG 형식 안내 배너
    if dataset_meta.get("dataset_format") == "oking":
        train_n    = dataset_meta["train_good_count"]
        ok_total   = dataset_meta.get("_oking_ok_count", 0)
        ng_total   = dataset_meta.get("_oking_ng_count", 0)
        ratio_pct  = int(_DEFAULT_TRAIN_RATIO * 100)
        st.info(
            f"**OK/NG 형식으로 로드됩니다.**  \n"
            f"OK {ok_total:,}장 중 {ratio_pct}% ({train_n:,}장)을 학습에, "
            f"나머지 {ok_total - train_n:,}장을 테스트(정상)에 사용합니다.  \n"
            f"NG {ng_total}장은 테스트(불량)로 사용합니다."
        )

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
        placeholder="예: C:/datasets/bolt  또는  C:/datasets/mvtec/screw",
        help="OK/NG 폴더 형식 또는 MVTec AD 형식 모두 지원합니다.",
    )
    if st.button("경로 확인", type="primary", key="_tab1_validate_btn"):
        _validate_and_load((path_value or "").strip())


# ---------------------------------------------------------------------------
# 경로 검증 + 데이터셋 메타 구성
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

    train_good = root / "train" / "good"

    # ── OK/NG 폴더 형식 감지 (MVTec 구조 없을 때)
    if not train_good.is_dir():
        ok_dir, ng_dir = detect_ok_ng_dirs(root)
        if ok_dir is not None:
            meta = _build_dataset_meta_oking(root, ok_dir, ng_dir)
            if meta["train_good_count"] == 0:
                st.error(f"`{ok_dir.name}/` 폴더에 유효한 이미지가 없습니다.")
                _clear_dataset_state()
                return
            _handle_path_change(str(root))
            st.session_state["dataset_path"] = str(root)
            st.session_state["dataset_meta"] = meta
            st.success(
                f"OK/NG 형식 데이터셋 확인 완료.  "
                f"(OK {meta['_oking_ok_count']:,}장 / NG {meta['_oking_ng_count']}장)"
            )
            return

        st.error(
            f"지원하지 않는 폴더 구조입니다.  \n"
            f"**OK/NG 형식**: `OK/`, `NG/` 폴더가 있어야 합니다.  \n"
            f"**MVTec AD 형식**: `train/good/`, `test/` 폴더가 있어야 합니다."
        )
        _clear_dataset_state()
        return

    # ── MVTec AD 형식 검증
    test_dir = root / "test"
    if not test_dir.is_dir():
        st.error(f"누락된 폴더: `test/` — {MSG['INVALID_FOLDER']}")
        _clear_dataset_state()
        return

    has_train_image = any(
        f.suffix.lower() in SUPPORTED_FORMATS for f in train_good.iterdir()
    )
    if not has_train_image:
        st.error("train/good/ 에 유효한 이미지가 없습니다.")
        _clear_dataset_state()
        return

    meta = _build_dataset_meta_mvtec(root)
    _handle_path_change(str(root))
    st.session_state["dataset_path"] = str(root)
    st.session_state["dataset_meta"] = meta
    st.success("MVTec AD 형식 데이터셋 확인 완료.")


# ---------------------------------------------------------------------------
# dataset_meta 구성 — OK/NG 형식
# ---------------------------------------------------------------------------

def _build_dataset_meta_oking(root: Path, ok_dir: Path, ng_dir: "Path | None") -> dict:
    ok_images = sorted(
        f for f in ok_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
    )
    ok_count = len(ok_images)
    ng_count = count_images(ng_dir) if ng_dir else 0

    train_n     = max(1, int(ok_count * _DEFAULT_TRAIN_RATIO)) if ok_count else 0
    test_good_n = max(0, ok_count - train_n)

    # 채널 감지
    channels = 3
    if ok_images:
        try:
            with Image.open(ok_images[0]) as img:
                channels = 1 if img.mode == "L" else 3
        except Exception:
            channels = 3

    ng_key       = ng_dir.name.lower() if ng_dir else "ng"
    test_counts  = {"good": test_good_n}
    defect_classes: list[str] = []
    if ng_dir and ng_count > 0:
        test_counts[ng_key] = ng_count
        defect_classes.append(ng_key)

    found_formats: set[str] = set()
    for f in ok_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            found_formats.add(f.suffix.lower())

    return {
        "dataset_path":      str(root),
        "dataset_format":    "oking",
        "train_good_count":  train_n,
        "test_counts":       test_counts,
        "gt_counts":         {},
        "total_test_count":  sum(test_counts.values()),
        "channels":          channels,
        "defect_classes":    defect_classes,
        "supported_formats": sorted(found_formats),
        "has_invalid_files": False,
        "_invalid_file_count": 0,
        # OK/NG 전용 메타
        "_oking_ok_dir":   ok_dir.name,
        "_oking_ng_dir":   ng_dir.name if ng_dir else None,
        "_oking_ok_count": ok_count,
        "_oking_ng_count": ng_count,
        "_train_ratio":    _DEFAULT_TRAIN_RATIO,
    }


# ---------------------------------------------------------------------------
# dataset_meta 구성 — MVTec AD 형식 (기존 로직)
# ---------------------------------------------------------------------------

def _build_dataset_meta_mvtec(root: Path) -> dict:
    train_good = root / "train" / "good"
    train_images = sorted(
        f for f in train_good.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
    )

    channels = 3
    if train_images:
        try:
            with Image.open(train_images[0]) as img:
                channels = 1 if img.mode == "L" else 3
        except Exception:
            channels = 3

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
        defect_classes.append(cls_dir.name)

    total_test_count = sum(test_counts.values())

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
        "dataset_path":      str(root),
        "dataset_format":    "mvtec",
        "train_good_count":  len(train_images),
        "test_counts":       test_counts,
        "gt_counts":         gt_counts,
        "total_test_count":  total_test_count,
        "channels":          channels,
        "defect_classes":    defect_classes,
        "supported_formats": sorted(found_formats),
        "has_invalid_files": invalid_count > 0,
        "_invalid_file_count": invalid_count,
    }


# 하위 호환: 이전 코드가 _build_dataset_meta를 호출하는 경우 대비
_build_dataset_meta = _build_dataset_meta_mvtec


# ---------------------------------------------------------------------------
# FR-T1-03: 폴더 트리 (최대 3단계)
# ---------------------------------------------------------------------------

def _build_tree_text(root: Path, meta: dict) -> str:
    lines: list[str] = [f"📂 {root.name}/"]

    if meta.get("dataset_format") == "oking":
        ok_name = meta.get("_oking_ok_dir", "OK")
        ng_name = meta.get("_oking_ng_dir")
        ok_n    = meta.get("_oking_ok_count", 0)
        ng_n    = meta.get("_oking_ng_count", 0)
        train_n = meta["train_good_count"]
        test_g  = meta["test_counts"].get("good", 0)
        lines.append(f"  📂 {ok_name}/ ({ok_n:,}장 전체)")
        lines.append(f"    ↳ 학습(train): {train_n:,}장  |  테스트(good): {test_g:,}장  ← 자동 분할")
        if ng_name:
            lines.append(f"  📂 {ng_name}/ ({ng_n}장) ← 테스트(불량)")
    else:
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

    rows.append({
        "클래스": "good (정상)",
        "학습(train)": meta["train_good_count"],
        "테스트(test)": meta["test_counts"].get("good", 0),
        "GT 마스크": meta["gt_counts"].get("good", 0),
    })

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
    st.subheader("클래스 대표 이미지")

    if meta.get("dataset_format") == "oking":
        cols_data: list[tuple[str, Path]] = []

        ok_name = meta.get("_oking_ok_dir", "OK")
        ok_dir  = root / ok_name
        if ok_dir.is_dir():
            cols_data.append(("OK (정상)", ok_dir))

        ng_name = meta.get("_oking_ng_dir")
        if ng_name:
            ng_dir = root / ng_name
            if ng_dir.is_dir():
                cols_data.append(("NG (불량)", ng_dir))

        if not cols_data:
            return
        cols = st.columns(len(cols_data))
        for col, (label, img_dir) in zip(cols, cols_data):
            images = sorted(
                f for f in img_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
            )
            if not images:
                col.write(f"{label}: 이미지 없음")
                continue
            try:
                img = Image.open(images[0]).convert("RGB").resize((150, 150), Image.LANCZOS)
                col.image(img, caption=label, width=150)
            except Exception:
                col.warning(f"{label}: 이미지 로드 실패")
        return

    # MVTec AD 형식 썸네일
    classes = meta["defect_classes"]
    if not classes:
        return
    test_dir = root / "test"
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
                img = Image.open(images[0]).convert("RGB").resize((150, 150), Image.LANCZOS)
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


def _clear_dataset_state() -> None:
    st.session_state["dataset_path"] = None
    st.session_state["dataset_meta"] = None

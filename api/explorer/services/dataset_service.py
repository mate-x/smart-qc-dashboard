"""
api/explorer/services/dataset_service.py

탭1 · 데이터셋:
    validate_dataset(path)           경로 검증 + 메타 구성 + 폴더 트리 생성
    get_thumbnail(dataset_path, class_name)  클래스 대표 썸네일 반환
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from api.explorer.state import get_state
from utils.dataset_converter import count_images, detect_ok_ng_dirs
from utils.image_utils import SUPPORTED_FORMATS

_DEFAULT_TRAIN_RATIO = 0.8

_NON_IMAGE_EXTS = {
    ".txt", ".json", ".yaml", ".yml", ".csv", ".log", ".md",
    ".xml", ".ini", ".cfg", ".toml", ".zip", ".tar", ".gz",
    ".py", ".sh", ".bat", ".exe", ".db", ".npy", ".npz",
}


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def validate_dataset(path: str, product_name: str = "") -> dict:
    """
    경로 검증 후 dataset_meta + folder_tree 를 포함한 dict 반환.

    Raises:
        FileNotFoundError: 경로 없음
        ValueError:        지원하지 않는 구조 또는 유효 이미지 없음
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")

    train_good = root / "train" / "good"

    if not train_good.is_dir():
        ok_dir, ng_dir = detect_ok_ng_dirs(root)
        if ok_dir is not None:
            meta = _build_oking_meta(root, ok_dir, ng_dir)
            if meta["train_good_count"] == 0:
                raise ValueError(f"`{ok_dir.name}/` 폴더에 유효한 이미지가 없습니다.")
            meta["folder_tree"] = _build_tree_text(root, meta)
            state = get_state()
            state["dataset_path"] = path
            state["product_name"] = product_name
            return meta
        raise ValueError(
            "지원하지 않는 폴더 구조입니다. "
            "OK/NG 형식: OK/, NG/ 폴더가 있어야 합니다. "
            "MVTec AD 형식: train/good/, test/ 폴더가 있어야 합니다."
        )

    test_dir = root / "test"
    if not test_dir.is_dir():
        raise ValueError("누락된 폴더: test/")

    has_train_image = any(
        f.suffix.lower() in SUPPORTED_FORMATS for f in train_good.iterdir()
    )
    if not has_train_image:
        raise ValueError("train/good/ 에 유효한 이미지가 없습니다.")

    meta = _build_mvtec_meta(root)
    meta["folder_tree"] = _build_tree_text(root, meta)
    state = get_state()
    state["dataset_path"] = path
    state["product_name"] = product_name
    return meta


def get_thumbnail(dataset_path: str, class_name: str) -> Image.Image:
    """
    클래스 대표 썸네일(150×150 RGB PNG) 반환.

    탐색 순서:
      1. {dataset_path}/test/{class_name}/  (MVTec AD)
      2. {dataset_path}/{class_name}/       (OK/NG)

    Raises:
        LookupError: 디렉토리 없거나 이미지 없음
    """
    root = Path(dataset_path)

    test_cls_dir = root / "test" / class_name
    if test_cls_dir.is_dir():
        img_dir = test_cls_dir
    else:
        direct_dir = root / class_name
        if direct_dir.is_dir():
            img_dir = direct_dir
        else:
            raise LookupError(f"클래스 디렉토리를 찾을 수 없습니다: {class_name}")

    images = sorted(
        f for f in img_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
    )
    if not images:
        raise LookupError(f"{class_name}: 이미지 없음")

    return Image.open(images[0]).convert("RGB").resize((150, 150), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_oking_meta(root: Path, ok_dir: Path, ng_dir: "Path | None") -> dict:
    ok_images = sorted(
        f for f in ok_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
    )
    ok_count = len(ok_images)
    ng_count = count_images(ng_dir) if ng_dir else 0

    train_n = max(1, int(ok_count * _DEFAULT_TRAIN_RATIO)) if ok_count else 0
    test_good_n = max(0, ok_count - train_n)

    channels = 3
    if ok_images:
        try:
            with Image.open(ok_images[0]) as img:
                channels = 1 if img.mode == "L" else 3
        except Exception:
            channels = 3

    ng_key = ng_dir.name.lower() if ng_dir else "ng"
    test_counts: dict[str, int] = {"good": test_good_n}
    defect_classes: list[str] = []
    if ng_dir and ng_count > 0:
        test_counts[ng_key] = ng_count
        defect_classes.append(ng_key)

    found_formats: set[str] = set()
    for f in ok_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            found_formats.add(f.suffix.lower())

    return {
        "dataset_format":  "oking",
        "channels":        channels,
        "train_good_count": train_n,
        "test_counts":     test_counts,
        "gt_counts":       {},
        "total_test_count": sum(test_counts.values()),
        "defect_classes":  defect_classes,
        "supported_formats": sorted(found_formats),
        "has_invalid_files": False,
        "invalid_file_count": 0,
        "oking_ok_dir":    ok_dir.name,
        "oking_ng_dir":    ng_dir.name if ng_dir else None,
        "oking_ok_count":  ok_count,
        "oking_ng_count":  ng_count,
        "train_ratio":     _DEFAULT_TRAIN_RATIO,
    }


def _build_mvtec_meta(root: Path) -> dict:
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
            if not f.is_file() or f.name.startswith("."):
                continue
            ext = f.suffix.lower()
            if ext in SUPPORTED_FORMATS:
                found_formats.add(ext)
            elif ext and ext not in _NON_IMAGE_EXTS:
                invalid_count += 1

    return {
        "dataset_format":  "mvtec",
        "channels":        channels,
        "train_good_count": len(train_images),
        "test_counts":     test_counts,
        "gt_counts":       gt_counts,
        "total_test_count": sum(test_counts.values()),
        "defect_classes":  defect_classes,
        "supported_formats": sorted(found_formats),
        "has_invalid_files": invalid_count > 0,
        "invalid_file_count": invalid_count,
        "oking_ok_dir":    None,
        "oking_ng_dir":    None,
        "oking_ok_count":  None,
        "oking_ng_count":  None,
        "train_ratio":     None,
    }


def _build_tree_text(root: Path, meta: dict) -> str:
    lines: list[str] = [f"📂 {root.name}/"]

    if meta.get("dataset_format") == "oking":
        ok_name = meta.get("oking_ok_dir", "OK")
        ng_name = meta.get("oking_ng_dir")
        ok_n    = meta.get("oking_ok_count", 0)
        ng_n    = meta.get("oking_ng_count", 0)
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

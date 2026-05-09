from __future__ import annotations

from pathlib import Path

from PIL import Image

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_dataset(dataset_path: str) -> dict:
    """
    MVTec AD 형식 데이터셋 스캔 → dataset_meta 반환.
    00_Global_Context 1.5절 스키마.

    Raises:
        ValueError: 필수 폴더 구조 미충족 또는 유효한 이미지 없음
    """
    root = Path(dataset_path).resolve()

    train_good = root / "train" / "good"
    test_dir = root / "test"

    if not train_good.exists():
        raise ValueError(
            f"MVTec AD 구조 미충족: train/good/ 디렉터리가 없습니다 ({root})"
        )
    if not test_dir.exists():
        raise ValueError(
            f"MVTec AD 구조 미충족: test/ 디렉터리가 없습니다 ({root})"
        )

    train_good_images = _list_images(train_good)
    if not train_good_images:
        raise ValueError(f"train/good/ 에 유효한 이미지가 없습니다 ({train_good})")
    train_good_count = len(train_good_images)

    # test/ 하위 클래스별 이미지 수
    test_counts: dict[str, int] = {}
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir():
            test_counts[class_dir.name] = len(_list_images(class_dir))

    total_test_count = sum(test_counts.values())
    defect_classes = [c for c in test_counts if c != "good"]

    # ground_truth/ 클래스별 마스크 수
    gt_dir = root / "ground_truth"
    gt_counts: dict[str, int] = {}
    if gt_dir.exists():
        for class_dir in sorted(gt_dir.iterdir()):
            if class_dir.is_dir():
                gt_counts[class_dir.name] = len(_list_images(class_dir))

    # 채널 감지 (train/good 첫 번째 이미지 기준)
    channels = _detect_channels(train_good_images[0])

    # 지원 포맷 감지 및 비지원 파일 존재 여부
    all_files: list[Path] = list(train_good.rglob("*"))
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            all_files.extend(class_dir.rglob("*"))

    found_formats: set[str] = set()
    has_invalid = False
    for f in all_files:
        if f.is_file():
            ext = f.suffix.lower()
            if ext in SUPPORTED_FORMATS:
                found_formats.add(ext)
            elif ext:
                has_invalid = True

    return {
        "dataset_path": str(root),
        "train_good_count": train_good_count,
        "test_counts": test_counts,
        "gt_counts": gt_counts,
        "total_test_count": total_test_count,
        "channels": channels,
        "defect_classes": defect_classes,
        "supported_formats": sorted(found_formats),
        "has_invalid_files": has_invalid,
    }


def _list_images(directory: Path) -> list[Path]:
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]


def _detect_channels(image_path: Path) -> int:
    try:
        img = Image.open(image_path)
        return 1 if img.mode in ("L", "1") else 3
    except Exception:
        return 3

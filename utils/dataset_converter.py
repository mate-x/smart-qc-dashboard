from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def detect_ok_ng_dirs(root: Path) -> tuple[Path | None, Path | None]:
    """
    루트 디렉터리에서 OK/NG 폴더를 대소문자 무관하게 탐색.

    반환: (ok_dir, ng_dir)
      ok_dir  — OK 계열 폴더 (OK, ok, Good, GOOD, normal ...)
      ng_dir  — NG 계열 폴더 (NG, ng, Bad, BAD, defect ...)
      찾지 못하면 None

    OK/NG 두 폴더가 모두 없으면 (None, None) 반환.
    OK만 있고 NG가 없어도 ok_dir만 반환 (NG-less 학습 허용).
    """
    OK_ALIASES = {"ok", "good", "normal", "pass", "neg"}
    NG_ALIASES = {"ng", "bad", "defect", "fail", "abnormal", "anomaly", "pos"}

    ok_dir = ng_dir = None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        low = d.name.lower()
        if low in OK_ALIASES:
            ok_dir = d
        elif low in NG_ALIASES:
            ng_dir = d

    return ok_dir, ng_dir


def count_images(directory: Path) -> int:
    """디렉터리 내 지원 이미지 파일 수 반환."""
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )


def convert_oking_to_mvtec(
    src_path: str,
    output_path: str,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    ng_class_name: str = "ng",
) -> dict:
    """
    OK/NG 폴더 구조를 MVTec AD 형식으로 변환.

    원본 파일은 건드리지 않고 하드링크(같은 드라이브) 또는
    파일 복사(다른 드라이브)로 output_path 에 새 구조를 생성.

    반환 dict:
      {
        "output_path":      str,
        "train_good_count": int,
        "test_good_count":  int,
        "test_ng_count":    int,
        "used_hardlink":    bool,
      }

    Raises:
        ValueError: OK 폴더가 없거나 이미지가 0장인 경우
    """
    src = Path(src_path).resolve()
    out = Path(output_path).resolve()

    ok_dir, ng_dir = detect_ok_ng_dirs(src)
    if ok_dir is None:
        raise ValueError(f"OK 계열 폴더를 찾을 수 없습니다: {src}")

    ok_images = sorted(
        f for f in ok_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    if not ok_images:
        raise ValueError(f"OK 폴더에 이미지가 없습니다: {ok_dir}")

    ng_images: list[Path] = []
    if ng_dir is not None:
        ng_images = sorted(
            f for f in ng_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )

    # OK 이미지를 train/test/good 으로 분할
    rng = random.Random(random_seed)
    shuffled = list(ok_images)
    rng.shuffle(shuffled)
    split_idx        = max(1, int(len(shuffled) * train_ratio))
    train_images     = shuffled[:split_idx]
    test_good_images = shuffled[split_idx:]

    # 출력 디렉터리 생성
    train_good_dir = out / "train" / "good"
    test_good_dir  = out / "test" / "good"
    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)
    if ng_images:
        (out / "test" / ng_class_name).mkdir(parents=True, exist_ok=True)

    used_hardlink = True

    def _link_or_copy(src_file: Path, dst_file: Path) -> bool:
        """hardlink 시도 → 실패 시 copy. hardlink 성공 여부 반환."""
        if dst_file.exists():
            return True
        try:
            os.link(src_file, dst_file)
            return True
        except OSError:
            shutil.copy2(src_file, dst_file)
            return False

    for f in train_images:
        if not _link_or_copy(f, train_good_dir / f.name):
            used_hardlink = False
    for f in test_good_images:
        if not _link_or_copy(f, test_good_dir / f.name):
            used_hardlink = False
    for f in ng_images:
        dst = out / "test" / ng_class_name / f.name
        if not _link_or_copy(f, dst):
            used_hardlink = False

    return {
        "output_path":      str(out),
        "train_good_count": len(train_images),
        "test_good_count":  len(test_good_images),
        "test_ng_count":    len(ng_images),
        "used_hardlink":    used_hardlink,
    }

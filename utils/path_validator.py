from pathlib import Path

ALLOWED_BASE_DIR = Path("/app/dataset")


def validate_dataset_path(user_input: str) -> Path:
    """
    사용자 입력 데이터셋 경로 검증 (10_Security B.1 — 경로 탈출 방지).
    Returns: resolve()된 절대 경로
    Raises: ValueError (빈 입력, 존재하지 않는 경로, 디렉터리 아님, 허용 범위 외 경로)
    """
    if not user_input or not user_input.strip():
        raise ValueError("경로를 입력해 주세요.")

    try:
        p = Path(user_input.strip()).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"유효하지 않은 경로입니다: {e}")

    if not str(p).startswith(str(ALLOWED_BASE_DIR)):
        raise ValueError(f"/app/dataset 하위 경로만 허용됩니다: {p}")

    if not p.exists():
        raise ValueError(f"경로가 존재하지 않습니다: {p}")

    if not p.is_dir():
        raise ValueError(f"디렉터리가 아닙니다: {p}")

    return p

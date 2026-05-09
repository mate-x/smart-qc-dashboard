import json
from pathlib import Path

HISTORY_FILE = Path("./experiments/history.json")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(records: list[dict]) -> None:
    # R-ATOMIC-01: tmpfile → rename
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_FILE)


def append_experiment(record: dict) -> None:
    records = load_history()
    records.append(record)
    save_history(records)


def validate_imagenet_penalty_dir(
    base: str = "./dataset/imagenet_penalty",
) -> None:
    p = Path(base)
    if not p.exists():
        raise ValueError(
            f"ImageNet penalty 디렉터리가 존재하지 않습니다: {p.resolve()}"
        )
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = [f for f in p.rglob("*") if f.suffix.lower() in supported]
    if not images:
        raise ValueError(
            f"ImageNet penalty 디렉터리에 이미지가 없습니다: {p.resolve()}"
        )

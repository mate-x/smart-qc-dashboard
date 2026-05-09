from pathlib import Path

REQUIRED_DIRS = [
    Path("./experiments"),
    Path("./models"),
    Path("./logs"),
    Path("./results"),
    Path("./dataset/imagenet_penalty"),
]


def ensure_required_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)

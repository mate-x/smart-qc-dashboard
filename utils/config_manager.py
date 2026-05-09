from pathlib import Path

import yaml


class ConfigLoadError(Exception):
    pass


def load_config(path: str = "./configs.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"ERR_CONFIG_LOAD_FAILED: {e}") from e


def save_config_section(
    section: str,
    data: dict,
    path: str = "./configs.yaml",
) -> None:
    # R-ATOMIC-01: tmpfile → rename
    config = load_config(path)
    config[section] = data
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    tmp.replace(p)


def get_preprocessing_config(path: str = "./configs.yaml") -> dict | None:
    return load_config(path).get("preprocessing")


def get_model_config(path: str = "./configs.yaml") -> dict | None:
    return load_config(path).get("model")

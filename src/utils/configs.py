from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(config_dir: str | Path = "configs") -> dict:
    config_dir = Path(config_dir)
    return {
        "data_config": load_yaml(config_dir / "data_config.yaml"),
        "model_config": load_yaml(config_dir / "model_config.yaml"),
        "retrieval_config": load_yaml(config_dir / "retrieval_config.yaml"),
    }

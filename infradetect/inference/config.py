"""Read trained-model YAML settings without importing training code."""

from __future__ import annotations

from pathlib import Path

import yaml


def config_from_file(path: Path) -> dict:
    """Resolve existing config-relative paths and check model class count."""

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data"]["root"] = str((path.parent / cfg["data"]["root"]).resolve())
    if cfg["data"].get("manifest"):
        cfg["data"]["manifest"] = str((path.parent / cfg["data"]["manifest"]).resolve())
    cfg["train"]["output_dir"] = str((path.parent / cfg["train"]["output_dir"]).resolve())
    if cfg["model"].get("thermal_checkpoint"):
        cfg["model"]["thermal_checkpoint"] = str(
            (path.parent / cfg["model"]["thermal_checkpoint"]).resolve()
        )
    if len(cfg["data"]["class_names"]) != cfg["model"]["classes"]:
        raise ValueError("class_names and model.classes disagree")
    return cfg

"""Tiny YAML config loader with single inheritance.

A config file may contain `base: other.yaml` (path relative to the file).
The base is loaded first and the keys of the current file are merged on
top of it recursively, so an ablation config only lists what it changes.
"""
from pathlib import Path

import yaml

from env.merge_env import merge_config


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    return merge_config(load_config(path.parent / base), cfg)


def density_for_progress(cfg: dict, progress: float) -> list:
    """Density range of the curriculum stage at training progress in [0, 1]."""
    stages = cfg.get("curriculum")
    if not stages:
        return cfg["env"]["density_range"]
    for stage in stages:
        if progress < stage["until"]:
            return stage["density_range"]
    return stages[-1]["density_range"]

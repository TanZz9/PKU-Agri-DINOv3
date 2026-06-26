from __future__ import annotations

from typing import Any, Dict

import yaml

from backbones.base import BaseBackbone
from paths import CONFIG_PATH, DINOV3_REPO, PROJECT_ROOT


def load_model_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    repo = cfg.get("dinov3_repo")
    if repo and not str(repo).startswith("/"):
        cfg["dinov3_repo"] = str((PROJECT_ROOT / repo).resolve())
    elif not cfg.get("dinov3_repo"):
        cfg["dinov3_repo"] = str(DINOV3_REPO)
    return cfg


def build_backbone(
    model_id: str,
    device: str = "cuda",
    checkpoint_override: str | None = None,
) -> BaseBackbone:
    cfg = load_model_config()
    if model_id not in cfg["models"]:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(cfg['models'])}")
    spec = dict(cfg["models"][model_id])
    family = spec["family"]
    if checkpoint_override:
        spec["checkpoint"] = checkpoint_override
        spec["weights"] = checkpoint_override

    if family == "dinov3":
        from backbones.dinov3 import DINOv3Backbone

        repo = cfg.get("dinov3_repo") or str(DINOV3_REPO)
        return DINOv3Backbone(model_id, spec, repo, device=device)
    raise ValueError(f"Unsupported family: {family}")

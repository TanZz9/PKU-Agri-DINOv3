from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms

from backbones.base import BaseBackbone


def _load_checkpoint(weights_path: str | Path):
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def _extract_backbone_state_dict(ckpt) -> dict[str, torch.Tensor]:
    """Support flat backbone .pth and finetune checkpoints with model/backbone.* keys."""
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        elif "teacher" in ckpt and isinstance(ckpt["teacher"], dict):
            ckpt = ckpt["teacher"]
        elif "student" in ckpt and isinstance(ckpt["student"], dict):
            ckpt = ckpt["student"]

    state: dict[str, torch.Tensor] = {}
    for key, value in ckpt.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = key.replace("module.", "")
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        if name.startswith("head."):
            continue
        state[name] = value
    return state


class DINOv3Backbone(BaseBackbone):
    def __init__(self, model_id: str, spec: Dict[str, Any], repo: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._feature_dim = spec.get("feature_dim", 768)
        crop = spec.get("crop_size", 224)
        resize = spec.get("resize_size", 256)
        self.crop_size = crop

        weights_path = spec.get("weights") or spec.get("checkpoint")
        repo_path = Path(repo)
        if not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv3 repo not found: {repo_path}. "
                "Clone https://github.com/facebookresearch/dinov3 and set DINOV3_REPO."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        self.model = torch.hub.load(
            str(repo_path),
            spec["hub_name"],
            source="local",
            pretrained=False,
        )
        if weights_path:
            ckpt = _load_checkpoint(weights_path)
            state_dict = _extract_backbone_state_dict(ckpt)
            if not state_dict:
                raise ValueError(f"No backbone tensors found in checkpoint: {weights_path}")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(
                    f"[DINOv3Backbone] loaded {weights_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
        self.model.eval().to(self.device)

        interpolation = transforms.InterpolationMode.BICUBIC
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=interpolation),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def get_preprocess(self):
        return self.preprocess

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(images, is_training=True)
        feats = out["x_norm_clstoken"]
        return F.normalize(feats.float(), dim=-1)

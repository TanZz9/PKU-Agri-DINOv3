from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseBackbone(ABC):
    model_id: str

    @abstractmethod
    def get_preprocess(self):
        """Return torchvision transform for eval."""

    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Input BCHW tensor on device; return BxD features."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        pass

    def to(self, device: torch.device):
        return self

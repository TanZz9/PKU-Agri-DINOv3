from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backbones.base import BaseBackbone
from data.path_utils import remap_split_items, resolve_image_path
from paths import CACHE_DIR, IMAGE_EXTS, SPLIT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagePathDataset(Dataset):
    def __init__(self, items: List[dict], transform, crop_size: int = 224):
        self.items = items
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        path = resolve_image_path(item["path"])
        label = item["label"]
        try:
            p = Path(path)
            if not p.exists() or p.suffix.lower() not in IMAGE_EXTS:
                raise FileNotFoundError("missing_or_not_image")
            with Image.open(p) as img:
                tensor = self.transform(img.convert("RGB"))
            return tensor, label, path, ""
        except Exception as e:
            tensor = torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32)
            return tensor, label, path, repr(e)


def _embed_split(
    items: List[dict],
    backbone: BaseBackbone,
    batch_size: int,
    workers: int,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    transform = backbone.get_preprocess()
    dataset = ImagePathDataset(items, transform, crop_size=crop_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    feats_list = []
    labels_list = []
    valid_paths = []
    device = getattr(backbone, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for images, labels, paths, errors in tqdm(loader, desc="extract", leave=False):
        valid_mask = [e == "" for e in errors]
        if any(valid_mask):
            images = images.to(device, non_blocking=True)
            batch_feats = backbone.extract_features(images)
            batch_feats = batch_feats.cpu().numpy().astype("float32", copy=False)
            valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
            feats_list.append(batch_feats[valid_indices])
            labels_list.extend([labels[i].item() for i in valid_indices])
            valid_paths.extend([paths[i] for i in valid_indices])

    if feats_list:
        X = np.concatenate(feats_list, axis=0)
        y = np.array(labels_list, dtype=np.int64)
    else:
        X = np.zeros((0, backbone.feature_dim), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
    return X, y, valid_paths


def load_split(dataset: str, seed: int = 42) -> dict:
    path = SPLIT_DIR / f"{dataset}_seed{seed}.json"
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    split["train"] = remap_split_items(split["train"])
    split["test"] = remap_split_items(split["test"])
    return split


def extract_and_cache(
    model_id: str,
    dataset: str,
    backbone: BaseBackbone,
    seed: int = 42,
    batch_size: int = 64,
    workers: int = 4,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> dict:
    split = load_split(dataset, seed)
    cache_path = CACHE_DIR / model_id / f"{dataset}_seed{seed}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and not refresh_cache and cache_path.exists():
        data = np.load(cache_path)
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "from_cache": True,
        }

    crop_size = getattr(backbone, "crop_size", None)
    if crop_size is None and hasattr(backbone, "preprocess"):
        from torchvision.transforms import CenterCrop

        crop_size = 224
        for t in backbone.preprocess.transforms:
            if isinstance(t, CenterCrop):
                crop_size = t.size if isinstance(t.size, int) else t.size[0]
                break
    crop_size = crop_size or 224

    X_train, y_train, _ = _embed_split(
        split["train"], backbone, batch_size, workers, crop_size
    )
    X_test, y_test, _ = _embed_split(
        split["test"], backbone, batch_size, workers, crop_size
    )

    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "from_cache": False,
    }

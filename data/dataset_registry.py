"""Load Agri-dataset subfolders into unified samples: path, class_name."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from paths import AGRI_ROOT, IMAGE_EXTS, WCS_ROOT

Sample = Dict[str, str]  # path, class_name


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _iter_images(root: Path):
    for p in root.rglob("*"):
        if _is_image(p):
            yield p


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _imagefolder_under(roots: List[Path], class_from_parent: bool = True) -> List[Sample]:
    """Collect images where immediate parent dir name is the class."""
    samples: List[Sample] = []
    for root in roots:
        if not root.exists():
            continue
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            images = list(_iter_images(class_dir))
            if not images:
                continue
            class_name = class_dir.name
            for img in images:
                samples.append({"path": str(img.resolve()), "class_name": class_name})
    return samples


def _imagefolder_nested(root: Path, depth: int = 2) -> List[Sample]:
    """Class = leaf folder name at given depth from root."""
    samples: List[Sample] = []
    if not root.exists():
        return samples

    def walk(current: Path, level: int, parts: List[str]):
        if level == depth:
            if any(_is_image(current / f) for f in current.iterdir() if f.is_file()):
                class_name = parts[-1] if parts else current.name
                for img in _iter_images(current):
                    samples.append({"path": str(img.resolve()), "class_name": class_name})
            return
        for child in sorted(current.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                walk(child, level + 1, parts + [child.name])

    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            walk(child, 1, [child.name])
    return samples


def _assign_label_ids(samples: List[Sample]) -> List[Sample]:
    classes = sorted({s["class_name"] for s in samples})
    name_to_id = {n: i for i, n in enumerate(classes)}
    out = []
    for s in samples:
        out.append(
            {
                "path": s["path"],
                "class_name": s["class_name"],
                "label_id": name_to_id[s["class_name"]],
            }
        )
    return out


def load_agrivision4() -> List[Sample]:
    root = AGRI_ROOT / "AgriVIsion4" / "Orginal_Dataset"
    samples = []
    for crop_dir in sorted(root.iterdir()) if root.exists() else []:
        if not crop_dir.is_dir():
            continue
        for class_dir in sorted(crop_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = list(_iter_images(class_dir))
            if not imgs:
                continue
            class_name = f"{crop_dir.name}/{class_dir.name}"
            for img in imgs:
                samples.append({"path": str(img.resolve()), "class_name": class_name})
    return _assign_label_ids(samples)


def load_corn_leaf() -> List[Sample]:
    root = AGRI_ROOT / "CornLeafDiseaseClassificationDataset"
    return _assign_label_ids(_imagefolder_under([root]))


LOADERS: Dict[str, Callable[[], List[Sample]]] = {
    "agrivision4": load_agrivision4,
    "corn_leaf": load_corn_leaf,
}


def load_samples(dataset_name: str) -> List[Sample]:
    if dataset_name not in LOADERS:
        raise KeyError(f"Unknown dataset: {dataset_name}. Available: {list(LOADERS)}")
    return LOADERS[dataset_name]()

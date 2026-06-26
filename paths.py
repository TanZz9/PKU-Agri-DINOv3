"""Project paths and dataset lists. Override roots via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset roots (override for your machine)
AGRI_ROOT = Path(
    os.environ.get(
        "AGRI_ROOT",
        "/data/tanzhi/Nongye_new/datasets/downstream/Agri-dataset",
    )
)
WCS_ROOT = Path(os.environ.get("WCS_ROOT", "/data/tanzhi/Nongye/data/wcs"))

# Roots baked into bundled split JSON files (used for path remapping)
ORIGINAL_AGRI_ROOT = Path(
    os.environ.get(
        "ORIGINAL_AGRI_ROOT",
        "/data/tanzhi/Nongye_new/datasets/downstream/Agri-dataset",
    )
)
ORIGINAL_WCS_ROOT = Path(
    os.environ.get("ORIGINAL_WCS_ROOT", "/data/tanzhi/Nongye/data/wcs")
)

# External DINOv3 repo (Meta hub, loaded via torch.hub.load source=local)
DINOV3_REPO = Path(
    os.environ.get("DINOV3_REPO", str(PROJECT_ROOT.parent / "dinov3"))
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ALL_DATASETS = [
    "plantdoc",
    "agrivision4",
    "corn_leaf",
    "seasonal_corn",
    "cucurbit",
    "manalagi_apple",
    "rice_leaf",
    "multicrop",
    "tcp",
    "crops_leafs",
    "corn_pests_early",
    "tom24",
    "wcs_cucumber",
    "wcs_tomato",
]

ALL_MODELS = ["dinov3_vitb16"]

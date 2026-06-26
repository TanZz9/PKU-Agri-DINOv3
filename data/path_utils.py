"""Remap image paths from bundled splits to local dataset roots."""

from __future__ import annotations

from pathlib import Path

from paths import AGRI_ROOT, ORIGINAL_AGRI_ROOT, ORIGINAL_WCS_ROOT, WCS_ROOT


def resolve_image_path(path: str) -> str:
    """Return an existing local path, remapping bundled absolute paths if needed."""
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate.resolve())

    path_str = str(path)
    if path_str.startswith(str(ORIGINAL_AGRI_ROOT)):
        remapped = AGRI_ROOT / Path(path_str).relative_to(ORIGINAL_AGRI_ROOT)
        if remapped.is_file():
            return str(remapped.resolve())

    return path_str


def remap_split_items(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        remapped = dict(item)
        remapped["path"] = resolve_image_path(item["path"])
        out.append(remapped)
    return out

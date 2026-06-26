#!/usr/bin/env python3
"""Extract DINOv3 CLS features from image(s) for downstream use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backbones.factory import build_backbone


def load_image(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def collect_images(input_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")
    images = sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Agri-DINOv3 feature inference")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--checkpoint", required=True, help="DINOv3 checkpoint (.pth)")
    parser.add_argument("--model-id", default="dinov3_vitb16")
    parser.add_argument("--output", default=None, help="Output .npz or .json path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_path = Path(args.input)
    image_paths = collect_images(input_path)

    backbone = build_backbone(
        args.model_id,
        device=args.device,
        checkpoint_override=args.checkpoint,
    )
    transform = backbone.get_preprocess()
    device = backbone.device

    feats_list = []
    paths_out = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        tensors = torch.stack([load_image(p, transform) for p in batch_paths]).to(device)
        with torch.inference_mode():
            feats = backbone.extract_features(tensors).cpu().numpy().astype("float32")
        feats_list.append(feats)
        paths_out.extend(str(p.resolve()) for p in batch_paths)

    features = np.concatenate(feats_list, axis=0)
    output = Path(
        args.output
        or (PROJECT_ROOT / "results" / f"features_{input_path.stem}.npz")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".json":
        payload = [
            {"path": p, "feature": f.tolist()}
            for p, f in zip(paths_out, features)
        ]
        with output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        np.savez_compressed(output, paths=np.array(paths_out), features=features)

    print(f"Saved {len(paths_out)} features -> {output}")
    print(f"feature shape: {features.shape}")


if __name__ == "__main__":
    main()
